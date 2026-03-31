"""
trainer.py  —  Complete two-phase training pipeline.

Phase 1 — Teacher Training
───────────────────────────
  Train TransformerLM normally on the target task (language modelling).
  Optimise:  L_teacher = CrossEntropy(logits, targets)
  Output:    frozen teacher weights  T*

Phase 2 — Hypernetwork Training
────────────────────────────────
  Train Hypernetwork H while T's LN / embedding weights are shared but
  its block weights are replaced by H's output.

  Optimise:  L = λ_task·L_task + λ_recon·L_recon + λ_distill·L_distill

  Three sub-phases:
    2a. Warm-up:  train only with L_recon  (pure weight matching)
    2b. Joint:    add L_task + L_distill
    2c. Fine-tune: anneal λ_recon to 0, focus on L_task

Data Pipeline
─────────────
  Uses HuggingFace datasets (wikitext2) or a simple synthetic dataset
  for testing without internet access.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .config import ExperimentConfig, TrainingConfig
from .hypernetwork import Hypernetwork, build_hypernetwork
from .losses import HypernetworkLoss, compute_perplexity, task_loss
from .target_model import TransformerLM, build_target_model

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Synthetic dataset (no internet required)
# ---------------------------------------------------------------------------

class SyntheticTokenDataset(Dataset):
    """Random token dataset for smoke-testing the pipeline."""

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int, seed: int = 42):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        self.data = torch.randint(
            0, vocab_size, (n_samples, seq_len), generator=rng
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.data[idx]
        return {"input_ids": x, "labels": x}


def make_dataloaders(
    cfg: ExperimentConfig,
) -> Tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader).  Falls back to synthetic if no data."""

    tc    = cfg.target
    tr    = cfg.training

    try:
        from datasets import load_dataset  # type: ignore
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token

        ds  = load_dataset("wikitext", "wikitext-2-raw-v1")

        def tokenize(examples):
            return tok(
                examples["text"], truncation=True,
                max_length=tr.seq_len, padding="max_length",
            )

        train_ds = ds["train"].map(tokenize, batched=True, remove_columns=["text"])
        val_ds   = ds["validation"].map(tokenize, batched=True, remove_columns=["text"])

        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        val_ds.set_format(type="torch",   columns=["input_ids", "attention_mask"])

        def collate(batch):
            ids = torch.stack([b["input_ids"] for b in batch])
            return {"input_ids": ids, "labels": ids.clone()}

        train_dl = DataLoader(
            train_ds, batch_size=tr.batch_size, shuffle=True,
            num_workers=tr.num_workers, collate_fn=collate,
        )
        val_dl = DataLoader(
            val_ds, batch_size=tr.batch_size, shuffle=False,
            num_workers=tr.num_workers, collate_fn=collate,
        )
        logger.info("✅  Loaded WikiText-2 dataset")
        return train_dl, val_dl

    except Exception as e:
        logger.warning(f"⚠️  Dataset load failed ({e}). Using synthetic data.")
        train_ds = SyntheticTokenDataset(tc.vocab_size, tr.seq_len, 10_000)
        val_ds   = SyntheticTokenDataset(tc.vocab_size, tr.seq_len, 1_000, seed=99)

        def collate_syn(batch):
            ids = torch.stack([b["input_ids"] for b in batch])
            return {"input_ids": ids, "labels": ids.clone()}

        train_dl = DataLoader(
            train_ds, batch_size=tr.batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_syn,
        )
        val_dl = DataLoader(
            val_ds, batch_size=tr.batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_syn,
        )
        return train_dl, val_dl


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def cosine_schedule_with_warmup(
    step:         int,
    warmup_steps: int,
    max_steps:    int,
    max_lr:       float,
    min_lr:       float,
) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


# ---------------------------------------------------------------------------
# Phase 1: Teacher training
# ---------------------------------------------------------------------------

class TeacherTrainer:
    """Trains the standalone TransformerLM (teacher model)."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg    = cfg
        self.device = torch.device(cfg.training.device
                                   if torch.cuda.is_available() else "cpu")
        self.model  = build_target_model(cfg.target).to(self.device)
        self.optim  = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.lr,
            betas=(cfg.training.beta1, cfg.training.beta2),
            weight_decay=cfg.training.weight_decay,
        )
        self.scaler = GradScaler(enabled=cfg.training.mixed_precision
                                  and self.device.type == "cuda")
        self.step   = 0
        self._best_val_loss = float("inf")

        os.makedirs(cfg.training.ckpt_dir, exist_ok=True)
        logger.info(f"Teacher:  {self.model.param_count/1e6:.2f} M params  "
                    f"on {self.device}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> TransformerLM:
        """Run teacher training. Returns trained model."""
        cfg = self.cfg.training
        self.model.train()

        data_iter: Iterator = iter(train_loader)
        log_losses: List[float] = []
        t0 = time.time()

        while self.step < cfg.teacher_steps:
            # Fetch batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch     = next(data_iter)

            input_ids = batch["input_ids"].to(self.device)
            labels    = batch["labels"].to(self.device)

            # LR schedule
            lr = cosine_schedule_with_warmup(
                self.step, cfg.warmup_steps, cfg.teacher_steps,
                cfg.lr, cfg.min_lr,
            )
            set_lr(self.optim, lr)

            # Forward + backward
            self.optim.zero_grad()
            amp_dtype = (torch.float16 if cfg.dtype == "float16" else torch.bfloat16)

            with autocast(
                enabled=cfg.mixed_precision and self.device.type == "cuda",
                dtype=amp_dtype,
            ):
                _, loss = self.model(input_ids, labels=labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.scaler.step(self.optim)
            self.scaler.update()

            log_losses.append(loss.item())
            self.step += 1

            # Logging
            if self.step % 200 == 0:
                avg   = sum(log_losses[-200:]) / len(log_losses[-200:])
                elapsed = time.time() - t0
                ppl   = math.exp(min(avg, 20))
                logger.info(
                    f"[Teacher] step {self.step:>6}/{cfg.teacher_steps}  "
                    f"lr={lr:.2e}  loss={avg:.4f}  ppl={ppl:.1f}  "
                    f"t={elapsed:.0f}s"
                )

            # Validation + checkpoint
            if self.step % cfg.eval_every == 0:
                val_loss = self._evaluate(val_loader)
                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    self._save("teacher_best.pt")
                self.model.train()

            if self.step % cfg.save_every == 0:
                self._save(f"teacher_step{self.step}.pt")

        self._save("teacher_final.pt")
        logger.info(f"✅  Teacher training complete.  Best val loss = {self._best_val_loss:.4f}")
        return self.model

    def _evaluate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch["input_ids"].to(self.device)
                lbls = batch["labels"].to(self.device)
                _, loss = self.model(ids, labels=lbls)
                if loss is not None:
                    total_loss += loss.item()
                    n_batches  += 1
                if n_batches >= 50:
                    break
        avg = total_loss / max(n_batches, 1)
        logger.info(f"  [eval] val_loss={avg:.4f}  ppl={math.exp(min(avg,20)):.1f}")
        return avg

    def _save(self, name: str) -> None:
        path = Path(self.cfg.training.ckpt_dir) / name
        torch.save({
            "step":       self.step,
            "model":      self.model.state_dict(),
            "optim":      self.optim.state_dict(),
            "val_loss":   self._best_val_loss,
        }, path)


# ---------------------------------------------------------------------------
# Phase 2: Hypernetwork training
# ---------------------------------------------------------------------------

class HypernetworkTrainer:
    """
    Trains the Hypernetwork H given a frozen teacher T*.

    The student model reuses the teacher's embeddings and LN layers;
    only the transformer block weights are generated by H.
    """

    def __init__(
        self,
        cfg:     ExperimentConfig,
        teacher: TransformerLM,
    ):
        self.cfg    = cfg
        self.device = torch.device(cfg.training.device
                                   if torch.cuda.is_available() else "cpu")

        # Frozen teacher
        self.teacher = teacher.to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        # Student model (shares non-generated weights with teacher)
        self.student = build_target_model(cfg.target).to(self.device)
        self._share_fixed_weights()

        # Hypernetwork
        self.hypernet = build_hypernetwork(cfg.target, cfg.hypernet).to(self.device)

        # Loss
        self.criterion = HypernetworkLoss(cfg.training)

        # Optimiser (only hypernetwork + shared student weights)
        train_params = list(self.hypernet.parameters())
        # Also allow LN params in student to adapt
        for block in self.student.blocks:
            train_params += list(block.ln1.parameters())
            train_params += list(block.ln2.parameters())
        train_params += list(self.student.ln_f.parameters())

        self.optim = torch.optim.AdamW(
            train_params,
            lr=cfg.training.lr,
            betas=(cfg.training.beta1, cfg.training.beta2),
            weight_decay=cfg.training.weight_decay,
        )
        self.scaler = GradScaler(
            enabled=cfg.training.mixed_precision and self.device.type == "cuda"
        )
        self.step   = 0
        self._best_val_loss = float("inf")

        logger.info(
            f"Hypernetwork: {self.hypernet.param_count/1e6:.3f} M params  "
            f"({self.hypernet.size_mb():.2f} MB FP32 | "
            f"{self.hypernet.size_mb(2):.2f} MB FP16)"
        )

    def _share_fixed_weights(self) -> None:
        """Copy teacher's embedding and LN weights to student (non-generated)."""
        sd = self.teacher.state_dict()
        # Share token/pos embeddings and final LN
        for key in ["token_embed.weight", "pos_embed.weight", "ln_f.weight", "ln_f.bias"]:
            if key in sd:
                student_param = dict(self.student.named_parameters()).get(key)
                if student_param is not None:
                    with torch.no_grad():
                        student_param.copy_(sd[key])
        # Copy per-block LN (these are not generated)
        for i, block in enumerate(self.student.blocks):
            for ln_name in ["ln1", "ln2"]:
                for suffix in ["weight", "bias"]:
                    src_key = f"blocks.{i}.{ln_name}.{suffix}"
                    if src_key in sd:
                        tgt = getattr(getattr(block, ln_name), suffix)
                        with torch.no_grad():
                            tgt.copy_(sd[src_key])

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> Tuple[Hypernetwork, TransformerLM]:
        """Train hypernetwork. Returns (hypernet, student_model)."""
        tr   = self.cfg.training
        t0   = time.time()

        # Warm-up phase boundaries
        warmup_recon_steps = min(5_000, tr.hypernetwork_steps // 10)

        data_iter = iter(train_loader)
        log_buf: Dict[str, List[float]] = {k: [] for k in
                                            ["total", "task", "recon", "distill"]}

        self.hypernet.train()
        self.student.train()

        while self.step < tr.hypernetwork_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch     = next(data_iter)

            input_ids = batch["input_ids"].to(self.device)
            labels    = batch["labels"].to(self.device)

            # Schedule
            lr = cosine_schedule_with_warmup(
                self.step, tr.warmup_steps, tr.hypernetwork_steps,
                tr.lr, tr.min_lr,
            )
            set_lr(self.optim, lr)

            # Dynamic loss weights: start recon-heavy, then shift to task
            if self.step < warmup_recon_steps:
                # Phase 2a: reconstruction warm-up
                lambda_task    = 0.0
                lambda_recon   = 1.0
                lambda_distill = 0.0
            elif self.step < tr.hypernetwork_steps // 2:
                # Phase 2b: joint training
                lambda_task    = tr.lambda_task
                lambda_recon   = tr.lambda_recon
                lambda_distill = tr.lambda_distill
            else:
                # Phase 2c: fine-tune — reduce recon pressure
                progress = (self.step - tr.hypernetwork_steps // 2) / (tr.hypernetwork_steps // 2)
                lambda_task    = tr.lambda_task
                lambda_recon   = tr.lambda_recon * (1 - progress) * 0.5
                lambda_distill = tr.lambda_distill

            self.optim.zero_grad()
            amp_dtype = torch.float16 if tr.dtype == "float16" else torch.bfloat16

            with autocast(
                enabled=tr.mixed_precision and self.device.type == "cuda",
                dtype=amp_dtype,
            ):
                # 1. Generate weights from hypernetwork
                gen_weights = self.hypernet.generate_all_weights()

                # 2. Student forward with generated weights
                logits_s, _ = self.student(input_ids, gen_weights)

                # 3. Teacher forward (no grad)
                with torch.no_grad():
                    logits_t, _ = self.teacher(input_ids)
                    tea_weights  = self.teacher.get_all_weight_dicts()

                # 4. Compute losses
                L_task    = task_loss(logits_s, labels)
                L_recon   = self._recon_loss(gen_weights, tea_weights)
                L_distill = self._distill_loss(logits_s, logits_t)

                total = (lambda_task    * L_task +
                         lambda_recon   * L_recon +
                         lambda_distill * L_distill)

            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.optim)
            nn.utils.clip_grad_norm_(
                list(self.hypernet.parameters()),
                tr.grad_clip,
            )
            self.scaler.step(self.optim)
            self.scaler.update()

            # Logging
            for k, v in [("total", total), ("task", L_task),
                          ("recon", L_recon), ("distill", L_distill)]:
                log_buf[k].append(v.item())
            self.step += 1

            if self.step % 200 == 0:
                avgs = {k: sum(v[-200:]) / len(v[-200:]) for k, v in log_buf.items()}
                ppl  = math.exp(min(avgs["task"], 20))
                logger.info(
                    f"[HyperNet] step {self.step:>6}/{tr.hypernetwork_steps}  "
                    f"lr={lr:.2e}  "
                    f"total={avgs['total']:.4f}  "
                    f"task={avgs['task']:.4f}  "
                    f"recon={avgs['recon']:.4f}  "
                    f"distill={avgs['distill']:.4f}  "
                    f"ppl={ppl:.1f}  "
                    f"t={time.time()-t0:.0f}s"
                )

            if self.step % tr.eval_every == 0:
                val_loss = self._evaluate(val_loader)
                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    self._save_best()
                self.hypernet.train()
                self.student.train()

            if self.step % tr.save_every == 0:
                self._save_checkpoint(f"hypernet_step{self.step}.pt")

        self._save_checkpoint("hypernet_final.pt")
        logger.info(f"✅  Hypernetwork training complete.  "
                    f"Best val loss = {self._best_val_loss:.4f}")
        return self.hypernet, self.student

    def _recon_loss(
        self,
        gen: List[Dict[str, torch.Tensor]],
        tea: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.device)
        count = 0
        for g, t in zip(gen, tea):
            for key in g:
                if key in t:
                    n     = g[key].numel()
                    total = total + ((g[key] - t[key].detach()) ** 2).sum() / n
                    count += 1
        return total / max(count, 1)

    def _distill_loss(
        self,
        logits_s: torch.Tensor,
        logits_t: torch.Tensor,
        T: float = 4.0,
    ) -> torch.Tensor:
        B, S, V = logits_s.shape
        ls  = (logits_s / T).view(-1, V)
        lt  = (logits_t / T).view(-1, V)
        kl  = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(ls, -1),
            torch.nn.functional.softmax(lt,  -1),
            reduction="batchmean",
        )
        return (T ** 2) * kl

    def _evaluate(self, val_loader: DataLoader) -> float:
        self.hypernet.eval()
        self.student.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            gen_weights = self.hypernet.generate_all_weights()
            for batch in val_loader:
                ids  = batch["input_ids"].to(self.device)
                lbls = batch["labels"].to(self.device)
                _, loss = self.student(ids, gen_weights, lbls)
                if loss is not None:
                    total_loss += loss.item()
                    n += 1
                if n >= 50:
                    break
        avg = total_loss / max(n, 1)
        logger.info(f"  [eval] val_loss={avg:.4f}  ppl={math.exp(min(avg,20)):.1f}")
        return avg

    def _save_best(self) -> None:
        self._save_checkpoint("hypernet_best.pt")

    def _save_checkpoint(self, name: str) -> None:
        path = Path(self.cfg.training.ckpt_dir) / name
        torch.save({
            "step":     self.step,
            "hypernet": self.hypernet.state_dict(),
            "student":  self.student.state_dict(),
            "optim":    self.optim.state_dict(),
            "val_loss": self._best_val_loss,
            "cfg":      self.cfg,
        }, path)
        logger.info(f"  💾  Saved → {path}")


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

def run_full_pipeline(cfg: ExperimentConfig) -> Tuple[TransformerLM, Hypernetwork]:
    """
    Runs both phases end-to-end.

    Phase 1: train teacher
    Phase 2: train hypernetwork
    Returns (teacher, hypernet)
    """
    logger.info(cfg.summary())

    torch.manual_seed(cfg.seed)

    train_dl, val_dl = make_dataloaders(cfg)

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("PHASE 1  —  Teacher Training")
    logger.info("="*60)

    teacher_trainer = TeacherTrainer(cfg)
    teacher         = teacher_trainer.train(train_dl, val_dl)

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("PHASE 2  —  Hypernetwork Training")
    logger.info("="*60)

    hypernet_trainer        = HypernetworkTrainer(cfg, teacher)
    hypernet, student_model = hypernet_trainer.train(train_dl, val_dl)

    # ── Final report ──────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("COMPRESSION REPORT")
    logger.info("="*60)
    teacher_mb  = teacher.size_mb()
    hypernet_mb = hypernet.size_mb(2)   # FP16
    logger.info(f"  Teacher model   : {teacher_mb:.1f} MB (FP32)")
    logger.info(f"  Hypernetwork    : {hypernet_mb:.2f} MB (FP16)")
    logger.info(f"  Compression     : {teacher_mb / hypernet_mb:.1f}×")
    logger.info(f"  Within 16 MB    : {'✅' if hypernet_mb <= 16 else '❌'}")

    return teacher, hypernet
