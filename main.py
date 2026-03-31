#!/usr/bin/env python3
"""
main.py  —  CLI entry point for the Hypernetwork system.

Usage examples
──────────────
  # Quick architecture check (no training, ~5 seconds)
  python main.py --mode check

  # Full experiment suite (ablations, benchmarks, no training)
  python main.py --mode experiments

  # Train teacher then hypernetwork (full pipeline)
  python main.py --mode train

  # Train with custom hyperparameters
  python main.py --mode train --rank 16 --n-layers 4 --hidden-dim 256

  # Phase 1 only (teacher)
  python main.py --mode teacher

  # Phase 2 only (hypernetwork), loading pre-trained teacher
  python main.py --mode hypernet --teacher-ckpt checkpoints/teacher_best.pt

  # Compare all weight strategies
  python main.py --mode strategies

  # Show parameter budget
  python main.py --mode budget
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

# Add parent to path for clean imports
sys.path.insert(0, str(Path(__file__).parent))

from hypernetwork import (
    ExperimentConfig,
    HypernetworkConfig,
    TargetModelConfig,
    TrainingConfig,
    build_hypernetwork,
    build_target_model,
    parameter_budget_breakdown,
    quick_eval_untrained,
    ablation_rank_sweep,
    run_experiment_suite,
    size_budget_check,
    quantize_to_fp16,
    quantize_to_int8,
    run_full_pipeline,
)
from hypernetwork.weight_strategies import compare_strategies, rank_sensitivity

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hypernetwork — on-the-fly weight generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--mode", default="check",
        choices=["check", "experiments", "train", "teacher", "hypernet",
                 "strategies", "budget", "benchmark"],
        help="Execution mode.",
    )

    # ── Target model ─────────────────────────────────────────────────────────
    m = p.add_argument_group("Target Model")
    m.add_argument("--vocab-size",  type=int, default=10_000)
    m.add_argument("--hidden-dim",  type=int, default=256)
    m.add_argument("--n-layers",    type=int, default=6)
    m.add_argument("--n-heads",     type=int, default=4)
    m.add_argument("--ffn-dim",     type=int, default=1024)
    m.add_argument("--max-seq-len", type=int, default=512)

    # ── Hypernetwork ─────────────────────────────────────────────────────────
    h = p.add_argument_group("Hypernetwork")
    h.add_argument("--rank",          type=int,   default=32)
    h.add_argument("--strategy",      type=str,   default="lowrank",
                   choices=["lowrank", "chunked", "implicit"])
    h.add_argument("--share-weights", action="store_true", default=True)
    h.add_argument("--max-size-mb",   type=float, default=16.0)

    # ── Training ─────────────────────────────────────────────────────────────
    t = p.add_argument_group("Training")
    t.add_argument("--dataset",           type=str,   default="wikitext2")
    t.add_argument("--batch-size",        type=int,   default=32)
    t.add_argument("--seq-len",           type=int,   default=128)
    t.add_argument("--lr",                type=float, default=1e-3)
    t.add_argument("--teacher-steps",     type=int,   default=50_000)
    t.add_argument("--hypernetwork-steps",type=int,   default=50_000)
    t.add_argument("--lambda-task",       type=float, default=1.0)
    t.add_argument("--lambda-recon",      type=float, default=0.1)
    t.add_argument("--lambda-distill",    type=float, default=0.5)
    t.add_argument("--temperature",       type=float, default=4.0)
    t.add_argument("--no-amp",            action="store_true")

    # ── Misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--device",       type=str, default="cuda")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--ckpt-dir",     type=str, default="checkpoints/")
    p.add_argument("--teacher-ckpt", type=str, default=None,
                   help="Path to pre-trained teacher checkpoint (for --mode hypernet)")
    p.add_argument("--name",         type=str, default="hypernetwork_v1",
                   help="Experiment name")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Config builder from args
# ─────────────────────────────────────────────────────────────────────────────

def build_config(args: argparse.Namespace) -> ExperimentConfig:
    target = TargetModelConfig(
        vocab_size  = args.vocab_size,
        hidden_dim  = args.hidden_dim,
        n_layers    = args.n_layers,
        n_heads     = args.n_heads,
        ffn_dim     = args.ffn_dim,
        max_seq_len = args.max_seq_len,
    )
    hypernet = HypernetworkConfig(
        rank         = args.rank,
        strategy     = args.strategy,
        share_weights= args.share_weights,
        max_size_mb  = args.max_size_mb,
    )
    training = TrainingConfig(
        dataset           = args.dataset,
        batch_size        = args.batch_size,
        seq_len           = args.seq_len,
        lr                = args.lr,
        teacher_steps     = args.teacher_steps,
        hypernetwork_steps= args.hypernetwork_steps,
        max_steps         = args.teacher_steps + args.hypernetwork_steps,
        lambda_task       = args.lambda_task,
        lambda_recon      = args.lambda_recon,
        lambda_distill    = args.lambda_distill,
        temperature       = args.temperature,
        mixed_precision   = not args.no_amp,
        device            = args.device,
        ckpt_dir          = args.ckpt_dir,
    )
    return ExperimentConfig(
        name     = args.name,
        seed     = args.seed,
        target   = target,
        hypernet = hypernet,
        training = training,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mode handlers
# ─────────────────────────────────────────────────────────────────────────────

def mode_check(args: argparse.Namespace) -> None:
    """Quick architecture and budget check — no training."""
    print("\n" + "━"*60)
    print("  MODE: Architecture Check")
    print("━"*60)

    cfg    = build_config(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(cfg.summary())

    target   = build_target_model(cfg.target).to(device)
    hypernet = build_hypernetwork(cfg.target, cfg.hypernet).to(device)

    print(f"\n{target}")
    print(f"\n{hypernet}")

    print("\n  ── Budget check ──")
    size_budget_check(hypernet, args.max_size_mb, "fp16")
    size_budget_check(hypernet, args.max_size_mb, "int8")

    # One forward pass
    B, S = 2, 64
    ids  = torch.randint(0, cfg.target.vocab_size, (B, S), device=device)
    with torch.no_grad():
        gen_w  = hypernet.generate_all_weights()
        logits, _ = target(ids, gen_w)
    print(f"\n  Forward pass  ✅  logits: {list(logits.shape)}")

    quick_eval_untrained(cfg, device=device)


def mode_experiments(args: argparse.Namespace) -> None:
    """Full ablation suite."""
    run_experiment_suite(save_dir="experiment_results/")


def mode_train(args: argparse.Namespace) -> None:
    """Full two-phase training pipeline."""
    cfg = build_config(args)
    teacher, hypernet = run_full_pipeline(cfg)
    # Final quantization check
    print("\n  Quantising to FP16 for storage...")
    hn_fp16 = quantize_to_fp16(hypernet)
    print(f"  FP16 size: {sum(p.numel() for p in hn_fp16.parameters())*2/1024**2:.2f} MB")


def mode_teacher(args: argparse.Namespace) -> None:
    """Phase 1 only: train teacher."""
    from hypernetwork.trainer import TeacherTrainer, make_dataloaders

    cfg      = build_config(args)
    torch.manual_seed(cfg.seed)

    train_dl, val_dl = make_dataloaders(cfg)
    trainer  = TeacherTrainer(cfg)
    teacher  = trainer.train(train_dl, val_dl)
    print(f"\n  Teacher trained. val loss = {trainer._best_val_loss:.4f}")


def mode_hypernet(args: argparse.Namespace) -> None:
    """Phase 2 only: train hypernetwork from pre-trained teacher checkpoint."""
    from hypernetwork.trainer import HypernetworkTrainer, make_dataloaders

    cfg = build_config(args)
    torch.manual_seed(cfg.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.teacher_ckpt is None:
        logger.error("--teacher-ckpt required for --mode hypernet")
        sys.exit(1)

    ckpt    = torch.load(args.teacher_ckpt, map_location=device)
    teacher = build_target_model(cfg.target).to(device)
    teacher.load_state_dict(ckpt["model"])
    logger.info(f"Loaded teacher from {args.teacher_ckpt}")

    train_dl, val_dl = make_dataloaders(cfg)
    trainer = HypernetworkTrainer(cfg, teacher)
    hypernet, student = trainer.train(train_dl, val_dl)
    print(f"\n  Hypernetwork trained. val loss = {trainer._best_val_loss:.4f}")


def mode_strategies(args: argparse.Namespace) -> None:
    """Compare weight generation strategies."""
    cfg = build_config(args)
    compare_strategies(cfg.target, cfg.hypernet, verbose=True)
    rank_sensitivity(cfg.target)


def mode_budget(args: argparse.Namespace) -> None:
    """Detailed parameter budget breakdown."""
    cfg      = build_config(args)
    device   = torch.device("cpu")
    hypernet = build_hypernetwork(cfg.target, cfg.hypernet)
    parameter_budget_breakdown(hypernet, verbose=True)

    print(f"\n  {'dtype':>6}  {'size (MB)':>12}  {'within 16 MB':>14}")
    print("  " + "-"*36)
    for dtype, bpp in [("FP32", 4), ("FP16", 2), ("INT8", 1)]:
        n    = sum(p.numel() for p in hypernet.parameters())
        mb   = n * bpp / (1024**2)
        ok   = "✅" if mb <= 16 else "❌"
        print(f"  {dtype:>6}  {mb:>12.3f}  {ok:>14}")

    print(f"\n  Rank sensitivity:")
    ablation_rank_sweep(cfg.target, verbose=True)


def mode_benchmark(args: argparse.Namespace) -> None:
    """Benchmark weight generation latency."""
    from hypernetwork.optimizer_utils import benchmark_generation, generate_all_weights_batched

    cfg      = build_config(args)
    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    hypernet = build_hypernetwork(cfg.target, cfg.hypernet).to(device)
    hypernet.eval()

    print(f"\n  Benchmarking on {device}...")
    benchmark_generation(
        hypernet, cfg.target.n_layers, cfg.target.all_weight_keys,
        n_warmup=10, n_runs=100,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

HANDLERS = {
    "check":       mode_check,
    "experiments": mode_experiments,
    "train":       mode_train,
    "teacher":     mode_teacher,
    "hypernet":    mode_hypernet,
    "strategies":  mode_strategies,
    "budget":      mode_budget,
    "benchmark":   mode_benchmark,
}


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    torch.manual_seed(args.seed)

    print(f"\n  🧠  Hypernetwork System  |  PyTorch {torch.__version__}  "
          f"|  device={args.device}")

    handler = HANDLERS.get(args.mode)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
