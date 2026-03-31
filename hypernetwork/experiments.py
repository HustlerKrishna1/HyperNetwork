"""
experiments.py  —  Experiment runner, ablation studies, and failure-mode analysis.

Experiment Plan
═══════════════

EXP-1: Baseline sanity check
    Dataset : WikiText-2  (small, fast)
    Metric  : Perplexity, compression ratio
    Goal    : Confirm hypernetwork can reconstruct teacher weights

EXP-2: Rank sweep (ablation)
    Ranks   : [4, 8, 16, 32, 64, 128]
    Metric  : PPL vs rank, size vs rank
    Goal    : Find pareto-optimal rank

EXP-3: Strategy comparison
    Strategies: lowrank, chunked, implicit
    Metric    : PPL, size, generation latency
    Goal      : Validate low-rank as primary choice

EXP-4: Loss weight ablation
    Vary λ_recon ∈ {0, 0.01, 0.1, 1.0}
    Vary λ_distill ∈ {0, 0.1, 0.5, 1.0}
    Metric: PPL
    Goal: Understand relative contribution of each loss

EXP-5: Scale-up study
    Scale to PTB, then OpenWebText
    Metric: PPL gap (teacher - student)
    Goal: Verify approach holds at larger scale

Failure Modes
════════════
1. Weight collapse: H generates near-zero weights
   Monitor: ‖W_gen‖_F per layer, gradient norms
2. Mode collapse: H ignores layer_id, generates same weights
   Monitor: cosine similarity between layer weights
3. Training instability: loss NaN / explosion
   Monitor: loss > 100 or NaN; grad norm > 10
4. Memory OOM: batched generation + backprop
   Monitor: GPU memory; reduce batch or chunk generation
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import (
    ExperimentConfig,
    HypernetworkConfig,
    TargetModelConfig,
    TrainingConfig,
)
from .hypernetwork import Hypernetwork, build_hypernetwork
from .losses import weight_reconstruction_report
from .optimizer_utils import (
    benchmark_generation,
    estimate_post_quant_size,
    parameter_budget_breakdown,
    size_budget_check,
)
from .target_model import TransformerLM, build_target_model
from .weight_strategies import compare_strategies, rank_sensitivity


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    name:            str
    config:          Dict[str, Any]
    teacher_ppl:     float = float("inf")
    student_ppl:     float = float("inf")
    ppl_gap:         float = float("inf")
    compression_ratio: float = 0.0
    teacher_mb:      float = 0.0
    hypernet_mb_fp32: float = 0.0
    hypernet_mb_fp16: float = 0.0
    hypernet_mb_int8: float = 0.0
    gen_latency_ms:  float = 0.0
    recon_errors:    Dict[str, float] = field(default_factory=dict)
    notes:           str = ""

    def print_summary(self) -> None:
        print(f"\n{'─'*55}")
        print(f"  Experiment : {self.name}")
        print(f"  Teacher PPL: {self.teacher_ppl:.2f}")
        print(f"  Student PPL: {self.student_ppl:.2f}")
        print(f"  PPL Gap    : {self.ppl_gap:.2f}  "
              f"({'✅' if self.ppl_gap < 5 else '⚠️' if self.ppl_gap < 20 else '❌'})")
        print(f"  Compression: {self.compression_ratio:.1f}×")
        print(f"  HN FP32    : {self.hypernet_mb_fp32:.2f} MB")
        print(f"  HN FP16    : {self.hypernet_mb_fp16:.2f} MB  "
              f"({'✅' if self.hypernet_mb_fp16 <= 16 else '❌'})")
        print(f"  HN INT8    : {self.hypernet_mb_int8:.2f} MB")
        print(f"  Gen latency: {self.gen_latency_ms:.2f} ms")
        if self.notes:
            print(f"  Notes      : {self.notes}")
        print(f"{'─'*55}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Quick-eval (no training — smoke test)
# ─────────────────────────────────────────────────────────────────────────────

def quick_eval_untrained(
    cfg: ExperimentConfig,
    device: Optional[torch.device] = None,
) -> ExperimentResult:
    """
    Instantiate teacher + hypernetwork, run one forward pass,
    measure sizes and latencies. No training required.

    Good for: CI smoke tests, budget validation, architecture checks.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher  = build_target_model(cfg.target).to(device)
    hypernet = build_hypernetwork(cfg.target, cfg.hypernet).to(device)

    # Budget check
    teacher_mb  = teacher.size_mb()
    hn_fp32     = hypernet.size_mb(4)
    hn_fp16     = hypernet.size_mb(2)
    hn_int8     = estimate_post_quant_size(hypernet, "int8")

    print(f"\n{'='*55}")
    print(f"  Quick Eval: {cfg.name}")
    print(f"  Teacher    : {teacher_mb:.1f} MB  ({teacher.param_count/1e6:.2f} M params)")
    print(f"  HyperNet   : {hn_fp32:.2f} MB FP32  /  "
          f"{hn_fp16:.2f} MB FP16  /  {hn_int8:.2f} MB INT8")
    size_budget_check(hypernet, cfg.hypernet.max_size_mb, "fp16")

    # Generation latency
    hypernet.eval()
    n_warmup, n_runs = 3, 20
    for _ in range(n_warmup):
        with torch.no_grad():
            hypernet.generate_all_weights()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            gen_weights = hypernet.generate_all_weights()
    if device.type == "cuda":
        torch.cuda.synchronize()
    lat_ms = (time.perf_counter() - t0) * 1000 / n_runs
    print(f"  Gen latency: {lat_ms:.2f} ms / forward-pass ({n_runs} runs)")

    # Forward pass through student
    B, S = 2, cfg.target.max_seq_len
    ids  = torch.randint(0, cfg.target.vocab_size, (B, min(S, 64)), device=device)
    with torch.no_grad():
        logits, _ = teacher(ids, gen_weights)
    print(f"  Student logits: {logits.shape}  ✅")

    # Reconstruction error (random weights → high error, just checking no crash)
    tea_w = teacher.get_all_weight_dicts()
    recon_report = weight_reconstruction_report(gen_weights, tea_w)

    result = ExperimentResult(
        name              = cfg.name,
        config            = {},
        compression_ratio = teacher_mb / max(hn_fp16, 1e-9),
        teacher_mb        = teacher_mb,
        hypernet_mb_fp32  = hn_fp32,
        hypernet_mb_fp16  = hn_fp16,
        hypernet_mb_int8  = hn_int8,
        gen_latency_ms    = lat_ms,
        recon_errors      = recon_report,
        notes             = "Untrained — architecture check only",
    )
    result.print_summary()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Ablation: Rank sweep
# ─────────────────────────────────────────────────────────────────────────────

def ablation_rank_sweep(
    target_cfg: TargetModelConfig,
    ranks: Optional[List[int]] = None,
    verbose: bool = True,
) -> Dict[int, Dict[str, float]]:
    """
    For each rank r, instantiate a hypernetwork and record size.
    (Training results must be plugged in manually from training runs.)
    """
    if ranks is None:
        ranks = [4, 8, 16, 32, 64, 128]

    results: Dict[int, Dict[str, float]] = {}

    if verbose:
        print(f"\n{'Rank Sweep — Parameter Budget':^60}")
        print("="*60)
        hdr = f"{'rank':>6} {'HN params':>12} {'FP32 MB':>10} {'FP16 MB':>10} {'Compr(D×D)':>12}"
        print(hdr)
        print("-"*60)

    for r in ranks:
        h_cfg = HypernetworkConfig(rank=r)
        h_cfg.share_weights = False
        hn  = build_hypernetwork(target_cfg, h_cfg)
        n   = sum(p.numel() for p in hn.parameters())
        mb4 = n * 4 / (1024**2)
        mb2 = n * 2 / (1024**2)
        D   = target_cfg.hidden_dim
        cr  = (D * D) / (r * 2 * D)

        results[r] = {
            "params":       n,
            "mb_fp32":      mb4,
            "mb_fp16":      mb2,
            "compression":  cr,
        }

        if verbose:
            budget_ok = "✅" if mb2 <= 16 else "❌"
            print(f"{r:>6} {n:>12,} {mb4:>10.2f} {mb2:>10.2f} {cr:>11.1f}×  {budget_ok}")

        del hn

    if verbose:
        rank_sensitivity(target_cfg, ranks)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Ablation: Strategy comparison
# ─────────────────────────────────────────────────────────────────────────────

def ablation_strategy_comparison(
    target_cfg:   TargetModelConfig,
    hypernet_cfg: HypernetworkConfig,
    verbose: bool = True,
) -> None:
    """Print strategy comparison table."""
    compare_strategies(target_cfg, hypernet_cfg, verbose=verbose)


# ─────────────────────────────────────────────────────────────────────────────
# Failure mode monitor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HealthMetrics:
    """Tracked during training to detect failure modes."""
    step:                int    = 0
    loss:                float  = float("inf")
    grad_norm:           float  = 0.0
    weight_norms:        Dict[str, float] = field(default_factory=dict)
    weight_cosine_sims:  Dict[str, float] = field(default_factory=dict)
    has_nan:             bool   = False
    has_explosion:       bool   = False

    def check(self) -> List[str]:
        """Returns list of detected failure modes."""
        issues: List[str] = []
        if self.has_nan:
            issues.append("🔴 NaN detected in loss or gradients")
        if self.has_explosion:
            issues.append("🔴 Loss explosion (> 100)")
        if self.grad_norm > 10.0:
            issues.append(f"🟡 High grad norm: {self.grad_norm:.2f}")
        for key, norm in self.weight_norms.items():
            if norm < 1e-6:
                issues.append(f"🔴 Weight collapse: {key} norm={norm:.2e}")
            if norm > 100:
                issues.append(f"🟡 Large weight: {key} norm={norm:.2f}")
        for key, sim in self.weight_cosine_sims.items():
            if sim > 0.99:
                issues.append(f"🟡 Possible mode collapse: {key} cosine_sim={sim:.4f}")
        return issues


def monitor_health(
    step:       int,
    loss:       torch.Tensor,
    gradients:  List[torch.Tensor],
    gen_weights: List[Dict[str, torch.Tensor]],
) -> HealthMetrics:
    """
    Compute health metrics from current training state.
    Call at every eval step or whenever debugging.
    """
    metrics = HealthMetrics(step=step)

    loss_val = loss.item() if not torch.isnan(loss) else float("nan")
    metrics.loss       = loss_val
    metrics.has_nan    = math.isnan(loss_val) or math.isinf(loss_val)
    metrics.has_explosion = loss_val > 100

    # Gradient norm
    valid_grads = [g for g in gradients if g is not None and not torch.isnan(g).any()]
    if valid_grads:
        total_norm = sum(g.norm(2).item() ** 2 for g in valid_grads) ** 0.5
        metrics.grad_norm = total_norm

    # Weight norms per matrix key
    if gen_weights:
        for key in gen_weights[0]:
            norms = [gen_weights[l][key].norm().item()
                     for l in range(len(gen_weights))
                     if key in gen_weights[l]]
            if norms:
                metrics.weight_norms[key] = sum(norms) / len(norms)

        # Cosine similarity between layer 0 and layer 1 (mode-collapse check)
        if len(gen_weights) >= 2:
            for key in gen_weights[0]:
                if gen_weights[0][key].dim() == 2:
                    w0 = gen_weights[0][key].flatten().float()
                    w1 = gen_weights[1][key].flatten().float()
                    sim = torch.nn.functional.cosine_similarity(
                        w0.unsqueeze(0), w1.unsqueeze(0)
                    ).item()
                    metrics.weight_cosine_sims[key] = sim

    return metrics


def print_health_report(metrics: HealthMetrics) -> None:
    issues = metrics.check()
    status = "✅ HEALTHY" if not issues else "⚠️ ISSUES DETECTED"
    print(f"\n  [step {metrics.step}] Health: {status}")
    print(f"    loss={metrics.loss:.4f}  grad_norm={metrics.grad_norm:.3f}")
    if issues:
        for issue in issues:
            print(f"    {issue}")


# ─────────────────────────────────────────────────────────────────────────────
# Full experiment suite runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_suite(save_dir: str = "experiment_results/") -> None:
    """
    Run all ablation experiments (architecture checks, no training).
    For training results, use run_full_pipeline from trainer.py.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "╔" + "═"*57 + "╗")
    print("║" + f"{'HYPERNETWORK EXPERIMENT SUITE':^57}" + "║")
    print("╚" + "═"*57 + "╝")
    print(f"  Device: {device}")
    print(f"  PyTorch: {torch.__version__}")

    target_cfg   = TargetModelConfig()
    hypernet_cfg = HypernetworkConfig()

    all_results: List[ExperimentResult] = []

    # ── EXP-1: Architecture sanity check ─────────────────────────────────────
    print("\n─── EXP-1: Architecture Sanity Check ───")
    cfg    = ExperimentConfig(name="exp1_baseline")
    result = quick_eval_untrained(cfg, device=device)
    all_results.append(result)

    # ── EXP-2: Rank sweep ────────────────────────────────────────────────────
    print("\n─── EXP-2: Rank Ablation ───")
    rank_results = ablation_rank_sweep(target_cfg, verbose=True)

    # ── EXP-3: Strategy comparison ───────────────────────────────────────────
    print("\n─── EXP-3: Strategy Comparison ───")
    ablation_strategy_comparison(target_cfg, hypernet_cfg, verbose=True)

    # ── EXP-4: Shared vs non-shared backbone ─────────────────────────────────
    print("\n─── EXP-4: Shared vs Unshared Backbone ───")
    for share in [True, False]:
        h_cfg = HypernetworkConfig(share_weights=share)
        hn = build_hypernetwork(target_cfg, h_cfg)
        n  = sum(p.numel() for p in hn.parameters())
        print(f"  share_weights={share}:  {n/1e6:.3f} M params  "
              f"({n*2/1024**2:.2f} MB FP16)")
        del hn

    # ── EXP-5: Generation latency ────────────────────────────────────────────
    print("\n─── EXP-5: Generation Latency Benchmark ───")
    hn = build_hypernetwork(target_cfg, hypernet_cfg).to(device)
    weight_keys = target_cfg.all_weight_keys
    benchmark_generation(hn, target_cfg.n_layers, weight_keys)
    del hn

    # ── EXP-6: Parameter budget breakdown ────────────────────────────────────
    print("\n─── EXP-6: Parameter Budget Breakdown ───")
    hn = build_hypernetwork(target_cfg, hypernet_cfg)
    parameter_budget_breakdown(hn, verbose=True)

    # ── Save results ─────────────────────────────────────────────────────────
    out = {
        "device":       str(device),
        "torch":        torch.__version__,
        "results":      [r.to_dict() for r in all_results],
        "rank_sweep":   {str(k): v for k, v in rank_results.items()},
    }
    out_path = Path(save_dir) / "experiment_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  ✅  Results saved → {out_path}")
    print("\n" + "═"*59)
    print(" Done. To run full training: python main.py --mode train")
    print("═"*59 + "\n")
