# 🧠 Hypernetwork — Tiny Generator, Full Model Weights

> Store a **1.3 MB generator** instead of a **28 MB model**.
> The generator creates all the weights on-the-fly, every forward pass.

---

## 💡 The Core Idea

Normal models store every weight matrix in full.
This project replaces them with a tiny network that **generates** weights using SVD factorization:

```
W = U @ diag(s) @ V^T
```

| What | Role | Size |
|------|------|------|
| `U`, `V` | Stored basis matrices (learned once) | small |
| `s` | Generated on-the-fly (32 values) | tiny |
| `W` | Reconstructed weight matrix | full size |

Only `s` is computed per-forward-pass. `U` and `V` live in the hypernetwork as parameters.

---

## 📦 Compression Result

| Component | Size |
|-----------|------|
| Original model (teacher) | ~28 MB |
| Hypernetwork (FP16) | ~1.3 MB |
| **Compression** | **~21× smaller** |

And it fits comfortably within a **16 MB budget**.

---

## 🗂️ Project Structure

```
Hypernetwork/
├── main.py                  ← CLI entry point
├── requirements.txt
└── hypernetwork/
    ├── config.py            ← All configs (model, hypernetwork, training)
    ├── hypernetwork.py      ← Core H(layer_id, weight_type, z) → W
    ├── target_model.py      ← Transformer that accepts generated weights
    ├── weight_strategies.py ← Low-rank, chunked, and implicit (SIREN) strategies
    ├── losses.py            ← Task loss, reconstruction loss, distillation loss
    ├── trainer.py           ← Two-phase training pipeline
    ├── optimizer_utils.py   ← Quantization, caching, LoRA hybrid, multi-task
    └── experiments.py       ← Ablations, benchmarks, failure-mode monitor
```

---

## ⚡ Quick Start

```bash
# Install dependencies
pip install torch datasets transformers

# 1. Architecture check — no GPU needed, runs in ~5 seconds
python main.py --mode check

# 2. Full training (teacher → hypernetwork)
python main.py --mode train

# 3. Run all ablation experiments
python main.py --mode experiments

# 4. See parameter budget breakdown
python main.py --mode budget

# 5. Compare all three weight generation strategies
python main.py --mode strategies
```

---

## 🏋️ Training Pipeline

Two phases, fully automated:

**Phase 1 — Train Teacher**
```
Standard transformer trained normally with cross-entropy loss.
Saved as frozen reference weights.
```

**Phase 2 — Train Hypernetwork**
```
Phase 2a (warm-up)  →  weight reconstruction loss only
Phase 2b (joint)    →  task loss + reconstruction + distillation
Phase 2c (fine-tune)→  task loss + distillation (reconstruction annealed)
```

Combined loss formula:
```
L = λ_task · CE(logits, targets)
  + λ_recon · MSE(W_generated, W_teacher)
  + λ_distill · T² · KL(softmax(logits_t/T) ‖ softmax(logits_s/T))
```

---

## 🔬 Three Weight Generation Strategies

| Strategy | How it works | Best for |
|----------|-------------|----------|
| **SVD Low-Rank** ✅ | `W = U @ diag(s) @ V^T` — only `s` generated | Primary method, best compression |
| **Chunked** | Matrix generated in row-blocks | No rank bottleneck, higher quality |
| **Implicit (SIREN)** | Coordinate-based MLP, `W[i,j] = f(i, j, cond)` | Smooth weight priors |

Run `python main.py --mode strategies` to compare all three side by side.

---

## 🧩 Advanced Variants

All included in `optimizer_utils.py`, ready to use:

**LoRA + Hypernetwork Hybrid**
The base weights are frozen (INT8). The hypernetwork generates only the residual delta `ΔW`.
Easier to train, lower rank needed.

**Token-Conditioned Weights**
Weights change based on the input context. The hypernetwork reads the tokens and adapts.
Useful for domain-specific specialization.

**Multi-Task Hypernetwork**
One hypernetwork, many tasks. Each task gets a learned latent code `z_task`.
Weights are specialized per task at inference time.

---

## 📊 Ablation Studies

Run automatically with `--mode experiments`:

| Experiment | What it tests |
|------------|--------------|
| Rank sweep (4 → 128) | Compression vs accuracy tradeoff |
| Strategy comparison | Low-rank vs chunked vs SIREN |
| Shared vs unshared backbone | Parameter savings from FiLM modulation |
| Generation latency | Sequential vs batched weight generation |
| Parameter budget | Per-component breakdown |

---

## ⚠️ Known Failure Modes

| Problem | Sign | Fix |
|---------|------|-----|
| Weight collapse | `‖W‖ → 0` | Add reconstruction loss warm-up |
| Mode collapse | Layer weights become identical | Monitor cosine similarity between layers |
| Training explosion | Loss > 100 or NaN | Lower LR, check grad clipping |
| OOM on GPU | CUDA out of memory | Reduce batch size or use chunked generation |

The `monitor_health()` function in `experiments.py` checks for all of these automatically.

---

## 🛠️ Custom Config

```python
from hypernetwork import ExperimentConfig, TargetModelConfig, HypernetworkConfig

cfg = ExperimentConfig(
    target=TargetModelConfig(
        hidden_dim=256, n_layers=6, n_heads=4, ffn_dim=1024
    ),
    hypernet=HypernetworkConfig(
        rank=32,           # SVD rank — higher = better quality, larger size
        strategy="lowrank",
        max_size_mb=16.0,
    ),
)
print(cfg.summary())
```

---

## 📐 Hypernetwork Architecture

```
layer_id ──┐
type_id  ──┼──▶  Embedding lookup  ──▶  [160-dim conditioning vector]
latent z ──┘
                        │
               MLP: 160 → 256 → 512 → 512  (SiLU activations)
                        │
                 ┌──────┴──────┐
              s-head         bias-head
           512 → rank       512 → bias_dim
                 │
          W = U @ diag(s) @ V^T
```

Total parameters: **~0.7 M** — fits in **1.3 MB FP16**.

---

## 📋 Requirements

```
torch >= 2.0
datasets >= 2.14   (optional, for WikiText-2)
transformers >= 4.35  (optional, for tokenizer)
```

Works without `datasets` / `transformers` — falls back to a synthetic token dataset automatically.

---

*Built as a research prototype for neural network compression via hypernetworks.*
