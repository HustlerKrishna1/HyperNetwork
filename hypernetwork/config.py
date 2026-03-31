"""
config.py  —  All configuration dataclasses for the HyperNetwork system.

Sections
--------
1. TargetModelConfig   – defines the transformer whose weights we generate
2. HypernetworkConfig  – defines H(layer_id, weight_type, z) → W
3. TrainingConfig      – optimisation hyper-parameters for both training phases
4. ExperimentConfig    – experiment / ablation bookkeeping
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import math


# ---------------------------------------------------------------------------
# 1. Target model
# ---------------------------------------------------------------------------

@dataclass
class TargetModelConfig:
    """
    A small but realistic transformer language model.

    Defaults give ≈15 M parameters — trainable on a single A100 / V100 in <1 h.
    """
    vocab_size:  int = 10_000
    hidden_dim:  int = 256          # d_model
    n_layers:    int = 6
    n_heads:     int = 4
    ffn_dim:     int = 1024         # 4 × hidden_dim
    max_seq_len: int = 512
    dropout:     float = 0.1
    tie_embeddings: bool = True     # tie input ↔ output embeddings

    # ── derived -----------------------------------------------------------------
    @property
    def head_dim(self) -> int:
        assert self.hidden_dim % self.n_heads == 0
        return self.hidden_dim // self.n_heads

    @property
    def param_count(self) -> int:
        d, f, V, L = self.hidden_dim, self.ffn_dim, self.vocab_size, self.n_layers
        embed    = V * d + self.max_seq_len * d
        attn     = 4 * d * d + 4 * d          # Q,K,V,O weights + biases
        ffn      = d * f + f + f * d + d       # FFN1 + FFN2 w/ biases
        ln       = 4 * d                        # 2 LayerNorm per layer (γ,β each)
        per_layer = attn + ffn + ln
        lm_head  = 0 if self.tie_embeddings else d * V
        return embed + L * per_layer + lm_head

    @property
    def weight_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Shape of every *generated* weight for a single transformer block."""
        d, f = self.hidden_dim, self.ffn_dim
        return {
            # Attention projections  (out_features, in_features)
            "q_weight":   (d, d),
            "k_weight":   (d, d),
            "v_weight":   (d, d),
            "o_weight":   (d, d),
            # Attention biases
            "q_bias":     (d,),
            "k_bias":     (d,),
            "v_bias":     (d,),
            "o_bias":     (d,),
            # Feed-forward projections
            "ffn1_weight": (f, d),
            "ffn2_weight": (d, f),
            "ffn1_bias":   (f,),
            "ffn2_bias":   (d,),
        }

    @property
    def matrix_weight_keys(self) -> List[str]:
        return ["q_weight", "k_weight", "v_weight", "o_weight",
                "ffn1_weight", "ffn2_weight"]

    @property
    def bias_weight_keys(self) -> List[str]:
        return ["q_bias", "k_bias", "v_bias", "o_bias",
                "ffn1_bias", "ffn2_bias"]

    @property
    def all_weight_keys(self) -> List[str]:
        return self.matrix_weight_keys + self.bias_weight_keys

    def size_mb(self, dtype_bytes: int = 4) -> float:
        """Full model size if all weights stored naïvely."""
        total = 0
        for shape in self.weight_shapes.values():
            total += math.prod(shape)
        total *= self.n_layers
        # add embedding + LM head
        total += self.vocab_size * self.hidden_dim * (1 if self.tie_embeddings else 2)
        total += self.max_seq_len * self.hidden_dim
        return total * dtype_bytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# 2. Hypernetwork
# ---------------------------------------------------------------------------

WEIGHT_TYPE_INDEX: Dict[str, int] = {
    "q_weight":    0,
    "k_weight":    1,
    "v_weight":    2,
    "o_weight":    3,
    "ffn1_weight": 4,
    "ffn2_weight": 5,
    "q_bias":      6,
    "k_bias":      7,
    "v_bias":      8,
    "o_bias":      9,
    "ffn1_bias":  10,
    "ffn2_bias":  11,
}


@dataclass
class HypernetworkConfig:
    """
    Configuration for H : (layer_id, weight_type, z) → W.

    16 MB budget in FP16 = 8 388 608 parameters.
    We target ≤ 4 M params (FP32) which is safely under budget after INT8 quant.

    Parameter Budget (default settings)
    ------------------------------------
    layer_embed:          6 × 64         =     384
    type_embed:          12 × 64         =     768
    latent_proj:          96 × 128       =  12 288
    MLP  128→256→512→512: 128×256 + 256×512 + 512×512
                                         = 425 984
    Low-rank heads (per weight type):
        matrix heads (6): 6 × 512 × 2r  = 6 × 512 × 64  = 196 608
        bias   heads (6): 6 × 512 × max_dim
    Total ≈ 0.7 M params  ← well within budget, leaving room to scale
    """
    # Conditioning embeddings
    layer_embed_dim: int = 64
    type_embed_dim:  int = 64
    latent_dim:      int = 32    # optional stochastic latent

    # MLP backbone hidden sizes
    hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 512])

    # Low-rank strategy
    rank: int = 32               # W ≈ A @ B,  A:[out,r], B:[r,in]

    # Generation strategy
    strategy: str = "lowrank"    # "lowrank" | "chunked" | "implicit"

    # Chunked strategy: how many output elements per forward pass
    chunk_size: int = 256

    # Size constraint
    max_size_mb: float = 16.0

    # Whether to also generate bias vectors
    generate_biases: bool = True

    # Number of distinct weight types (matches WEIGHT_TYPE_INDEX above)
    n_weight_types: int = 12

    # Temperature for weight initialisation scaling
    init_scale: float = 0.02

    @property
    def cond_dim(self) -> int:
        """Dimensionality after concatenating all conditioning signals."""
        return self.layer_embed_dim + self.type_embed_dim + self.latent_dim

    @property
    def mlp_input_dim(self) -> int:
        return self.cond_dim

    @property
    def mlp_output_dim(self) -> int:
        return self.hidden_dims[-1]


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Hyper-parameters for both training phases."""

    # ── Data -------------------------------------------------------------------
    dataset:   str = "wikitext2"   # "wikitext2" | "ptb" | "openwebtext"
    data_dir:  str = "data/"
    batch_size: int = 32
    seq_len:    int = 128
    num_workers: int = 4

    # ── Optimiser --------------------------------------------------------------
    lr:            float = 1e-3
    min_lr:        float = 1e-5
    weight_decay:  float = 1e-4
    grad_clip:     float = 1.0
    beta1:         float = 0.9
    beta2:         float = 0.95

    # ── Schedule ---------------------------------------------------------------
    max_steps:    int = 100_000
    warmup_steps: int = 5_000

    # ── Phase split -----------------------------------------------------------
    teacher_steps:      int = 50_000   # Phase-1: train teacher alone
    hypernetwork_steps: int = 50_000   # Phase-2: train hypernetwork

    # ── Loss weights ----------------------------------------------------------
    lambda_task:    float = 1.0    # cross-entropy on token prediction
    lambda_recon:   float = 0.1    # MSE between generated & teacher weights
    lambda_distill: float = 0.5    # KL of soft labels (knowledge distillation)
    temperature:    float = 4.0    # distillation temperature T

    # ── Mixed precision -------------------------------------------------------
    mixed_precision: bool = True
    dtype: str = "float16"         # "float16" | "bfloat16"

    # ── Checkpointing ---------------------------------------------------------
    save_every: int = 5_000
    eval_every: int = 1_000
    ckpt_dir:   str = "checkpoints/"

    # ── Logging ---------------------------------------------------------------
    log_dir:    str = "runs/hypernetwork"
    use_wandb:  bool = False
    project:    str = "hypernetwork"

    # ── Device ----------------------------------------------------------------
    device: str = "cuda"


# ---------------------------------------------------------------------------
# 4. Experiment
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Top-level config that groups all three sub-configs."""
    name:        str = "hypernetwork_v1"
    seed:        int = 42
    target:      TargetModelConfig  = field(default_factory=TargetModelConfig)
    hypernet:    HypernetworkConfig = field(default_factory=HypernetworkConfig)
    training:    TrainingConfig     = field(default_factory=TrainingConfig)

    def summary(self) -> str:
        lines = [
            f"Experiment : {self.name}",
            f"  Target params   : {self.target.param_count / 1e6:.2f} M",
            f"  Target size     : {self.target.size_mb():.1f} MB (FP32)",
            f"  HyperNet budget : {self.hypernet.max_size_mb} MB",
            f"  Strategy        : {self.hypernet.strategy}",
            f"  Rank            : {self.hypernet.rank}",
            f"  Dataset         : {self.training.dataset}",
        ]
        return "\n".join(lines)
from dataclasses import dataclass

@dataclass
class HypernetworkConfig:
    rank: int
    # other fields...

    share_weights: bool = False   # ✅ ADD THIS