"""
optimizer_utils.py  —  Quantization, weight caching, parameter budget tools.

Sections
════════
1. Parameter budget analysis
2. FP16 / INT8 quantization helpers
3. Weight cache (avoid re-generating weights on every forward pass)
4. Batch weight generation (generate all layers in one batched forward)
5. LoRA + Hypernetwork hybrid (Advanced Extension)
6. Token-conditioned weights (Advanced Extension)
7. Multi-task hypernetwork adapter (Advanced Extension)
"""

from __future__ import annotations

import time
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import HypernetworkConfig, TargetModelConfig, WEIGHT_TYPE_INDEX
from .hypernetwork import Hypernetwork


# ─────────────────────────────────────────────────────────────────────────────
# 1. Parameter Budget Analysis
# ─────────────────────────────────────────────────────────────────────────────

def parameter_budget_breakdown(
    hypernet: Hypernetwork,
    verbose:  bool = True,
) -> Dict[str, int]:
    """
    Detailed parameter count by component.

    Returns dict {component_name: n_params}
    """
    budget: Dict[str, int] = {}

    for name, module in hypernet.named_children():
        n = sum(p.numel() for p in module.parameters())
        budget[name] = n

    total = sum(budget.values())
    budget["_total"] = total

    if verbose:
        print("\n" + "="*55)
        print(f"{'Hypernetwork Parameter Budget':^55}")
        print("="*55)
        for comp, n in budget.items():
            if comp.startswith("_"):
                continue
            pct = 100 * n / max(total, 1)
            bar = "█" * int(pct / 2)
            print(f"  {comp:<22} {n:>10,}  ({pct:5.1f}%)  {bar}")
        print("-"*55)
        print(f"  {'TOTAL':<22} {total:>10,}")
        print(f"  {'FP32 size':<22} {total*4/1024**2:>9.2f} MB")
        print(f"  {'FP16 size':<22} {total*2/1024**2:>9.2f} MB")
        print(f"  {'INT8 size':<22} {total*1/1024**2:>9.2f} MB")
        print("="*55 + "\n")

    return budget


def size_budget_check(
    hypernet: Hypernetwork,
    max_mb:   float = 16.0,
    dtype:    str   = "fp16",
) -> bool:
    """Returns True if hypernetwork fits within max_mb at given dtype."""
    bytes_per_param = {"fp32": 4, "fp16": 2, "int8": 1}[dtype.lower()]
    n     = sum(p.numel() for p in hypernet.parameters())
    size  = n * bytes_per_param / (1024**2)
    ok    = size <= max_mb
    print(f"  Size at {dtype.upper()}: {size:.2f} MB  "
          f"{'✅ within' if ok else '❌ exceeds'} {max_mb} MB budget")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 2. Quantization
# ─────────────────────────────────────────────────────────────────────────────

def quantize_to_fp16(model: nn.Module) -> nn.Module:
    """
    Cast all parameters and buffers to FP16 in-place.

    This halves storage.  Inference must be done in FP16 context.
    """
    return model.half()


def quantize_to_int8(model: nn.Module) -> nn.Module:
    """
    Dynamic INT8 quantization using torch.quantization.

    Maps  Linear layers → quantized Linear.
    Reduces model to ~25 % of FP32 size.
    NOTE: requires CPU (dynamic quant) or CUDA + bitsandbytes for GPU int8.
    """
    try:
        import torch.quantization as tq
        model_cpu = model.cpu()
        quantized = tq.quantize_dynamic(
            model_cpu,
            {nn.Linear},
            dtype=torch.qint8,
        )
        print(f"  ✅  INT8 dynamic quantization applied.")
        return quantized
    except Exception as e:
        print(f"  ⚠️  INT8 quantization failed: {e}")
        return model


def estimate_post_quant_size(
    model: nn.Module,
    dtype: str = "int8",
) -> float:
    """Estimate size (MB) after quantization without actually quantizing."""
    n             = sum(p.numel() for p in model.parameters())
    bytes_map     = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
    bytes_per_p   = bytes_map.get(dtype.lower(), 4)
    return n * bytes_per_p / (1024**2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Weight Cache (LRU)
# ─────────────────────────────────────────────────────────────────────────────

class WeightCache:
    """
    LRU cache for generated weights.

    Use this when:
    * You run multiple forward passes with the same latent z.
    * You want to amortise hypernetwork inference over many token steps.

    Cache key:  (latent_hash, layer_idx, weight_key)
    """

    def __init__(self, max_size: int = 256):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._hits  = 0
        self._misses = 0

    def _key(self, z_hash: int, layer_idx: int, weight_key: str) -> str:
        return f"{z_hash}:{layer_idx}:{weight_key}"

    def get(
        self,
        z_hash:     int,
        layer_idx:  int,
        weight_key: str,
    ) -> Optional[torch.Tensor]:
        k = self._key(z_hash, layer_idx, weight_key)
        if k in self._cache:
            self._cache.move_to_end(k)
            self._hits += 1
            return self._cache[k]
        self._misses += 1
        return None

    def put(
        self,
        z_hash:     int,
        layer_idx:  int,
        weight_key: str,
        tensor:     torch.Tensor,
    ) -> None:
        k = self._key(z_hash, layer_idx, weight_key)
        self._cache[k] = tensor.detach()
        self._cache.move_to_end(k)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        return {
            "hits":     self._hits,
            "misses":   self._misses,
            "hit_rate": self.hit_rate,
            "size":     len(self._cache),
        }


def z_hash(z: Optional[torch.Tensor]) -> int:
    """Stable hash for a latent tensor (or None → 0)."""
    if z is None:
        return 0
    return hash(z.detach().cpu().numpy().tobytes())


# ─────────────────────────────────────────────────────────────────────────────
# 4. Batched Weight Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_weights_batched(
    hypernet:   Hypernetwork,
    n_layers:   int,
    weight_keys: List[str],
    z:          Optional[torch.Tensor] = None,
    cache:      Optional[WeightCache]  = None,
) -> List[Dict[str, torch.Tensor]]:
    """
    Generate all layer weights in a SINGLE batched forward pass through H.

    Instead of L × K separate forward passes, we batch all (layer, type) pairs
    together:  batch_size = L × K.

    For default config (6 layers × 12 weight types) = 72 forward items,
    all processed in one GPU kernel launch.

    Speedup: ~5–10× vs. sequential generation on GPU.
    """
    L   = n_layers
    K   = len(weight_keys)
    N   = L * K

    device = next(hypernet.parameters()).device

    # Build batched conditioning indices
    layer_ids   = []
    type_ids    = []
    for l in range(L):
        for k_idx, key in enumerate(weight_keys):
            layer_ids.append(l)
            type_ids.append(WEIGHT_TYPE_INDEX[key])

    layer_t = torch.tensor(layer_ids, dtype=torch.long, device=device)  # [N]
    type_t  = torch.tensor(type_ids,  dtype=torch.long, device=device)  # [N]

    # Expand z to batch if provided
    z_batch = None
    if z is not None:
        z_batch = z.expand(N, -1) if z.dim() == 1 else z.repeat(N, 1)

    # Single forward pass through hypernetwork — this is the key optimisation
    with torch.no_grad():
        W_batch, b_batch = hypernet(layer_t, type_t, z_batch)
        # W_batch : [N, out, in]  (matrix keys)  or  [N, out]  (bias keys)

    # Unpack back to list[dict]
    all_weights: List[Dict[str, torch.Tensor]] = [{} for _ in range(L)]
    for idx, (l, key) in enumerate(
        [(l, k) for l in range(L) for k in weight_keys]
    ):
        if W_batch.dim() == 3:       # [N, out, in]
            all_weights[l][key] = W_batch[idx]
        else:                        # [N, out]
            all_weights[l][key] = W_batch[idx]

        if b_batch is not None:
            bias_key = key.replace("_weight", "_bias")
            if bias_key in WEIGHT_TYPE_INDEX and bias_key not in all_weights[l]:
                all_weights[l][bias_key] = b_batch[idx]

    return all_weights


def benchmark_generation(
    hypernet:    Hypernetwork,
    n_layers:    int,
    weight_keys: List[str],
    n_warmup:    int = 5,
    n_runs:      int = 50,
) -> Dict[str, float]:
    """
    Benchmark sequential vs batched weight generation.

    Returns timing results in milliseconds.
    """
    hypernet.eval()
    device = next(hypernet.parameters()).device

    # Warmup
    for _ in range(n_warmup):
        hypernet.generate_all_weights()

    # Sequential (baseline)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            hypernet.generate_all_weights()
    if device.type == "cuda":
        torch.cuda.synchronize()
    seq_ms = (time.perf_counter() - t0) * 1000 / n_runs

    # Batched
    t0 = time.perf_counter()
    for _ in range(n_runs):
        generate_all_weights_batched(hypernet, n_layers, weight_keys)
    if device.type == "cuda":
        torch.cuda.synchronize()
    bat_ms = (time.perf_counter() - t0) * 1000 / n_runs

    results = {
        "sequential_ms": seq_ms,
        "batched_ms":    bat_ms,
        "speedup":       seq_ms / max(bat_ms, 1e-6),
    }
    print(f"\n  Generation benchmark ({n_runs} runs):")
    print(f"    Sequential : {seq_ms:.2f} ms/step")
    print(f"    Batched    : {bat_ms:.2f} ms/step")
    print(f"    Speedup    : {results['speedup']:.1f}×")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. LoRA + Hypernetwork Hybrid (Advanced Extension)
# ─────────────────────────────────────────────────────────────────────────────

class LoRAHypernetwork(Hypernetwork):
    """
    LoRA + Hypernetwork Hybrid.

    The base weight W_base is fixed (teacher weight, stored compressed).
    The hypernetwork generates only the LoRA delta: ΔW = A_H @ B_H.

    Final weight:  W = W_base + ΔW   (no weight copying, just addition)

    Benefits:
    * Lower rank needed (δ only, not full W)
    * Base weights can be INT8-quantised; delta stays FP16
    * Easier convergence (hypernetwork learns residual)

    Usage:
        hyp = LoRAHypernetwork(target_cfg, hypernet_cfg, teacher_weights)
        delta_weights = hyp.generate_all_deltas()
        final_weights = hyp.combine_with_base(delta_weights)
    """

    def __init__(
        self,
        target_cfg:     TargetModelConfig,
        hypernet_cfg:   HypernetworkConfig,
        base_weights:   List[Dict[str, torch.Tensor]],  # frozen teacher weights
    ):
        super().__init__(target_cfg, hypernet_cfg)

        # Store base weights as non-parameter buffers
        self.n_layers_target = target_cfg.n_layers
        for l_idx, layer_w in enumerate(base_weights):
            for key, w in layer_w.items():
                buf_name = f"base_L{l_idx}_{key.replace('.', '_')}"
                self.register_buffer(buf_name, w.detach().clone())

    def _get_base(self, layer_idx: int, key: str) -> torch.Tensor:
        buf_name = f"base_L{layer_idx}_{key.replace('.', '_')}"
        return getattr(self, buf_name)

    def generate_all_deltas(
        self,
        z: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate only the residual delta weights."""
        return self.generate_all_weights(z=z)

    def combine_with_base(
        self,
        deltas: List[Dict[str, torch.Tensor]],
        scale:  float = 0.1,        # LoRA α/r scaling factor
    ) -> List[Dict[str, torch.Tensor]]:
        """Add deltas to frozen base weights."""
        combined: List[Dict[str, torch.Tensor]] = []
        for l, delta_layer in enumerate(deltas):
            layer_w: Dict[str, torch.Tensor] = {}
            for key, dw in delta_layer.items():
                base = self._get_base(l, key)
                layer_w[key] = base + scale * dw
            combined.append(layer_w)
        return combined


# ─────────────────────────────────────────────────────────────────────────────
# 6. Token-Conditioned Weights (Advanced Extension)
# ─────────────────────────────────────────────────────────────────────────────

class TokenConditionedHypernetwork(Hypernetwork):
    """
    Generates weights conditioned on the CURRENT input tokens.

    Context vector:  c_token = MeanPool(token_embeddings)  ∈ ℝ^D
    Projected to:    z_token = Proj(c_token)                ∈ ℝ^latent_dim

    Then: H(layer_id, weight_type, z_token) → W

    Use case: Different weight specialisation per document/domain/task.

    WARNING: This is expensive — weights change per forward pass.
             Best used with caching keyed on the context vector.
    """

    def __init__(
        self,
        target_cfg:   TargetModelConfig,
        hypernet_cfg: HypernetworkConfig,
        context_proj: Optional[nn.Module] = None,
    ):
        super().__init__(target_cfg, hypernet_cfg)
        D       = target_cfg.hidden_dim
        lat_dim = hypernet_cfg.latent_dim

        # Project pooled token embeddings → latent space
        self.context_proj = context_proj or nn.Sequential(
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, lat_dim),
        )

    def encode_context(
        self,
        token_embeddings: torch.Tensor,   # [B, S, D]
        attention_mask:   Optional[torch.Tensor] = None,  # [B, S]
    ) -> torch.Tensor:
        """Mean-pool token embeddings → latent z  [B, latent_dim]."""
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            pooled = token_embeddings.mean(1)           # [B, D]
        return self.context_proj(pooled)                # [B, latent_dim]

    def generate_weights_from_tokens(
        self,
        token_embeddings: torch.Tensor,
        attention_mask:   Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate all weights conditioned on the input context."""
        # Take mean over batch for weight generation (weights are shared in batch)
        z = self.encode_context(token_embeddings, attention_mask)
        z_mean = z.mean(0, keepdim=True)                # [1, latent_dim]
        return self.generate_all_weights(z=z_mean)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Multi-Task Hypernetwork (Advanced Extension)
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskHypernetwork(Hypernetwork):
    """
    Supports multiple tasks via per-task latent codes.

    Each task gets a learned embedding: z_task ∈ ℝ^latent_dim.
    The hypernetwork generates task-specialised weights.

    Example tasks: LM, classification, summarisation, Q&A.

    Usage:
        hyp = MultiTaskHypernetwork(target_cfg, hypernet_cfg, n_tasks=4)
        weights = hyp.generate_for_task(task_id=2)
    """

    def __init__(
        self,
        target_cfg:   TargetModelConfig,
        hypernet_cfg: HypernetworkConfig,
        n_tasks:      int = 4,
        task_names:   Optional[List[str]] = None,
    ):
        super().__init__(target_cfg, hypernet_cfg)
        self.n_tasks    = n_tasks
        self.task_names = task_names or [f"task_{i}" for i in range(n_tasks)]

        # Learnable task embeddings
        self.task_embed = nn.Embedding(n_tasks, hypernet_cfg.latent_dim)
        nn.init.normal_(self.task_embed.weight, std=0.01)

    def generate_for_task(
        self,
        task_id: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate weights specialised for a given task."""
        device = next(self.parameters()).device
        z = self.task_embed(torch.tensor([task_id], device=device))  # [1, latent_dim]
        return self.generate_all_weights(z=z)

    def get_task_weights_all(self) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        """Generate weights for all tasks. Returns {task_name: weight_list}."""
        return {
            name: self.generate_for_task(i)
            for i, name in enumerate(self.task_names)
        }
