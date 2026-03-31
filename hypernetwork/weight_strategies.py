"""
weight_strategies.py  —  Three weight-generation strategies compared.

Strategy 1 — Low-Rank Factorisation (PRIMARY ✅)
================================================
    W = A @ B      A ∈ ℝ^{out × r},  B ∈ ℝ^{r × in}

    Pros:  simple, differentiable, rank controls param count linearly
    Cons:  rank bottleneck — very low rank degrades accuracy

    Parameter cost:  out*r + r*in = r*(out+in)
    For D=256, r=32:  32*(256+256) = 16 384 per head  (vs 65 536 full)
    Compression:  4×  (for square D×D matrices)

Strategy 2 — Chunked Generation
================================
    Split W into K chunks of size [chunk_size, in].
    H generates each chunk conditioned on (layer_id, weight_type, chunk_idx).

    Pros:  no rank bottleneck, any matrix size supported cleanly
    Cons:  K forward passes per weight matrix → latency overhead

    Parameter cost:  H params (fixed) + chunk positional embedding

Strategy 3 — Implicit Neural Representation (INR)
==================================================
    Map coordinate (i, j) → W[i,j] via a SIREN or NeRF-style network.

    W[i,j] = MLP(ω₀ * [i/out, j/in, layer_id, weight_type])

    Pros:  arbitrary matrix sizes, smooth inductive bias
    Cons:  slow (one pass per element unless batched), training instability
           with vanilla MLPs (requires careful ω₀ tuning)

Decision: LOW-RANK is primary. Used in Hypernetwork class.
          Chunked and INR provided here as drop-in generators for experiments.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HypernetworkConfig, TargetModelConfig


# ---------------------------------------------------------------------------
# Strategy 1 (reference) — Low-Rank Generator
# ---------------------------------------------------------------------------

class LowRankWeightGenerator(nn.Module):
    """
    Standalone low-rank weight generator for a single weight matrix shape.

    W = A @ B,   A:[out, rank],  B:[rank, in]

    Input:  conditioning vector c  of dimension cond_dim
    Output: W  of shape  (out_dim, in_dim)

    Example
    -------
    >>> gen = LowRankWeightGenerator(cond_dim=160, out_dim=256, in_dim=256, rank=32)
    >>> c   = torch.randn(4, 160)          # batch of 4
    >>> W   = gen(c)                       # [4, 256, 256]
    """

    def __init__(
        self,
        cond_dim: int,
        out_dim:  int,
        in_dim:   int,
        rank:     int,
        hidden_dim: int = 256,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.in_dim  = in_dim
        self.rank    = rank

        # Two separate heads per the A and B factors
        self.head_A = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim * rank),
        )
        self.head_B = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, rank * in_dim),
        )

        # Orthogonal init for A, small init for B
        for m in self.head_A.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=init_scale)
                if m.bias is not None: nn.init.zeros_(m.bias)
        for m in self.head_B.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=init_scale / math.sqrt(rank))
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        c : [B, cond_dim]
        Returns W : [B, out_dim, in_dim]
        """
        B = c.size(0)
        A = self.head_A(c).view(B, self.out_dim, self.rank)   # [B, out, r]
        B_ = self.head_B(c).view(B, self.rank, self.in_dim)   # [B, r, in]
        return torch.bmm(A, B_)                                # [B, out, in]

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def compression_ratio(self) -> float:
        full = self.out_dim * self.in_dim
        low_rank = self.rank * (self.out_dim + self.in_dim)
        return full / low_rank


# ---------------------------------------------------------------------------
# Strategy 2 — Chunked Generator
# ---------------------------------------------------------------------------

class ChunkedWeightGenerator(nn.Module):
    """
    Generates a weight matrix W[out, in] in vertical chunks of 'chunk_size' rows.

    For each chunk k (k = 0 … ceil(out/chunk_size)):
        chunk_cond = concat(cond, chunk_embed[k])
        W[k*cs : (k+1)*cs, :] = MLP(chunk_cond)

    Pros:  no rank bottleneck; the full row capacity is generated.
    Cons:  O(out/chunk_size) forward passes per weight.

    Shapes
    ------
    cond       : [B, cond_dim]
    chunk_embed: learned [n_chunks, chunk_embed_dim]
    MLP output : [B, chunk_size * in_dim]  →  reshape  →  [B, chunk_size, in_dim]
    Full W     : [B, out_dim, in_dim]
    """

    def __init__(
        self,
        cond_dim:        int,
        out_dim:         int,
        in_dim:          int,
        chunk_size:      int = 64,
        chunk_embed_dim: int = 32,
        hidden_dim:      int = 256,
    ):
        super().__init__()
        self.out_dim    = out_dim
        self.in_dim     = in_dim
        self.chunk_size = chunk_size
        self.n_chunks   = math.ceil(out_dim / chunk_size)

        # Chunk positional embedding
        self.chunk_embed = nn.Embedding(self.n_chunks, chunk_embed_dim)

        total_cond = cond_dim + chunk_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_cond, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, chunk_size * in_dim),
        )

        self._init()

    def _init(self):
        nn.init.normal_(self.chunk_embed.weight, std=0.01)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        cond : [B, cond_dim]
        Returns W : [B, out_dim, in_dim]
        """
        B      = cond.size(0)
        device = cond.device
        chunks: List[torch.Tensor] = []

        for k in range(self.n_chunks):
            ce      = self.chunk_embed(
                torch.tensor([k], device=device).expand(B)
            )                                        # [B, chunk_embed_dim]
            c_full  = torch.cat([cond, ce], dim=-1)  # [B, cond+embed]
            chunk   = self.mlp(c_full)               # [B, chunk_size*in_dim]
            chunk   = chunk.view(B, self.chunk_size, self.in_dim)
            chunks.append(chunk)

        W = torch.cat(chunks, dim=1)                 # [B, n_chunks*chunk_size, in]
        W = W[:, :self.out_dim, :]                   # trim padding
        return W                                     # [B, out_dim, in_dim]


# ---------------------------------------------------------------------------
# Strategy 3 — Implicit Neural Representation (SIREN-like)
# ---------------------------------------------------------------------------

class SineLayer(nn.Module):
    """Single SIREN layer: y = sin(ω₀ · Wx + b)."""
    def __init__(self, in_f: int, out_f: int, omega_0: float = 30.0,
                 is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear  = nn.Linear(in_f, out_f)
        self._init(in_f, is_first)

    def _init(self, in_f: int, is_first: bool):
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_f, 1 / in_f)
            else:
                bound = math.sqrt(6 / in_f) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class INRWeightGenerator(nn.Module):
    """
    Implicit Neural Representation (INR) weight generator.

    Maps (row_coord, col_coord, layer_cond, type_cond) → W[i,j]

    Coordinates are normalised to [-1, 1].

    Usage note:  This is the SLOWEST strategy but has strong inductive bias
                 for smooth weight matrices. Best for small matrices.

    Batched inference:  We flatten the (out × in) grid into a single batch
                        of coordinates, making it a single forward pass.
    """

    def __init__(
        self,
        out_dim:     int,
        in_dim:      int,
        cond_dim:    int,
        hidden_dim:  int = 128,
        n_layers:    int = 4,
        omega_0:     float = 30.0,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.in_dim  = in_dim

        # Coordinate input: (row_norm, col_norm) + cond  →  2 + cond_dim
        coord_dim   = 2 + cond_dim
        self.layers = nn.ModuleList()
        self.layers.append(SineLayer(coord_dim, hidden_dim, omega_0, is_first=True))
        for _ in range(n_layers - 2):
            self.layers.append(SineLayer(hidden_dim, hidden_dim, omega_0))
        # Final linear (no sine activation on last layer)
        self.final = nn.Linear(hidden_dim, 1)
        nn.init.uniform_(self.final.weight, -math.sqrt(6/hidden_dim)/omega_0,
                          math.sqrt(6/hidden_dim)/omega_0)

        # Pre-compute coordinate grid  [out*in, 2]
        rows = torch.linspace(-1, 1, out_dim)
        cols = torch.linspace(-1, 1, in_dim)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")  # [out,in] each
        # coords : [out*in, 2]
        self.register_buffer(
            "coords",
            torch.stack([grid_r.flatten(), grid_c.flatten()], dim=-1)
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        cond : [B, cond_dim]
        Returns W : [B, out_dim, in_dim]
        """
        B      = cond.size(0)
        N      = self.out_dim * self.in_dim   # total matrix elements
        device = cond.device

        # Expand coords to batch: [N, 2] → [B, N, 2]
        coords_exp = self.coords.unsqueeze(0).expand(B, -1, -1)   # [B, N, 2]

        # Expand cond to all positions: [B, cond_dim] → [B, N, cond_dim]
        cond_exp   = cond.unsqueeze(1).expand(-1, N, -1)

        # Input to network: [B, N, 2+cond_dim]
        inp = torch.cat([coords_exp, cond_exp], dim=-1)
        inp = inp.view(B * N, -1)                 # [B*N, 2+cond_dim]

        # SIREN forward
        x = inp
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)                         # [B*N, 1]

        W = x.view(B, self.out_dim, self.in_dim)
        return W


# ---------------------------------------------------------------------------
# Strategy Comparison
# ---------------------------------------------------------------------------

def compare_strategies(
    target_cfg:   TargetModelConfig,
    hypernet_cfg: HypernetworkConfig,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Print parameter counts and compression ratios for all three strategies
    applied to each weight type.

    Returns dict of {strategy_name: {weight_key: stats}}
    """
    cond_dim = (hypernet_cfg.layer_embed_dim +
                hypernet_cfg.type_embed_dim +
                hypernet_cfg.latent_dim)
    rank      = hypernet_cfg.rank
    chunk_sz  = hypernet_cfg.chunk_size
    shapes    = target_cfg.weight_shapes
    results: Dict[str, Dict] = {}

    for strategy_name in ["lowrank", "chunked", "implicit"]:
        results[strategy_name] = {}
        total_gen_params = 0

        for key in target_cfg.matrix_weight_keys:
            out_d, in_d = shapes[key]
            full_params  = out_d * in_d

            if strategy_name == "lowrank":
                gen_params = rank * (out_d + in_d) + 2 * 256  # approx MLP heads
                cr = full_params / (rank * (out_d + in_d))
            elif strategy_name == "chunked":
                n_chunks   = math.ceil(out_d / chunk_sz)
                gen_params = n_chunks * chunk_sz * in_d + 2 * 256  # approx
                cr = 1.0   # chunked generates full matrix, no compression
            else:  # implicit
                gen_params = 4 * 128 * 128  # SIREN hidden layers
                cr = full_params / gen_params

            results[strategy_name][key] = {
                "out_dim":     out_d,
                "in_dim":      in_d,
                "full_params": full_params,
                "gen_params":  gen_params,
                "compression": cr,
            }
            total_gen_params += gen_params

        results[strategy_name]["_total_gen_params"] = total_gen_params
        results[strategy_name]["_size_mb_fp32"]     = total_gen_params * 4 / (1024**2)

    if verbose:
        print("\n" + "="*70)
        print(f"{'Weight Generation Strategy Comparison':^70}")
        print("="*70)
        hdr = f"{'Key':<16} {'Shape':<14} {'Full':>8} {'LowRnk':>8} {'Chunk':>8} {'INR':>8}"
        print(hdr)
        print("-"*70)
        for key in target_cfg.matrix_weight_keys:
            s   = shapes[key]
            lr  = results['lowrank'][key]
            ch  = results['chunked'][key]
            inr = results['implicit'][key]
            print(f"{key:<16} {str(s):<14} "
                  f"{s[0]*s[1]:>8,} "
                  f"{lr['gen_params']:>8,} "
                  f"{ch['gen_params']:>8,} "
                  f"{inr['gen_params']:>8,}")
        print("-"*70)
        for s_name in ["lowrank", "chunked", "implicit"]:
            tp   = results[s_name]["_total_gen_params"]
            mb   = results[s_name]["_size_mb_fp32"]
            print(f"[{s_name:>8}]  Total gen params: {tp:>10,}  ({mb:.2f} MB FP32)")
        print("="*70)
        print("\n✅ RECOMMENDATION: LowRank — best compression/accuracy tradeoff")
        print("   Rank 32 → 4× compression for square D×D weight matrices")
        print("   Rank 16 → 8× compression  (some accuracy loss at rank < 16)")
        print()

    return results


# ---------------------------------------------------------------------------
# Rank sensitivity analysis
# ---------------------------------------------------------------------------

def rank_sensitivity(
    target_cfg: TargetModelConfig,
    ranks: Optional[List[int]] = None,
) -> None:
    """Print compression ratios and approximate parameter budget per rank."""
    if ranks is None:
        ranks = [4, 8, 16, 32, 64, 128]

    D, F = target_cfg.hidden_dim, target_cfg.ffn_dim
    L    = target_cfg.n_layers

    print(f"\n{'Rank':<6} {'D×D compr':>10} {'D×F compr':>10} "
          f"{'Total weight params':>20} {'Approx MB FP32':>15}")
    print("-" * 65)

    for r in ranks:
        # Square: Q, K, V, O  (4 per layer)
        sq_full     = D * D
        sq_lr       = r * (D + D)
        sq_compr    = sq_full / sq_lr

        # Rectangular: FFN1 (F×D) and FFN2 (D×F)  (2 per layer)
        rect_full   = F * D
        rect_lr     = r * (F + D)
        rect_compr  = rect_full / rect_lr

        # Total per layer (weights only, excl biases)
        per_layer = 4 * sq_lr + 2 * rect_lr
        total     = L * per_layer
        mb        = total * 4 / (1024**2)

        print(f"{r:<6} {sq_compr:>10.1f}×  {rect_compr:>10.1f}× "
              f"{total:>20,} {mb:>14.2f}")
