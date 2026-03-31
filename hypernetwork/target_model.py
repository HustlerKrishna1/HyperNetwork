"""
target_model.py  —  Transformer T that accepts externally-generated weights.

Design goals
============
* Standard decoder-only transformer (GPT-style) for language modelling.
* Two operating modes:
    1. STANDALONE  – maintains its own nn.Parameter weights (teacher training).
    2. GENERATED   – accepts a dict of weight tensors produced by Hypernetwork.
* Both modes share identical forward logic so gradients flow correctly in
  either mode.

Tensor shapes throughout (single sample, seq_len=S, hidden=D=256):
  x             : [B, S]         token ids
  embed         : [B, S, D]
  after LN      : [B, S, D]
  Q / K / V     : [B, n_heads, S, head_dim]  = [B, 4, S, 64]
  attn_scores   : [B, 4, S, S]
  attn_out      : [B, S, D]
  ffn hidden    : [B, S, ffn_dim]  = [B, S, 1024]
  logits        : [B, S, vocab_size]
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TargetModelConfig


# ---------------------------------------------------------------------------
# Causal self-attention (functional — receives weight tensors as args)
# ---------------------------------------------------------------------------

def causal_self_attention(
    x:        torch.Tensor,          # [B, S, D]
    q_weight: torch.Tensor,          # [D, D]
    k_weight: torch.Tensor,          # [D, D]
    v_weight: torch.Tensor,          # [D, D]
    o_weight: torch.Tensor,          # [D, D]
    q_bias:   Optional[torch.Tensor],# [D]
    k_bias:   Optional[torch.Tensor],# [D]
    v_bias:   Optional[torch.Tensor],# [D]
    o_bias:   Optional[torch.Tensor],# [D]
    n_heads:  int,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Multi-head causal self-attention using functional weights.

    Returns attn_out : [B, S, D]
    """
    B, S, D = x.shape
    head_dim = D // n_heads

    # Linear projections  [B, S, D]
    Q = F.linear(x, q_weight, q_bias)
    K = F.linear(x, k_weight, k_bias)
    V = F.linear(x, v_weight, v_bias)

    # Reshape to [B, n_heads, S, head_dim]
    def split_heads(t: torch.Tensor) -> torch.Tensor:
        return t.view(B, S, n_heads, head_dim).transpose(1, 2)

    Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

    # Scaled dot-product attention with causal mask
    scale   = math.sqrt(head_dim)
    scores  = torch.matmul(Q, K.transpose(-2, -1)) / scale   # [B, h, S, S]

    # Causal mask (upper triangular = -inf)
    mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))

    attn  = F.softmax(scores, dim=-1)
    attn  = F.dropout(attn, p=dropout_p, training=training)

    # Context  [B, h, S, head_dim]  →  [B, S, D]
    ctx  = torch.matmul(attn, V)
    ctx  = ctx.transpose(1, 2).contiguous().view(B, S, D)

    out  = F.linear(ctx, o_weight, o_bias)
    return out


# ---------------------------------------------------------------------------
# Feed-forward block (functional)
# ---------------------------------------------------------------------------

def feed_forward(
    x:          torch.Tensor,          # [B, S, D]
    ffn1_weight: torch.Tensor,         # [ffn_dim, D]
    ffn2_weight: torch.Tensor,         # [D, ffn_dim]
    ffn1_bias:  Optional[torch.Tensor],# [ffn_dim]
    ffn2_bias:  Optional[torch.Tensor],# [D]
    dropout_p:  float = 0.0,
    training:   bool = False,
) -> torch.Tensor:
    """
    Two-layer MLP with GELU activation.

    Returns [B, S, D]
    """
    h = F.linear(x, ffn1_weight, ffn1_bias)      # [B, S, ffn_dim]
    h = F.gelu(h)
    h = F.dropout(h, p=dropout_p, training=training)
    h = F.linear(h, ffn2_weight, ffn2_bias)       # [B, S, D]
    return h


# ---------------------------------------------------------------------------
# Transformer block (standalone mode — stores its own parameters)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Single transformer decoder block.

    Stores its own weights as nn.Parameters for standalone (teacher) training.
    Can also run in 'generated' mode by calling forward_generated().
    """

    def __init__(self, cfg: TargetModelConfig):
        super().__init__()
        D, F, H = cfg.hidden_dim, cfg.ffn_dim, cfg.n_heads
        self.n_heads   = H
        self.dropout_p = cfg.dropout

        # Layer norms (always stored — small, not generated)
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)

        # Attention projections
        self.q_weight = nn.Parameter(torch.empty(D, D))
        self.k_weight = nn.Parameter(torch.empty(D, D))
        self.v_weight = nn.Parameter(torch.empty(D, D))
        self.o_weight = nn.Parameter(torch.empty(D, D))
        self.q_bias   = nn.Parameter(torch.zeros(D))
        self.k_bias   = nn.Parameter(torch.zeros(D))
        self.v_bias   = nn.Parameter(torch.zeros(D))
        self.o_bias   = nn.Parameter(torch.zeros(D))

        # Feed-forward
        self.ffn1_weight = nn.Parameter(torch.empty(F, D))
        self.ffn2_weight = nn.Parameter(torch.empty(D, F))
        self.ffn1_bias   = nn.Parameter(torch.zeros(F))
        self.ffn2_bias   = nn.Parameter(torch.zeros(D))

        self._init_weights(D, F)

    def _init_weights(self, D: int, F: int):
        for w, fan_in in [
            (self.q_weight, D), (self.k_weight, D),
            (self.v_weight, D), (self.o_weight, D),
        ]:
            nn.init.normal_(w, std=0.02)
        for w, fan_in in [(self.ffn1_weight, D), (self.ffn2_weight, F)]:
            nn.init.normal_(w, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standalone forward using own parameters."""
        # Attention sub-block
        attn_out = causal_self_attention(
            self.ln1(x),
            self.q_weight, self.k_weight, self.v_weight, self.o_weight,
            self.q_bias,   self.k_bias,   self.v_bias,   self.o_bias,
            self.n_heads, self.dropout_p, self.training,
        )
        x = x + attn_out

        # FFN sub-block
        ffn_out = feed_forward(
            self.ln2(x),
            self.ffn1_weight, self.ffn2_weight,
            self.ffn1_bias,   self.ffn2_bias,
            self.dropout_p, self.training,
        )
        x = x + ffn_out
        return x

    def forward_generated(
        self,
        x:       torch.Tensor,
        weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass using externally-generated weights.
        'weights' must contain all keys from TargetModelConfig.weight_shapes.
        """
        attn_out = causal_self_attention(
            self.ln1(x),
            weights["q_weight"], weights["k_weight"],
            weights["v_weight"], weights["o_weight"],
            weights.get("q_bias"), weights.get("k_bias"),
            weights.get("v_bias"), weights.get("o_bias"),
            self.n_heads, self.dropout_p, self.training,
        )
        x = x + attn_out

        ffn_out = feed_forward(
            self.ln2(x),
            weights["ffn1_weight"], weights["ffn2_weight"],
            weights.get("ffn1_bias"), weights.get("ffn2_bias"),
            self.dropout_p, self.training,
        )
        x = x + ffn_out
        return x

    def get_weight_dict(self) -> Dict[str, torch.Tensor]:
        """Export own weights in the same format as the hypernetwork produces."""
        return {
            "q_weight":    self.q_weight,
            "k_weight":    self.k_weight,
            "v_weight":    self.v_weight,
            "o_weight":    self.o_weight,
            "q_bias":      self.q_bias,
            "k_bias":      self.k_bias,
            "v_bias":      self.v_bias,
            "o_bias":      self.o_bias,
            "ffn1_weight": self.ffn1_weight,
            "ffn2_weight": self.ffn2_weight,
            "ffn1_bias":   self.ffn1_bias,
            "ffn2_bias":   self.ffn2_bias,
        }


# ---------------------------------------------------------------------------
# Full decoder transformer
# ---------------------------------------------------------------------------

class TransformerLM(nn.Module):
    """
    Decoder-only transformer language model.

    Can be run in two modes:
    * standalone  (generated_weights=None)  — uses its own block parameters.
    * generated   (generated_weights=list)  — uses supplied weight dicts.
    """

    def __init__(self, cfg: TargetModelConfig):
        super().__init__()
        self.cfg = cfg
        D, V, S = cfg.hidden_dim, cfg.vocab_size, cfg.max_seq_len

        self.token_embed = nn.Embedding(V, D)
        self.pos_embed   = nn.Embedding(S, D)
        self.embed_drop  = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        self.ln_f    = nn.LayerNorm(D)
        # Language model head — optionally tied to token_embed
        self.lm_head = nn.Linear(D, V, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight,   std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.lm_head:
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── forward (standalone) ─────────────────────────────────────────────────

    def forward(
        self,
        input_ids:        torch.Tensor,                          # [B, S]
        generated_weights: Optional[List[Dict[str, torch.Tensor]]] = None,
        labels:           Optional[torch.Tensor] = None,         # [B, S]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        input_ids         : [B, S] token ids
        generated_weights : list of L weight dicts (one per block) or None
        labels            : [B, S] targets for cross-entropy loss

        Returns
        -------
        logits : [B, S, vocab_size]
        loss   : scalar or None
        """
        B, S = input_ids.shape
        device = input_ids.device

        pos = torch.arange(S, device=device).unsqueeze(0)      # [1, S]
        x   = self.token_embed(input_ids) + self.pos_embed(pos)
        x   = self.embed_drop(x)

        for i, block in enumerate(self.blocks):
            if generated_weights is not None:
                x = block.forward_generated(x, generated_weights[i])
            else:
                x = block(x)

        x      = self.ln_f(x)
        logits = self.lm_head(x)              # [B, S, V]

        loss = None
        if labels is not None:
            # Shift: predict token t+1 from token t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    # ── hidden states (for distillation) ─────────────────────────────────────

    def forward_with_hidden(
        self,
        input_ids:        torch.Tensor,
        generated_weights: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Returns (logits, list_of_hidden_states_per_block)."""
        B, S = input_ids.shape
        device = input_ids.device

        pos = torch.arange(S, device=device).unsqueeze(0)
        x   = self.token_embed(input_ids) + self.pos_embed(pos)
        x   = self.embed_drop(x)

        hiddens: List[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            if generated_weights is not None:
                x = block.forward_generated(x, generated_weights[i])
            else:
                x = block(x)
            hiddens.append(x)

        x      = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, hiddens

    # ── teacher weight extraction ─────────────────────────────────────────────

    def get_all_weight_dicts(self) -> List[Dict[str, torch.Tensor]]:
        """Export per-block weight dicts (used as distillation targets)."""
        return [block.get_weight_dict() for block in self.blocks]

    # ── utilities ─────────────────────────────────────────────────────────────

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def size_mb(self, dtype_bytes: int = 4) -> float:
        return self.param_count * dtype_bytes / (1024 ** 2)

    def __repr__(self) -> str:  # type: ignore[override]
        return (
            f"TransformerLM(\n"
            f"  vocab_size = {self.cfg.vocab_size}\n"
            f"  hidden_dim = {self.cfg.hidden_dim}\n"
            f"  n_layers   = {self.cfg.n_layers}\n"
            f"  n_heads    = {self.cfg.n_heads}\n"
            f"  ffn_dim    = {self.cfg.ffn_dim}\n"
            f"  params     = {self.param_count / 1e6:.2f} M\n"
            f"  size (FP32)= {self.size_mb():.1f} MB\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_target_model(cfg: TargetModelConfig) -> TransformerLM:
    return TransformerLM(cfg)
