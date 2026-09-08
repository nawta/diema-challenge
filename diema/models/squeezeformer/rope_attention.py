"""Rotary Position Embedding (RoPE) + Multi-Head Self-Attention.

LLaMA-style rotary position embedding applied to Q/K (not V) inside a
standard multi-head self-attention module. Cached cos/sin buffers for
fast inference up to ``max_seq_len`` tokens.

Reference:
  - Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864
  - LLaMA (Touvron et al., 2023) — applies RoPE per attention head
  - ASL Fingerspelling 1st place solution (Dieter / Christof Henkel)
    uses this exact pattern in place of relative position encoding
    for ~2x training / ~3x tflite speedup.

Related:
  - diema/models/skateformer/partition_attention.py (vanilla MHSA baseline)
  - diema/models/squeezeformer/squeezeformer_block.py (consumer)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables for rotary position embedding.

    Implements the standard RoPE formulation:

        theta_i = 1.0 / (base ** (2i / dim)), i = 0 .. dim/2 - 1
        For position m:  (x_{2i}, x_{2i+1}) rotated by angle m * theta_i.

    The cos/sin tables are registered as persistent=False buffers so they
    move with the module across devices but are not saved to checkpoints.

    Args:
        dim: per-head feature dimension (must be even).
        max_seq_len: maximum sequence length to precompute for.
        base: frequency base (default 10000, as in LLaMA / RoFormer).
    """

    def __init__(self, dim: int, max_seq_len: int = 256, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, f"RotaryEmbedding requires even dim, got {dim}"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        # (max_seq_len, dim/2)
        freqs = torch.einsum("m,i->mi", positions, inv_freq)
        # Expand each pair: (max_seq_len, dim) via interleaving cos/sin-friendly format.
        cos = freqs.cos()
        sin = freqs.sin()
        # Store as (1, 1, max_seq_len, dim/2) for easy broadcast to (B, H, N, d/2)
        self.register_buffer("cos_cached", cos[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", sin[None, None, :, :], persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) slices shaped (1, 1, seq_len, dim/2)."""
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Requested RoPE length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to a Q or K tensor.

    Args:
        x: (B, H, N, D) where D is per-head dim (even).
        cos, sin: (1, 1, N, D/2) frequency tables.

    Returns:
        Rotated tensor of the same shape as ``x``.
    """
    # Split into the (even, odd) pair halves: x_2i and x_{2i+1}.
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    # Apply 2D rotation: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    # Interleave back into the original layout.
    out = torch.stack((rot1, rot2), dim=-1)
    return out.flatten(-2)


class RoPEMultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with LLaMA-style rotary position embedding.

    Standard pre-activation MHSA:
      - Linear(d, 3d) -> split into q, k, v
      - Apply RoPE to q and k only (NOT v)
      - Scaled dot-product attention (via torch.nn.functional.sdpa)
      - Linear(d, d) projection + dropout

    Args:
        dim: model dimension (must be divisible by num_heads).
        num_heads: number of attention heads.
        max_seq_len: max cached RoPE length (default 256, > T=64 here).
        attn_drop: dropout on attention output (applied via sdpa's dropout_p
            during training).
        proj_drop: dropout on the output projection.

    See:; ASL FS 1st place uses num_heads=4, d=192.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        max_seq_len: int = 256,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0, (
            f"head_dim must be even for RoPE, got {self.head_dim}"
        )
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop_p = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run MHSA + RoPE on sequence tensor.

        Args:
            x: (B, N, d_model) input sequence.
        Returns:
            (B, N, d_model) tensor of the same shape.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # (3, B, H, N, D)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rotary(N)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        dropout_p = self.attn_drop_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
