"""Partitioned attention modules for SkateFormer.

The original SkateFormer uses 4 partitioned attention types:
  - Skeletal-temporal local (intra-body-part, intra-window)
  - Skeletal-temporal global (intra-body-part, inter-window)
  - Skeletal local (inter-body-part, intra-window)
  - Skeletal global (inter-body-part, inter-window)

This is a simplified adaptation that captures the key idea: factorize
attention along body-part groups and temporal windows so that the cost
scales linearly in T*V instead of quadratically.

Tensor convention: (B, T, V, C) for clarity in this module.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_btvc(x: torch.Tensor) -> torch.Tensor:
    """(N, C, T, V) → (N, T, V, C)."""
    return x.permute(0, 2, 3, 1).contiguous()


def _to_nctv(x: torch.Tensor) -> torch.Tensor:
    """(N, T, V, C) → (N, C, T, V)."""
    return x.permute(0, 3, 1, 2).contiguous()


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C) where N = number of tokens
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (B, H, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class PartitionedAttention(nn.Module):
    """Apply self-attention within body-part × temporal-window partitions.

    Steps:
      1. Reshape (N, T, V, C) to (N * num_parts * num_windows, T_window * V_part, C)
      2. Run self-attention on each partition (token count is small)
      3. Reshape back to (N, T, V, C)

    Args:
        dim: feature dimension
        num_heads: attention heads
        body_parts: list of node-index lists, must cover all V nodes
        window_size: temporal window length (T must be divisible by window_size)
        attn_drop, proj_drop: dropout rates
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        body_parts: Sequence[Sequence[int]],
        window_size: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.body_parts = [list(p) for p in body_parts]
        self.window_size = window_size
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_drop, proj_drop)

        # Validate parts cover [0, V-1]
        all_nodes = sorted(n for p in self.body_parts for n in p)
        assert len(all_nodes) == len(set(all_nodes)), "body parts must be disjoint"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, V, C)
        N, T, V, C = x.shape
        ws = self.window_size
        # Pad T to multiple of window_size
        if T % ws != 0:
            pad = ws - (T % ws)
            x = F.pad(x, (0, 0, 0, 0, 0, pad))   # pad along T dim
            T_pad = T + pad
        else:
            T_pad = T
        num_windows = T_pad // ws

        out = torch.zeros_like(x)
        for part in self.body_parts:
            # Gather nodes for this body part: (N, T_pad, V_part, C)
            idx = torch.tensor(part, dtype=torch.long, device=x.device)
            xp = x.index_select(2, idx)
            V_part = len(part)

            # Window the time axis: (N, num_windows, ws, V_part, C)
            xp = xp.view(N, num_windows, ws, V_part, C)
            # Flatten to per-partition tokens: (N * num_windows, ws * V_part, C)
            xp = xp.reshape(N * num_windows, ws * V_part, C)

            yp = self.attn(xp)
            yp = yp.reshape(N, num_windows, ws, V_part, C)
            yp = yp.reshape(N, T_pad, V_part, C)

            # Scatter back into the output tensor
            out.index_copy_(2, idx, yp)

        # Crop back to original T length
        return out[:, :T]


class PartitionedAttentionBlock(nn.Module):
    """Standard pre-norm Transformer block with partitioned attention + MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        body_parts: Sequence[Sequence[int]],
        window_size: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PartitionedAttention(
            dim, num_heads, body_parts, window_size, attn_drop, proj_drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_drop),
        )
        self.drop_path_rate = drop_path

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_path_rate <= 0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_path_rate
        # Bernoulli mask per sample
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, V, C)
        x = x + self._drop_path(self.attn(self.norm1(x)))
        x = x + self._drop_path(self.mlp(self.norm2(x)))
        return x
