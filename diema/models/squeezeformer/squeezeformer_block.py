"""Squeezeformer block: Macaron FFN + RoPE MHSA + Conv Module + Macaron FFN.

Each of the four sub-blocks is wrapped in a pre-LayerNorm + residual with a
learnable scalar gamma (the "improved / squeeze" macaron variant used by the
ASL Fingerspelling 1st place solution). DropPath (stochastic depth) is
applied per sub-block output with a linearly increasing per-layer rate.

Reference:
  - Squeezeformer (Kim et al., 2022): https://arxiv.org/abs/2206.00888
  - Conformer (Gulati et al., 2020): https://arxiv.org/abs/2005.08100
  - ASL Fingerspelling 1st place solution (Dieter / Christof Henkel)

Related:
  - diema/models/squeezeformer/rope_attention.py (RoPE MHSA)
  - diema/models/squeezeformer/squeezeformer_model.py (full stack)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diema.models.squeezeformer.rope_attention import RoPEMultiHeadAttention


class DropPath(nn.Module):
    """Stochastic depth per sample.

    Drops the residual branch of a sub-block with probability ``drop_prob``.
    At inference (eval) mode this is an identity operation.

    Args:
        drop_prob: probability of dropping the residual branch.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class FeedForwardModule(nn.Module):
    """Macaron Feed-Forward sub-block.

    Structure: LayerNorm -> Linear(d, d*mlp_ratio) -> Swish -> Dropout
               -> Linear(d*mlp_ratio, d) -> Dropout.

    The two macaron halves each scale the residual by 0.5 at the block level.

    Args:
        dim: model dimension.
        mlp_ratio: hidden expansion ratio (default 4).
        dropout: dropout rate applied inside the FFN.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()  # Swish
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvolutionModule(nn.Module):
    """Conformer-style convolution sub-block.

    Structure (applied along the temporal axis N):
        LayerNorm
        -> Pointwise Conv1d(d -> 2d)
        -> GLU (split 2d into two halves along channels)
        -> Depthwise Conv1d(d -> d, kernel=k, groups=d, padding='same')
        -> BatchNorm1d
        -> Swish
        -> Pointwise Conv1d(d -> d)
        -> Dropout

    Args:
        dim: model dimension.
        kernel_size: depthwise conv kernel size (must be odd for "same" padding).
        dropout: dropout applied at the output.
    """

    def __init__(self, dim: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"
        padding = (kernel_size - 1) // 2

        self.norm = nn.LayerNorm(dim)
        self.pw_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1, bias=True)
        self.glu = nn.GLU(dim=1)
        self.dw_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=True,
        )
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()  # Swish
        self.pw_conv2 = nn.Conv1d(dim, dim, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model)
        x = self.norm(x)
        # (B, N, C) -> (B, C, N) for Conv1d
        x = x.transpose(1, 2)
        x = self.pw_conv1(x)
        x = self.glu(x)
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        # back to (B, N, C)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return x


class SqueezeformerBlock(nn.Module):
    """Improved Squeezeformer block (macaron FFN + MHSA + Conv + FFN).

    Four sub-blocks, each wrapped as:
        out = x + drop_path(gamma_i * sub_block(x))     # for MHSA and Conv
        out = x + 0.5 * drop_path(gamma_i * FFN(x))      # for macaron FFN halves

    The learnable per-sub-block gamma scalars (initialized to 1.0) are the
    "improved / squeeze" macaron variant from the ASL FS 1st place solution.
    The final LayerNorm over all blocks is applied at the model level.

    Args:
        dim: model dimension.
        num_heads: attention heads.
        mlp_ratio: FFN hidden expansion ratio.
        conv_kernel_size: depthwise conv kernel size (15 for DIEM-A T=64).
        dropout: dropout rate inside each sub-block.
        drop_path: stochastic depth rate for this layer's residuals.
        max_seq_len: max cached RoPE length.

    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        max_seq_len: int = 256,
    ):
        super().__init__()

        self.ffn1 = FeedForwardModule(dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.attn_norm = nn.LayerNorm(dim)
        # RoPEMultiHeadAttention already applies proj_drop to its output,
        # so we don't wrap another nn.Dropout around it — otherwise the
        # effective attention signal is halved at (1-p)^2.
        self.attn = RoPEMultiHeadAttention(
            dim=dim, num_heads=num_heads, max_seq_len=max_seq_len,
            attn_drop=dropout, proj_drop=dropout,
        )
        self.conv = ConvolutionModule(dim, kernel_size=conv_kernel_size, dropout=dropout)
        self.ffn2 = FeedForwardModule(dim, mlp_ratio=mlp_ratio, dropout=dropout)

        # Learnable scaling scalars (one per sub-block) — "improved" macaron variant.
        self.gamma_ffn1 = nn.Parameter(torch.ones(1))
        self.gamma_attn = nn.Parameter(torch.ones(1))
        self.gamma_conv = nn.Parameter(torch.ones(1))
        self.gamma_ffn2 = nn.Parameter(torch.ones(1))

        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model)

        # Macaron FFN (first half)
        x = x + 0.5 * self.drop_path(self.gamma_ffn1 * self.ffn1(x))

        # RoPE MHSA (RoPEMultiHeadAttention already applies proj_drop).
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.drop_path(self.gamma_attn * attn_out)

        # Convolution module
        x = x + self.drop_path(self.gamma_conv * self.conv(x))

        # Macaron FFN (second half)
        x = x + 0.5 * self.drop_path(self.gamma_ffn2 * self.ffn2(x))

        return x
