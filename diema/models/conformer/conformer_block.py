"""Conformer block building blocks: Macaron FFN, Conv module, ConformerBlock.

Faithful (pure-PyTorch) reimplementation of the Conformer block from
"Conformer: Convolution-augmented Transformer for Speech Recognition"
(Gulati et al., INTERSPEECH 2020, https://arxiv.org/abs/2005.08100).

The canonical block order is:
  1. 1/2 step Macaron FFN  (residual: x = x + 0.5 * FFN(x))
  2. Multi-Head Self-Attention (pre-norm, residual)
  3. Convolution Module (pre-norm, residual):
       pointwise Conv1d -> GLU -> depthwise Conv1d -> BN -> Swish ->
       pointwise Conv1d -> Dropout
  4. 1/2 step Macaron FFN  (residual: x = x + 0.5 * FFN(x))
  5. Final LayerNorm

All submodules use pre-LayerNorm and DropPath (stochastic depth) on each
residual branch, following the ASL FS 11th place adaptation (bliao).

Related:
  - diema/models/skateformer/partition_attention.py  (MultiHeadSelfAttention)
  - diema/models/conformer/conformer_model.py        (Conformer_Model)

"""

import torch
import torch.nn as nn

from diema.models.skateformer.partition_attention import MultiHeadSelfAttention


class DropPath(nn.Module):
    """Per-sample stochastic depth (drop_path).

    Implements the same behaviour as timm.models.layers.DropPath but without
    the extra dependency. For training mode only: during eval it is a no-op.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Shape (N, 1, 1, ...) so the mask broadcasts over all non-batch dims.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class FeedForwardModule(nn.Module):
    """Macaron feed-forward (FFN) with Swish activation.

    Used with a 1/2 residual scale on both sides of the Conformer block
    (hence "Macaron"). The expansion ratio is controlled by mlp_dim.

    Input/Output shape: (N, T, d_model).
    """

    def __init__(self, d_model: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, mlp_dim)
        self.activation = nn.SiLU()   # SiLU == Swish
        self.drop1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.linear1(y)
        y = self.activation(y)
        y = self.drop1(y)
        y = self.linear2(y)
        y = self.drop2(y)
        return y


class ConvolutionModule(nn.Module):
    """Conformer convolution module.

    Sequence:
        LayerNorm ->
        Pointwise Conv1d (d_model -> 2 * d_model) ->
        GLU (-> d_model) ->
        Depthwise Conv1d (kernel_size, symmetric padding) ->
        BatchNorm1d ->
        Swish ->
        Pointwise Conv1d (d_model -> d_model) ->
        Dropout

    Input/Output shape: (N, T, d_model).

    Note: we use symmetric (non-causal) padding so that the module preserves
    the time length and can operate on the whole clip at once. The kernel
    size must be odd; the DIEM-A adaptation uses kernel_size=15
    (the original paper uses 31; we reduce because clip_length=64).
    """

    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, (
            f"ConvolutionModule requires odd kernel_size, got {kernel_size}"
        )
        padding = (kernel_size - 1) // 2

        self.norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(
            d_model, 2 * d_model, kernel_size=1, stride=1, padding=0, bias=True,
        )
        # GLU along channel dim: split 2d -> (d, d) -> a * sigmoid(b)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=d_model,
            bias=True,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()   # Swish
        self.pointwise_conv2 = nn.Conv1d(
            d_model, d_model, kernel_size=1, stride=1, padding=0, bias=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, d_model)
        y = self.norm(x)
        # -> (N, d_model, T) for Conv1d
        y = y.transpose(1, 2)
        y = self.pointwise_conv1(y)       # (N, 2*d, T)
        y = self.glu(y)                   # (N, d, T)
        y = self.depthwise_conv(y)        # (N, d, T)
        y = self.batch_norm(y)
        y = self.activation(y)
        y = self.pointwise_conv2(y)       # (N, d, T)
        y = self.dropout(y)
        # -> (N, T, d_model)
        return y.transpose(1, 2)


class ConformerBlock(nn.Module):
    """Single Conformer encoder block.

    Implements the macaron-style residual structure:
        x = x + 0.5 * FFN_1(x)
        x = x + MHSA(x)
        x = x + Conv(x)
        x = x + 0.5 * FFN_2(x)
        x = LayerNorm(x)

    Each residual branch is wrapped with a DropPath (stochastic depth) layer
    for regularisation, matching the bliao recipe.

    Input/Output shape: (N, T, d_model).
    """

    def __init__(
        self,
        d_model: int,
        mlp_dim: int,
        num_heads: int,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, mlp_dim, dropout=dropout)
        self.mhsa_norm = nn.LayerNorm(d_model)
        self.mhsa = MultiHeadSelfAttention(
            dim=d_model,
            num_heads=num_heads,
            attn_drop=attn_dropout,
            proj_drop=dropout,
        )
        self.mhsa_drop = nn.Dropout(dropout)
        self.conv = ConvolutionModule(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        self.ffn2 = FeedForwardModule(d_model, mlp_dim, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model)

        # Separate DropPath per residual branch: all share the same drop prob
        # within a block but are drawn independently each step.
        self.drop_path_ffn1 = DropPath(drop_path)
        self.drop_path_mhsa = DropPath(drop_path)
        self.drop_path_conv = DropPath(drop_path)
        self.drop_path_ffn2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, d_model)
        x = x + 0.5 * self.drop_path_ffn1(self.ffn1(x))
        x = x + self.drop_path_mhsa(self.mhsa_drop(self.mhsa(self.mhsa_norm(x))))
        x = x + self.drop_path_conv(self.conv(x))
        x = x + 0.5 * self.drop_path_ffn2(self.ffn2(x))
        x = self.final_norm(x)
        return x
