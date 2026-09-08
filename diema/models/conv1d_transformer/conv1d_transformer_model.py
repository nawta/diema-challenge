"""Conv1D + Transformer hybrid (hoyso48 / ASL Signs 1st place).

Faithful PyTorch port of the ASL Signs 1st place solution (hoyso48),
originally a 1.85M-param TensorFlow/Keras model. The architecture is a
stack of causal depthwise Conv1D blocks interleaved with BatchNorm+Swish
Transformer blocks, followed by a top Conv1D and global pooling.

Architecture (matching the original):
    stem: Conv1d(in_feat -> 192, k=1)
    Block A: 3x Conv1DBlock(kernel=17, causal) + 1x TransformerBlock
    Block B: 3x Conv1DBlock(kernel=17, causal) + 1x TransformerBlock
    top:  Conv1d(192 -> 384, k=1)
    GAP over T -> LateDropout(p=0.8, start_step=1000) -> Linear(384 -> num_class)

DIEM-A adaptation:
    The original model consumes per-frame 2D landmarks `(T, V*2)`.
    DIEM-A feeds `(N, C=6, T=64, V=25)` skeleton tensors, so we flatten
    the (C, V) axes into a single per-frame feature dim of C*V (=150 by
    default) and operate purely on the temporal dimension.

Compatible with diema's BaseModel contract:
    Input : (N, C, T, V)
    Output: {"logits": (N, num_class)}

Related: diema/models/skateformer/skateformer_model.py (peer transformer model),
         diema/models/ctrgcn/ctrgcn_model.py (BaseModel template)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diema.models.base import BaseModel


class Swish(nn.Module):
    """Swish / SiLU activation: x * sigmoid(x).

    Thin wrapper around F.silu used by Conv1DBlock and
    BNSwishTransformerBlock. See:
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution with causal (left-only) padding.

    The original hoyso48 solution uses causal padding so that each output
    position only depends on the current and past frames, matching the
    streaming-friendly behavior of the Conformer-style conv module.

    Padding rule: `F.pad(x, (kernel_size - 1, 0))` before the Conv1d,
    which has `padding=0` itself. See:
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=0,
            groups=channels,   # depthwise
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T). Pad only on the left side for causality.
        x = F.pad(x, (self.kernel_size - 1, 0))
        return self.conv(x)


class Conv1DBlock(nn.Module):
    """Depthwise + pointwise Conv1D block with Swish and residual.

    Mirrors the `Conv1DBlock` from hoyso48's ASL Signs 1st place writeup:
        x -> DepthwiseCausalConv1d -> BN -> Swish
          -> PointwiseConv1d -> BN -> Swish
          -> Dropout
          -> + residual

    """

    def __init__(self, channels: int, kernel_size: int = 17, drop_rate: float = 0.2):
        super().__init__()
        self.dw = CausalDepthwiseConv1d(channels, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = Swish()
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = Swish()
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T)
        residual = x
        h = self.dw(x)
        h = self.bn1(h)
        h = self.act1(h)
        h = self.pw(h)
        h = self.bn2(h)
        h = self.act2(h)
        h = self.drop(h)
        return h + residual


class BNSwishTransformerBlock(nn.Module):
    """Transformer block with BatchNorm1d + Swish in place of LayerNorm+GELU.

    Operates on `(N, C, T)` channels-first tensors:
        MHSA(4 heads, d_model=C) -> BN -> + residual
        FFN(C -> mlp*C -> C)     -> BN -> + residual  (Swish activation)

    Following the original hoyso48 solution we use BatchNorm1d rather than
    LayerNorm for better tflite compatibility; the rest is a standard
    post-norm Transformer block.

    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"BNSwishTransformerBlock: dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        self.dim = dim
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.bn_attn = nn.BatchNorm1d(dim)

        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            Swish(),
            nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity(),
            nn.Linear(hidden, dim),
        )
        self.bn_ffn = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T) — channels-first, matching the surrounding Conv1D blocks.
        # Attention: convert to (N, T, C) for nn.MultiheadAttention(batch_first=True).
        residual = x
        h = x.transpose(1, 2)                       # (N, T, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.transpose(1, 2)                       # (N, C, T)
        h = self.bn_attn(h)
        h = self.drop(h)
        x = residual + h

        residual = x
        h = x.transpose(1, 2)                       # (N, T, C)
        h = self.ffn(h)
        h = h.transpose(1, 2)                       # (N, C, T)
        h = self.bn_ffn(h)
        h = self.drop(h)
        x = residual + h
        return x


class LateDropout(nn.Module):
    """Dropout that only activates after `start_step` training steps.

    Matches hoyso48's "late dropout" trick: the final classification head
    is kept fully deterministic for the first `start_step` training steps,
    after which a high-rate dropout (p=0.8 by default) is applied. The
    module tracks its own internal counter; call sites should simply wrap
    their pre-classifier features, e.g. `feat = self.late_drop(feat)`.

    At eval / inference time no dropout is ever applied (standard behavior).

    """

    def __init__(self, p: float = 0.8, start_step: int = 1000):
        super().__init__()
        self.p = float(p)
        self.start_step = int(start_step)
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        # Count every training-mode forward as one step.
        self._step += 1
        if int(self._step.item()) < self.start_step:
            return x
        return F.dropout(x, p=self.p, training=True)

    def extra_repr(self) -> str:
        return f"p={self.p}, start_step={self.start_step}"


class Conv1DTransformer_Model(BaseModel):
    """hoyso48-style 1DCNN + Transformer hybrid for DIEM-A emotion classes.

    Architecture:
        (N, C, T, V) ->  reshape  -> (N, C*V, T)
                     ->  stem     -> Conv1d(C*V -> dim, k=1)
                     ->  Block x num_blocks:
                              3 x Conv1DBlock(dim, kernel_size)
                              1 x BNSwishTransformerBlock(dim)
                     ->  top      -> Conv1d(dim -> 2*dim, k=1)
                     ->  pool (GAP over T)
                     ->  LateDropout(p=0.8, start_step)
                     ->  Linear(2*dim -> num_class)

    Args:
        num_class: number of output classes (DIEM-A: 12)
        in_channels: input channels per joint (e.g., 6 for 6D rotation)
        num_nodes: number of skeleton joints (DIEM-A: 25)
        dim: model width (stem channel count). Default 192.
        num_blocks: number of (3xConv + 1xTransformer) macro blocks. Default 2.
        kernel_size: depthwise Conv1D kernel size. Default 17 (hoyso48 used 17).
        num_heads: transformer attention heads. Default 4.
        mlp_ratio: transformer FFN expansion. Default 2.0 (dim -> 2*dim -> dim).
        drop_rate: dropout inside Conv1DBlock and TransformerBlock. Default 0.2.
        late_dropout: probability of the final LateDropout. Default 0.8.
        late_dropout_start_step: steps before LateDropout activates. Default 1000.
        pool_type: "gap" (mean) or "gmp" (max) along the time axis. Default "gap".
    """

    def __init__(
        self,
        num_class: int,
        in_channels: int = 6,
        num_nodes: int = 25,
        dim: int = 192,
        num_blocks: int = 2,
        kernel_size: int = 17,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.2,
        late_dropout: float = 0.8,
        late_dropout_start_step: int = 1000,
        pool_type: str = "gap",
    ):
        super().__init__()
        self.num_class = num_class
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.dim = dim
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.pool_type = pool_type.lower()
        if self.pool_type not in ("gap", "gmp"):
            raise ValueError(f"pool_type must be 'gap' or 'gmp', got {pool_type}")

        in_feat = in_channels * num_nodes

        # Channel-wise BatchNorm on the flattened input features.
        self.data_bn = nn.BatchNorm1d(in_feat)

        # Stem: 1x1 projection from C*V -> dim.
        self.stem = nn.Conv1d(in_feat, dim, kernel_size=1, bias=False)

        # Macro blocks: each is 3 x Conv1DBlock + 1 x BNSwishTransformerBlock.
        blocks: list[nn.Module] = []
        for _ in range(num_blocks):
            for _ in range(3):
                blocks.append(Conv1DBlock(dim, kernel_size=kernel_size, drop_rate=drop_rate))
            blocks.append(
                BNSwishTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    attn_drop=drop_rate,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Top: 1x1 projection dim -> 2*dim before pooling (hoyso48 used 192 -> 384).
        top_dim = dim * 2
        self.top_conv = nn.Conv1d(dim, top_dim, kernel_size=1, bias=False)
        self.top_bn = nn.BatchNorm1d(top_dim)
        self.top_act = Swish()

        # Late dropout + final classifier.
        self.late_drop = LateDropout(p=late_dropout, start_step=late_dropout_start_step)
        self.fc = nn.Linear(top_dim, num_class)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Classifier follows the STGCN/CTR-GCN convention.
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / self.num_class))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_input(x)
        N, C, T, V = x.size()
        if C != self.in_channels or V != self.num_nodes:
            raise ValueError(
                f"Conv1DTransformer_Model expected (N, {self.in_channels}, T, {self.num_nodes}), "
                f"got {tuple(x.shape)}"
            )

        # (N, C, T, V) -> (N, C*V, T).
        # We transpose so that the per-frame feature is [v0_c0, v0_c1, ..., v24_c5].
        h = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        h = self.data_bn(h)

        h = self.stem(h)                         # (N, dim, T)

        for blk in self.blocks:
            h = blk(h)                           # (N, dim, T)

        h = self.top_conv(h)                     # (N, 2*dim, T)
        h = self.top_bn(h)
        h = self.top_act(h)

        if self.pool_type == "gap":
            h = h.mean(dim=2)                    # (N, 2*dim)
        else:  # gmp
            h = h.max(dim=2).values              # (N, 2*dim)

        features = h                             # (N, 2*dim) pre-classifier
        h = self.late_drop(h)
        logits = self.fc(h)
        return {"logits": logits, "features": features}

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the pre-classifier feature vector."""
        return self.dim * 2

    @property
    def output_dim(self) -> int:
        return self.num_class

    @classmethod
    def from_config(cls, config) -> "Conv1DTransformer_Model":
        return cls(
            num_class=config.model.num_class,
            in_channels=getattr(config.model, "in_channels", 6),
            num_nodes=getattr(config.skeleton, "num_nodes", 25),
            dim=getattr(config.model, "dim", 192),
            num_blocks=getattr(config.model, "num_blocks", 2),
            kernel_size=getattr(config.model, "kernel_size", 17),
            num_heads=getattr(config.model, "num_heads", 4),
            mlp_ratio=getattr(config.model, "mlp_ratio", 2.0),
            drop_rate=getattr(config.model, "drop_rate", 0.2),
            late_dropout=getattr(config.model, "late_dropout", 0.8),
            late_dropout_start_step=getattr(config.model, "late_dropout_start_step", 1000),
            pool_type=getattr(config.model, "pool_type", "gap"),
        )


from diema.models.registry import register_model  # noqa: E402

register_model("conv1d_transformer", Conv1DTransformer_Model)
