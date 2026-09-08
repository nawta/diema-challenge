"""Squeezeformer + RoPE model for DIEM-A 12-class emotion classification.

17-layer Improved Squeezeformer encoder (ASL FS 1st place / Dieter /
Christof Henkel) adapted to the DIEM-A skeleton input contract:

  Input:  (N, C, T, V) = (N, 6, 64, 25)  — 6D rotation, 64 frames, 25 nodes.
  Output: {"logits": (N, num_class)}

Pipeline:
  1. TokenEmbedding: Conv2d(C -> d, kernel=(3, 1)) + BN -> (N, d, T, V)
  2. Average over V                            -> (N, d, T)
  3. Permute                                   -> (N, T, d)
  4. N_layers Squeezeformer blocks (RoPE MHSA) -> (N, T, d)
  5. Final LayerNorm
  6. GAP over T                                -> (N, d)
  7. Linear(d -> num_class)

No absolute position embedding — rotary position embedding is applied
inside each RoPE MHSA sub-block.

Reference:
  - Squeezeformer: https://arxiv.org/abs/2206.00888
  - ASL FS 1st place:
      https://www.kaggle.com/competitions/asl-fingerspelling/writeups/darragh-dieter-1st-place-solution-improved-squeeze

Related:
  - diema/models/squeezeformer/rope_attention.py
  - diema/models/squeezeformer/squeezeformer_block.py
  - diema/models/skateformer/skateformer_model.py (TokenEmbedding pattern)
  - diema/models/ctrgcn/ctrgcn_model.py (from_config pattern)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from diema.models.base import BaseModel
from diema.models.squeezeformer.squeezeformer_block import SqueezeformerBlock


class TokenEmbedding(nn.Module):
    """Per-frame token embedding from raw skeleton channels.

    Uses a small Conv2d(C -> d_model, kernel=(3, 1)) for local temporal
    smoothing, followed by BatchNorm2d. The joint dimension V is then
    averaged out in the main model to produce a pure (T, d_model) sequence
    for the Squeezeformer encoder.

    Args:
        in_channels: input feature channels (6 for 6D rotation).
        d_model: output embedding dimension.
    """

    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, d_model, kernel_size=(3, 1), padding=(1, 0), bias=False,
        )
        self.bn = nn.BatchNorm2d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V) -> (N, d_model, T, V)
        return self.bn(self.conv(x))


class Squeezeformer_Model(BaseModel):
    """17-layer Squeezeformer + RoPE encoder for DIEM-A classification.

    Matches the BaseModel contract:
        Input:  (N, 6, 64, 25)
        Output: {"logits": (N, num_class)}

    Architecture is described in the module docstring. The encoder uses
    17 blocks of Macaron FFN + RoPE MHSA + Conv + Macaron FFN (see
    SqueezeformerBlock). Stochastic depth (DropPath) is applied with a
    linearly increasing per-layer rate from 0 to ``drop_path``.

    Args:
        num_class: number of output classes (12 for DIEM-A).
        in_channels: raw input channels (6 for 6D rotation).
        num_layers: number of Squeezeformer blocks (default 17, per ASL FS).
        d_model: model dimension (default 192).
        num_heads: attention heads (default 4).
        mlp_ratio: FFN hidden expansion ratio (default 4).
        conv_kernel_size: depthwise conv kernel (default 15; reduced from
            the ASL FS 31 because DIEM-A T=64 is shorter than 384).
        dropout: dropout rate inside each sub-block (default 0.1).
        drop_path: max stochastic depth rate (linearly scaled per layer).
        max_seq_len: max cached RoPE length (default 256 >= T=64).

    """

    def __init__(
        self,
        num_class: int,
        in_channels: int = 6,
        num_layers: int = 17,
        d_model: int = 192,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.num_class = num_class
        self.d_model = d_model
        self.num_layers = num_layers

        self.embed = TokenEmbedding(in_channels, d_model)

        # Linearly increasing per-layer drop_path rates.
        dpr = [drop_path * i / max(num_layers - 1, 1) for i in range(num_layers)]
        self.blocks = nn.ModuleList([
            SqueezeformerBlock(
                dim=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                drop_path=dpr[i],
                max_seq_len=max_seq_len,
            )
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_input(x)
        N, C, T, V = x.size()

        # Token embedding: (N, C, T, V) -> (N, d, T, V)
        h = self.embed(x)
        # Average over joints V: (N, d, T, V) -> (N, d, T)
        h = h.mean(dim=-1)
        # -> (N, T, d)
        h = h.transpose(1, 2).contiguous()

        for blk in self.blocks:
            h = blk(h)

        h = self.norm(h)
        # GAP over T -> (N, d)
        h = h.mean(dim=1)
        logits = self.fc(h)
        return {"logits": logits}

    @property
    def output_dim(self) -> int:
        return self.num_class

    @classmethod
    def from_config(cls, config) -> "Squeezeformer_Model":
        return cls(
            num_class=config.model.num_class,
            in_channels=getattr(config.model, "in_channels", 6),
            num_layers=getattr(config.model, "num_layers", 17),
            d_model=getattr(config.model, "d_model", 192),
            num_heads=getattr(config.model, "num_heads", 4),
            mlp_ratio=getattr(config.model, "mlp_ratio", 4.0),
            conv_kernel_size=getattr(config.model, "conv_kernel_size", 15),
            dropout=getattr(config.model, "dropout", 0.1),
            drop_path=getattr(config.model, "drop_path", 0.1),
            max_seq_len=getattr(config.model, "max_seq_len", 256),
        )


from diema.models.registry import register_model  # noqa: E402

register_model("squeezeformer", Squeezeformer_Model)
