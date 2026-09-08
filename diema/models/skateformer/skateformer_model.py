"""SkateFormer skeletal-temporal Transformer model.

Simplified faithful adaptation of the SkateFormer architecture (ECCV 2024).
Uses partitioned attention along (body_part, temporal_window) groups for
efficient skeleton-time attention.

Compatible with diema's BaseModel:
  Input: (N, C, T, V), Output: {"logits": (N, num_class)}

"""

import math
from typing import Sequence

import torch
import torch.nn as nn

from diema.models.base import BaseModel
from diema.models.heads import ArcMarginHead
from diema.models.skateformer.partition_attention import PartitionedAttentionBlock
from diema.models.skeleton_graph import DIEMA_BODY_PARTS


class TokenEmbedding(nn.Module):
    """Per-(t, v) token embedding from raw input channels.

    Uses a small Conv2d (kernel 3x1) for local temporal smoothing then a 1x1
    projection to the model dimension.
    """

    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, dim, kernel_size=(3, 1), padding=(1, 0))
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V) → (N, dim, T, V)
        return self.bn(self.conv(x))


class SkateFormer_Model(BaseModel):
    """Skeletal-temporal Transformer for DIEM-A 25-node skeleton.

    Args:
        num_class: number of output classes
        in_channels: input channels (e.g., 6 for 6D rotation)
        dim: model dimension (must be divisible by num_heads)
        depth: number of transformer blocks
        num_heads: attention heads
        mlp_ratio: MLP hidden ratio
        window_size: temporal window length used by partitioned attention
        body_parts: dict mapping body part name → CTV node indices
        dropout: classifier head dropout
        attn_drop, proj_drop, drop_path: regularization in attention blocks
    """

    def __init__(
        self,
        num_class: int,
        in_channels: int = 6,
        dim: int = 128,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        window_size: int = 16,
        body_parts: dict | None = None,
        dropout: float = 0.3,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.1,
        num_nodes: int = 25,
        pool_type: str = "gap",
        use_arcface_head: bool = False,
    ):
        super().__init__()
        if pool_type not in ("gap", "gmp", "gap_gmp"):
            raise ValueError(f"pool_type must be 'gap'/'gmp'/'gap_gmp', got {pool_type}")
        self.pool_type = pool_type
        self.use_arcface_head = use_arcface_head
        self.num_class = num_class
        self.num_nodes = num_nodes

        if body_parts is None:
            body_parts = DIEMA_BODY_PARTS
        body_parts_list = [body_parts[k] for k in sorted(body_parts.keys())]

        self.embed = TokenEmbedding(in_channels, dim)

        # Learned positional embedding over (T_max, V) — kept small to avoid blow-up
        # We use separate temporal and joint embeddings broadcasted at inference time.
        self.t_pos_embed = nn.Parameter(torch.zeros(1, 256, 1, dim))   # supports up to T=256
        self.v_pos_embed = nn.Parameter(torch.zeros(1, 1, num_nodes, dim))
        nn.init.trunc_normal_(self.t_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.v_pos_embed, std=0.02)

        # Drop-path schedule
        dpr = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            PartitionedAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                body_parts=body_parts_list,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.drop_out = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if self.use_arcface_head:
            # cosine classifier head for ArcFace-style losses.
            self.fc = ArcMarginHead(dim, num_class)
        else:
            self.fc = nn.Linear(dim, num_class)
            nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_input(x)
        N, C, T, V = x.size()
        # Embed: (N, C, T, V) → (N, dim, T, V)
        h = self.embed(x)
        # → (N, T, V, dim)
        h = h.permute(0, 2, 3, 1).contiguous()
        # Add pos embeddings (broadcast)
        h = h + self.t_pos_embed[:, :T, :, :] + self.v_pos_embed[:, :, :V, :]

        for blk in self.blocks:
            h = blk(h)

        h = self.norm(h)
        # Global pool over T and V → (N, dim)
        if self.pool_type == "gap":
            h = h.mean(dim=(1, 2))
        elif self.pool_type == "gmp":
            # max over T and V (flatten then amax)
            h = h.flatten(1, 2).amax(dim=1)
        else:  # gap_gmp
            h_avg = h.mean(dim=(1, 2))
            h_max = h.flatten(1, 2).amax(dim=1)
            h = 0.5 * (h_avg + h_max)
        h = self.drop_out(h)
        logits = self.fc(h)
        return {"logits": logits}

    @property
    def output_dim(self) -> int:
        return self.num_class

    @classmethod
    def from_config(cls, config) -> "SkateFormer_Model":
        return cls(
            num_class=config.model.num_class,
            in_channels=config.model.in_channels,
            dim=getattr(config.model, "dim", 128),
            depth=getattr(config.model, "depth", 6),
            num_heads=getattr(config.model, "num_heads", 8),
            mlp_ratio=getattr(config.model, "mlp_ratio", 4.0),
            window_size=getattr(config.model, "window_size", 16),
            dropout=getattr(config.model, "dropout", 0.3),
            attn_drop=getattr(config.model, "attn_drop", 0.0),
            proj_drop=getattr(config.model, "proj_drop", 0.0),
            drop_path=getattr(config.model, "drop_path", 0.1),
            num_nodes=getattr(config.skeleton, "num_nodes", 25),
            pool_type=getattr(config.model, "pool_type", "gap"),
            use_arcface_head=getattr(config.model, "use_arcface_head", False),
        )


from diema.models.registry import register_model  # noqa: E402

register_model("skateformer", SkateFormer_Model)
