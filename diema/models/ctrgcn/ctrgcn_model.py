"""CTR-GCN full model: stacked TCN_GCN_unit blocks for skeleton classification.

Architecture (10-layer, NTU-style channel progression):
  in_channels → 64 (x4) → 128 (x3, with stride=2 first) → 256 (x3, with stride=2 first)

Compatible with diema's BaseModel contract:
  Input: (N, C, T, V), Output: {"logits": (N, num_class)}

Reference: ICCV 2021, https://github.com/Uason-Chen/CTR-GCN
"""

import math

import torch
import torch.nn as nn

from diema.models.base import BaseModel
from diema.models.ctrgcn.ctrgcn_units import TCN_GCN_unit, conv_init, bn_init
from diema.models.heads import ArcMarginHead
from diema.models.skeleton_graph import build_adjacency


class CTR_GCN_Model(BaseModel):
    """CTR-GCN classification model.

    Args:
        num_class: number of output classes
        edge_index: list of (child, parent) tuples (CTV node index space)
        num_nodes: number of skeleton nodes (default: 25 for DIEM-A)
        in_channels: input feature channels (default: 6 for 6D rotation)
        base_channels: starting channel count (default: 64)
        adj_strategy: adjacency strategy ("aagcn" / "spatial_config")
        dropout: classifier head dropout
        adaptive: whether the adjacency is learnable per layer
        kernel_size: temporal kernel size in MS-TCN
    """

    def __init__(
        self,
        num_class: int,
        edge_index: list,
        num_nodes: int = 25,
        in_channels: int = 6,
        base_channels: int = 64,
        adj_strategy: str = "aagcn",
        dropout: float = 0.5,
        adaptive: bool = True,
        kernel_size: int = 5,
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

        # Build adjacency once and pass to each block (each block clones internally as Parameter)
        A = build_adjacency(edge_index, num_nodes, strategy=adj_strategy)

        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.layers = nn.ModuleList([
            TCN_GCN_unit(in_channels, c1, A, residual=False, adaptive=adaptive, kernel_size=kernel_size),
            TCN_GCN_unit(c1, c1, A, adaptive=adaptive, kernel_size=kernel_size),
            TCN_GCN_unit(c1, c1, A, adaptive=adaptive, kernel_size=kernel_size),
            TCN_GCN_unit(c1, c1, A, adaptive=adaptive, kernel_size=kernel_size),
            TCN_GCN_unit(c1, c2, A, stride=2, adaptive=adaptive, kernel_size=kernel_size),
            TCN_GCN_unit(c2, c2, A, adaptive=adaptive, kernel_size=kernel_size),
            TCN_GCN_unit(c2, c2, A, adaptive=adaptive, kernel_size=kernel_size),
            TCN_GCN_unit(c2, c3, A, stride=2, adaptive=adaptive, kernel_size=kernel_size),
            TCN_GCN_unit(c3, c3, A, adaptive=adaptive, kernel_size=kernel_size),
            TCN_GCN_unit(c3, c3, A, adaptive=adaptive, kernel_size=kernel_size),
        ])

        if self.use_arcface_head:
            # cosine classifier head for ArcFace-style losses.
            self.fc = ArcMarginHead(c3, num_class)
        else:
            self.fc = nn.Linear(c3, num_class)
            nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        self.drop_out = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                bn_init(m)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_input(x)
        N, C, T, V = x.size()
        # Data BN: (N, C, T, V) → (N, V*C, T) → BN → (N, C, T, V)
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        for layer in self.layers:
            x = layer(x)

        # Global pooling over T and V → (N, c_new)
        c_new = x.size(1)
        x = x.view(N, c_new, -1)
        if self.pool_type == "gap":
            x = x.mean(dim=2)
        elif self.pool_type == "gmp":
            x = x.amax(dim=2)
        else:  # gap_gmp
            x = 0.5 * (x.mean(dim=2) + x.amax(dim=2))
        x = self.drop_out(x)
        logits = self.fc(x)
        return {"logits": logits}

    @property
    def output_dim(self) -> int:
        return self.num_class

    @classmethod
    def from_config(cls, config) -> "CTR_GCN_Model":
        return cls(
            num_class=config.model.num_class,
            edge_index=config.skeleton.inward_edges,
            num_nodes=config.skeleton.num_nodes,
            in_channels=config.model.in_channels,
            base_channels=getattr(config.model, "base_channels", 64),
            adj_strategy=getattr(config.model, "adj_strategy", "aagcn"),
            dropout=getattr(config.model, "dropout", 0.5),
            adaptive=getattr(config.model, "adaptive", True),
            kernel_size=getattr(config.model, "kernel_size", 5),
            pool_type=getattr(config.model, "pool_type", "gap"),
            use_arcface_head=getattr(config.model, "use_arcface_head", False),
        )


from diema.models.registry import register_model  # noqa: E402

register_model("ctrgcn", CTR_GCN_Model)
