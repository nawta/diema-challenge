"""ProtoGCN: CTR-GCN backbone + per-body-part prototype matching head.

Architecture:
  - Backbone: CTR-GCN 10-layer (reuses TCN_GCN_unit blocks)
  - Output features: (N, 256, T', V=25) before global pooling
  - PerBodyPartPrototype: 6 body parts × 16 prototypes/part → (N, 6*256)
  - Linear classifier → (N, 12)

The prototype head replaces the standard GAP+FC of CTR-GCN, exposing
per-body-part discriminative features for fine-grained emotion classification.

Compatible with diema's BaseModel:
  Input: (N, C, T, V), Output: {"logits": (N, num_class)}

"""

import math

import torch
import torch.nn as nn

from diema.models.base import BaseModel
from diema.models.ctrgcn.ctrgcn_units import TCN_GCN_unit, conv_init, bn_init
from diema.models.protogcn.prototype_module import PerBodyPartPrototype
from diema.models.skeleton_graph import DIEMA_BODY_PARTS, build_adjacency


class ProtoGCN_Model(BaseModel):
    """Prototype GCN classification model.

    Args:
        num_class: number of output classes
        edge_index: list of (child, parent) tuples (CTV node indices)
        num_nodes: number of skeleton nodes (default 25)
        in_channels: input feature channels (default 6 for 6D)
        base_channels: starting channel count (default 64)
        adj_strategy: adjacency strategy
        dropout: classifier head dropout
        adaptive: learnable adjacency
        kernel_size: temporal kernel size
        body_parts: dict mapping part name → CTV node indices
        num_proto: prototypes per body part
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
        body_parts: dict | None = None,
        num_proto: int = 16,
    ):
        super().__init__()
        self.num_class = num_class
        self.num_nodes = num_nodes

        if body_parts is None:
            body_parts = DIEMA_BODY_PARTS
        self.body_parts_list = [body_parts[k] for k in sorted(body_parts.keys())]

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

        self.proto_head = PerBodyPartPrototype(
            body_parts=self.body_parts_list,
            feat_dim=c3,
            num_proto=num_proto,
        )

        self.drop_out = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        proto_out_dim = c3 * len(self.body_parts_list)
        self.fc = nn.Linear(proto_out_dim, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                bn_init(m)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_input(x)
        N, C, T, V = x.size()

        # Data BN
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # CTR-GCN backbone
        for layer in self.layers:
            x = layer(x)
        # x: (N, c3, T', V)

        # Prototype head: per-body-part feature aggregation
        proto_feat = self.proto_head(x)        # (N, P * c3)
        proto_feat = self.drop_out(proto_feat)
        logits = self.fc(proto_feat)
        return {"logits": logits}

    @property
    def output_dim(self) -> int:
        return self.num_class

    @classmethod
    def from_config(cls, config) -> "ProtoGCN_Model":
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
            num_proto=getattr(config.model, "num_proto", 16),
        )


from diema.models.registry import register_model  # noqa: E402

register_model("protogcn", ProtoGCN_Model)
