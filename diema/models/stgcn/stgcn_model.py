"""Full STGCN classification model.

Stacks 10 STGCN_Units to classify skeleton sequences into emotion
categories. Supports both basic TCN and multi-branch STGCN++ variant.

See: diema_challenge_implementation_spec.md §6.4 M1 — STGCN++ benchmark reproduction
Related: diema/models/base.py, diema/models/stgcn/st_units.py

Ported from: an internal baseline
Architecture: Input -> BN -> 10x STGCN_Unit -> GAP -> Dropout -> FC -> logits
Channel progression: in -> 64 -> 64 -> 64 -> 64 -> 128(s2) -> 128 -> 128 -> 256(s2) -> 256 -> 256
"""

import math

import torch
import torch.nn as nn

from diema.models.base import BaseModel
from diema.models.heads import ArcMarginHead
from diema.models.stgcn.st_units import STGCN_Unit
from diema.models.stgcn.adj_matrix import GraphAAGCN


class STGCN_Model(BaseModel):
    """Full ST-GCN classification model.

    Args:
        num_class: number of output classes
        edge_index: list of (child, parent) edge tuples
        num_nodes: number of joints in the skeleton
        in_channels: number of input channels (e.g., 6 for 6D rotation)
        dropout: dropout rate for classifier head
        edge_weighting: if True, make adjacency matrix learnable
        plusplus: if True, use multi-branch TCN (STGCN++)
        unit_dropout: dropout within TCN_Unit_plus branches
    """

    def __init__(
        self,
        num_class: int,
        edge_index: list,
        num_nodes: int,
        in_channels: int,
        dropout: float = 0.5,
        edge_weighting: bool = True,
        plusplus: bool = True,
        unit_dropout: float = 0.1,
        pool_type: str = "gap",
        use_arcface_head: bool = False,
    ):
        super().__init__()
        if pool_type not in ("gap", "gmp", "gap_gmp"):
            raise ValueError(f"pool_type must be 'gap'/'gmp'/'gap_gmp', got {pool_type}")
        self.pool_type = pool_type
        self.use_arcface_head = use_arcface_head
        graph = GraphAAGCN(edge_index, num_nodes)
        A = torch.as_tensor(graph.A).clone().float()
        self.register_buffer("A", A)
        A = nn.Parameter(A, requires_grad=edge_weighting)
        self.num_class = num_class
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        self.st_gcn_networks = nn.Sequential(
            STGCN_Unit(in_channels, 64, A, plusplus, residual=False, unit_dropout=unit_dropout),
            STGCN_Unit(64, 64, A, plusplus, unit_dropout=unit_dropout),
            STGCN_Unit(64, 64, A, plusplus, unit_dropout=unit_dropout),
            STGCN_Unit(64, 64, A, plusplus, unit_dropout=unit_dropout),
            STGCN_Unit(64, 128, A, plusplus, stride=2, unit_dropout=unit_dropout),
            STGCN_Unit(128, 128, A, plusplus, unit_dropout=unit_dropout),
            STGCN_Unit(128, 128, A, plusplus, unit_dropout=unit_dropout),
            STGCN_Unit(128, 256, A, plusplus, stride=2, unit_dropout=unit_dropout),
            STGCN_Unit(256, 256, A, plusplus, unit_dropout=unit_dropout),
            STGCN_Unit(256, 256, A, plusplus, unit_dropout=unit_dropout),
        )
        if self.use_arcface_head:
            # cosine classifier head for ArcFace-style losses.
            self.fc = ArcMarginHead(256, num_class)
        else:
            self.fc = nn.Linear(256, num_class)
            nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        self.drop_out = nn.Dropout(dropout) if dropout > 0 else lambda x: x

    def forward(self, x):
        self._validate_input(x)
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        x = self.st_gcn_networks(x)
        c_new = x.size(1)
        x = x.view(N, c_new, -1)
        if self.pool_type == "gap":
            x = x.mean(2)
        elif self.pool_type == "gmp":
            x = x.amax(2)
        else:  # gap_gmp
            x = 0.5 * (x.mean(2) + x.amax(2))
        x = self.drop_out(x)
        x = self.fc(x)
        return {"logits": x}

    @property
    def output_dim(self):
        return self.num_class

    @classmethod
    def from_config(cls, config):
        return cls(
            num_class=config.model.num_class,
            edge_index=config.skeleton.inward_edges,
            num_nodes=config.skeleton.num_nodes,
            in_channels=config.model.in_channels,
            edge_weighting=getattr(config.model, "edge_weighting", False),
            plusplus=getattr(config.model, "plusplus", False),
            dropout=getattr(config.model, "dropout", 0.5),
            unit_dropout=getattr(config.model, "unit_dropout", 0.1),
            pool_type=getattr(config.model, "pool_type", "gap"),
            use_arcface_head=getattr(config.model, "use_arcface_head", False),
        )


from diema.models.registry import register_model  # noqa: E402

register_model("stgcn", STGCN_Model)
