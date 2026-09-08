"""Spatio-Temporal Graph Convolutional Unit.

Combines spatial (GCN) and temporal (TCN) convolutions in sequence
with a residual connection.

See: diema_challenge_implementation_spec.md §6.4 M1
Related: diema/models/stgcn/spatial_units.py, diema/models/stgcn/tempo_units.py

Ported from: an internal baseline
"""

import torch
import torch.nn as nn

from diema.models.stgcn.spatial_units import Basic_GCN_Unit
from diema.models.stgcn.tempo_units import Basic_TCN_Unit, TCN_Unit_plus


class STGCN_Unit(nn.Module):
    """Spatio-Temporal Graph Convolutional Unit.

    Processing flow:
      1. Spatial: GCN aggregates features from neighboring joints
      2. Temporal: TCN captures dependencies across frames
      3. Residual: skip connection to prevent gradient degradation
      4. ReLU activation

    Args:
        in_channels: input feature channels
        out_channels: output feature channels
        A: adjacency matrix (num_subset, V, V)
        plusplus: if True, use multi-branch TCN (STGCN++)
        stride: temporal stride for downsampling
        residual: if True, apply residual connection
        unit_dropout: dropout within TCN_Unit_plus branches
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.FloatTensor,
        plusplus: bool = False,
        stride: int = 1,
        residual: bool = True,
        unit_dropout: float = 0.3,
    ):
        super().__init__()
        self.gcn1 = Basic_GCN_Unit(in_channels, out_channels, A)

        if not plusplus:
            self.tcn1 = Basic_TCN_Unit(out_channels, out_channels, stride=stride)
        else:
            self.tcn1 = TCN_Unit_plus(out_channels, out_channels, dropout=unit_dropout, stride=stride)

        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = Basic_TCN_Unit(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.gcn1(x)
        y = self.tcn1(y)
        y = y + self.residual(x)
        y = self.relu(y)
        return y
