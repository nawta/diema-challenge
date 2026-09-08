"""Spatial graph convolution unit (GCN) for skeleton-based recognition.

Performs multi-subset graph convolution where neighboring joints are
classified into subsets (self, inward, outward). Each subset has its
own learnable convolution parameters.

See: diema_challenge_implementation_spec.md §6.4 M1
Related: diema/models/stgcn/adj_matrix.py, diema/models/stgcn/st_units.py

Ported from: an internal baseline
"""

import math

import torch
import torch.nn as nn

from diema.models.weights_init import conv_init, bn_init


class Basic_GCN_Unit(nn.Module):
    """Basic Graph Convolutional Network Unit.

    Multi-subset spatial graph convolution:
      1. For each adjacency subset A[i], aggregate features from connected joints
      2. Apply learnable 1x1 convolution per subset
      3. Sum all subset outputs
      4. Add residual connection

    Args:
        in_channels: number of input feature channels
        out_channels: number of output feature channels
        A: adjacency matrix of shape (num_subset, V, V)
        num_subset: number of spatial subsets (default: 3)
    """

    def __init__(self, in_channels: int, out_channels: int, A: torch.FloatTensor, num_subset: int = 3):
        super().__init__()
        self.num_subset = num_subset
        self.A = A

        self.conv_d = nn.ModuleList()
        for _ in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self._init_conv_bn()

    def _conv_branch_init(self, conv, branches):
        weight = conv.weight
        n = weight.size(0)
        k1 = weight.size(1)
        k2 = weight.size(2)
        nn.init.normal_(weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
        nn.init.constant_(conv.bias, 0)

    def _init_conv_bn(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            self._conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        y = None
        for i in range(self.num_subset):
            A1 = self.A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        y = self.bn(y)
        y = y + self.down(x)
        y = self.relu(y)
        return y
