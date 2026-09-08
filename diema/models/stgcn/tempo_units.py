"""Temporal convolution units for skeleton-based recognition.

Provides both basic single-kernel TCN and multi-branch STGCN++ TCN.

See: diema_challenge_implementation_spec.md §6.4 M1
Related: diema/models/stgcn/spatial_units.py, diema/models/stgcn/st_units.py

Ported from: an internal baseline
"""

import torch
import torch.nn as nn

from diema.models.weights_init import conv_init, bn_init


class Basic_TCN_Unit(nn.Module):
    """Basic Temporal Convolutional Network Unit.

    Applies 1D temporal convolution (kernel_size, 1) along the time dimension.

    Args:
        in_channels: input feature channels
        out_channels: output feature channels
        kernel_size: temporal kernel size (default: 9)
        stride: temporal stride (default: 1)
        dilation: temporal dilation (default: 1)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1, dilation: int = 1):
        super().__init__()
        pad = int((kernel_size + (kernel_size - 1) * (dilation - 1) - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1),
            dilation=(dilation, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.bn(self.conv(x))


class TCN_Unit_plus(nn.Module):
    """Multi-branch Temporal Convolutional Network Unit (STGCN++).

    6 parallel branches capturing multi-scale temporal patterns:
      - Branches 0-3: 3x1 conv with dilations 1-4
      - Branch 4: max pooling 3x1
      - Branch 5: 1x1 conv (channel transform only)

    Outputs are concatenated and refined through a final 1x1 conv.

    Args:
        in_channels: input feature channels
        out_channels: output feature channels
        dropout: dropout probability (default: 0.3)
        stride: temporal stride (default: 1)
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.3, stride: int = 1):
        super().__init__()
        ms_cfg = [(3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"]
        num_branches = len(ms_cfg)
        self.relu = nn.ReLU()

        mid_channels = out_channels // num_branches
        rem_mid_channels = out_channels - mid_channels * (num_branches - 1)

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == "1x1":
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == "max":
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1),
                        nn.BatchNorm2d(branch_c),
                        self.relu,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0)),
                    )
                )
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1),
                nn.BatchNorm2d(branch_c),
                self.relu,
                Basic_TCN_Unit(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1]),
            )
            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        self.transform = nn.Sequential(
            nn.BatchNorm2d(out_channels), self.relu, nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)
        self._init_weights()

    def _init_weights(self):
        for branch in self.branches:
            if isinstance(branch, nn.Conv2d):
                conv_init(branch)
            elif isinstance(branch, nn.Sequential):
                for layer in branch:
                    if isinstance(layer, nn.Conv2d):
                        conv_init(layer)
                    elif isinstance(layer, nn.BatchNorm2d):
                        bn_init(layer, 1)
        for layer in self.transform:
            if isinstance(layer, nn.Conv2d):
                conv_init(layer)
            elif isinstance(layer, nn.BatchNorm2d):
                bn_init(layer, 1)
        bn_init(self.bn, 1)

    def forward(self, x):
        branch_outs = [tempconv(x) for tempconv in self.branches]
        feat = torch.cat(branch_outs, dim=1)
        feat = self.transform(feat)
        out = self.bn(feat)
        return self.drop(out)
