"""CTR-GCN spatial/temporal units.

Faithful adaptation of the official CTR-GCN repo (Uason-Chen/CTR-GCN, ICCV 2021):
  - CTRGC: channel-wise topology refinement spatial conv
  - unit_tcn: simple temporal conv
  - MultiScale_TemporalConv: 6-branch dilated temporal block (CTR-GCN++ style)
  - TCN_GCN_unit: combined ST block with residual

Tensor layout: (N, C, T, V).
"""

import math

import torch
import torch.nn as nn


def conv_init(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def bn_init(m: nn.Module, scale: float = 1.0) -> None:
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, scale)
        nn.init.constant_(m.bias, 0)


class CTRGC(nn.Module):
    """Channel-wise Topology Refinement Graph Convolution layer.

    For each output channel, the spatial topology is refined as:
        A_c = A + alpha * (phi(x) - psi(x).T)
    where phi and psi are 1x1 conv projections, and the result is fed
    through a tanh nonlinearity. The output is the channel-wise weighted
    aggregation of features over neighbors.

    Args:
        in_channels: input feature channels
        out_channels: output channels
        rel_reduction: reduction factor for refinement projections (default: 8)
    """

    def __init__(self, in_channels: int, out_channels: int, rel_reduction: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = max(in_channels // rel_reduction, 1)
            self.mid_channels = max(in_channels // rel_reduction, 1)

        self.conv1 = nn.Conv2d(in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, scale=1.0)

    def forward(self, x: torch.Tensor, A: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        # x: (N, C, T, V), A: (V, V) shared subset
        # x1, x2: pooled over time → (N, rel, V)
        x1 = self.conv1(x).mean(dim=2)
        x2 = self.conv2(x).mean(dim=2)
        # Pairwise difference, then refinement projection over rel_channels
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))   # (N, rel, V, V)
        x1 = self.conv4(x1) * alpha + A.unsqueeze(0).unsqueeze(0)  # (N, out, V, V)
        # Aggregate features
        x_proj = self.conv3(x)                     # (N, out, T, V)
        x_out = torch.einsum("nctv,ncvw->nctw", x_proj, x1)
        return x_out


class unit_tcn(nn.Module):
    """Simple temporal conv: Conv2d kernel (kernel_size, 1)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class MultiScale_TemporalConv(nn.Module):
    """Multi-branch temporal conv used in CTR-GCN++ for stronger temporal modeling.

    Branches: 4 dilated kxk convs + 1 maxpool + 1 1x1 conv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        dilations: list[int] | None = None,
        residual: bool = True,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 3, 4]
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if branch_channels * self.num_branches != out_channels:
            # ensure exact division
            branch_channels = max(out_channels // self.num_branches, 1)

        # Dilated branches
        self.branches = nn.ModuleList()
        for d in dilations:
            pad = (kernel_size + (kernel_size - 1) * (d - 1) - 1) // 2
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    branch_channels, branch_channels,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding=(pad, 0),
                    dilation=(d, 1),
                ),
                nn.BatchNorm2d(branch_channels),
            )
            self.branches.append(branch)

        # Maxpool branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels),
        ))
        # 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels),
        ))

        # Output channel after concat may not equal out_channels exactly; project to align
        actual_out = branch_channels * self.num_branches
        self.proj = None
        if actual_out != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(actual_out, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

        # Residual
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        outs = [b(x) for b in self.branches]
        out = torch.cat(outs, dim=1)
        if self.proj is not None:
            out = self.proj(out)
        out = out + res
        return out


class unit_gcn(nn.Module):
    """Spatial unit: K parallel CTRGC layers (one per A subset) summed and BN'd."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        adaptive: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subset = A.shape[0]

        self.convs = nn.ModuleList([CTRGC(in_channels, out_channels) for _ in range(self.num_subset)])

        if adaptive:
            # Learnable adjacency
            self.PA = nn.Parameter(A.clone().float(), requires_grad=True)
        else:
            self.register_buffer("PA", A.clone().float())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.down = nn.Identity()
        else:
            self.down = lambda x: 0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m)
        bn_init(self.bn, scale=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = None
        for i, conv in enumerate(self.convs):
            z = conv(x, self.PA[i], self.alpha)
            y = z if y is None else y + z
        y = self.bn(y)
        y = y + self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    """One CTR-GCN block: spatial GCN → temporal multi-scale conv → residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
        adaptive: bool = True,
        kernel_size: int = 5,
        dilations: list[int] | None = None,
    ):
        super().__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilations=dilations,
            residual=False,
        )
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gcn1(x)
        y = self.tcn1(y) + self.residual(x)
        return self.relu(y)
