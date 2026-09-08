"""Weight initialization utilities for neural network modules.

Ported from: an internal baseline
"""

import torch.nn as nn


def conv_init(conv: nn.Conv2d) -> None:
    """Initialize Conv2d with Kaiming normal."""
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    nn.init.constant_(conv.bias, 0)


def bn_init(bn: nn.BatchNorm2d, scale: float) -> None:
    """Initialize BatchNorm2d with given scale."""
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)
