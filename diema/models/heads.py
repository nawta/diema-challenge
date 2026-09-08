"""Classification heads for emotion recognition models.

Provides:
  - ClassificationHead: standard Dropout -> Linear head
  - ArcMarginHead: cosine classifier for ArcFace loss (unit-normalised
    features and weights). Use this when training with
    ``diema.training.losses.ArcFaceLoss`` or ``ArcFaceCELoss``.

See: diema_challenge_implementation_spec.md §6.4,
Related: diema/models/base.py, diema/training/losses.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """Simple classification head: Dropout -> Linear.

    Args:
        in_features: input feature dimension
        num_classes: number of output classes
        dropout: dropout probability
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)


class ArcMarginHead(nn.Module):
    """Cosine classifier head used with ArcFace-style losses.

    Computes ``cos(theta) = normalize(features) @ normalize(weights).T``.
    The output is in [-1, 1] and must be fed to ``ArcFaceLoss`` or
    ``ArcFaceCELoss`` (NOT plain CE — the scale is wrong for plain CE).

    There is an intentionally duplicated (identical) implementation in
    ``diema.training.losses`` for backward compatibility — the canonical
    location is here in the models package. New callers should import from
    ``diema.models.heads``.

    Args:
        in_features: feature dimension
        num_classes: output classes
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x, w)
