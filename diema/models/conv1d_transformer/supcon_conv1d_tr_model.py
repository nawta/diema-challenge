"""Conv1D+Transformer with SupCon projection head.

Wraps a backbone that exposes ``"features"`` (e.g., Conv1DTransformer_Model
or RegionAwareConv1DTransformer_Model) and adds a non-linear projection
head for Supervised Contrastive Learning.

The projection head is train-time only — at inference the ``z_sc`` output
can be ignored and only ``logits`` is needed.

See: (exp040)
Related: diema/training/supcon_trainer.py,
         diema/training/losses.py::SupConLoss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diema.models.base import BaseModel


class SupConConv1DTr_Model(BaseModel):
    """Backbone + SupCon projection head.

    Forward output::

        {
            "logits":   (N, num_class),
            "features": (N, feat_dim),   # pre-classifier backbone features
            "z_sc":     (N, proj_dim),   # L2-normalised projection for SupCon
        }

    Args:
        backbone: a model whose ``forward(x)`` returns a dict with
            ``"logits"`` and ``"features"`` keys.
        proj_dim: output dimension of the projection head (default 128).
        proj_hidden: hidden dimension; defaults to backbone's feature_dim.
    """

    def __init__(
        self,
        backbone: BaseModel,
        proj_dim: int = 128,
        proj_hidden: int | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_class = backbone.output_dim
        self.num_nodes = backbone.num_nodes
        feat_dim = backbone.feature_dim
        hidden = proj_hidden if proj_hidden is not None else feat_dim

        self.proj_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim),
        )
        self._proj_dim = proj_dim

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.backbone(x)
        features = out["features"]
        z_sc = self.proj_head(features)
        z_sc = F.normalize(z_sc, dim=1)  # L2-normalise onto unit hypersphere
        out["z_sc"] = z_sc
        return out

    @property
    def feature_dim(self) -> int:
        return self.backbone.feature_dim

    @property
    def output_dim(self) -> int:
        return self.backbone.output_dim

    @classmethod
    def from_config(cls, config) -> "SupConConv1DTr_Model":
        from diema.models import build_model
        backbone = build_model(config)
        return cls(
            backbone=backbone,
            proj_dim=getattr(config.model, "proj_dim", 128),
            proj_hidden=getattr(config.model, "proj_hidden", None),
        )


from diema.models.registry import register_model  # noqa: E402

register_model("supcon_conv1d_tr", SupConConv1DTr_Model)
