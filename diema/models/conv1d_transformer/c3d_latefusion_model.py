""" (exp056): Conv1D+Tr backbone with C3D stats late fusion.

Related:
- diema/features/c3d_stats.py (produces the 51-D feature vector)
- diema/models/conv1d_transformer/conv1d_transformer_model.py (backbone)
- experiments/exp056_c3d_stats_latefusion/ (consumer)

Wraps a backbone that exposes ``features`` (all registered BaseModel subclasses
do) and late-fuses it with a small MLP encoder of the pre-computed C3D
statistics. Concatenation + residual-style fusion keeps the motion-only
inference path usable by zeroing the C3D head.

Forward signature *differs* from the standard BaseModel: in addition to the
``(N, C, T, V)`` motion tensor it takes a ``c3d_vec`` ``(N, D_c3d)`` tensor
produced by ``diema.features.c3d_stats.stats_to_vector``. The training loop
is responsible for plumbing the C3D features in. When ``c3d_vec`` is None
the fusion branch is zeroed out (the model reverts to pure motion), which
keeps the checkpoint usable under the standard contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diema.models.base import BaseModel


class C3DLateFusion_Model(BaseModel):
    """Backbone + C3D MLP + concatenated linear classifier.

    Forward output::

        {
            "logits":        (N, num_class),
            "features":      (N, feat_dim),      # backbone features
            "fused":         (N, feat_dim + c3d_proj_dim),
            "c3d_proj":      (N, c3d_proj_dim),  # projected C3D vector
        }
    """

    def __init__(
        self,
        backbone: BaseModel,
        c3d_in_dim: int = 51,
        c3d_hidden: int = 128,
        c3d_proj_dim: int = 64,
        c3d_dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_class = backbone.output_dim
        self.num_nodes = backbone.num_nodes
        feat_dim = backbone.feature_dim

        self.c3d_head = nn.Sequential(
            nn.Linear(c3d_in_dim, c3d_hidden),
            nn.LayerNorm(c3d_hidden),
            nn.GELU(),
            nn.Dropout(c3d_dropout),
            nn.Linear(c3d_hidden, c3d_proj_dim),
        )
        self._c3d_in_dim = c3d_in_dim
        self._c3d_proj_dim = c3d_proj_dim

        # Late-fusion classifier over concat(backbone_features, c3d_proj).
        self.fused_classifier = nn.Linear(feat_dim + c3d_proj_dim, self.num_class)

    def forward(
        self,
        x: torch.Tensor,
        c3d_vec: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.backbone(x)
        features = out["features"]  # (N, feat_dim)
        if c3d_vec is None:
            # Pure-motion fallback: zero the C3D projection.
            c3d_proj = features.new_zeros(features.size(0), self._c3d_proj_dim)
        else:
            if c3d_vec.dim() != 2 or c3d_vec.size(1) != self._c3d_in_dim:
                raise ValueError(
                    f"c3d_vec must be (N, {self._c3d_in_dim}); got {tuple(c3d_vec.shape)}"
                )
            c3d_proj = self.c3d_head(c3d_vec)
        fused = torch.cat([features, c3d_proj], dim=1)
        logits = self.fused_classifier(fused)
        out["logits"] = logits
        out["fused"] = fused
        out["c3d_proj"] = c3d_proj
        return out

    @property
    def feature_dim(self) -> int:
        return self.backbone.feature_dim + self._c3d_proj_dim

    @property
    def output_dim(self) -> int:
        return self.num_class

    @classmethod
    def from_config(cls, config) -> "C3DLateFusion_Model":
        """Build via the registered backbone_name (never recurse)."""
        from copy import deepcopy
        from types import SimpleNamespace

        from diema.models import build_model

        backbone_name = getattr(config.model, "backbone_name", "conv1d_transformer")
        if backbone_name == "c3d_late_fusion":
            raise ValueError(
                "backbone_name cannot be 'c3d_late_fusion' — would recurse."
            )
        if isinstance(config, SimpleNamespace):
            sub = deepcopy(config)
            sub.model.name = backbone_name
        else:
            sub = SimpleNamespace(
                model=SimpleNamespace(**{
                    k: getattr(config.model, k) for k in dir(config.model)
                    if not k.startswith("_")
                }),
                skeleton=config.skeleton,
            )
            sub.model.name = backbone_name
        backbone = build_model(sub)
        return cls(
            backbone=backbone,
            c3d_in_dim=getattr(config.model, "c3d_in_dim", 51),
            c3d_hidden=getattr(config.model, "c3d_hidden", 128),
            c3d_proj_dim=getattr(config.model, "c3d_proj_dim", 64),
            c3d_dropout=getattr(config.model, "c3d_dropout", 0.2),
        )


from diema.models.registry import register_model  # noqa: E402

register_model("c3d_late_fusion", C3DLateFusion_Model)
