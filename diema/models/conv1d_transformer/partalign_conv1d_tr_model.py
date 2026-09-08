""" exp052: backbone + part-wise projection head.

Wraps any ``BaseModel`` that exposes ``"features"`` (Conv1DTransformer_Model
or RegionAwareConv1DTransformer_Model) and adds a 6 × text_dim projection
head for part-wise alignment against rationale embeddings.

At inference only ``"logits"`` is required; ``z_parts`` is train-time only
and is stripped by ``tools/strip_text_branch.py`` before submission.

See: (exp052)
Related:
- diema/models/conv1d_transformer/textalign_conv1d_tr_model.py (global analogue)
- diema/training/losses_motion_text.py::PartAlignmentLoss
- diema/features/rationale_text_cache.py (feeds z_parts target at batch time)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diema.models.base import BaseModel


class PartAlignConv1DTr_Model(BaseModel):
    """Backbone + motion-to-part projection head.

    Forward output::

        {
            "logits":   (N, num_class),
            "features": (N, feat_dim),
            "z_parts":  (N, num_parts, text_dim),   # L2-normalized per part
        }

    Args:
        backbone: a :class:`BaseModel` whose forward returns ``"features"``.
        text_dim: text embedding dimension (default 384 = MiniLM-L6-v2).
        num_parts: number of body parts to project into (default 6:
            head, torso, left_arm, right_arm, left_leg, right_leg).
        proj_hidden: hidden width between shared trunk and the part fan-out.
            Defaults to ``backbone.feature_dim``.

    Design note — shared trunk, per-part head:
        The two-layer shared trunk (``Linear → BN → ReLU``) keeps the
        parameter count close to ``TextAlignConv1DTr_Model`` while the
        final ``Linear(hidden → P * text_dim)`` lets each body part learn
        its own affine direction in the text space. Splitting early (e.g.,
        P independent 2-layer heads) over-parameterizes the head relative
        to the backbone and historically destabilizes exp020-class
        Conv1D+Tr backbones.
    """

    def __init__(
        self,
        backbone: BaseModel,
        text_dim: int = 384,
        num_parts: int = 6,
        proj_hidden: int | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_class = backbone.output_dim
        self.num_nodes = backbone.num_nodes
        feat_dim = backbone.feature_dim
        hidden = proj_hidden if proj_hidden is not None else feat_dim
        self._text_dim = int(text_dim)
        self._num_parts = int(num_parts)

        self.part_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self._num_parts * self._text_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.backbone(x)
        features = out["features"]                              # (N, F)
        flat = self.part_head(features)                         # (N, P*D)
        z = flat.view(flat.shape[0], self._num_parts, self._text_dim)
        # Expose the raw projection so part-importance analysis
        # can recover the pre-normalize magnitude per (sample, part). The
        # normalized vector is what the cosine / InfoNCE loss consumes.
        out["z_parts_raw"] = z                                   # (N, P, D)
        out["z_parts"] = F.normalize(z, dim=2)                  # (N, P, D)
        return out

    @property
    def feature_dim(self) -> int:
        return self.backbone.feature_dim

    @property
    def output_dim(self) -> int:
        return self.backbone.output_dim

    @property
    def text_dim(self) -> int:
        return self._text_dim

    @property
    def num_parts(self) -> int:
        return self._num_parts
