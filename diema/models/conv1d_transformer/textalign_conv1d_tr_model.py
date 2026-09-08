"""Conv1D+Transformer with scenario-text projection head.

Wraps a backbone that exposes ``"features"`` (Conv1DTransformer_Model or
RegionAwareConv1DTransformer_Model) and adds a projection head mapping the
backbone's features into the sentence-transformer embedding space (default
384-d for MiniLM-L6-v2).

At inference, only ``"logits"`` is required; the text projection head and the
``z_text`` output are train-time only. ``tools/strip_text_branch.py`` (Phase
3AE-7) will remove them from submitted checkpoints.

See: (exp051)
Related:
- diema/training/losses_motion_text.py (GlobalInfoNCELoss)
- diema/features/text_cache.py (ScenarioTextCache)
- diema/training/text_align_trainer.py (LightningModel wrapper)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diema.models.base import BaseModel


class TextAlignConv1DTr_Model(BaseModel):
    """Backbone + motion-to-text projection head.

    Forward output::

        {
            "logits":   (N, num_class),
            "features": (N, feat_dim),
            "z_text":   (N, text_dim),   # L2-normalized motion projection
        }

    Args:
        backbone: a :class:`BaseModel` whose forward returns ``"features"``.
        text_dim: text embedding dimension (default 384 = MiniLM-L6-v2).
        proj_hidden: hidden layer width; defaults to backbone.feature_dim.
    """

    def __init__(
        self,
        backbone: BaseModel,
        text_dim: int = 384,
        proj_hidden: int | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_class = backbone.output_dim
        self.num_nodes = backbone.num_nodes
        feat_dim = backbone.feature_dim
        hidden = proj_hidden if proj_hidden is not None else feat_dim
        self._text_dim = text_dim

        # Follow the SupCon head pattern (Linear-BN-ReLU-Linear) so the
        # projection head fits a comparable regularization budget.
        self.text_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, text_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.backbone(x)
        features = out["features"]
        z_text = self.text_head(features)
        out["z_text"] = F.normalize(z_text, dim=1)
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

    @classmethod
    def from_config(cls, config) -> "TextAlignConv1DTr_Model":
        """Construct the wrapped backbone + text head from a flat config.

        Uses ``config.model.backbone_name`` (default ``conv1d_transformer``) to
        decide which backbone to build. ``config.model.name`` is reserved for
        the registry lookup that dispatched us here; reading it back into
        ``build_model`` would recurse into this same ``from_config``.
        """
        from copy import deepcopy
        from types import SimpleNamespace

        from diema.models import build_model

        backbone_name = getattr(config.model, "backbone_name", "conv1d_transformer")
        if backbone_name == "textalign_conv1d_tr":
            raise ValueError(
                "backbone_name cannot be 'textalign_conv1d_tr' — would recurse. "
                "Set config.model.backbone_name to a leaf model name "
                "(e.g. 'conv1d_transformer' or 'region_aware_conv1d_transformer')."
            )

        # Deep-copy and swap the name so we do not mutate the caller's config.
        if isinstance(config, SimpleNamespace):
            sub = deepcopy(config)
            sub.model.name = backbone_name
        else:
            # OmegaConf DictConfig path: avoid deepcopy quirks by rebuilding
            # a SimpleNamespace view.
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
            text_dim=getattr(config.model, "text_dim", 384),
            proj_hidden=getattr(config.model, "proj_hidden", None),
        )


from diema.models.registry import register_model  # noqa: E402

register_model("textalign_conv1d_tr", TextAlignConv1DTr_Model)
