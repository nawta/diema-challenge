"""Content/Style Dual-Head wrapper (exp036).

Wraps an arbitrary backbone that exposes a pre-classifier feature vector
and produces a *content* embedding and a *style* embedding from it. The
two embeddings feed separate heads:

  - ``emotion_head(z_content) -> emo_logits``   — primary task
  - ``style_head(z_style)     -> style_logits`` — auxiliary "let style
    be style" supervision (performer classification)
  - ``adv_head(GRL(z_content)) -> adv_logits``  — weak adversarial branch
    that *discourages* performer-identity from leaking into the content
    embedding

The wrapper itself does NOT compute the joint loss; that is the
:class:`DualHeadLightningModel` subclass's job (which owns the
performer-label map and the loss weights). This model is
just the forward graph.

Design notes:
    - RegionAwareConv1DTransformer_Model exposes a
      ``features`` key in its forward output and a ``feature_dim``
      property. Any backbone that follows this contract is compatible.
    - The adversarial head is ONLY applied to ``z_content`` (weak GRL).
      ``z_style`` has no GRL — we want it to absorb style, not lose it.
    - Orthogonality is enforced by the joint loss via
      :func:`diema.training.losses.orthogonality_loss`, not by this
      module's forward (the wrapper just provides the two embeddings).

Related: diema/training/grl.py, diema/training/losses.py::orthogonality_loss,
         diema/models/conv1d_transformer/region_aware_conv1d_transformer_model.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from diema.models.base import BaseModel
from diema.training.grl import gradient_reversal


class _ProjHead(nn.Module):
    """Plain linear projection used for both content and style branches.

     specifies Linear(backbone_dim, 256) for both
    projections, so this is intentionally minimal: a single affine layer
    followed by optional dropout. A deeper MLP would triple the
    parameter count of the wrapper and is not what the spec
    asked for.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(x))


class DualHead_Model(BaseModel):
    """Backbone + content/style dual-head wrapper.

    Args:
        backbone: pre-constructed BaseModel whose ``forward(x)`` returns a
            dict containing a ``"features"`` key of shape ``(N, feature_dim)``.
            The backbone's own logits head is kept intact but ignored by
            this wrapper — we only consume ``out["features"]``.
        feature_dim: dimension of the backbone's pre-classifier feature
            vector. If ``None``, the wrapper reads ``backbone.feature_dim``.
        num_class: number of emotion classes (DIEM-A: 12).
        num_performer: number of performer IDs for the style/adversarial
            classifiers (DIEM-A: 92). Set to 0 to disable the style and
            adversarial heads (useful for ablations).
        content_dim: width of the content embedding. Default 256.
        style_dim:   width of the style embedding. Default 256.
        proj_dropout: dropout inside each projection MLP. Default 0.1.
        grl_lambda: strength of the adversarial GRL on ``z_content``.
            Default 0.05 (weak). Can be scheduled externally by setting
            ``self.grl_lambda`` at runtime.

    Forward output dict:
        ``logits``       : (N, num_class) — primary emotion logits
        ``features``     : (N, feature_dim) — raw backbone features (alias of backbone output)
        ``z_content``    : (N, content_dim)
        ``z_style``      : (N, style_dim)
        ``style_logits`` : (N, num_performer) if num_performer > 0 else missing
        ``adv_logits``   : (N, num_performer) if num_performer > 0 else missing
    """

    def __init__(
        self,
        backbone: BaseModel,
        feature_dim: int | None = None,
        num_class: int = 12,
        num_performer: int = 92,
        content_dim: int = 256,
        style_dim: int = 256,
        proj_dropout: float = 0.1,
        grl_lambda: float = 0.05,
    ):
        super().__init__()
        if feature_dim is None:
            feature_dim = int(getattr(backbone, "feature_dim", 0))
        if feature_dim <= 0:
            raise ValueError(
                "DualHead_Model requires a positive feature_dim. Either pass "
                "feature_dim explicitly or ensure the backbone exposes a "
                "`.feature_dim` property."
            )
        if num_class <= 0:
            raise ValueError(f"num_class must be positive, got {num_class}")
        if num_performer < 0:
            raise ValueError(f"num_performer must be >= 0, got {num_performer}")

        self.backbone = backbone
        self.feature_dim_ = feature_dim
        self.num_class = num_class
        self.num_performer = num_performer
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.grl_lambda = float(grl_lambda)

        # Content / style projections.
        self.content_proj = _ProjHead(feature_dim, content_dim, dropout=proj_dropout)
        self.style_proj = _ProjHead(feature_dim, style_dim, dropout=proj_dropout)

        # Primary emotion head on z_content.
        self.emotion_head = nn.Linear(content_dim, num_class)

        # Performer-style head on z_style (no GRL — we WANT style info here).
        # Adversarial branch on z_content (with GRL — we want to LOSE style info here).
        if num_performer > 0:
            self.style_head = nn.Linear(style_dim, num_performer)
            self.adv_head = nn.Linear(content_dim, num_performer)
        else:
            self.style_head = None
            self.adv_head = None

        self._init_dual_weights()

    def _init_dual_weights(self) -> None:
        """Initialize only the dual-head branches. The backbone's own init
        is left untouched (it was set up when the backbone was built)."""
        for m in (self.content_proj, self.style_proj):
            for sub in m.modules():
                if isinstance(sub, nn.Linear):
                    nn.init.trunc_normal_(sub.weight, std=0.02)
                    if sub.bias is not None:
                        nn.init.zeros_(sub.bias)
        # Classification heads: use the same init convention as
        # region_aware_conv1d_transformer_model.fc.
        for head, out_size in (
            (self.emotion_head, self.num_class),
            (self.style_head, self.num_performer),
            (self.adv_head, self.num_performer),
        ):
            if head is None:
                continue
            nn.init.normal_(head.weight, 0, math.sqrt(2.0 / max(out_size, 1)))
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_input(x)
        backbone_out = self.backbone(x)
        if "features" not in backbone_out:
            raise RuntimeError(
                "DualHead_Model expects the backbone's forward() to return a "
                "dict with a 'features' key. The backbone "
                f"{type(self.backbone).__name__} returned keys "
                f"{list(backbone_out.keys())} — extend its forward() to also "
                "return the pre-classifier feature vector."
            )
        features = backbone_out["features"]
        if features.ndim != 2 or features.size(1) != self.feature_dim_:
            raise RuntimeError(
                f"backbone features shape mismatch: expected (N, {self.feature_dim_}), "
                f"got {tuple(features.shape)}"
            )

        z_content = self.content_proj(features)
        z_style = self.style_proj(features)

        logits = self.emotion_head(z_content)

        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "features": features,
            "z_content": z_content,
            "z_style": z_style,
        }

        if self.style_head is not None:
            out["style_logits"] = self.style_head(z_style)
        if self.adv_head is not None:
            # Weak GRL on z_content: the forward pass is identity, so the
            # adversarial classifier sees the real content embedding and
            # tries to predict the performer; the reversed gradient then
            # pushes z_content *away* from performer-discriminative features.
            adv_in = gradient_reversal(z_content, lambda_=self.grl_lambda)
            out["adv_logits"] = self.adv_head(adv_in)

        return out

    @property
    def output_dim(self) -> int:
        return self.num_class

    @classmethod
    def from_config(cls, config) -> "DualHead_Model":
        """Build a DualHead_Model from a config namespace.

        Expects ``config.model.backbone`` to be an already-constructed
        BaseModel (the run.py script is responsible for building it via
        ``build_model`` on the backbone sub-config). The remaining
        ``config.model.*`` keys control the dual-head sizes and GRL
        strength.
        """
        backbone = getattr(config.model, "backbone", None)
        if backbone is None:
            raise ValueError(
                "DualHead_Model.from_config requires config.model.backbone "
                "to be a pre-built BaseModel instance. Construct the backbone "
                "first via build_model(backbone_cfg), then pass it in."
            )
        return cls(
            backbone=backbone,
            feature_dim=getattr(config.model, "feature_dim", None),
            num_class=config.model.num_class,
            num_performer=getattr(config.model, "num_performer", 92),
            content_dim=getattr(config.model, "content_dim", 256),
            style_dim=getattr(config.model, "style_dim", 256),
            proj_dropout=getattr(config.model, "proj_dropout", 0.1),
            grl_lambda=getattr(config.model, "grl_lambda", 0.05),
        )


from diema.models.registry import register_model  # noqa: E402

register_model("dual_head", DualHead_Model)
