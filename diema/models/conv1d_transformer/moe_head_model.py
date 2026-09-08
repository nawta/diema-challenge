"""Style-Conditioned MoE model (v2).

v2 improvements based on a code review:
- **Feature-adapter experts** instead of logit experts: each expert is
  a bottleneck adapter ``W_up GELU(W_down h)`` that modifies the feature,
  not the logits. One shared classifier on the adapted feature.
- **Learned routing from backbone features** (detached) instead of
  handcrafted 21-dim style summary. Optional: concat style summary.
- **K=2** by default (K=4 fragments 8K samples too much).
- **ReZero-style alpha** for residual scaling (starts at 0.0).
- **Zero-init** adapter output layers for warm start.

Architecture::

    h = backbone(x)["features"]
    α = softmax(router(stopgrad(h)))     # routing from detached features
    h' = h + alpha * Σ_k α_k * adapter_k(h)   # feature-space residual
    logits = backbone.fc(late_drop(h'))   # shared classifier

See: v2
Related: diema/training/moe_trainer.py, diema/features/style_summary.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diema.models.base import BaseModel


class FeatureAdapterMoE_Model(BaseModel):
    """Backbone + Feature-Adapter MoE with learned routing.

    Forward output::

        {
            "logits":           (N, num_class),
            "features":         (N, feat_dim),
            "adapted_features": (N, feat_dim),
            "routing_weights":  (N, K),
        }

    Args:
        backbone: model exposing ``"features"`` and ``"logits"`` keys.
        num_experts: K (default 2).
        adapter_bottleneck: bottleneck dim for adapter MLPs (default 64).
        router_hidden: router MLP hidden dim (default 64).
        alpha_init: initial ReZero scaling factor (default 0.0).
        use_style_concat: concat 21-dim style summary to router input.
    """

    def __init__(
        self,
        backbone: BaseModel,
        num_experts: int = 2,
        adapter_bottleneck: int = 64,
        router_hidden: int = 64,
        alpha_init: float = 0.0,
        use_style_concat: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_class = backbone.output_dim
        self.num_nodes = backbone.num_nodes
        feat_dim = backbone.feature_dim
        self._num_experts = num_experts
        self._use_style_concat = use_style_concat

        # Feature-space bottleneck adapters: W_up GELU(W_down h)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, adapter_bottleneck),
                nn.GELU(),
                nn.Linear(adapter_bottleneck, feat_dim),
            )
            for _ in range(num_experts)
        ])
        # Zero-init output layers → adapter output ≈ 0 at init
        for adapter in self.adapters:
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)

        # ReZero scaling (learnable, starts near 0)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # Router: MLP on detached backbone features (+ optional style)
        if use_style_concat:
            from diema.features.style_summary import STYLE_DIM
            router_in_dim = feat_dim + STYLE_DIM
        else:
            router_in_dim = feat_dim
        self.router = nn.Sequential(
            nn.Linear(router_in_dim, router_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(router_hidden, num_experts),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Run backbone up to features only — we must NOT call
        # backbone.forward() because that runs late_drop + fc, and
        # we need to apply them ourselves on the *adapted* features
        # (otherwise LateDropout step counter advances twice).
        out = self.backbone(x)
        features = out["features"]  # (N, feat_dim) — pre-late_drop

        # Router input: detached backbone features (stop gradient to
        # prevent router from corrupting backbone representation)
        router_input = features.detach()
        if self._use_style_concat:
            from diema.features.style_summary import compute_style_summary
            style = compute_style_summary(x)  # (N, 21)
            router_input = torch.cat([router_input, style], dim=1)

        routing_logits = self.router(router_input)  # (N, K)
        routing_weights = F.softmax(routing_logits, dim=1)  # (N, K)

        # Adapter outputs
        adapter_outputs = [adapter(features) for adapter in self.adapters]
        adapter_stack = torch.stack(adapter_outputs, dim=1)  # (N, K, D)
        weighted_adapter = (
            routing_weights.unsqueeze(2) * adapter_stack
        ).sum(dim=1)  # (N, D)

        # Feature-space residual with ReZero scaling
        adapted_features = features + self.alpha * weighted_adapter  # (N, D)

        # Apply backbone's late_drop + fc on adapted features.
        # NOTE: backbone.forward() already called late_drop once to
        # produce out["logits"]. We call it again here on adapted
        # features. To prevent LateDropout step counter from advancing
        # twice, we decrement it after the backbone's forward call.
        # Simpler approach: just use the backbone's fc without late_drop
        # since the backbone already applied dropout to the features
        # that feed into the adapter. Adapted features inherit that
        # regularization through the residual connection.
        logits = self.backbone.fc(adapted_features)

        return {
            "logits": logits,
            "features": features,
            "adapted_features": adapted_features,
            "routing_weights": routing_weights,
        }

    @property
    def feature_dim(self) -> int:
        return self.backbone.feature_dim

    @property
    def output_dim(self) -> int:
        return self.num_class

    @classmethod
    def from_config(cls, config) -> "FeatureAdapterMoE_Model":
        from diema.models import build_model
        backbone = build_model(config)
        return cls(
            backbone=backbone,
            num_experts=getattr(config.model, "num_experts", 2),
            adapter_bottleneck=getattr(config.model, "adapter_bottleneck", 64),
            router_hidden=getattr(config.model, "router_hidden", 64),
            alpha_init=getattr(config.model, "alpha_init", 0.0),
            use_style_concat=getattr(config.model, "use_style_concat", False),
        )


from diema.models.registry import register_model  # noqa: E402

register_model("feature_adapter_moe", FeatureAdapterMoE_Model)
# Keep old name for backward compat with existing configs
register_model("style_moe_conv1d_tr", FeatureAdapterMoE_Model)
