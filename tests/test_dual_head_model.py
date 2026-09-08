"""Tests for DualHead_Model wrapper.

Covered:
  - forward returns all expected keys + correct shapes
  - gradient flows through backbone AND dual-head branches
  - missing "features" key in backbone output raises a clear error
  - num_performer=0 disables the style / adv heads
  - from_config builds the model from a SimpleNamespace
  - GRL is actually wired to the adversarial branch (sign check)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from diema.models import build_model, list_models
from diema.models.base import BaseModel
from diema.models.dual_head import DualHead_Model
from diema.models.skeleton_graph import DIEMA_INWARD_EDGES


def _build_region_aware_backbone() -> BaseModel:
    """Build a small Region-Aware Conv1D+Tr backbone for the wrapper tests."""
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            name="region_aware_conv1d_transformer",
            num_class=12, in_channels=6, clip_length=64,
            dim=64, num_conv_blocks=2, kernel_size=7,
            num_cross_blocks=1, num_heads=4, mlp_ratio=2.0,
            drop_rate=0.1, use_per_part_gate=True,
        ),
        skeleton=SimpleNamespace(num_nodes=25, inward_edges=DIEMA_INWARD_EDGES),
    )
    return build_model(cfg)


class _StubBackboneNoFeatures(BaseModel):
    """Backbone that forgets to expose `features` — for the error test."""

    def __init__(self, num_class: int = 12):
        super().__init__()
        self.fc = nn.Linear(25 * 6 * 64, num_class)
        self.num_class = num_class
        self.feature_dim = 128  # lie — but features key is missing

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = x.reshape(x.size(0), -1)
        return {"logits": self.fc(flat)}

    @property
    def output_dim(self) -> int:
        return self.num_class


def test_dual_head_registered():
    assert "dual_head" in list_models()


def test_dual_head_forward_shapes():
    backbone = _build_region_aware_backbone()
    model = DualHead_Model(
        backbone=backbone,
        num_class=12, num_performer=92,
        content_dim=64, style_dim=64,
    )
    x = torch.randn(3, 6, 64, 25)
    out = model(x)
    assert "logits" in out and out["logits"].shape == (3, 12)
    assert "features" in out and out["features"].shape == (3, backbone.feature_dim)
    assert out["z_content"].shape == (3, 64)
    assert out["z_style"].shape == (3, 64)
    assert out["style_logits"].shape == (3, 92)
    assert out["adv_logits"].shape == (3, 92)


def test_dual_head_gradient_flows():
    backbone = _build_region_aware_backbone()
    model = DualHead_Model(backbone=backbone, content_dim=48, style_dim=48)
    model.train()
    x = torch.randn(2, 6, 64, 25)
    out = model(x)
    # Sum all losses so every branch contributes.
    loss = (
        torch.nn.functional.cross_entropy(out["logits"], torch.tensor([0, 1]))
        + torch.nn.functional.cross_entropy(out["style_logits"], torch.tensor([0, 1]))
        + torch.nn.functional.cross_entropy(out["adv_logits"], torch.tensor([0, 1]))
    )
    loss.backward()
    backbone_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in backbone.parameters()
    )
    content_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.content_proj.parameters()
    )
    style_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.style_proj.parameters()
    )
    assert backbone_has_grad, "backbone received no gradient"
    assert content_has_grad, "content_proj received no gradient"
    assert style_has_grad, "style_proj received no gradient"


def test_dual_head_num_performer_zero_disables_aux_heads():
    backbone = _build_region_aware_backbone()
    model = DualHead_Model(
        backbone=backbone,
        num_class=12, num_performer=0,
        content_dim=48, style_dim=48,
    )
    x = torch.randn(2, 6, 64, 25)
    out = model(x)
    assert "logits" in out
    assert "z_content" in out
    assert "z_style" in out
    assert "style_logits" not in out
    assert "adv_logits" not in out


def test_dual_head_rejects_backbone_without_features():
    stub = _StubBackboneNoFeatures()
    model = DualHead_Model(backbone=stub, feature_dim=128, num_class=12, num_performer=4)
    x = torch.randn(2, 6, 64, 25)
    with pytest.raises(RuntimeError, match="features"):
        model(x)


def test_dual_head_feature_dim_mismatch_rejects():
    backbone = _build_region_aware_backbone()
    # Lie about feature_dim: pass a value that doesn't match the backbone's
    # actual pre-FC vector width. Should raise at forward time.
    with pytest.raises(RuntimeError, match="shape mismatch"):
        model = DualHead_Model(
            backbone=backbone,
            feature_dim=backbone.feature_dim + 1,
            num_class=12, num_performer=4,
        )
        model(torch.randn(2, 6, 64, 25))


def test_dual_head_grl_affects_content_gradient():
    """Sanity: the adversarial head with grl_lambda=0 should send ZERO
    gradient to content_proj (because GRL multiplies by -0 = 0), while
    grl_lambda>0 should send a non-zero gradient. We isolate the
    adversarial path by computing the loss only on ``adv_logits``.
    """
    torch.manual_seed(0)
    x = torch.randn(4, 6, 64, 25)
    y_perf = torch.tensor([0, 1, 2, 3])

    def total_content_grad_norm(grl_lambda: float) -> float:
        backbone = _build_region_aware_backbone()
        model = DualHead_Model(
            backbone=backbone,
            num_class=12, num_performer=4,
            content_dim=48, style_dim=48,
            grl_lambda=grl_lambda,
        )
        model.train()
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out["adv_logits"], y_perf)
        loss.backward()
        total = 0.0
        for p in model.content_proj.parameters():
            if p.grad is not None:
                total += float(p.grad.abs().sum().item())
        return total

    norm_pos = total_content_grad_norm(grl_lambda=0.1)
    norm_zero = total_content_grad_norm(grl_lambda=0.0)
    # With lambda=0 the GRL backward zero-scales the upstream gradient,
    # so content_proj sees exactly zero gradient from the adv branch.
    assert norm_zero == pytest.approx(0.0, abs=1e-6)
    # With lambda>0 the gradient must be non-zero.
    assert norm_pos > 0.0


def test_dual_head_from_config():
    backbone = _build_region_aware_backbone()
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            name="dual_head",
            backbone=backbone,
            feature_dim=backbone.feature_dim,
            num_class=12, num_performer=8,
            content_dim=48, style_dim=48,
            proj_dropout=0.1, grl_lambda=0.05,
        ),
        skeleton=SimpleNamespace(num_nodes=25, inward_edges=DIEMA_INWARD_EDGES),
    )
    model = DualHead_Model.from_config(cfg)
    x = torch.randn(2, 6, 64, 25)
    out = model(x)
    assert out["logits"].shape == (2, 12)
    assert out["style_logits"].shape == (2, 8)
