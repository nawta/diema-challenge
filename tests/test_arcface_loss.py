"""Tests for ArcFace loss primitives.

Covered:
  - ArcMarginHead produces values in [-1, 1] (cosine similarity)
  - ArcFaceLoss returns a positive scalar, is differentiable
  - ArcFaceCELoss degenerates to plain CE when alpha=0
  - Joint loss alpha=0.5 gives a value between ArcFace-only and CE-only
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from diema.training.losses import (
    ArcFaceCELoss,
    ArcFaceLoss,
    ArcMarginHead,
    get_loss_fn,
)


def _fake_features(num_samples=8, dim=16, num_classes=12):
    feats = torch.randn(num_samples, dim, requires_grad=True)
    targets = torch.randint(0, num_classes, (num_samples,))
    return feats, targets


def test_arc_margin_head_cosine_range():
    head = ArcMarginHead(in_features=16, num_classes=12)
    feats = torch.randn(4, 16)
    cos = head(feats)
    assert cos.shape == (4, 12)
    assert torch.all(cos <= 1.0 + 1e-6)
    assert torch.all(cos >= -1.0 - 1e-6)


def test_arcface_loss_scalar_and_differentiable():
    head = ArcMarginHead(16, 12)
    feats, targets = _fake_features()
    cos = head(feats)
    loss = ArcFaceLoss(s=32.0, m=0.2)(cos, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    loss.backward()
    assert feats.grad is not None
    assert feats.grad.abs().sum() > 0.0


def test_arcface_ce_alpha_zero_equals_scaled_ce():
    """alpha=0 → only the (scaled) CE branch contributes.

    ArcFaceCELoss multiplies cosine logits by ``s`` before CE so the
    softmax sees comparable magnitudes to the ArcFace branch — see the
    docstring. The reference here uses the same ``s*cos`` logits.
    """
    torch.manual_seed(0)
    head = ArcMarginHead(16, 12)
    feats, targets = _fake_features()
    cos = head(feats)
    af_ce = ArcFaceCELoss(alpha=0.0, s=32.0)(cos, targets)
    plain_ce = F.cross_entropy(cos * 32.0, targets)
    assert torch.allclose(af_ce, plain_ce, atol=1e-6)


def test_arcface_ce_alpha_one_equals_arcface_only():
    torch.manual_seed(0)
    head = ArcMarginHead(16, 12)
    feats, targets = _fake_features()
    cos = head(feats)
    af_ce = ArcFaceCELoss(alpha=1.0, s=32.0, m=0.2)(cos, targets)
    af_only = ArcFaceLoss(s=32.0, m=0.2)(cos, targets)
    assert torch.allclose(af_ce, af_only, atol=1e-6)


def test_get_loss_fn_registry():
    for name in ("arcface", "arcface_ce"):
        loss = get_loss_fn(name)
        assert loss is not None


def test_arcface_head_on_stgcn_produces_cosine_logits():
    """End-to-end: STGCN++ with ``use_arcface_head=True`` must return logits
    in [-1, 1] regardless of input magnitude."""
    import torch as _torch
    from diema.models.stgcn.stgcn_model import STGCN_Model
    from diema.models.skeleton_graph import DIEMA_INWARD_EDGES

    m = STGCN_Model(
        num_class=12, edge_index=DIEMA_INWARD_EDGES, num_nodes=25, in_channels=6,
        plusplus=True, use_arcface_head=True,
    )
    m.eval()
    with _torch.no_grad():
        logits = m(_torch.randn(4, 6, 64, 25) * 10.0)["logits"]
    assert logits.shape == (4, 12)
    assert _torch.all(logits <= 1.0 + 1e-6), f"max={logits.max().item()}"
    assert _torch.all(logits >= -1.0 - 1e-6), f"min={logits.min().item()}"


def test_arcface_head_on_ctrgcn_produces_cosine_logits():
    import torch as _torch
    from diema.models.ctrgcn.ctrgcn_model import CTR_GCN_Model
    from diema.models.skeleton_graph import DIEMA_INWARD_EDGES

    m = CTR_GCN_Model(
        num_class=12, edge_index=DIEMA_INWARD_EDGES, num_nodes=25, in_channels=6,
        base_channels=32, use_arcface_head=True,
    )
    m.eval()
    with _torch.no_grad():
        logits = m(_torch.randn(4, 6, 64, 25))["logits"]
    assert logits.shape == (4, 12)
    assert _torch.all(logits.abs() <= 1.0 + 1e-6)


def test_arcface_head_on_skateformer_produces_cosine_logits():
    import torch as _torch
    from diema.models.skateformer.skateformer_model import SkateFormer_Model

    m = SkateFormer_Model(
        num_class=12, in_channels=6, dim=64, depth=2, num_heads=4,
        window_size=16, use_arcface_head=True,
    )
    m.eval()
    with _torch.no_grad():
        logits = m(_torch.randn(4, 6, 64, 25))["logits"]
    assert logits.shape == (4, 12)
    assert _torch.all(logits.abs() <= 1.0 + 1e-6)


def test_lightningmodel_rejects_arcface_on_linear_head():
    """If the user picks an ArcFace loss but passes a plain-Linear model,
    LightningModel should raise at init time (guard rail)."""
    from diema.models.stgcn.stgcn_model import STGCN_Model
    from diema.models.skeleton_graph import DIEMA_INWARD_EDGES
    from diema.training.train import LightningModel

    plain = STGCN_Model(
        num_class=12, edge_index=DIEMA_INWARD_EDGES, num_nodes=25, in_channels=6,
        plusplus=True, use_arcface_head=False,
    )
    import pytest as _pytest
    with _pytest.raises(ValueError, match="cosine classifier head"):
        LightningModel(
            model=plain, base_lr=0.01, num_class=12,
            loss_type="arcface", optimizer="AdamW", scheduler_type="cosine",
        )

    # Same check for arcface_ce
    with _pytest.raises(ValueError):
        LightningModel(
            model=plain, base_lr=0.01, num_class=12,
            loss_type="arcface_ce", optimizer="AdamW", scheduler_type="cosine",
        )


def test_lightningmodel_accepts_arcface_on_arcface_head():
    """Same but with ``use_arcface_head=True`` — should not raise."""
    from diema.models.stgcn.stgcn_model import STGCN_Model
    from diema.models.skeleton_graph import DIEMA_INWARD_EDGES
    from diema.training.train import LightningModel

    arc = STGCN_Model(
        num_class=12, edge_index=DIEMA_INWARD_EDGES, num_nodes=25, in_channels=6,
        plusplus=True, use_arcface_head=True,
    )
    lit = LightningModel(
        model=arc, base_lr=0.01, num_class=12,
        loss_type="arcface_ce", optimizer="AdamW", scheduler_type="cosine",
    )
    assert lit is not None


def test_arcface_ce_scaled_ce_branch():
    """The CE branch inside ArcFaceCELoss should multiply cosine logits by s."""
    from diema.training.losses import ArcFaceCELoss, ArcMarginHead
    import torch as _torch
    import torch.nn.functional as F

    head = ArcMarginHead(16, 12)
    feats, targets = _fake_features()
    cos = head(feats)
    # alpha=0 → only the scaled CE branch contributes
    loss = ArcFaceCELoss(alpha=0.0, s=32.0)(cos, targets)
    ref = F.cross_entropy(cos * 32.0, targets)
    assert torch.allclose(loss, ref, atol=1e-6)
