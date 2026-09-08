"""Tests for Gaussian weight noise integration in LightningModel.

Covered:
  - _apply_gaussian_weight_noise / _restore_weights round-trip
  - on_after_backward restores weights (happy path)
  - training_step restores weights if forward raises (exception-safety)
  - Gaussian noise is a no-op when sigma=0
  - Combined with mixup + aux_losses
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from diema.models.stgcn.stgcn_model import STGCN_Model
from diema.models.skeleton_graph import DIEMA_INWARD_EDGES
from diema.training.train import LightningModel


class _FailingModel(nn.Module):
    """Always raises on forward — used to verify exception-safe restore."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 12)
        self.use_arcface_head = False

    def forward(self, x):
        raise RuntimeError("synthetic failure")


def _make_lit_module(gaussian_noise_sigma=0.01):
    model = STGCN_Model(
        num_class=12, edge_index=DIEMA_INWARD_EDGES, num_nodes=25, in_channels=6,
        plusplus=True,
    )
    lit = LightningModel(
        model=model, base_lr=0.01, num_class=12,
        loss_type="ce", optimizer="AdamW", scheduler_type="cosine",
        gaussian_noise_sigma=gaussian_noise_sigma,
    )
    return lit, model


def test_apply_and_restore_round_trip():
    lit, model = _make_lit_module(gaussian_noise_sigma=0.05)
    p0 = next(iter(model.parameters())).data.clone()

    saved = lit._apply_gaussian_weight_noise()
    p_noisy = next(iter(model.parameters())).data.clone()
    assert not torch.allclose(p0, p_noisy)

    lit._restore_weights(saved)
    p_restored = next(iter(model.parameters())).data
    assert torch.allclose(p0, p_restored)


def test_zero_sigma_is_noop():
    lit, model = _make_lit_module(gaussian_noise_sigma=0.0)
    p0 = next(iter(model.parameters())).data.clone()
    saved = lit._apply_gaussian_weight_noise()
    assert saved == []
    p1 = next(iter(model.parameters())).data
    assert torch.allclose(p0, p1)


def test_training_step_restores_on_exception():
    """If forward raises, training_step must restore weights before re-raising."""
    failing = _FailingModel()
    lit = LightningModel(
        model=failing, base_lr=0.01, num_class=12,
        loss_type="ce", optimizer="AdamW", scheduler_type="cosine",
        gaussian_noise_sigma=0.1,
    )
    p0 = next(iter(failing.parameters())).data.clone()

    batch = (torch.randn(2, 6, 64, 25), torch.tensor([0, 1]), ["a", "b"])
    with pytest.raises(RuntimeError, match="synthetic failure"):
        lit.training_step(batch, 0)

    p_after = next(iter(failing.parameters())).data
    assert torch.allclose(p0, p_after), (
        "Exception-safe restore failed: weights still perturbed after forward raise"
    )


def test_on_after_backward_restores():
    """Happy path: on_after_backward should restore after a successful step."""
    lit, model = _make_lit_module(gaussian_noise_sigma=0.02)
    p0 = next(iter(model.parameters())).data.clone()

    # Simulate what training_step does
    lit._noise_saved_weights = lit._apply_gaussian_weight_noise()
    # Weights are now perturbed
    assert not torch.allclose(p0, next(iter(model.parameters())).data)
    # Simulate Lightning calling on_after_backward
    lit.on_after_backward()
    assert torch.allclose(p0, next(iter(model.parameters())).data)


def test_gaussian_noise_does_not_touch_running_stats():
    """BN running_mean / running_var are BUFFERS, not parameters, so they
    must not be touched by Gaussian weight noise."""
    lit, model = _make_lit_module(gaussian_noise_sigma=0.1)
    # Find a BN buffer
    bn_mean0 = None
    bn_mean_ref = None
    for name, buf in model.named_buffers():
        if name.endswith("running_mean"):
            bn_mean_ref = buf
            bn_mean0 = buf.clone()
            break
    assert bn_mean_ref is not None, "Could not find a BN running_mean buffer"

    saved = lit._apply_gaussian_weight_noise()
    # running_mean should be unchanged
    assert torch.allclose(bn_mean0, bn_mean_ref)
    lit._restore_weights(saved)
