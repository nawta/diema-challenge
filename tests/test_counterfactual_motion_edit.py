""" unit tests — counterfactual edit math on synthetic tensors.

Related: tools/counterfactual_motion_edit.py
"""
from __future__ import annotations

import pytest
import torch

from tools.counterfactual_motion_edit import EDIT_TYPES, PARTS_ORDERED, _apply_edit


def test_freeze_sets_temporal_variation_to_zero() -> None:
    """After ``freeze``, the selected nodes' temporal variance is exactly 0."""
    x = torch.randn(2, 6, 64, 25)
    nodes = (5, 6, 7, 8)  # head
    x_edit = _apply_edit(x, nodes, "freeze")
    # For edited nodes the per-frame values equal the temporal mean
    slc = x_edit[:, :, :, list(nodes)]
    mean = slc.mean(dim=2, keepdim=True)
    assert torch.allclose(slc, mean.expand_as(slc), atol=1e-6)
    # Variance is zero on the time axis
    assert torch.allclose(slc.var(dim=2), torch.zeros_like(slc.var(dim=2)), atol=1e-6)


def test_freeze_does_not_alter_other_nodes() -> None:
    x = torch.randn(2, 6, 64, 25)
    nodes = (5, 6, 7, 8)
    x_edit = _apply_edit(x, nodes, "freeze")
    other = [n for n in range(25) if n not in nodes]
    assert torch.allclose(x_edit[:, :, :, other], x[:, :, :, other])


def test_damp_halves_temporal_variation_around_mean() -> None:
    x = torch.randn(3, 6, 64, 25)
    nodes = (0, 1, 2)  # torso-ish subset
    x_edit = _apply_edit(x, nodes, "damp_0.5")
    slc_orig = x[:, :, :, list(nodes)]
    slc_edit = x_edit[:, :, :, list(nodes)]
    mean_orig = slc_orig.mean(dim=2, keepdim=True)
    mean_edit = slc_edit.mean(dim=2, keepdim=True)
    # Mean is preserved (pose intact)
    assert torch.allclose(mean_orig, mean_edit, atol=1e-6)
    # Variation is exactly halved
    var_ratio = slc_edit.var(dim=2) / slc_orig.var(dim=2).clamp_min(1e-8)
    expected = torch.full_like(var_ratio, 0.25)  # (0.5)^2
    assert torch.allclose(var_ratio, expected, atol=1e-5)


def test_amplify_doubles_temporal_variation_around_mean() -> None:
    x = torch.randn(3, 6, 64, 25)
    nodes = (17, 18, 19, 20)  # r_leg
    x_edit = _apply_edit(x, nodes, "amplify_2.0")
    slc_orig = x[:, :, :, list(nodes)]
    slc_edit = x_edit[:, :, :, list(nodes)]
    mean_orig = slc_orig.mean(dim=2, keepdim=True)
    mean_edit = slc_edit.mean(dim=2, keepdim=True)
    assert torch.allclose(mean_orig, mean_edit, atol=1e-6)
    var_ratio = slc_edit.var(dim=2) / slc_orig.var(dim=2).clamp_min(1e-8)
    expected = torch.full_like(var_ratio, 4.0)  # (2.0)^2
    assert torch.allclose(var_ratio, expected, atol=1e-5)


def test_edit_returns_fresh_tensor() -> None:
    """The tool must not mutate the caller's input (same as explain_part_masking)."""
    x = torch.randn(1, 6, 64, 25)
    x_ref = x.clone()
    _ = _apply_edit(x, (5, 6, 7, 8), "freeze")
    assert torch.allclose(x, x_ref), "_apply_edit mutated caller's tensor"


def test_unknown_edit_raises() -> None:
    x = torch.zeros(1, 6, 64, 25)
    with pytest.raises(ValueError, match="unknown edit"):
        _apply_edit(x, (0,), "zeroify")


def test_constants_cover_six_parts_and_three_edits() -> None:
    """Regression guard: anyone changing PARTS_ORDERED or EDIT_TYPES must
    update downstream figure / table sizing."""
    assert len(PARTS_ORDERED) == 6
    assert set(EDIT_TYPES) == {"freeze", "damp_0.5", "amplify_2.0"}
