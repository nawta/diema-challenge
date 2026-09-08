""" unit tests — body-part zero-mask logic.

Related: tools/explain_part_masking.py, diema/models/skeleton_graph.py

Tests verify the masking semantics without requiring a real checkpoint.
"""
from __future__ import annotations

import numpy as np
import torch

from diema.models.skeleton_graph import DIEMA_BODY_PARTS
from tools.explain_part_masking import PARTS_ORDERED, _auc_drop, _macro_f1, _per_class_f1


def test_parts_ordered_matches_skeleton_graph() -> None:
    """PARTS_ORDERED must be the same 6 parts as DIEMA_BODY_PARTS."""
    assert set(PARTS_ORDERED) == set(DIEMA_BODY_PARTS.keys())
    assert len(PARTS_ORDERED) == 6


def test_parts_cover_all_25_nodes() -> None:
    """All body-part node unions must equal {0, ..., 24}."""
    all_nodes: list[int] = []
    for p in PARTS_ORDERED:
        all_nodes.extend(DIEMA_BODY_PARTS[p])
    assert sorted(set(all_nodes)) == list(range(25))


def test_mask_zeroes_requested_nodes() -> None:
    """Replicate the tool's masking in isolation and confirm it does nothing else."""
    x = torch.randn(2, 6, 64, 25)
    x_orig = x.clone()
    mask_nodes = tuple(DIEMA_BODY_PARTS["head"])  # [5, 6, 7, 8]
    for n in mask_nodes:
        x[:, :, :, n] = 0.0
    # Masked nodes are 0.
    for n in mask_nodes:
        assert torch.all(x[:, :, :, n] == 0)
    # All other nodes are untouched.
    other = [n for n in range(25) if n not in mask_nodes]
    for n in other:
        assert torch.allclose(x[:, :, :, n], x_orig[:, :, :, n])


def test_macro_f1_matches_sklearn_direct() -> None:
    """Quick regression: sklearn's call and our wrapper match."""
    from sklearn.metrics import f1_score
    y = np.array([0, 1, 2, 0, 1, 2])
    p = np.array([0, 2, 2, 0, 1, 1])
    assert _macro_f1(y, p) == f1_score(y, p, average="macro", zero_division=0)


def test_per_class_f1_has_12_entries() -> None:
    y = np.array([0, 1, 2, 3, 4])
    p = np.array([0, 1, 2, 3, 4])
    res = _per_class_f1(y, p)
    assert res.shape == (12,)
    assert np.all(res[:5] == 1.0)
    assert np.all(res[5:] == 0.0)  # absent classes


def test_auc_drop_zero_when_curve_flat() -> None:
    """Flat curve = no drop = AUC = 0."""
    assert _auc_drop([0.3, 0.3, 0.3, 0.3]) == 0.0


def test_auc_drop_positive_when_curve_decreasing() -> None:
    """Monotone decrease → positive AUC."""
    curve = [0.5, 0.4, 0.3, 0.2]
    auc = _auc_drop(curve)
    # drops = [0.1, 0.2, 0.3] → mean = 0.2
    assert abs(auc - 0.2) < 1e-6


def test_auc_drop_handles_single_point_curve() -> None:
    """Edge case: only the baseline, no masking steps."""
    assert _auc_drop([0.4]) == 0.0
