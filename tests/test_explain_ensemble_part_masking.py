""" unit tests — ensemble-level body-part zero-mask faithfulness.

Related:
- tools/explain_ensemble_part_masking.py  (the tool under test)
- tools/explain_part_masking.py  (single-model baseline pattern this generalises)
- experiments/exp090_ensemble_patterns/build_logitmean_submission.py
    (canonical logit-mean fusion the ensemble must match)

Tests verify the fusion math, mask semantics, and faithfulness curve logic
without requiring real checkpoints. The model-loading and fold-iteration
paths are exercised in a separate smoke run.
"""
from __future__ import annotations

import numpy as np
import torch

from diema.models.skeleton_graph import DIEMA_BODY_PARTS
from tools.explain_ensemble_part_masking import (
    PARTS_ORDERED,
    _logit_mean_fuse,
    _logit_mean_fuse_argmax,
    _ensemble_macro_f1,
)


# --------------------------------------------------------------------------- #
# Fusion math
# --------------------------------------------------------------------------- #


def test_logit_mean_fuse_matches_canonical_submission_formula() -> None:
    """The fused log-prob must equal ``log(softmax(L)).mean(0)``.

    This is the canonical convention in
    ``experiments/exp090_ensemble_patterns/build_logitmean_submission.py``::

        P = softmax(logits)         # per-member probs, shape (M, N, C)
        final = softmax(log(P).mean(0))

    Our helper returns ``log(P).mean(0)`` (unnormalised log-probs). The argmax
    is invariant to the final softmax so we test the log-prob directly.
    """
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(3, 5, 12)).astype(np.float64)
    # Reference: softmax → log → mean
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=axis, keepdims=True)
    P = softmax(logits, axis=-1)  # (M, N, C)
    ref = np.log(np.clip(P, 1e-12, 1.0)).mean(axis=0)  # (N, C)
    got = _logit_mean_fuse(list(logits))
    assert got.shape == (5, 12)
    assert np.allclose(got, ref, atol=1e-10), (got - ref).max()


def test_logit_mean_fuse_identity_with_single_member() -> None:
    """When M=1, fused argmax = single-member argmax."""
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(7, 12)).astype(np.float64)
    pred_solo = logits.argmax(1)
    pred_fused = _logit_mean_fuse_argmax([logits])
    assert np.array_equal(pred_solo, pred_fused)


def test_logit_mean_fuse_invariant_to_member_order() -> None:
    """Mean is commutative, so the ensemble prediction is order-invariant."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(4, 12)).astype(np.float64)
    b = rng.normal(size=(4, 12)).astype(np.float64)
    c = rng.normal(size=(4, 12)).astype(np.float64)
    lp_abc = _logit_mean_fuse([a, b, c])
    lp_cba = _logit_mean_fuse([c, b, a])
    assert np.allclose(lp_abc, lp_cba, atol=1e-12)


def test_logit_mean_fuse_argmax_matches_softmax_of_logprob() -> None:
    """argmax of log-prob == argmax of softmax(log-prob) (softmax is monotone)."""
    rng = np.random.default_rng(3)
    logits_list = [rng.normal(size=(8, 12)).astype(np.float64) for _ in range(11)]
    lp = _logit_mean_fuse(logits_list)
    # softmax of log-prob → renormalised probs
    e = np.exp(lp - lp.max(axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)
    assert np.array_equal(lp.argmax(1), probs.argmax(1))
    assert np.array_equal(_logit_mean_fuse_argmax(logits_list), lp.argmax(1))


def test_logit_mean_fuse_rejects_misaligned_shapes() -> None:
    """Stacking must fail loudly when members produce different (N, C)."""
    a = np.zeros((4, 12), dtype=np.float64)
    b = np.zeros((5, 12), dtype=np.float64)  # wrong N
    import pytest
    with pytest.raises((ValueError, AssertionError)):
        _logit_mean_fuse([a, b])


def test_ensemble_macro_f1_matches_sklearn_on_synthetic() -> None:
    """End-to-end Macro-F1 on toy ensemble predictions."""
    from sklearn.metrics import f1_score
    labels = np.array([0, 1, 2, 0, 1, 2, 3, 3])
    # Member A votes correctly for the first 6, wrong on last 2
    # Member B votes correctly on last 4, wrong on first 4
    # Mean log-prob arbitrates by majority; we test the helper directly.
    rng = np.random.default_rng(4)
    a = -5.0 * np.ones((8, 12)) + rng.normal(scale=1e-2, size=(8, 12))
    b = -5.0 * np.ones((8, 12)) + rng.normal(scale=1e-2, size=(8, 12))
    for i, y in enumerate(labels):
        a[i, y] = +10.0 if i < 6 else -10.0
        b[i, y] = -10.0 if i < 4 else +10.0
    f1 = _ensemble_macro_f1([a, b], labels)
    pred = _logit_mean_fuse_argmax([a, b])
    assert f1 == f1_score(labels, pred, average="macro", zero_division=0)


# --------------------------------------------------------------------------- #
# Body-part mask semantics (inherited contract from single-model tool)
# --------------------------------------------------------------------------- #


def test_parts_ordered_inherited_from_single_tool() -> None:
    """Same 6-part partition as the single-model tool."""
    assert set(PARTS_ORDERED) == set(DIEMA_BODY_PARTS.keys())
    assert len(PARTS_ORDERED) == 6


def test_mask_zeroes_requested_nodes_for_ctv25_input() -> None:
    """Sanity: zero-mask the joint axis (V=25) without touching other axes."""
    x = torch.randn(2, 6, 64, 25)
    x_orig = x.clone()
    mask_nodes = tuple(DIEMA_BODY_PARTS["r_arm"])  # [9, 10, 11, 12]
    for n in mask_nodes:
        x[:, :, :, n] = 0.0
    for n in mask_nodes:
        assert torch.all(x[:, :, :, n] == 0)
    other = [n for n in range(25) if n not in mask_nodes]
    for n in other:
        assert torch.allclose(x[:, :, :, n], x_orig[:, :, :, n])


# --------------------------------------------------------------------------- #
# Faithfulness curve & AUC
# --------------------------------------------------------------------------- #


def test_auc_drop_matches_single_tool_definition() -> None:
    """Reuse the same AUC definition as the single-model tool for comparability."""
    from tools.explain_part_masking import _auc_drop
    # Monotone decrease 0.5 → 0.2: drops mean = 0.2
    assert abs(_auc_drop([0.5, 0.4, 0.3, 0.2]) - 0.2) < 1e-6
    # Flat curve = 0
    assert _auc_drop([0.4] * 4) == 0.0


def test_ordering_helpers_agree_with_drops() -> None:
    """important_first / reverse / random must produce the expected orderings."""
    from tools.explain_ensemble_part_masking import _orderings_for_drops
    drops = {"head": 0.10, "r_arm": 0.05, "torso": 0.02,
             "l_arm": 0.04, "r_leg": 0.07, "l_leg": 0.01}
    important, reverse, random = _orderings_for_drops(drops, seed=0)
    assert important == ["head", "r_leg", "r_arm", "l_arm", "torso", "l_leg"]
    assert reverse == list(reversed(important))
    assert sorted(random) == sorted(drops.keys())
    assert len(random) == 6


# --------------------------------------------------------------------------- #
# Mask-mode semantics (zero vs shuffle)
# --------------------------------------------------------------------------- #


def test_zero_mask_sets_indicated_joints_to_zero() -> None:
    """``mode='zero'`` zeroes the requested joints, leaves others intact."""
    from tools.explain_ensemble_part_masking import _apply_mask_inplace
    x = torch.randn(4, 6, 64, 25)
    x_orig = x.clone()
    nodes = (5, 6, 7, 8)  # head
    _apply_mask_inplace(x, nodes, mode="zero")
    for n in nodes:
        assert torch.all(x[:, :, :, n] == 0)
    for n in [v for v in range(25) if v not in nodes]:
        assert torch.allclose(x[:, :, :, n], x_orig[:, :, :, n])


def test_shuffle_mask_permutes_indicated_joints_across_batch() -> None:
    """``mode='shuffle'`` replaces each sample's masked joints with another's."""
    from tools.explain_ensemble_part_masking import _apply_mask_inplace
    # Distinct per-batch signal: x[b, :, :, n] = b * 10 + n so we can spot
    # the permutation.
    x = torch.zeros(5, 1, 1, 25)
    for b in range(5):
        for n in range(25):
            x[b, 0, 0, n] = b * 10 + n
    x_orig = x.clone()
    nodes = (5, 6, 7, 8)
    perm = torch.tensor([4, 3, 2, 1, 0])  # reverse
    _apply_mask_inplace(x, nodes, mode="shuffle", perm=perm)
    for b in range(5):
        for n in nodes:
            expected = (4 - b) * 10 + n  # b-th sample takes (4-b)-th's joint
            assert x[b, 0, 0, n].item() == expected, (
                f"shuffle: x[{b},...,{n}] = {x[b,0,0,n]} ≠ {expected}"
            )
        for n in [v for v in range(25) if v not in nodes]:
            assert x[b, 0, 0, n].item() == x_orig[b, 0, 0, n].item()


def test_shuffle_mask_preserves_marginal_distribution() -> None:
    """Shuffle preserves the per-joint marginal exactly (it's a permutation)."""
    from tools.explain_ensemble_part_masking import _apply_mask_inplace
    rng = np.random.default_rng(7)
    x = torch.from_numpy(rng.normal(size=(64, 6, 32, 25)).astype(np.float32))
    x_orig = x.clone()
    nodes = (9, 10, 11, 12)  # r_arm
    perm = torch.from_numpy(rng.permutation(64))
    _apply_mask_inplace(x, nodes, mode="shuffle", perm=perm)
    # Each masked joint's set of (sample, time) values is preserved as a multiset.
    for n in nodes:
        before = x_orig[:, :, :, n].numpy().ravel()
        after = x[:, :, :, n].numpy().ravel()
        assert np.array_equal(np.sort(before), np.sort(after))


def test_shuffle_mask_rejects_missing_perm() -> None:
    from tools.explain_ensemble_part_masking import _apply_mask_inplace
    x = torch.zeros(4, 6, 64, 25)
    import pytest
    with pytest.raises(ValueError, match="permutation"):
        _apply_mask_inplace(x, (5,), mode="shuffle", perm=None)


def test_unknown_mask_mode_raises() -> None:
    from tools.explain_ensemble_part_masking import _apply_mask_inplace
    x = torch.zeros(2, 6, 64, 25)
    import pytest
    with pytest.raises(ValueError, match="unknown mask mode"):
        _apply_mask_inplace(x, (5,), mode="gaussian")


# --------------------------------------------------------------------------- #
# Post-C3-fix tests: C3D grouped permutation should use ONE donor per call
# --------------------------------------------------------------------------- #


def test_c3d_grouped_permutation_uses_one_donor() -> None:
    """``_apply_marker_mask(mode='shuffle')`` must use a single donor
    for the whole masked-part group (post-C3 fix 2026-06-12); the
    legacy behaviour of per-marker independent donor was rejected by
    the a review (breaks within-part coherence)."""
    from tools.explain_c3d_part_masking import _apply_marker_mask
    # Distinct per-clip signal: marker m of clip c has values (c, m, c+m, ...)
    F, M = 4, 6
    clip_labels = [f"MK{m}" for m in range(M)]
    target = np.zeros((F, M, 3), dtype=np.float32)
    for m in range(M):
        target[:, m, :] = -1.0  # target marker values are -1
    donors = []
    for c in range(3):
        d = np.zeros((F, M, 3), dtype=np.float32)
        for m in range(M):
            d[:, m, :] = 100 * (c + 1) + m  # each donor's marker has unique sentinel
        donors.append((d, clip_labels))

    rng = np.random.default_rng(0)
    out = _apply_marker_mask(
        target, clip_labels, ("MK1", "MK3", "MK5"), mode="shuffle",
        donor_markers_and_labels=donors, rng=rng,
    )
    # Verify the masked markers all share the SAME donor offset (100, 200, or 300):
    base_offsets = set()
    for m_name in ("MK1", "MK3", "MK5"):
        m = clip_labels.index(m_name)
        # marker values are 100*(c+1) + m, so subtracting m gives the donor offset.
        marker_offset = out[0, m, 0] - m
        base_offsets.add(int(marker_offset))
    assert len(base_offsets) == 1, (
        f"Expected one donor offset across all masked markers; got {base_offsets}"
    )

    # Verify un-masked markers are still the target value (-1).
    for m_name in ("MK0", "MK2", "MK4"):
        m = clip_labels.index(m_name)
        assert (out[:, m, :] == -1.0).all()


def test_c3d_shuffle_falls_back_to_independent_donor_when_no_full_donor() -> None:
    """When no single donor has ALL the masked markers, we fall back to
    per-marker best-effort (each marker may be replaced from a different
    donor; if the per-marker 5-try retry also fails because the rng
    happens to pick the donor missing that marker every time, we land
    on NaN — that's a documented worst case, not a bug).

    We aggregate across multiple seeds so a single unlucky seed doesn't
    flake the test.
    """
    from tools.explain_c3d_part_masking import _apply_marker_mask
    F, M = 3, 4
    target_labels = ["A", "B", "C", "D"]
    target = np.zeros((F, M, 3), dtype=np.float32)
    donor1 = (np.full((F, M, 3), 100.0, dtype=np.float32), ["A", "X", "C", "D"])  # no B
    donor2 = (np.full((F, M, 3), 200.0, dtype=np.float32), ["A", "B", "X", "D"])  # no C

    successes = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        out = _apply_marker_mask(
            target, target_labels, ("B", "C"), mode="shuffle",
            donor_markers_and_labels=[donor1, donor2], rng=rng,
        )
        # A and D must always be untouched.
        assert (out[:, 0, :] == 0).all()
        assert (out[:, 3, :] == 0).all()
        # Count seeds where BOTH B and C ended up replaced (probability
        # ≈ (1 - 0.5^5)^2 ≈ 94 %).
        if not np.isnan(out[:, 1, :]).any() and not np.isnan(out[:, 2, :]).any():
            successes += 1
    # At least 80 % of seeds should successfully replace both markers.
    assert successes >= 16, (
        f"only {successes}/20 seeds replaced both markers; "
        f"the per-marker fallback is too brittle"
    )
