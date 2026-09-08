""" tests: motion↔text retrieval metrics.

Covers recall_at_k, median_rank, same-class consistency, cross-group
clustering, and the full motion_to_text_retrieval aggregate — each against
synthetic constructions with known ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diema.evaluation.retrieval import (  # noqa: E402
    cross_group_topk_fraction,
    median_rank,
    motion_to_text_retrieval,
    recall_at_k,
    same_class_topk_fraction,
)


# -- recall / medr on a handcrafted sim ------------------------------------

def test_recall_at_k_perfect():
    # 3 queries, 3 candidates. Diagonal is argmax.
    sim = torch.tensor([
        [9.0, 0.0, 0.0],
        [0.0, 9.0, 0.0],
        [0.0, 0.0, 9.0],
    ])
    pos = torch.tensor([0, 1, 2])
    out = recall_at_k(sim, pos, ks=(1, 2, 3))
    assert out["recall_at_1"] == 1.0
    assert out["recall_at_2"] == 1.0
    assert out["recall_at_3"] == 1.0


def test_recall_at_k_no_hit():
    # Positive is always the last-ranked item.
    sim = torch.tensor([
        [0.0, 1.0, 2.0],   # pos=0 → rank 3
        [0.0, 1.0, 2.0],   # pos=0 → rank 3
    ])
    pos = torch.tensor([0, 0])
    out = recall_at_k(sim, pos, ks=(1, 2))
    assert out["recall_at_1"] == 0.0
    assert out["recall_at_2"] == 0.0


def test_recall_at_k_partial():
    sim = torch.tensor([
        [9.0, 0.0, 5.0],   # pos=0 → rank 1 (hit@1)
        [1.0, 2.0, 3.0],   # pos=0 → rank 3 (miss@2)
        [5.0, 8.0, 9.0],   # pos=1 → rank 2 (hit@2, miss@1)
    ])
    pos = torch.tensor([0, 0, 1])
    out = recall_at_k(sim, pos, ks=(1, 2))
    assert out["recall_at_1"] == pytest.approx(1 / 3)
    assert out["recall_at_2"] == pytest.approx(2 / 3)


def test_recall_at_k_rejects_bad_k():
    sim = torch.randn(4, 3)
    with pytest.raises(ValueError, match="k=0"):
        recall_at_k(sim, torch.zeros(4, dtype=torch.long), ks=(0,))
    with pytest.raises(ValueError, match="k=4"):
        recall_at_k(sim, torch.zeros(4, dtype=torch.long), ks=(4,))


def test_median_rank_matches_hand_computation():
    # Distinct similarity values so argsort is deterministic (no tie-break dependence).
    sim = torch.tensor([
        [9.0, 5.0, 1.0],   # pos=0 → rank 1
        [1.0, 5.0, 9.0],   # pos=0 → rank 3
        [1.0, 9.0, 5.0],   # pos=0 → rank 3
    ])
    pos = torch.tensor([0, 0, 0])
    # Ranks = [1, 3, 3] → median = 3.
    assert median_rank(sim, pos) == 3.0


# -- same-class consistency ------------------------------------------------

def test_same_class_topk_all_same():
    # All 4 points have the same label; any neighbor is same-label (except self-mask).
    z = torch.randn(4, 8)
    sim = z @ z.t()
    labels = torch.tensor([0, 0, 0, 0])
    assert same_class_topk_fraction(sim, labels, k=3) == 1.0


def test_same_class_topk_diverse():
    # 4 points, 4 classes. 0 same-class neighbors achievable → 0.
    z = torch.randn(4, 8)
    sim = z @ z.t()
    labels = torch.tensor([0, 1, 2, 3])
    assert same_class_topk_fraction(sim, labels, k=3) == 0.0


def test_same_class_topk_self_mask():
    # Without self-mask we'd always count the query itself; check that's excluded.
    # Construct a case where self-sim is high but no other same-label points.
    N = 3
    sim = torch.eye(N) * 100.0  # each row is all zeros except its self
    labels = torch.tensor([0, 1, 2])
    # Self-mask forces neighbors to be cross-label since there are no intra-label pairs.
    assert same_class_topk_fraction(sim, labels, k=1) == 0.0


# -- cross-group clustering -----------------------------------------------

def test_cross_group_topk_all_different_groups():
    N = 4
    sim = torch.rand(N, N)
    groups = torch.tensor([0, 1, 2, 3])  # all distinct
    # Any neighbor is from a different group (no ties).
    assert cross_group_topk_fraction(sim, groups, k=3) == 1.0


def test_cross_group_topk_all_same_group():
    N = 4
    sim = torch.rand(N, N)
    groups = torch.zeros(N, dtype=torch.long)  # everyone in group 0
    # Every neighbor shares the group → 0 cross-group.
    assert cross_group_topk_fraction(sim, groups, k=3) == 0.0


def test_cross_group_topk_with_label_filter():
    # 4 points. Labels: [0, 0, 1, 1]. Groups: [A, A, B, B].
    # Within same-label pairs, all neighbors are same-group → 0.
    sim = torch.tensor([
        [0, 9, 3, 2],
        [9, 0, 2, 1],
        [3, 2, 0, 9],
        [2, 1, 9, 0],
    ], dtype=torch.float)
    labels = torch.tensor([0, 0, 1, 1])
    groups = torch.tensor([0, 0, 1, 1])
    # k=1 neighbor for each query:
    #   q0: top is idx 1 (label 0, group 0) → same label, same group → not cross-group
    #   q1: top is idx 0 (label 0, group 0) → same label, same group → not cross-group
    #   q2: top is idx 3 (label 1, group 1) → same label, same group → not cross-group
    #   q3: top is idx 2 (label 1, group 1) → same label, same group → not cross-group
    # Total same-label neighbors: 4, cross-group: 0.
    assert cross_group_topk_fraction(sim, groups, labels=labels, k=1) == 0.0


def test_cross_group_topk_zero_same_label_edge():
    # Forced singleton classes; no same-label neighbor pairs exist.
    sim = torch.randn(3, 3)
    labels = torch.tensor([0, 1, 2])
    groups = torch.tensor([0, 0, 0])
    # No same-label pairs → numerator 0 / denominator 0 → 0.0.
    assert cross_group_topk_fraction(sim, groups, labels=labels, k=2) == 0.0


# -- aggregate wrapper ----------------------------------------------------

def test_motion_to_text_retrieval_perfect_diagonal():
    N, D = 6, 8
    torch.manual_seed(42)
    z = torch.randn(N, D)
    # Make t_text a perfect match to z_motion.
    t = z.clone()
    pos = torch.arange(N)
    out = motion_to_text_retrieval(z, t, pos, ks=(1, 3))
    assert out["recall_at_1"] == 1.0
    assert out["recall_at_3"] == 1.0
    assert out["medr"] == 1.0
    assert out["n_valid_queries"] == N
    assert out["n_queries"] == N


def test_motion_to_text_retrieval_with_missing_positive():
    N, D = 5, 8
    z = torch.randn(N, D)
    t = z.clone()
    pos = torch.tensor([0, 1, -1, 3, 4])  # -1 marks no positive
    out = motion_to_text_retrieval(z, t, pos, ks=(1,))
    assert out["n_valid_queries"] == 4
    assert out["recall_at_1"] == 1.0  # only the 4 valid ones count


def test_motion_to_text_retrieval_no_valid_queries():
    N, D = 3, 4
    z = torch.randn(N, D)
    t = torch.randn(N, D)
    pos = torch.tensor([-1, -1, -1])
    out = motion_to_text_retrieval(z, t, pos, ks=(1,))
    assert out["n_valid_queries"] == 0
    assert out["recall_at_1"] == 0.0
    assert torch.isnan(torch.tensor(out["medr"]))


def test_motion_to_text_retrieval_with_labels_and_groups():
    N, D = 4, 8
    torch.manual_seed(0)
    z = torch.randn(N, D)
    t = z.clone()
    pos = torch.arange(N)
    labels = torch.tensor([0, 0, 1, 1])
    groups = torch.tensor([0, 1, 0, 1])
    # N=4 candidates, so ks must be ≤ 4.
    out = motion_to_text_retrieval(
        z, t, pos,
        ks=(1, 3),
        labels=labels, groups=groups,
        same_class_k=1, cross_group_k=1,
    )
    assert "same_class_top1_frac" in out
    assert "cross_group_top1_frac" in out
    # Perfect diagonal → recall@1 == 1.0 for the retrieval part.
    assert out["recall_at_1"] == 1.0


def test_motion_to_text_retrieval_dim_mismatch_raises():
    z = torch.randn(3, 4)
    t = torch.randn(3, 5)
    with pytest.raises(ValueError, match="embedding dim mismatch"):
        motion_to_text_retrieval(z, t, [0, 1, 2])


def test_motion_to_text_retrieval_positive_length_mismatch_raises():
    z = torch.randn(3, 4)
    t = torch.randn(3, 4)
    with pytest.raises(ValueError, match="positive_map"):
        motion_to_text_retrieval(z, t, [0, 1])  # too short
