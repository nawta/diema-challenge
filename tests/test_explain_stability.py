""" unit tests — rank ordering + Spearman correlation helpers.

Related: tools/explain_stability.py
"""
from __future__ import annotations

from tools.explain_stability import _rank_order, _spearman, PARTS_ORDERED


def test_rank_order_highest_drop_gets_rank_zero() -> None:
    drops = {p: 0.0 for p in PARTS_ORDERED}
    drops[PARTS_ORDERED[2]] = 1.0  # most important
    ranks = _rank_order(drops)
    # The part that was assigned 1.0 must be ranked 0
    assert ranks[2] == 0
    # Ranks must cover 0..5 exactly once
    assert sorted(ranks) == list(range(6))


def test_rank_order_ties_produce_stable_order() -> None:
    """When all drops tie, _rank_order is deterministic (matches PARTS_ORDERED)."""
    drops = {p: 0.5 for p in PARTS_ORDERED}
    r1 = _rank_order(drops)
    r2 = _rank_order(drops)
    assert r1 == r2


def test_spearman_same_rank_returns_one() -> None:
    r = [0, 1, 2, 3, 4, 5]
    assert _spearman(r, r) == 1.0


def test_spearman_reverse_rank_returns_minus_one() -> None:
    a = [0, 1, 2, 3, 4, 5]
    b = [5, 4, 3, 2, 1, 0]
    assert _spearman(a, b) == -1.0


def test_spearman_partial_overlap_intermediate() -> None:
    """A small swap yields a correlation between -1 and 1 (not either extreme)."""
    a = [0, 1, 2, 3, 4, 5]
    b = [1, 0, 2, 3, 4, 5]  # swap first two
    rho = _spearman(a, b)
    assert -1.0 < rho < 1.0
    assert rho > 0.9  # still highly correlated (5 of 6 positions match)
