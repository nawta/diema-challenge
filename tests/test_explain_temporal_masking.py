""" unit tests — temporal-window ranking and mask construction.

Related: tools/explain_temporal_masking.py
"""
from __future__ import annotations

import numpy as np
import torch

from tools.explain_temporal_masking import (
    _mask_for_top_k,
    _window_ranking,
)


def test_window_ranking_simple_case() -> None:
    """Saliency [4, 1, 2, 3] over 4 windows of 1 frame each → rank order [0, 3, 2, 1]."""
    sal = torch.tensor([[4.0, 1.0, 2.0, 3.0]])
    ranks = _window_ranking(sal, num_windows=4)
    # Rank 0 = window with highest mean → window 0 (value 4)
    # Rank 1 = window 3 (value 3), rank 2 = window 2 (value 2), rank 3 = window 1
    assert ranks.tolist() == [[0, 3, 2, 1]]


def test_window_ranking_groups_frames_by_window() -> None:
    """T=8, num_windows=4 → 2 frames per window. Mean decides rank."""
    # Windows: [0,1] mean=0.5; [2,3] mean=2.5; [4,5] mean=4.5; [6,7] mean=6.5
    sal = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    ranks = _window_ranking(sal, num_windows=4)
    assert ranks.tolist() == [[3, 2, 1, 0]]  # window 3 (frames 6,7) is most salient


def test_mask_for_top_k_salient_first_covers_highest_windows() -> None:
    """top_k=2 salient_first should zero frames in the 2 highest-mean windows."""
    sal = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    ranks = _window_ranking(sal, num_windows=4)
    mask = _mask_for_top_k(ranks, num_windows=4, T=8, top_k=2,
                           order="salient_first", seed=0)
    # Windows 3 (frames 6-7) and 2 (frames 4-5) are top-2
    expected = torch.tensor([[False, False, False, False, True, True, True, True]])
    assert torch.equal(mask, expected)


def test_mask_for_top_k_reverse_picks_lowest_windows() -> None:
    """top_k=2 reverse masks the lowest-saliency windows."""
    sal = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    ranks = _window_ranking(sal, num_windows=4)
    mask = _mask_for_top_k(ranks, num_windows=4, T=8, top_k=2,
                           order="reverse", seed=0)
    # Windows 0 (frames 0-1) and 1 (frames 2-3) are bottom-2
    expected = torch.tensor([[True, True, True, True, False, False, False, False]])
    assert torch.equal(mask, expected)


def test_mask_for_top_k_random_mask_deterministic_per_seed() -> None:
    """Same seed must produce same mask; different seeds may differ."""
    sal = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    ranks = _window_ranking(sal, num_windows=4)
    m1 = _mask_for_top_k(ranks, num_windows=4, T=8, top_k=2, order="random", seed=0)
    m2 = _mask_for_top_k(ranks, num_windows=4, T=8, top_k=2, order="random", seed=0)
    assert torch.equal(m1, m2)
    # Mask must cover exactly top_k windows = 2 × 2 frames = 4 True entries per row
    assert m1.sum(dim=1).tolist() == [4]


def test_mask_for_top_k_zero_returns_all_false() -> None:
    sal = torch.randn(3, 8)
    ranks = _window_ranking(sal, num_windows=4)
    mask = _mask_for_top_k(ranks, num_windows=4, T=8, top_k=0,
                           order="salient_first", seed=0)
    assert mask.shape == (3, 8)
    assert not mask.any()


def test_mask_top_k_num_windows_handles_nondivisible_T() -> None:
    """T=10, num_windows=3 → windows of 3, 3, 4 frames; last absorbs remainder."""
    sal = torch.zeros(1, 10)
    sal[0, 7] = 1.0  # make window 2 (frames 6..9) most salient
    ranks = _window_ranking(sal, num_windows=3)
    assert ranks[0, 0].item() == 2
    mask = _mask_for_top_k(ranks, num_windows=3, T=10, top_k=1,
                           order="salient_first", seed=0)
    # Window 2 = frames 6..9 (4 frames)
    assert mask[0, 6:10].all()
    assert not mask[0, :6].any()
    assert mask.sum().item() == 4


def test_mask_dtype_bool() -> None:
    sal = torch.randn(2, 8)
    ranks = _window_ranking(sal, num_windows=4)
    mask = _mask_for_top_k(ranks, num_windows=4, T=8, top_k=2,
                           order="salient_first", seed=0)
    assert mask.dtype == torch.bool
