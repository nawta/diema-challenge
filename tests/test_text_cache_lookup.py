""" smoke tests for ScenarioTextCache.

Related: diema/features/text_cache.py, tools/build_scenario_cache.py

Tests skip cleanly when the masked cache or train CSV is not present.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

MASKED_CACHE = Path("output/scenario_text_cache_masked.pt")
TRAIN_CSV = Path("data/diema_challenge/raw/train_data.csv")

pytestmark = pytest.mark.skipif(
    not MASKED_CACHE.exists() or not TRAIN_CSV.exists(),
    reason="requires masked scenario cache and train CSV",
)


@pytest.fixture(scope="module")
def cache():
    from diema.features.text_cache import ScenarioTextCache
    return ScenarioTextCache(cache_path=MASKED_CACHE, csv_path=TRAIN_CSV)


def test_basic_shape(cache) -> None:
    assert cache.num_unique > 1000
    assert cache.embedding_dim == 384
    assert cache.embedding_matrix.shape == (cache.num_unique, cache.embedding_dim)
    assert cache.num_filenames == 7992  # full DIEM-A train split


def test_filename_lookup_is_stable(cache) -> None:
    idx1 = cache.idx_for("JP_06_joy_1_L.bvh")
    idx2 = cache.idx_for("JP_06_joy_1_L")  # extension stripped
    assert idx1 == idx2
    assert idx1 >= 0


def test_same_scenario_shares_idx(cache) -> None:
    """Different intensities of the same scenario must share scenario_idx."""
    idx_L = cache.idx_for("JP_06_joy_1_L.bvh")
    idx_M = cache.idx_for("JP_06_joy_1_M.bvh")
    idx_H = cache.idx_for("JP_06_joy_1_H.bvh")
    assert idx_L == idx_M == idx_H


def test_unknown_filename_returns_minus_one(cache) -> None:
    assert cache.idx_for("test0001") == -1
    assert cache.idx_for("nonexistent_clip.bvh") == -1


def test_embedding_is_l2_normalized_by_default(cache) -> None:
    emb = cache.embedding_for("JP_06_joy_1_L.bvh")
    assert torch.isclose(emb.norm(), torch.tensor(1.0), atol=1e-4)
    # Unknown → zero vector
    emb_unk = cache.embedding_for("test0001")
    assert emb_unk.norm().item() == 0.0


def test_batch_index_tensor_shape(cache) -> None:
    batch = ["JP_06_joy_1_L.bvh", "JP_06_joy_1_M.bvh", "test0001"]
    idx = cache.idx_for_batch(batch)
    assert idx.shape == (3,)
    assert idx.dtype == torch.long
    assert idx[0].item() == idx[1].item()
    assert idx[2].item() == -1


def test_positive_mask_policy(cache) -> None:
    """Multi-positive mask: same-scenario + same-emotion joins L/M/H variants."""
    fns = [
        "JP_06_joy_1_L.bvh",
        "JP_06_joy_1_M.bvh",  # same scenario + same emotion
        "JP_06_anger_1_L.bvh",  # same performer, different emotion
    ]
    scenario_idx = cache.idx_for_batch(fns)
    # Emotion labels: derived outside the class (parser.EMOTION_TO_IDX).
    from diema.data.parser import EMOTION_TO_IDX
    emo = torch.tensor([EMOTION_TO_IDX["joy"], EMOTION_TO_IDX["joy"], EMOTION_TO_IDX["anger"]])

    mask = cache.positive_mask_for_batch(scenario_idx, emo, policy="same_scenario_and_emotion")
    assert mask.shape == (3, 3)
    # Diagonal is always True
    assert mask[0, 0] and mask[1, 1] and mask[2, 2]
    # (0, 1) same scenario + same emotion → True
    assert mask[0, 1] and mask[1, 0]
    # (0, 2) same performer but different emotion AND different scenario → False
    assert not mask[0, 2]
    assert not mask[1, 2]


def test_meta_records_mask_mode(cache) -> None:
    assert cache.meta.get("mask_mode") == "emotion"
    assert cache.meta.get("mask_token") == "[MASK]"
