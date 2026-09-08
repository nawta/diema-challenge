""" unit tests — per-performer aggregation on synthetic OOF data.

See: performer analysis
Related: tools/analyze_per_performer.py
"""
from __future__ import annotations

import numpy as np
import pytest

from tools.analyze_per_performer import (
    _accuracy,
    _macro_f1,
    _perf_id,
    build_per_performer_table,
)


def test_perf_id_parses_diema_stem() -> None:
    out = _perf_id("JP_06_joy_1_L")
    assert out == ("JP_06", "JP")
    out2 = _perf_id("TW_33_anger_2_H.bvh")
    assert out2 == ("TW_33", "TW")


def test_perf_id_returns_none_on_garbage() -> None:
    assert _perf_id("not-a-diema-name") is None
    assert _perf_id("abc") is None


def test_perf_id_pads_performer_number_to_two_digits() -> None:
    """JP_6 (unpadded) stem should still produce JP_06 for downstream grouping."""
    out = _perf_id("JP_6_joy_1_L")
    assert out == ("JP_06", "JP")


def test_macro_f1_all_correct_is_one() -> None:
    y = np.array([0, 1, 2, 3])
    assert _macro_f1(y, y) == 1.0


def test_macro_f1_all_wrong_is_zero() -> None:
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([1, 2, 3, 0])
    assert _macro_f1(y_true, y_pred) == 0.0


def test_accuracy_matches_numpy_mean() -> None:
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 1, 1])
    assert _accuracy(y_true, y_pred) == pytest.approx(4 / 5)


def test_build_per_performer_groups_by_id() -> None:
    """Mixed 2-performer batch: each performer gets their own row."""
    np.random.seed(0)
    # 3 samples from JP_06 (perfect), 3 from JP_07 (all wrong)
    filenames = [
        "JP_06_joy_1_L", "JP_06_joy_1_M", "JP_06_joy_1_H",
        "JP_07_anger_2_L", "JP_07_anger_2_M", "JP_07_anger_2_H",
    ]
    labels = np.array([4, 4, 4, 0, 0, 0])
    # JP_06 predictions all correct; JP_07 predictions all wrong
    logits = np.zeros((6, 12))
    logits[:3, 4] = 10.0  # JP_06 → predict joy
    logits[3:, 5] = 10.0  # JP_07 → predict sadness (wrong)

    df = build_per_performer_table(logits, labels, filenames)
    assert set(df["performer"]) == {"JP_06", "JP_07"}
    jp06 = df[df["performer"] == "JP_06"].iloc[0]
    jp07 = df[df["performer"] == "JP_07"].iloc[0]
    assert jp06["macro_f1"] == pytest.approx(1.0)  # 1 class, all correct
    assert jp06["accuracy"] == 1.0
    assert jp06["n_samples"] == 3
    assert jp06["country"] == "JP"
    assert jp07["accuracy"] == 0.0
    assert jp07["macro_f1"] == 0.0


def test_build_per_performer_skips_unparseable_filenames() -> None:
    filenames = [
        "JP_06_joy_1_L",
        "garbage",
        "TW_33_anger_2_H",
    ]
    labels = np.array([4, 4, 0])
    logits = np.zeros((3, 12))
    logits[:, 0] = 1.0  # everyone predicts anger
    df = build_per_performer_table(logits, labels, filenames)
    # Only the 2 parseable performers appear
    assert set(df["performer"]) == {"JP_06", "TW_33"}
    # JP_06 has 1 sample (joy, predicted anger) → F1=0, n=1
    jp06 = df[df["performer"] == "JP_06"].iloc[0]
    assert jp06["n_samples"] == 1
    assert jp06["macro_f1"] == 0.0
    # TW_33 has 1 sample (anger, predicted anger) → F1=1, n=1
    tw33 = df[df["performer"] == "TW_33"].iloc[0]
    assert tw33["n_samples"] == 1
    assert tw33["macro_f1"] == 1.0


def test_build_per_performer_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="filename count"):
        build_per_performer_table(
            np.zeros((2, 12)), np.array([0, 1]), ["JP_06_joy_1_L"],
        )
