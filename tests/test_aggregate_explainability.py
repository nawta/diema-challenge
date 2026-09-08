""" unit tests for the explainability aggregator.

Related: tools/aggregate_explainability.py

Tests verify the loader functions tolerate missing files and parse the
schemas produced by ``explain_part_masking.py`` / ``explain_temporal_masking.py``
/ ``explain_stability.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tools.aggregate_explainability import (
    _load_cross_cultural_summary,
    _load_part_masking_summary,
    _load_stability_summary,
    _load_temporal_masking_summary,
)


def test_part_masking_summary_missing(tmp_path: Path) -> None:
    res = _load_part_masking_summary(tmp_path, "exp_nonexistent")
    assert res["status"] == "missing"


def test_part_masking_summary_ok(tmp_path: Path) -> None:
    """Minimal fixture reproducing the schema from explain_part_masking.py."""
    exp = "fake_exp"
    parts = ["torso", "head", "r_arm", "l_arm", "r_leg", "l_leg"]
    rows = []
    for f in range(2):
        for p in parts:
            rows.append({
                "experiment": exp, "fold": f, "country": "ALL", "part": p,
                "baseline_f1": 0.30, "masked_f1": 0.25,
                "f1_drop": 0.05 if p != "head" else 0.27,
                **{f"f1_drop_{e}": 0.01 for e in (
                    "anger", "contempt", "disgust", "fear", "gratitude",
                    "guilt", "jealousy", "joy", "pride", "sadness", "shame",
                    "surprise",
                )},
            })
    pd.DataFrame(rows).to_csv(tmp_path / f"{exp}_part_importance.csv", index=False)
    # Minimal faithfulness JSON
    for f in range(2):
        (tmp_path / f"{exp}_fold{f:02d}_faithfulness.json").write_text(
            json.dumps({
                "baseline_f1": 0.30,
                "aucs": {"important_first": 0.20, "reverse": 0.10, "random": 0.15},
                "curves": {},
            })
        )
    res = _load_part_masking_summary(tmp_path, exp)
    assert res["status"] == "ok"
    assert res["num_folds"] == 2
    assert abs(res["baseline_f1"] - 0.30) < 1e-6
    assert res["auc_mean"]["important_first"] == pytest.approx(0.20)
    assert res["part_drops_mean"]["head"] == pytest.approx(0.27, abs=1e-3)


def test_temporal_masking_summary_missing(tmp_path: Path) -> None:
    res = _load_temporal_masking_summary(tmp_path, "none")
    assert res["status"] == "missing"


def test_temporal_masking_summary_ok(tmp_path: Path) -> None:
    exp = "fake_exp"
    rows = [
        {"experiment": exp, "fold": 0, "baseline_f1": 0.30, "num_windows": 4,
         "auc_salient_first": 0.25, "auc_reverse": 0.24, "auc_random": 0.24},
        {"experiment": exp, "fold": 1, "baseline_f1": 0.32, "num_windows": 4,
         "auc_salient_first": 0.26, "auc_reverse": 0.25, "auc_random": 0.25},
    ]
    pd.DataFrame(rows).to_csv(tmp_path / f"{exp}_temporal_importance.csv", index=False)
    res = _load_temporal_masking_summary(tmp_path, exp)
    assert res["status"] == "ok"
    assert res["num_folds"] == 2
    assert res["num_windows"] == 4
    assert res["baseline_f1"] == pytest.approx(0.31)
    assert res["auc_salient_first_mean"] == pytest.approx(0.255)


def test_stability_summary_missing(tmp_path: Path) -> None:
    res = _load_stability_summary(tmp_path, "none")
    assert res["status"] == "missing"


def test_stability_summary_ok(tmp_path: Path) -> None:
    exp = "fake_exp"
    rows = []
    parts = ["torso", "head", "r_arm", "l_arm", "r_leg", "l_leg"]
    for f in range(2):
        for s in (0.0, 0.005, 0.01):
            rows.append({
                "experiment": exp, "fold": f, "sigma": s,
                "baseline_f1": 0.30,
                "spearman_vs_ref": 1.0 if s == 0.0 else 0.8,
                **{f"drop_{p}": 0.05 for p in parts},
            })
    pd.DataFrame(rows).to_csv(tmp_path / f"{exp}_stability.csv", index=False)
    res = _load_stability_summary(tmp_path, exp)
    assert res["status"] == "ok"
    assert res["num_folds"] == 2
    assert set(res["sigmas"]) == {0.0, 0.005, 0.01}
    assert res["spearman_mean"][0.005] == pytest.approx(0.8)
    assert res["spearman_mean"][0.0] == pytest.approx(1.0)


def test_cross_cultural_summary_requires_both_splits(tmp_path: Path) -> None:
    exp = "fake_exp"
    parts = ["torso", "head", "r_arm", "l_arm", "r_leg", "l_leg"]
    # Only JP present
    rows = [{"experiment": exp, "fold": 0, "country": "JP", "part": p,
             "baseline_f1": 0.3, "masked_f1": 0.25, "f1_drop": 0.05}
            for p in parts]
    pd.DataFrame(rows).to_csv(tmp_path / f"{exp}_part_importance_JP.csv", index=False)
    res = _load_cross_cultural_summary(tmp_path, exp)
    assert res["status"] == "missing"
    assert res["jp_present"] is True and res["tw_present"] is False


def test_cross_cultural_summary_diff(tmp_path: Path) -> None:
    exp = "fake_exp"
    parts = ["torso", "head", "r_arm", "l_arm", "r_leg", "l_leg"]
    jp_rows = [{"experiment": exp, "fold": 0, "country": "JP", "part": p,
                "baseline_f1": 0.3, "masked_f1": 0.25,
                "f1_drop": 0.2 if p == "head" else 0.05}
               for p in parts]
    tw_rows = [{"experiment": exp, "fold": 0, "country": "TW", "part": p,
                "baseline_f1": 0.3, "masked_f1": 0.27,
                "f1_drop": 0.1 if p == "head" else 0.05}
               for p in parts]
    pd.DataFrame(jp_rows).to_csv(tmp_path / f"{exp}_part_importance_JP.csv", index=False)
    pd.DataFrame(tw_rows).to_csv(tmp_path / f"{exp}_part_importance_TW.csv", index=False)

    res = _load_cross_cultural_summary(tmp_path, exp)
    assert res["status"] == "ok"
    assert res["jp_drops"]["head"] == pytest.approx(0.2)
    assert res["tw_drops"]["head"] == pytest.approx(0.1)
    assert res["jp_minus_tw_drop"]["head"] == pytest.approx(0.1)
