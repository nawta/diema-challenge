"""TDD for exp091 submission-robustness diagnostic.

Pure-function contracts (fast, no IO) + one cheap integration check that the
locked 36.94% reproduces from cached OOF logits. Heavy steps (2000x bootstrap)
are exercised by the CLI, not the unit suite.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "exp091_diag",
    Path(__file__).resolve().parents[1]
    / "experiments/exp091_submission_robustness_diagnostic/diagnostic.py",
)
diag = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diag)


# ── pure: softmax / combine ───────────────────────────────────────────────
def test_softmax_normalizes():
    x = np.random.RandomState(0).randn(5, 12)
    p = diag.softmax(x)
    assert np.allclose(p.sum(-1), 1.0)
    assert (p >= 0).all()


def test_logit_mean_equals_geometric_mean():
    rng = np.random.RandomState(1)
    probs = [diag.softmax(rng.randn(20, 12)) for _ in range(11)]
    lm = diag.combine(probs, "logit_mean")
    gm = diag.combine(probs, "geometric_mean")
    assert np.allclose(lm, gm), "logit-mean must equal geometric-mean exactly"
    assert np.allclose(lm.sum(-1), 1.0)


def test_combine_median_trimmed_renormalize():
    rng = np.random.RandomState(2)
    probs = [diag.softmax(rng.randn(8, 12)) for _ in range(11)]
    for m in ("median", "trimmed_mean", "softmax_mean", "rank_mean"):
        out = diag.combine(probs, m)
        assert out.shape == (8, 12)
        assert np.allclose(out.sum(-1), 1.0), f"{m} not normalized"


def test_trimmed_mean_requires_three_members():
    probs = [diag.softmax(np.random.randn(4, 12)) for _ in range(2)]
    with pytest.raises(ValueError):
        diag.combine(probs, "trimmed_mean")


def test_macro_f1_range():
    y = np.array([0, 1, 2, 3] * 3)
    probs = diag.softmax(np.random.RandomState(3).randn(12, 12))
    assert 0.0 <= diag.macro_f1(probs, y) <= 1.0
    assert len(diag.per_class_f1(probs, y)) == 12


# ── pure: confusion-pair label indices ────────────────────────────────────
def test_confusion_pairs_indices():
    pairs = {(diag.IDX_TO_EMOTION[a], diag.IDX_TO_EMOTION[b], d)
             for a, b, d in diag.CONFUSION_PAIRS}
    assert ("jealousy", "contempt", False) in pairs
    assert ("guilt", "sadness", True) in pairs       # directed
    assert ("fear", "surprise", False) in pairs


def test_pair_errors_directed_vs_undirected():
    y = np.array([9, 9, 5, 5])          # guilt, guilt, sadness, sadness
    pred = np.array([5, 9, 9, 5])       # guilt->sadness x1 ; sadness->guilt x1
    assert diag._pair_errors(pred, y, 9, 5, True) == 1   # only guilt->sadness
    assert diag._pair_errors(pred, y, 9, 5, False) == 2  # both directions


# ── pure: decision rule (first-match-wins) ────────────────────────────────
def _ok_repro():
    return {"reproducible": True}


def _ok_boot():
    return {"ci_lo_gt_7way": True, "ci_lo": 0.35}


def _loose_fold():
    return {"concentrated": False, "max_single_fold_share": 0.2}


def _no_gain_sweep():
    return {"median": {"f1": 0.369, "delta_pp": 0.01,
                       "classes_regressed_gt_0_30pp": 0},
            "trimmed_mean": {"f1": 0.369, "delta_pp": -0.1,
                             "classes_regressed_gt_0_30pp": 1}}


def test_decision_D0_guard_on_bad_ci():
    d = diag.decide(_ok_repro(), _loose_fold(),
                    {"ci_lo_gt_7way": False, "ci_lo": 0.33},
                    _no_gain_sweep(), {"any_stable_rescuer": True},
                    {}, np.array([0]), np.array([0]))
    assert d["action"] == "SHIP_LOCKED" and d["branch"] == "D0_guard"


def test_decision_D0_guard_on_concentration():
    d = diag.decide(_ok_repro(),
                    {"concentrated": True, "max_single_fold_share": 0.8},
                    _ok_boot(), _no_gain_sweep(),
                    {"any_stable_rescuer": False},
                    {}, np.array([0]), np.array([0]))
    assert d["branch"] == "D0_guard"


def test_decision_D3_default_ship_locked():
    d = diag.decide(_ok_repro(), _loose_fold(), _ok_boot(),
                    _no_gain_sweep(), {"any_stable_rescuer": False},
                    {}, np.array([0]), np.array([0]))
    assert d["action"] == "SHIP_LOCKED" and d["branch"] == "D3_default"


def test_decision_D2_unlock_exp092():
    d = diag.decide(_ok_repro(), _loose_fold(), _ok_boot(),
                    _no_gain_sweep(),
                    {"any_stable_rescuer": True,
                     "pairs": [{"stable": True, "pair": "guilt->sadness"}]},
                    {}, np.array([0]), np.array([0]))
    assert d["action"] == "UNLOCK_EXP092" and d["branch"] == "D2_stable_rescuer"


def test_decision_D1_ships_strictly_better_variant():
    # 11 identical members -> every combine identical -> variant >= locked on
    # all folds, 0 regressions; sweep reports a >=+0.20pp win -> D1.
    n = 40
    P = diag.softmax(np.random.RandomState(7).randn(n, 12))
    src = {m: P.copy() for m in diag.MEMBERS11}
    y = np.arange(n) % 12
    fold_ids = np.arange(n) % 10
    sweep = {"median": {"f1": diag.LOCKED_F1 + 0.005, "delta_pp": 0.5,
                        "classes_regressed_gt_0_30pp": 0},
             "trimmed_mean": {"f1": 0.0, "delta_pp": -9.0,
                              "classes_regressed_gt_0_30pp": 9}}
    d = diag.decide(_ok_repro(), _loose_fold(), _ok_boot(), sweep,
                    {"any_stable_rescuer": True}, src, y, fold_ids)
    assert d["action"] == "SHIP_VARIANT" and d["variant"] == "median"
    assert d["folds_nonneg"] >= diag.MIN_FOLDS_NONNEG


def test_decision_precedence_D0_beats_D1_and_D2():
    # bad CI must win even if a variant and a rescuer also qualify
    sweep = {"median": {"f1": 9.9, "delta_pp": 9.9,
                        "classes_regressed_gt_0_30pp": 0}}
    d = diag.decide(_ok_repro(), _loose_fold(),
                    {"ci_lo_gt_7way": False, "ci_lo": 0.30}, sweep,
                    {"any_stable_rescuer": True}, {}, np.array([0]),
                    np.array([0]))
    assert d["branch"] == "D0_guard"


# ── cheap integration: locked 36.94% reproduces from cached logits ────────
_SPLITS = "data/diema_challenge/processed/split_lpo_10fold.pkl"


@pytest.mark.skipif(not os.path.exists(_SPLITS),
                    reason="OOF artifacts not on this machine")
def test_locked_submission_reproduces():
    import pickle
    with open(_SPLITS, "rb") as f:
        splits = pickle.load(f)
    src, y, performer, fold_ids = diag.load_sources(splits)
    r = diag.reproducibility(src, y)
    assert r["reproducible"], r
    assert r["logit_eq_geometric"]
    assert abs(r["logit_mean_f1"] - diag.LOCKED_F1) < 1e-6
    assert len(np.unique(performer)) == 74
    assert set(np.unique(fold_ids)) == set(range(10))
