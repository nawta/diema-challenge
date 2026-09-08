""" tests: deterministic 200-clip audit sampler.

Covers stem parsing, baseline × overlay composition, exact 200-count
invariant, layer-label distribution, determinism under the same seed,
and the overlay-composition / overlay-count tables' internal consistency.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.sample_quality_audit_clips import (  # noqa: E402
    _OVERLAY,
    _OVERLAY_COMPOSITION,
    BASELINE_PER_CELL,
    parse_stem,
    pick_baseline_and_overlay,
)

EMOTIONS = [
    "anger", "contempt", "disgust", "fear", "gratitude", "guilt",
    "jealousy", "joy", "pride", "sadness", "shame", "surprise",
]
NATIONALITIES = ["JP", "TW"]
INTENSITIES = ["L", "M", "H"]


def _synthetic_records(performers_per_cell: int = 10) -> list[dict]:
    """Synthesize enough clips per cell to cover baseline + overlay."""
    recs = []
    for emo, nat, inten in product(EMOTIONS, NATIONALITIES, INTENSITIES):
        for pid in range(performers_per_cell):
            for take in range(3):
                stem = f"{nat}_{pid:02d}_{emo}_{take}_{inten}"
                recs.append({
                    "stem": stem,
                    "nationality": nat,
                    "performer": f"{nat}_{pid:02d}",
                    "emotion": emo,
                    "intensity": inten,
                    "token_cap_hit": 1 if (pid % 4 == 0 and inten == "H") else 0,
                    "leg_motion_proxy": float(pid),
                    "duration_sec": 6.4 + (pid * 0.01),
                })
    return recs


# -- stem parser ---------------------------------------------------------

def test_parse_stem_basic():
    got = parse_stem("JP_06_anger_1_H")
    assert got == {
        "stem": "JP_06_anger_1_H",
        "nationality": "JP",
        "performer": "JP_06",
        "emotion": "anger",
        "intensity": "H",
    }


def test_parse_stem_malformed_returns_none():
    assert parse_stem("JP_06") is None
    assert parse_stem("") is None


# -- composition table sanity ---------------------------------------------

def test_overlay_table_sums_to_56():
    assert sum(_OVERLAY.values()) == 56


def test_overlay_composition_matches_overlay_totals():
    for key, total in _OVERLAY.items():
        n_h, n_legs = _OVERLAY_COMPOSITION[key]
        assert n_h + n_legs == total, key


def test_overlay_hr_h_and_legs_global_totals():
    hr_h = sum(v[0] for v in _OVERLAY_COMPOSITION.values())
    hr_legs = sum(v[1] for v in _OVERLAY_COMPOSITION.values())
    assert hr_h == 24
    assert hr_legs == 32


def test_overlay_covers_all_emotion_nationality_pairs():
    expected = {(e, n) for e in EMOTIONS for n in NATIONALITIES}
    assert set(_OVERLAY) == expected
    assert set(_OVERLAY_COMPOSITION) == expected


# -- end-to-end picker ----------------------------------------------------

def test_pick_produces_exactly_200():
    recs = _synthetic_records()
    out = pick_baseline_and_overlay(recs, seed=42)
    assert len(out) == 200


def test_pick_layer_counts():
    recs = _synthetic_records()
    out = pick_baseline_and_overlay(recs, seed=42)
    counts = Counter(p["layer"] for p in out)
    assert counts["baseline"] == 144
    assert counts["high_risk_intensity_extreme"] == 24
    assert counts["high_risk_legs_placeholder_high"] == 32


def test_pick_is_deterministic_under_same_seed():
    recs = _synthetic_records()
    a = pick_baseline_and_overlay(recs, seed=1)
    b = pick_baseline_and_overlay(recs, seed=1)
    assert [p["stem"] for p in a] == [p["stem"] for p in b]


def test_pick_differs_across_seeds():
    recs = _synthetic_records()
    a = pick_baseline_and_overlay(recs, seed=1)
    b = pick_baseline_and_overlay(recs, seed=2)
    assert [p["stem"] for p in a] != [p["stem"] for p in b]


def test_pick_baseline_is_2_per_cell():
    recs = _synthetic_records()
    out = pick_baseline_and_overlay(recs, seed=42)
    baseline_cells = Counter(
        (p["emotion"], p["nationality"], p["intensity"])
        for p in out if p["layer"] == "baseline"
    )
    # 72 cells × 2 clips.
    assert len(baseline_cells) == 72
    for c, n in baseline_cells.items():
        assert n == BASELINE_PER_CELL, c


def test_pick_overlay_is_on_intensity_H_only():
    recs = _synthetic_records()
    out = pick_baseline_and_overlay(recs, seed=42)
    for p in out:
        if p["layer"].startswith("high_risk_"):
            assert p["intensity"] == "H", p


def test_pick_marginals_match_spec():
    recs = _synthetic_records()
    out = pick_baseline_and_overlay(recs, seed=42)
    # Expected: JP=102, TW=98, L=48, M=48, H=104 from the protocol card.
    by_nat = Counter(p["nationality"] for p in out)
    by_int = Counter(p["intensity"] for p in out)
    assert by_nat["JP"] == 102
    assert by_nat["TW"] == 98
    assert by_int["L"] == 48
    assert by_int["M"] == 48
    assert by_int["H"] == 104


def test_pick_honors_token_cap_proxy_in_overlay_ranking():
    # Build a small cell where only 2 records have token_cap_hit=1;
    # those should land in the HR picks ahead of the zero-proxy rest.
    emo, nat, inten = "anger", "JP", "H"
    recs = []
    for i in range(10):
        stem = f"{nat}_{i:02d}_{emo}_0_{inten}"
        recs.append({
            "stem": stem,
            "nationality": nat,
            "performer": f"{nat}_{i:02d}",
            "emotion": emo,
            "intensity": inten,
            "token_cap_hit": 1 if i < 2 else 0,
            "leg_motion_proxy": 0.0,
            "duration_sec": 6.4,
        })
    # Fill the other cells with non-H so the baseline pass doesn't chew the
    # token_cap_hit candidates before overlay sees them. We still need the
    # baseline pass to have ≥ 2 clips per cell, so pad all 72 strata.
    for e, n, t in product(EMOTIONS, NATIONALITIES, INTENSITIES):
        if (e, n, t) == (emo, nat, inten):
            continue
        for i in range(5):
            recs.append({
                "stem": f"{n}_{i:02d}_{e}_0_{t}",
                "nationality": n,
                "performer": f"{n}_{i:02d}",
                "emotion": e,
                "intensity": t,
                "token_cap_hit": 0,
                "leg_motion_proxy": 0.0,
                "duration_sec": 6.4,
            })
    out = pick_baseline_and_overlay(recs, seed=0)
    # The anger/JP/H overlay should contain 3 total clips — at least 1 of them
    # should have a token_cap_hit=1 stem (i=0 or 1) since those rank highest.
    anger_jp_h_overlay = [
        p["stem"] for p in out
        if p["emotion"] == "anger" and p["nationality"] == "JP"
        and p["intensity"] == "H" and p["layer"].startswith("high_risk_")
    ]
    # Stem prefix check: i=0 → JP_00, i=1 → JP_01.
    assert any(s.startswith("JP_00") or s.startswith("JP_01")
               for s in anger_jp_h_overlay), anger_jp_h_overlay
