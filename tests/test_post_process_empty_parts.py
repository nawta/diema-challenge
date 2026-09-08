""" guard: ensure token-cap-eaten rationales are flagged as errors.

Covers the ``_post_process_output`` branch that rejects rationales whose 6
body-part strings are all empty — a silent failure mode observed 6/218 times
in the pilot at max_new_tokens=4096. Bumping to 6144 + this
detector together drop the failure to near-zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.generate_part_rationales_qwenvl_vllm import (  # noqa: E402
    _post_process_output,
)


def _full_rationale_json(empty_parts: bool = False) -> str:
    parts = {
        "head": "", "torso": "", "left_arm": "", "right_arm": "",
        "left_leg": "", "right_leg": "",
    } if empty_parts else {
        "head": "Head tilts forward.",
        "torso": "Torso leans right.",
        "left_arm": "Left arm rises.",
        "right_arm": "Right arm rises symmetrically.",
        "left_leg": "Left leg steps forward.",
        "right_leg": "Right leg stays planted.",
    }
    payload = {
        "global_motion": "asymmetric upper-body motion" if not empty_parts else "",
        **parts,
        "salient_intervals": [],
        "uncertainty_notes": "",
    }
    return json.dumps(payload)


def test_post_process_valid_rationale_parses_cleanly():
    raw = "<think>reasoning…</think>\n" + _full_rationale_json()
    rat, err = _post_process_output(raw)
    assert err is None
    assert rat["head"].startswith("Head tilts")
    assert rat["uncertainty_notes"] == ""


def test_post_process_all_parts_empty_is_flagged():
    # Simulates token-cap consumption: JSON shell emitted, all parts empty.
    raw = "<think>thinking…</think>\n" + _full_rationale_json(empty_parts=True)
    rat, err = _post_process_output(raw)
    assert err is not None, "all-empty rationale must be rejected"
    assert "all-parts-empty" in err
    assert rat == {}


def test_post_process_whitespace_only_part_counts_as_empty():
    payload = {
        "global_motion": "some motion",
        "head": "   ", "torso": "", "left_arm": "\n",
        "right_arm": "\t", "left_leg": "", "right_leg": "",
        "salient_intervals": [],
        "uncertainty_notes": "",
    }
    rat, err = _post_process_output(json.dumps(payload))
    assert err is not None
    assert "all-parts-empty" in err


def test_post_process_partial_coverage_is_kept():
    # One valid part is enough to keep the rationale.
    payload = {
        "global_motion": "x",
        "head": "Head nods.", "torso": "",
        "left_arm": "", "right_arm": "",
        "left_leg": "", "right_leg": "",
        "salient_intervals": [],
        "uncertainty_notes": "",
    }
    rat, err = _post_process_output(json.dumps(payload))
    assert err is None
    assert rat["head"] == "Head nods."


def test_post_process_missing_keys_auto_filled_then_emptiness_checked():
    # Malformed: missing 5 part keys. Auto-fill leaves them empty,
    # then the empty-detector trips because now all 6 parts are empty.
    payload = {
        "global_motion": "x",
        "head": "",
        "salient_intervals": [],
    }
    rat, err = _post_process_output(json.dumps(payload))
    assert err is not None
    assert "all-parts-empty" in err


def test_post_process_bad_json_returns_parse_error():
    raw = "not json at all"
    rat, err = _post_process_output(raw)
    assert err is not None
    assert "all-parts-empty" not in err  # different error class
    assert rat == {}
