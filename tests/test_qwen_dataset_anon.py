"""TDD for the HELD Qwen-rationale dataset anonymization (Phase D).

Privacy-critical: the release tree must contain ZERO raw performer ids,
zero content-hash keys, and the de-anon map must live OUTSIDE the release.
Skips if the staged build is absent.

Run: conda run -n acii2026 pytest tests/test_qwen_dataset_anon.py -q
"""

import json
import re
from pathlib import Path

import pytest

REL = Path("data/dataset_release/qwen_rationale_motion_v1")
PRIV = Path("data/private_deanon_map")
RAW_ID = re.compile(r"\b(JP|TW)_\d\d")

pytestmark = pytest.mark.skipif(
    not REL.exists(), reason="run tools/build_qwen_rationale_dataset.py")


def test_no_raw_performer_ids_anywhere_in_release():
    hits = []
    for p in REL.rglob("*"):
        if p.is_file() and p.suffix in {".json", ".jsonl", ".md", ".txt"}:
            if RAW_ID.search(p.read_text(errors="ignore")):
                hits.append(str(p))
    assert hits == [], f"raw JP_/TW_ ids leaked into release: {hits[:5]}"


def test_no_content_hash_keys_in_released_rationales():
    for p in list((REL / "rationales").glob("*.json"))[:300]:
        d = json.loads(p.read_text())
        assert "rationale_cache_key" not in d
        assert "render_cache_key" not in d
        assert d["id"].startswith("rat_")
        assert re.fullmatch(r"P\d{3}_[a-z]+_\d+_[A-Z]", d["clip"])


def test_manifest_consistent_and_pcodes():
    lines = [json.loads(x) for x in
             (REL / "dataset_manifest.jsonl").read_text().splitlines() if x]
    n_rat = len(list((REL / "rationales").glob("*.json")))
    assert len(lines) == n_rat and n_rat > 0
    for r in lines:
        assert re.fullmatch(r"P\d{3}", r["performer"])
        assert r["has_rationale"] is True


def test_deanon_map_is_outside_release_tree():
    # the de-anon key must NOT be anywhere under the (uploadable) release
    assert not any("deanon" in p.name.lower() or "PRIVATE" in str(p)
                   for p in REL.rglob("*"))
    assert (PRIV / "qwen_rationale_motion_v1_deanon.json").exists()
    assert PRIV not in REL.parents and REL not in PRIV.parents


def test_datasheet_states_held_and_residual_risk():
    ds = (REL / "DATASHEET.md").read_text()
    assert "HELD" in ds and "re-identifi" in ds.lower()
    assert "License" in ds
