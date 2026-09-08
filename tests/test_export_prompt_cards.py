""" sanity tests for tools/export_prompt_cards.py.

Covers: semver parsing, rationale stat aggregation (version filter + tokens /
elapsed + generation window), render-meta ingestion, and end-to-end
card rendering against a synthetic fixture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.export_prompt_cards import (  # noqa: E402
    gather_rationale_stats,
    load_one_render_meta,
    load_prompt_yaml,
    parse_semver,
    render_card_md,
)


def test_parse_semver_accepts_two_and_three_segments():
    assert parse_semver("1.0") == (1, 0, 0)
    assert parse_semver("2.3.4") == (2, 3, 4)


@pytest.mark.parametrize("bad", ["", "v1.0", "1", "1.a", "1.0.0.0", "latest"])
def test_parse_semver_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_semver(bad)


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _minimal_prompt_yaml(version: str = "1.0") -> str:
    return (
        f"schema_version: \"{version}\"\n"
        "model: \"Qwen/Qwen3-VL-30B-A3B-Thinking\"\n"
        "system: |\n  sys text\n"
        "user: |\n  user text\n"
    )


def test_load_prompt_yaml_requires_all_keys(tmp_path: Path):
    p = tmp_path / "missing.yaml"
    _write_yaml(p, "schema_version: \"1.0\"\nmodel: m\nsystem: s\n")  # no user
    with pytest.raises(Exception) as excinfo:
        load_prompt_yaml(p)
    assert "user" in str(excinfo.value)


def test_load_prompt_yaml_rejects_bad_semver(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    _write_yaml(
        p,
        "schema_version: latest\nmodel: m\nsystem: s\nuser: u\n",
    )
    with pytest.raises(ValueError):
        load_prompt_yaml(p)


def test_gather_rationale_stats_version_filter(tmp_path: Path):
    for i, (ver, tokens, elapsed, gen) in enumerate([
        ("1.0", 1000, 5.0, "2026-04-24T01:00:00+00:00"),
        ("1.0", 2000, 7.0, "2026-04-24T02:00:00+00:00"),
        ("0.9", 9999, 99.0, "2020-01-01T00:00:00+00:00"),  # wrong version; ignored
    ]):
        (tmp_path / f"{i}.json").write_text(
            json.dumps({
                "rationale_cache_key": f"k{i}",
                "prompt_version": ver,
                "model_version": "Qwen/Qwen3-VL-30B-A3B-Thinking",
                "generated_at": gen,
                "output_tokens": tokens,
                "elapsed_sec_chunk_avg": elapsed,
                "rationale": {},
            }),
            encoding="utf-8",
        )
    stats = gather_rationale_stats(tmp_path, prompt_version="1.0")
    assert stats["n_total"] == 3
    assert stats["n_match"] == 2
    assert stats["output_tokens"]["n"] == 2
    assert stats["output_tokens"]["mean"] == pytest.approx(1500.0)
    assert stats["elapsed_sec_chunk_avg"]["n"] == 2
    assert stats["models_seen"] == ["Qwen/Qwen3-VL-30B-A3B-Thinking"]
    assert stats["generation_window"]["first"].startswith("2026-04-24T01:00:00")
    assert stats["generation_window"]["last"].startswith("2026-04-24T02:00:00")


def test_gather_rationale_stats_empty_dir(tmp_path: Path):
    stats = gather_rationale_stats(tmp_path, prompt_version="1.0")
    assert stats["n_total"] == 0
    assert stats["n_match"] == 0
    assert stats["output_tokens"] == {}
    assert stats["generation_window"] == {}


def test_load_one_render_meta_first(tmp_path: Path):
    meta = {
        "cache_key": "abc",
        "render_settings": {
            "schema_version": "1.0", "fps": 10, "view": "front_side_2view",
            "slowmo": 1.0, "bg": "white", "scale": "hip_width",
            "line_width": 3, "max_frames": 64,
        },
        "frames_rendered": 64,
    }
    (tmp_path / "aaa.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    got = load_one_render_meta(tmp_path)
    assert got is not None
    assert got["fps"] == 10
    assert got["view"] == "front_side_2view"


def test_load_one_render_meta_missing(tmp_path: Path):
    assert load_one_render_meta(tmp_path) is None


def test_render_card_md_end_to_end(tmp_path: Path):
    prompt_yaml = tmp_path / "motion_rationale_qwenvl_v1.0.yaml"
    _write_yaml(prompt_yaml, _minimal_prompt_yaml("1.0"))
    card = load_prompt_yaml(prompt_yaml)
    md = render_card_md(
        phase="3AD-4",
        prompt_yaml_path=prompt_yaml,
        prompt_card=card,
        render_settings={
            "schema_version": "1.0", "fps": 10, "view": "front_side_2view",
            "bg": "white", "scale": "hip_width", "line_width": 3,
            "max_frames": 64, "slowmo": 1.0,
        },
        model_commit="d0ed0380729be07a546fdefafbb4fe411f341e92",
        stats={
            "n_total": 10, "n_match": 10,
            "output_tokens": {"n": 10, "mean": 1500.0, "median": 1500.0, "p05": 1000.0, "p95": 2000.0, "min": 1000.0, "max": 2000.0},
            "elapsed_sec_chunk_avg": {"n": 10, "mean": 10.0, "median": 10.0, "p05": 5.0, "p95": 15.0, "min": 5.0, "max": 15.0},
            "models_seen": ["Qwen/Qwen3-VL-30B-A3B-Thinking"],
            "generation_window": {"first": "2026-04-24T01:00:00+00:00", "last": "2026-04-24T04:00:00+00:00"},
        },
        now_iso="2026-04-24T12:00:00+00:00",
    )
    # Anchor checks
    assert "# Prompt card — 3AD-4 v1.0" in md
    assert "d0ed0380729be07a546fdefafbb4fe411f341e92" in md
    assert "| patch | typo / wording only | cache stays valid |" in md
    assert "front_side_2view" in md
    assert "| `max_frames` | `64` |" in md
    assert "salient_intervals" in md
    assert "check_rationale_leakage.py" in md
    # No blocking-word leakage of a specific emotion in the template itself
    for forbidden in ["angry", "happy", "sad"]:
        # "angry" etc shouldn't appear unless as examples in the system prompt.
        # The minimal fixture doesn't include them, so assert absence here.
        assert forbidden not in md.lower()
