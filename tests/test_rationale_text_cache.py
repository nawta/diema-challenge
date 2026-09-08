""" tests: rationale text cache builder + runtime loader.

Covers ``tools.build_rationale_text_cache`` (placeholder detection, render
index parsing, build_cache end-to-end with a fake encoder) and
``diema.features.rationale_text_cache.RationaleTextCache`` (batch lookup,
normalization, coverage stats).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diema.features.rationale_text_cache import RationaleTextCache  # noqa: E402
from tools.build_rationale_text_cache import (  # noqa: E402
    PART_NAMES,
    build_cache,
    collect_rationales,
    is_placeholder,
    load_render_index,
)


@pytest.mark.parametrize("text,expected", [
    ("no distinctive motion", True),
    ("No distinctive motion.", True),
    ("NONE", True),
    ("n/a", True),
    ("na", True),
    ("", True),
    ("   ", True),
    ("Head moves up and down.", False),
    ("no", False),  # not the exact placeholder phrasing
])
def test_is_placeholder(text: str, expected: bool):
    assert is_placeholder(text) is expected


def test_load_render_index_parses_and_skips_junk(tmp_path: Path):
    p = tmp_path / "render_index.jsonl"
    p.write_text(
        json.dumps({"cache_key": "abc", "stem": "JP_06_anger_1_H"}) + "\n"
        + "\n"  # blank line
        + "not json\n"
        + json.dumps({"cache_key": "def", "stem": "TW_05_joy_2_M"}) + "\n",
        encoding="utf-8",
    )
    m = load_render_index(p)
    assert m == {"abc": "JP_06_anger_1_H", "def": "TW_05_joy_2_M"}


def test_load_render_index_missing_returns_empty(tmp_path: Path):
    assert load_render_index(tmp_path / "nope.jsonl") == {}


def test_collect_rationales_version_filter(tmp_path: Path):
    (tmp_path / "a.json").write_text(
        json.dumps({"prompt_version": "1.0", "rationale": {"head": "x"}, "render_cache_key": "ck0"}),
        encoding="utf-8",
    )
    (tmp_path / "b.json").write_text(
        json.dumps({"prompt_version": "0.9", "rationale": {"head": "y"}, "render_cache_key": "ck1"}),
        encoding="utf-8",
    )
    (tmp_path / "c.json").write_text("not json", encoding="utf-8")
    rows = collect_rationales(tmp_path, prompt_version="1.0")
    assert len(rows) == 1
    rows = collect_rationales(tmp_path, prompt_version=None)
    assert len(rows) == 2


class _FakeEncoder:
    """Deterministic 4-d encoder: vector depends only on the input text."""
    def encode(self, texts, normalize=True):
        vecs = torch.zeros(len(texts), 4, dtype=torch.float32)
        for i, t in enumerate(texts):
            # Stable but text-sensitive content.
            h = sum(ord(c) for c in str(t))
            vecs[i] = torch.tensor([h, h + 1, h + 2, h + 3], dtype=torch.float32)
        return vecs


def _make_rationale(
    render_key: str, stem_text: str, all_placeholder: bool = False,
) -> dict:
    if all_placeholder:
        parts = {p: "no distinctive motion" for p in PART_NAMES}
    else:
        parts = {p: f"{p} moves {stem_text}" for p in PART_NAMES}
    # Sprinkle one placeholder in for masking coverage.
    parts["left_leg"] = "no distinctive motion"
    return {
        "prompt_version": "1.0",
        "render_cache_key": render_key,
        "rationale": parts,
    }


def test_build_cache_assigns_rows_and_masks_placeholders():
    rationales = [
        _make_rationale("ck0", "fast"),
        _make_rationale("ck1", "slow"),
    ]
    stem_lookup = {"ck0": "JP_06_anger_1_H", "ck1": "TW_05_joy_2_M"}
    payload = build_cache(rationales, stem_lookup, _FakeEncoder())
    assert payload["embeddings"].shape == (2, 6, 4)
    assert payload["part_mask"].shape == (2, 6)
    # left_leg (index 4) is placeholder for both rows → mask False.
    ll = PART_NAMES.index("left_leg")
    assert payload["part_mask"][:, ll].tolist() == [False, False]
    # All other parts are valid.
    for i, p in enumerate(PART_NAMES):
        if p == "left_leg":
            continue
        assert payload["part_mask"][:, i].all().item(), p
    assert payload["stem_to_idx"] == {
        "JP_06_anger_1_H": 0,
        "TW_05_joy_2_M": 1,
    }
    assert payload["n_dropped"] == 0


def test_build_cache_skips_unresolvable(tmp_path: Path):
    rationales = [
        _make_rationale("ck0", "a"),
        _make_rationale("ck_missing", "b"),  # not in render_index
    ]
    stem_lookup = {"ck0": "JP_stem"}
    payload = build_cache(rationales, stem_lookup, _FakeEncoder())
    assert payload["embeddings"].shape == (1, 6, 4)
    assert payload["n_dropped"] == 1


def test_build_cache_raises_when_empty():
    import click
    with pytest.raises(click.ClickException):
        build_cache([], {}, _FakeEncoder())


def _write_payload(tmp_path: Path, n: int = 2) -> Path:
    stems = [f"S{i:02d}" for i in range(n)]
    embeddings = torch.arange(n * 6 * 4, dtype=torch.float32).reshape(n, 6, 4)
    part_mask = torch.ones(n, 6, dtype=torch.bool)
    # Knock out left_leg for row 1 to exercise mask plumbing.
    if n > 1:
        part_mask[1, PART_NAMES.index("left_leg")] = False
    path = tmp_path / "cache.pt"
    torch.save({
        "embeddings": embeddings,
        "part_mask": part_mask,
        "stem_to_idx": {stem: i for i, stem in enumerate(stems)},
        "part_names": list(PART_NAMES),
        "meta": {"schema_version": "1.0"},
    }, path)
    return path


def test_rationale_text_cache_loads_and_reports_shape(tmp_path: Path):
    p = _write_payload(tmp_path, n=3)
    cache = RationaleTextCache(p)
    assert cache.num_rows == 3
    assert cache.num_parts == 6
    assert cache.embedding_dim == 4
    assert cache.part_names == list(PART_NAMES)


def test_rationale_text_cache_batch_lookup(tmp_path: Path):
    p = _write_payload(tmp_path, n=2)
    cache = RationaleTextCache(p)
    z, m, valid = cache.embedding_for_batch(
        ["S00", "S01", "UNKNOWN"]
    )
    assert z.shape == (3, 6, 4)
    assert m.shape == (3, 6)
    assert valid.tolist() == [True, True, False]
    # Unknown row is zero-filled; its mask is all-False.
    assert (z[2] == 0).all()
    assert (m[2] == False).all()
    # Row 1's left_leg was knocked out in the fixture.
    ll = PART_NAMES.index("left_leg")
    assert m[1, ll].item() is False
    # Other parts for row 1 are valid.
    for i in range(6):
        if i == ll:
            continue
        assert m[1, i].item() is True


def test_rationale_text_cache_normalize(tmp_path: Path):
    p = _write_payload(tmp_path, n=1)
    cache = RationaleTextCache(p)
    z, _, _ = cache.embedding_for_batch(["S00"], normalize=True)
    norms = z.norm(dim=-1)
    # Non-zero rows should be unit-norm after normalize.
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_rationale_text_cache_rejects_bad_shape(tmp_path: Path):
    bad = tmp_path / "bad.pt"
    torch.save({
        "embeddings": torch.zeros(2, 4),  # 2-D, should be 3-D
        "part_mask": torch.ones(2, dtype=torch.bool),
        "stem_to_idx": {"a": 0, "b": 1},
        "part_names": list(PART_NAMES),
    }, bad)
    with pytest.raises(ValueError, match="3-D"):
        RationaleTextCache(bad)


def test_coverage_stats(tmp_path: Path):
    p = _write_payload(tmp_path, n=2)
    cache = RationaleTextCache(p)
    stats = cache.coverage_stats()
    # 12 cells, 11 True (one knocked out)
    assert stats["cells"] == 12
    assert stats["true_fraction"] == pytest.approx(11 / 12)
    assert len(stats["per_part"]) == 6
