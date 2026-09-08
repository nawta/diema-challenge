""" tests: eval_motion_text_retrieval CLI + target-builder wrappers.

Covers the file-based two-stage workflow end-to-end with a synthetic
embeddings .pt and a synthetic rationale cache (scenario variant would
require a CSV + sentence-transformer invocation, which the pure-tensor
engine test already exercises).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from click.testing import CliRunner

from tools.eval_motion_text_retrieval import (  # noqa: E402
    _country_from_stem,
    _performer_group_from_stem,
    _load_embeddings,
    build_rationale_targets,
    main as cli_main,
)


def test_country_parser():
    assert _country_from_stem("JP_06_anger_1_H") == "JP"
    assert _country_from_stem("TW_05_joy_2_M") == "TW"
    assert _country_from_stem("noprefix") == "UNK"


def test_performer_group_parser():
    assert _performer_group_from_stem("JP_06_anger_1_H") == "JP_06"
    assert _performer_group_from_stem("TW_12_guilt_2_L") == "TW_12"
    assert _performer_group_from_stem("solo") == "solo"


def test_load_embeddings_requires_keys(tmp_path: Path):
    bad = tmp_path / "bad.pt"
    torch.save({"z_motion": torch.randn(2, 4)}, bad)  # no labels, no stems
    with pytest.raises(click.ClickException, match="missing required key"):
        _load_embeddings(bad)


def test_load_embeddings_full(tmp_path: Path):
    good = tmp_path / "ok.pt"
    torch.save({
        "z_motion": torch.randn(3, 8),
        "labels": torch.tensor([0, 1, 2], dtype=torch.long),
        "stems": ["JP_06_anger_1_H", "JP_06_joy_2_M", "TW_05_pride_3_L"],
    }, good)
    p = _load_embeddings(good)
    assert p["z_motion"].shape == (3, 8)


def test_build_rationale_targets_masks_placeholders(tmp_path: Path):
    from tools.build_rationale_text_cache import PART_NAMES
    # 2 cached clips, 6 parts, 4-d.
    emb = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[0, 3] = False  # placeholder cell
    cache = tmp_path / "rat.pt"
    torch.save({
        "embeddings": emb,
        "part_mask": mask,
        "stem_to_idx": {"JP_06_anger_1_H": 0, "TW_05_joy_2_M": 1},
        "part_names": list(PART_NAMES),
        "meta": {},
    }, cache)

    stems = ["JP_06_anger_1_H", "TW_05_joy_2_M", "UNKNOWN"]
    t, pos = build_rationale_targets(stems, cache)
    assert t.shape == (2, 4)  # (N_cached, D)
    # L2-normalized.
    norms = t.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # Positive lookup: unknown is -1.
    assert pos.tolist() == [0, 1, -1]


def test_cli_end_to_end_rationale(tmp_path: Path):
    """Integration-style: synthetic embeddings + rationale cache → runs the tool."""
    from tools.build_rationale_text_cache import PART_NAMES

    # Embeddings: 4 clips in the val fold, D=4.
    stems = ["JP_06_anger_1_H", "JP_06_joy_2_M", "TW_05_pride_3_L", "TW_05_sadness_1_H"]
    emb_path = tmp_path / "z_motion.pt"
    torch.save({
        "z_motion": torch.randn(4, 4),
        "labels": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "stems": stems,
    }, emb_path)

    # Rationale cache: exactly the 4 stems + a 5th extra.
    all_stems = stems + ["JP_99_orphan_1_H"]
    rat_emb = torch.randn(len(all_stems), 6, 4)
    rat_mask = torch.ones(len(all_stems), 6, dtype=torch.bool)
    rat_path = tmp_path / "rat.pt"
    torch.save({
        "embeddings": rat_emb,
        "part_mask": rat_mask,
        "stem_to_idx": {s: i for i, s in enumerate(all_stems)},
        "part_names": list(PART_NAMES),
        "meta": {},
    }, rat_path)

    out_md = tmp_path / "report.md"
    out_json = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(cli_main, [
        "--motion-embeddings", str(emb_path),
        "--target", "rationale",
        "--rationale-cache", str(rat_path),
        "--output-md", str(out_md),
        "--output-json", str(out_json),
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output

    assert out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["target"] == "rationale"
    assert payload["n_queries"] == 4
    assert "recall_at_1" in payload["metrics"]
    # Per-country breakdown should split JP (2) and TW (2).
    assert set(payload["per_country"]) == {"JP", "TW"}

    md_text = out_md.read_text(encoding="utf-8")
    assert "Retrieval metrics —" in md_text
    assert "## Per-country" in md_text
    assert "| JP " in md_text or "| JP|" in md_text


def test_cli_errors_on_mismatched_shapes(tmp_path: Path):
    emb_path = tmp_path / "bad.pt"
    torch.save({
        "z_motion": torch.randn(3, 4),       # 3 rows
        "labels": torch.tensor([0, 1], dtype=torch.long),  # 2 rows
        "stems": ["a", "b", "c"],             # 3 entries
    }, emb_path)
    runner = CliRunner()
    result = runner.invoke(cli_main, [
        "--motion-embeddings", str(emb_path),
        "--target", "rationale",
        "--rationale-cache", str(emb_path),  # any existing path, won't be reached
    ])
    assert result.exit_code != 0
    assert "mismatched shapes" in result.output
