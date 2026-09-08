""" tests: compute_part_importance + PartAlignConv1DTr_Model output.

Covers:
- z_parts_raw is exposed by the forward and remains unnormalized.
- compute_part_importance handles N=0 cells, mixed valid/placeholder,
  single-sample per emotion, and per-emotion cosine in the [-1, 1] range.
- The CLI markdown renderer includes every section.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from click.testing import CliRunner  # noqa: E402

from diema.evaluation.part_importance import compute_part_importance  # noqa: E402
from diema.models.base import BaseModel  # noqa: E402
from diema.models.conv1d_transformer.partalign_conv1d_tr_model import (  # noqa: E402
    PartAlignConv1DTr_Model,
)
from tools.export_part_importance import (  # noqa: E402
    DEFAULT_PART_NAMES,
    _render_markdown,
    main as cli_main,
)


class _FakeBackbone(BaseModel):
    def __init__(self, num_class: int = 12, feat_dim: int = 64, num_nodes: int = 25):
        super().__init__()
        self._num_class = num_class
        self._feat_dim = feat_dim
        self._num_nodes = num_nodes
        self.in_proj = torch.nn.Linear(num_nodes * 6 * 64, feat_dim)
        self.cls_head = torch.nn.Linear(feat_dim, num_class)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        N = x.shape[0]
        flat = x.reshape(N, -1)
        feat = torch.relu(self.in_proj(flat))
        return {"logits": self.cls_head(feat), "features": feat}

    @property
    def feature_dim(self) -> int:
        return self._feat_dim

    @property
    def output_dim(self) -> int:
        return self._num_class

    @property
    def num_nodes(self) -> int:
        return self._num_nodes


def test_partalign_forward_emits_both_z_parts_and_raw():
    torch.manual_seed(0)
    model = PartAlignConv1DTr_Model(_FakeBackbone(), text_dim=16, num_parts=6)
    model.eval()
    x = torch.randn(3, 6, 64, 25)
    with torch.no_grad():
        out = model(x)
    assert "z_parts_raw" in out
    assert "z_parts" in out
    assert out["z_parts_raw"].shape == (3, 6, 16)
    # Normalized output should be unit-norm; raw should NOT be.
    assert torch.allclose(out["z_parts"].norm(dim=-1),
                          torch.ones(3, 6), atol=1e-5)
    assert not torch.allclose(out["z_parts_raw"].norm(dim=-1),
                              torch.ones(3, 6), atol=1e-3)


def test_compute_part_importance_shape_and_fields():
    torch.manual_seed(42)
    N, P, D = 12, 6, 8
    z = torch.randn(N, P, D)
    mask = torch.ones(N, P, dtype=torch.bool)
    mask[3, 2] = False  # knock out one placeholder
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    parts = list(DEFAULT_PART_NAMES)
    stats = compute_part_importance(z, mask, labels, parts)
    assert stats["part_names"] == parts
    assert len(stats["norm_stats"]) == P
    # 4 emotions × 6 parts = 24 cells.
    assert len(stats["per_emotion_norm"]) == 4 * P
    assert len(stats["per_emotion_cosine"]) == 4 * P
    # Sample index 3 has label=1 (labels[3]=1), so the knockout affects
    # emotion=1's left_arm cell, not emotion=0's.
    cell = next(r for r in stats["per_emotion_norm"]
                if r["emotion"] == "1" and r["part"] == "left_arm")
    assert cell["n"] == 2  # 3 samples in emotion 1, minus 1 placeholder
    # Meanwhile emotion 0's left_arm is untouched.
    cell_e0 = next(r for r in stats["per_emotion_norm"]
                   if r["emotion"] == "0" and r["part"] == "left_arm")
    assert cell_e0["n"] == 3


def test_compute_part_importance_empty_emotion_does_not_crash():
    z = torch.randn(3, 6, 4)
    mask = torch.zeros(3, 6, dtype=torch.bool)  # all placeholders
    labels = torch.tensor([0, 0, 0])
    stats = compute_part_importance(z, mask, labels, list(DEFAULT_PART_NAMES))
    for row in stats["norm_stats"]:
        assert row["n"] == 0
        assert row["mean"] == 0.0
    for row in stats["per_emotion_norm"]:
        assert row["n"] == 0


def test_compute_part_importance_single_sample_per_emotion_std_is_zero():
    z = torch.randn(4, 6, 4)
    mask = torch.ones(4, 6, dtype=torch.bool)
    labels = torch.tensor([0, 1, 2, 3])
    stats = compute_part_importance(z, mask, labels, list(DEFAULT_PART_NAMES))
    for r in stats["per_emotion_norm"]:
        assert r["n"] == 1
        assert r["std"] == 0.0


def test_compute_part_importance_cosine_range():
    torch.manual_seed(123)
    z = torch.randn(24, 6, 8)
    mask = torch.ones(24, 6, dtype=torch.bool)
    labels = torch.arange(24) % 3
    stats = compute_part_importance(z, mask, labels, list(DEFAULT_PART_NAMES))
    for r in stats["per_emotion_cosine"]:
        assert -1.01 <= r["cosine_to_global"] <= 1.01


def test_compute_part_importance_uses_emotion_names():
    z = torch.randn(6, 6, 4)
    mask = torch.ones(6, 6, dtype=torch.bool)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    stats = compute_part_importance(
        z, mask, labels, list(DEFAULT_PART_NAMES),
        emotion_names=["anger", "joy"],
    )
    assert stats["emotion_names"] == ["anger", "joy"]
    seen = {r["emotion"] for r in stats["per_emotion_norm"]}
    assert seen == {"anger", "joy"}


def test_render_markdown_has_all_sections():
    # Minimal stats dict.
    stats = {
        "part_names": ["head", "torso"],
        "emotion_names": ["anger"],
        "emotion_ids": [0],
        "norm_stats": [
            {"part": "head", "n": 3, "mean": 0.5, "std": 0.1},
            {"part": "torso", "n": 3, "mean": 0.7, "std": 0.2},
        ],
        "per_emotion_norm": [
            {"emotion": "anger", "part": "head", "n": 3, "mean": 0.5, "std": 0.1},
            {"emotion": "anger", "part": "torso", "n": 3, "mean": 0.7, "std": 0.2},
        ],
        "per_emotion_cosine": [
            {"emotion": "anger", "part": "head", "cosine_to_global": 0.9},
            {"emotion": "anger", "part": "torso", "cosine_to_global": 0.8},
        ],
    }
    md = _render_markdown(stats, source=Path("dummy.pt"))
    assert "## 1. Per-part projection norm" in md
    assert "## 2. Per-(emotion, part) mean norm" in md
    assert "## 3. Per-(emotion, part) cosine" in md
    assert "| anger | 0.500 | 0.700 |" in md
    assert "| anger | 0.900 | 0.800 |" in md


def test_cli_end_to_end(tmp_path: Path):
    payload = {
        "z_parts_raw": torch.randn(8, 6, 4),
        "part_mask": torch.ones(8, 6, dtype=torch.bool),
        "labels": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        "part_names": list(DEFAULT_PART_NAMES),
        "emotion_names": ["anger", "joy", "fear", "sadness"],
    }
    input_path = tmp_path / "parts.pt"
    torch.save(payload, input_path)

    out_md = tmp_path / "report.md"
    out_csv = tmp_path / "report.csv"
    out_json = tmp_path / "report.json"

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        "--input", str(input_path),
        "--output-md", str(out_md),
        "--output-csv", str(out_csv),
        "--output-json", str(out_json),
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output

    md_text = out_md.read_text(encoding="utf-8")
    assert "Per-part projection norm" in md_text
    assert "anger" in md_text and "joy" in md_text

    csv_text = out_csv.read_text(encoding="utf-8")
    assert csv_text.startswith("emotion,part,")
    # 4 emotions × 6 parts + header = 25 lines.
    assert len(csv_text.strip().splitlines()) == 4 * 6 + 1

    j = json.loads(out_json.read_text(encoding="utf-8"))
    assert "norm_stats" in j
    assert len(j["norm_stats"]) == 6


def test_cli_rejects_missing_keys(tmp_path: Path):
    bad = tmp_path / "bad.pt"
    torch.save({"z_parts_raw": torch.randn(2, 6, 4)}, bad)
    runner = CliRunner()
    result = runner.invoke(cli_main, ["--input", str(bad)])
    assert result.exit_code != 0
    assert "missing required key" in result.output
