""" tests: strip_text_branch removes only train-time weights.

Mandated assertions (TODO 3AE-7):
- model.eval(), dropout off, text input=None path
- ``torch.allclose(logits_before, logits_after, atol=1e-6, rtol=1e-5)``
- ``set(after.keys()) ⊆ set(before.keys())``
- ``max_abs_diff < 1e-5`` is recorded (here: asserted + printed)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diema.models.base import BaseModel  # noqa: E402
from diema.models.conv1d_transformer.partalign_conv1d_tr_model import (  # noqa: E402
    PartAlignConv1DTr_Model,
)
from tools.strip_text_branch import (  # noqa: E402
    classify_key,
    strip_state_dict,
    write_report,
)


class _FakeBackbone(BaseModel):
    def __init__(self, num_class: int = 12, feat_dim: int = 64, num_nodes: int = 25):
        super().__init__()
        self._num_class = num_class
        self._feat_dim = feat_dim
        self._num_nodes = num_nodes
        self.in_proj = nn.Linear(num_nodes * 6 * 64, feat_dim)
        self.cls_head = nn.Linear(feat_dim, num_class)

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


# -- unit tests on the key classifier ----------------------------------------

@pytest.mark.parametrize("key,expected", [
    ("model.text_head.0.weight", True),
    ("model.text_head.1.bias", True),
    ("model.part_head.2.running_mean", True),
    ("text_head.0.weight", True),
    ("part_head.0.weight", True),
    ("scenario_embedding_matrix", True),
    ("part_embedding_tensor", True),
    ("part_mask_tensor", True),
    ("model.backbone.in_proj.weight", False),
    ("model.backbone.cls_head.bias", False),
    ("model.some_layer.weight", False),
    ("other_buffer", False),
])
def test_classify_key(key: str, expected: bool):
    assert classify_key(key) is expected


def test_strip_state_dict_partitions_correctly():
    sd = {
        "model.backbone.conv.weight": torch.randn(8, 8),
        "model.part_head.0.weight": torch.randn(16, 16),
        "model.part_head.0.bias": torch.randn(16),
        "part_embedding_tensor": torch.randn(3, 6, 4),
        "part_mask_tensor": torch.ones(3, 6, dtype=torch.bool),
    }
    kept, removed = strip_state_dict(sd)
    kept_keys = set(kept.keys())
    removed_keys = {k for k, _ in removed}
    assert kept_keys == {"model.backbone.conv.weight"}
    assert removed_keys == {
        "model.part_head.0.weight",
        "model.part_head.0.bias",
        "part_embedding_tensor",
        "part_mask_tensor",
    }
    # All removed entries carry their shape for the report.
    assert all(isinstance(s, tuple) for _, s in removed)


# -- forward-equivalence invariants (TODO 3AE-7 mandated) --------------------

def _build_part_align_model() -> PartAlignConv1DTr_Model:
    backbone = _FakeBackbone(num_class=12, feat_dim=64, num_nodes=25)
    return PartAlignConv1DTr_Model(backbone, text_dim=32, num_parts=6)


def test_forward_equivalence_before_after_strip(tmp_path: Path):
    torch.manual_seed(0)
    model_full = _build_part_align_model()
    model_full.eval()

    # Baseline forward (dropout off since we called eval()).
    x = torch.randn(3, 6, 64, 25)
    with torch.no_grad():
        logits_before = model_full(x)["logits"].clone()

    # Save + strip.
    ckpt_path = tmp_path / "full.ckpt"
    stripped_path = tmp_path / "stripped.ckpt"
    # Mimic the Lightning wrapper layout by nesting under "state_dict".
    torch.save({
        "state_dict": {f"model.{k}": v for k, v in model_full.state_dict().items()},
        "hyper_parameters": {
            "scenario_embedding_matrix": torch.zeros(1, 1),  # to scrub
        },
    }, ckpt_path)

    # Invoke the tool via its public API (not CLI).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    kept, removed = strip_state_dict(ckpt["state_dict"])
    ckpt["state_dict"] = kept
    if isinstance(ckpt.get("hyper_parameters"), dict):
        ckpt["hyper_parameters"].pop("scenario_embedding_matrix", None)
    torch.save(ckpt, stripped_path)

    # Reload into a fresh backbone-only model for inference parity.
    inference_model = _FakeBackbone(num_class=12, feat_dim=64, num_nodes=25)
    loaded = torch.load(stripped_path, map_location="cpu", weights_only=False)
    state = {k.replace("model.backbone.", ""): v
             for k, v in loaded["state_dict"].items()
             if k.startswith("model.backbone.")}
    inference_model.load_state_dict(state, strict=True)
    inference_model.eval()
    with torch.no_grad():
        logits_after = inference_model(x)["logits"].clone()

    # TODO-mandated tolerance: atol=1e-6, rtol=1e-5.
    assert torch.allclose(logits_before, logits_after, atol=1e-6, rtol=1e-5), (
        "Logits changed after strip; inference path is not bit-close"
    )
    max_abs_diff = (logits_before - logits_after).abs().max().item()
    assert max_abs_diff < 1e-5, f"max_abs_diff={max_abs_diff} exceeds 1e-5 gate"

    # Subset property: stripped keys are a subset of originals.
    before = set(model_full.state_dict().keys())
    # Rebuild inference-model key names to match original backbone namespace.
    after = {k.replace("model.backbone.", "backbone.") for k in loaded["state_dict"]}
    # Translate: inference state names the layers as e.g. "in_proj.weight";
    # the original model named them "backbone.in_proj.weight". Align.
    after_aligned = {f"backbone.{k}" for k in state.keys()}
    assert after_aligned.issubset({f"backbone.{k.split('backbone.', 1)[1]}"
                                   for k in before if k.startswith("backbone.")})

    # And the dropped head keys are absent.
    removed_names = {k for k, _ in removed}
    for dropped in removed_names:
        assert dropped not in loaded["state_dict"]


def test_write_report_renders_markdown(tmp_path: Path):
    report_path = tmp_path / "report.md"
    write_report(
        report_path,
        input_path=Path("/tmp/in.ckpt"),
        output_path=Path("/tmp/out.ckpt"),
        removed=[("model.part_head.0.weight", (32, 64)),
                 ("part_embedding_tensor", (8, 6, 4))],
        n_kept=42,
    )
    text = report_path.read_text(encoding="utf-8")
    assert "Strip-text-branch report" in text
    assert "kept state_dict keys: **42**" in text
    assert "removed state_dict keys: **2**" in text
    assert "`model.part_head.0.weight`" in text
    assert "`part_embedding_tensor`" in text


def test_write_report_handles_empty(tmp_path: Path):
    report_path = tmp_path / "empty.md"
    write_report(
        report_path,
        input_path=Path("/tmp/in.ckpt"),
        output_path=Path("/tmp/out.ckpt"),
        removed=[], n_kept=100,
    )
    text = report_path.read_text(encoding="utf-8")
    assert "inference-clean" in text
