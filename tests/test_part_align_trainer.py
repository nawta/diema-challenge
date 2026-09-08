""" tests: PartAlignLightningModel batch-idx lookup + loss plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diema.training.part_align_trainer import PartAlignLightningModel  # noqa: E402


class _TinyBackboneWithZ(nn.Module):
    """Model whose forward returns {logits, features, z_parts}."""

    def __init__(self, num_class: int = 12, parts: int = 6, text_dim: int = 4):
        super().__init__()
        self.num_class = num_class
        self._parts = parts
        self._text_dim = text_dim
        self.fc = nn.Linear(16, num_class)
        self.proj = nn.Linear(16, parts * text_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        N = x.shape[0]
        feat = x.reshape(N, -1)[:, :16]
        logits = self.fc(feat)
        z = self.proj(feat).view(N, self._parts, self._text_dim)
        z = torch.nn.functional.normalize(z, dim=-1)
        return {"logits": logits, "features": feat, "z_parts": z}


def _make_trainer(
    part_loss_weight: float = 0.1,
    part_loss_mode: str = "cosine",
    n_cache: int = 3,
    parts: int = 6,
    text_dim: int = 4,
) -> PartAlignLightningModel:
    emb = torch.randn(n_cache, parts, text_dim)
    mask = torch.ones(n_cache, parts, dtype=torch.bool)
    mask[1, 2] = False  # knock out one cell for masking coverage
    return PartAlignLightningModel(
        model=_TinyBackboneWithZ(parts=parts, text_dim=text_dim),
        base_lr=1e-3,
        num_class=12,
        filename_to_part_idx={"SA": 0, "SB": 1, "SC": 2},
        part_embedding_tensor=emb,
        part_mask_tensor=mask,
        part_loss_weight=part_loss_weight,
        part_loss_mode=part_loss_mode,
    )


def test_invalid_mode_raises():
    emb = torch.randn(2, 6, 4)
    mask = torch.ones(2, 6, dtype=torch.bool)
    with pytest.raises(ValueError, match="part_loss_mode"):
        PartAlignLightningModel(
            model=_TinyBackboneWithZ(),
            base_lr=1e-3,
            num_class=12,
            filename_to_part_idx={"A": 0, "B": 1},
            part_embedding_tensor=emb,
            part_mask_tensor=mask,
            part_loss_mode="bogus",
        )


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="3-D"):
        PartAlignLightningModel(
            model=_TinyBackboneWithZ(),
            base_lr=1e-3,
            num_class=12,
            filename_to_part_idx={"A": 0},
            part_embedding_tensor=torch.randn(2, 4),  # wrong rank
            part_mask_tensor=torch.ones(2, 4, dtype=torch.bool),
        )

    with pytest.raises(ValueError, match="does not match"):
        PartAlignLightningModel(
            model=_TinyBackboneWithZ(),
            base_lr=1e-3,
            num_class=12,
            filename_to_part_idx={"A": 0},
            part_embedding_tensor=torch.randn(2, 6, 4),
            part_mask_tensor=torch.ones(3, 6, dtype=torch.bool),  # wrong N
        )


def test_batch_part_idx_handles_unknown():
    lit = _make_trainer()
    # Two known, one unknown.
    idx = lit._batch_part_idx(["SA", "UNK", "SB"], device=torch.device("cpu"))
    assert idx.tolist() == [0, -1, 1]


def test_lookup_targets_zero_fills_unknowns_and_carries_mask():
    lit = _make_trainer()
    idx = torch.tensor([0, -1, 1], dtype=torch.long)
    t, m = lit._lookup_targets(idx)
    assert t.shape == (3, 6, 4)
    assert m.shape == (3, 6)
    # Row 0 (maps to cache row 0): mask all True
    assert m[0].all().item() is True
    # Row 1 (unknown): all zero targets, all-False mask
    assert (t[1] == 0).all().item() is True
    assert (m[1] == False).all().item() is True
    # Row 2 (maps to cache row 1): cell [2] was knocked out in fixture
    assert m[2, 2].item() is False
    for p in (0, 1, 3, 4, 5):
        assert m[2, p].item() is True


def test_lookup_targets_extension_stripped():
    lit = _make_trainer()
    # Only the stem is used for lookup; extensions are stripped.
    idx = lit._batch_part_idx(["SA.bvh", "SB.c3d"], device=torch.device("cpu"))
    assert idx.tolist() == [0, 1]


def test_buffers_registered_and_persistent():
    lit = _make_trainer()
    # Verifies torch registers both tensors as buffers on the lightning module.
    state = dict(lit.named_buffers())
    assert "part_embedding_tensor" in state
    assert "part_mask_tensor" in state
    # Tensor shapes match what we handed in.
    assert state["part_embedding_tensor"].shape == (3, 6, 4)
    assert state["part_mask_tensor"].shape == (3, 6)
    assert state["part_mask_tensor"].dtype == torch.bool
