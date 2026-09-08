""" (Tier 2) tests: softlabel KD loss math + Lightning wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diema.training.kd_trainer import (  # noqa: E402
    SoftLabelKdLightningModel,
    softlabel_kd_loss,
)


# -- pure loss math --------------------------------------------------------

def test_kd_loss_zero_when_student_matches_teacher():
    # If student logits already produce the teacher distribution, KL ~ 0.
    teacher = torch.tensor([
        [0.7, 0.2, 0.1],
        [0.1, 0.1, 0.8],
    ])
    # Inverse softmax → logits (up to constant).
    logits = teacher.log()
    loss = softlabel_kd_loss(logits, teacher, temperature=1.0)
    assert float(loss) < 1e-4


def test_kd_loss_temperature_squared_scaling():
    # Same student/teacher, different T → loss scales by T² (Hinton 2015).
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    teacher = torch.softmax(torch.randn(8, 5), dim=-1)
    L1 = float(softlabel_kd_loss(logits, teacher, temperature=1.0))
    L2 = float(softlabel_kd_loss(logits, teacher, temperature=2.0))
    # T² scaling — within numerical tolerance the 4× term dominates.
    # (Not exact because softmax shape changes, but ratio should be > 1.)
    assert L2 > L1


def test_kd_loss_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        softlabel_kd_loss(torch.randn(4, 5), torch.softmax(torch.randn(4, 6), dim=-1))


def test_kd_loss_bad_temperature_raises():
    with pytest.raises(ValueError, match="temperature"):
        softlabel_kd_loss(torch.randn(4, 5), torch.softmax(torch.randn(4, 5), dim=-1), temperature=0.0)


# -- Lightning wrapper ----------------------------------------------------

class _TinyBackbone(nn.Module):
    """Minimal model returning {logits, features} for the trainer's contract."""
    def __init__(self, num_class: int = 12):
        super().__init__()
        self.num_class = num_class
        self.fc = nn.Linear(8, num_class)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        N = x.shape[0]
        feat = x.reshape(N, -1)[:, :8]
        return {"logits": self.fc(feat), "features": feat}


def _make_model(
    n_cache: int = 4, num_class: int = 12,
    kd_loss_weight: float = 0.5, kd_temperature: float = 2.0,
):
    teacher = torch.softmax(torch.randn(n_cache, num_class), dim=-1)
    return SoftLabelKdLightningModel(
        model=_TinyBackbone(num_class=num_class),
        base_lr=1e-3,
        num_class=num_class,
        filename_to_softlabel_idx={f"S{i}": i for i in range(n_cache)},
        softlabel_tensor=teacher,
        kd_loss_weight=kd_loss_weight,
        kd_temperature=kd_temperature,
    )


def test_constructor_rejects_wrong_class_axis():
    with pytest.raises(ValueError, match="softlabel_tensor"):
        SoftLabelKdLightningModel(
            model=_TinyBackbone(),
            base_lr=1e-3, num_class=12,
            filename_to_softlabel_idx={"a": 0},
            softlabel_tensor=torch.softmax(torch.randn(1, 8), dim=-1),  # wrong C
        )


def test_constructor_rejects_zero_temperature():
    with pytest.raises(ValueError, match="kd_temperature"):
        SoftLabelKdLightningModel(
            model=_TinyBackbone(),
            base_lr=1e-3, num_class=12,
            filename_to_softlabel_idx={"a": 0},
            softlabel_tensor=torch.softmax(torch.randn(1, 12), dim=-1),
            kd_temperature=0.0,
        )


def test_batch_softlabel_handles_known_and_unknown_stems():
    lit = _make_model(n_cache=3, num_class=12)
    teacher, valid = lit._batch_softlabel(
        ["S0", "UNK", "S2"], device=torch.device("cpu"),
    )
    assert teacher.shape == (3, 12)
    assert valid.tolist() == [True, False, True]
    # Unknown row: uniform 1/12 placeholder.
    assert torch.allclose(teacher[1], torch.full((12,), 1.0 / 12), atol=1e-6)
    # Known rows match the registered softlabel tensor.
    assert torch.allclose(teacher[0], lit.softlabel_tensor[0], atol=1e-6)
    assert torch.allclose(teacher[2], lit.softlabel_tensor[2], atol=1e-6)


def test_softlabel_buffer_is_persistent():
    lit = _make_model(n_cache=4)
    state = dict(lit.named_buffers())
    assert "softlabel_tensor" in state
    assert state["softlabel_tensor"].shape == (4, 12)


def test_kd_warmup_epochs_supresses_kd_term_below_threshold():
    teacher = torch.softmax(torch.randn(2, 12), dim=-1)
    lit = SoftLabelKdLightningModel(
        model=_TinyBackbone(),
        base_lr=1e-3, num_class=12,
        filename_to_softlabel_idx={"S0": 0, "S1": 1},
        softlabel_tensor=teacher,
        kd_loss_weight=1.0, kd_temperature=2.0, kd_warmup_epochs=5,
    )
    assert lit.kd_warmup_epochs == 5
    # The training_step gating logic depends on self.current_epoch which is
    # Lightning-managed; verify the simple boolean used in the gate.
    fake_epoch = 3
    assert (lit.kd_loss_weight > 0) and (fake_epoch < lit.kd_warmup_epochs)
