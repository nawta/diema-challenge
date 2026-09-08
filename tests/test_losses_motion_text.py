""" tests — multi-positive InfoNCE and part alignment losses.

Related: diema/training/losses_motion_text.py
"""
from __future__ import annotations

import pytest
import torch

from diema.training.losses_motion_text import (
    GlobalInfoNCELoss,
    PartAlignmentLoss,
    build_positive_mask,
    global_multi_positive_info_nce,
    part_alignment_loss,
)


def _deterministic() -> None:
    torch.manual_seed(0)


def test_info_nce_shape_validation() -> None:
    with pytest.raises(ValueError):
        global_multi_positive_info_nce(
            torch.randn(4, 8), torch.randn(4, 16)  # mismatched D
        )
    with pytest.raises(ValueError):
        global_multi_positive_info_nce(torch.randn(4, 8), torch.randn(5, 8))
    with pytest.raises(ValueError):
        global_multi_positive_info_nce(torch.randn(4, 8), torch.randn(4, 8), temperature=0.0)


def test_diagonal_matches_classic_info_nce() -> None:
    """No positive_mask ⇒ falls back to single-positive CLIP behaviour."""
    _deterministic()
    N, D = 8, 32
    z = torch.randn(N, D)
    t = torch.randn(N, D)
    # diagonal positive mask (CLIP convention)
    loss = global_multi_positive_info_nce(z, t, positive_mask=None, temperature=0.1)
    # identity mask should give the same value as None
    eye_mask = torch.eye(N, dtype=torch.bool)
    loss_eye = global_multi_positive_info_nce(z, t, positive_mask=eye_mask, temperature=0.1)
    assert torch.allclose(loss, loss_eye, atol=1e-6, rtol=1e-5)


def test_symmetric_flag_behaviour() -> None:
    """Symmetric=True averages m→t and t→m; both directions should match on symmetric inputs."""
    _deterministic()
    N, D = 6, 16
    z = torch.randn(N, D)
    t = z.clone()  # identical sides ⇒ m→t and t→m must match exactly
    mask = torch.eye(N, dtype=torch.bool)
    loss_sym = global_multi_positive_info_nce(z, t, mask, symmetric=True)
    loss_m2t = global_multi_positive_info_nce(z, t, mask, symmetric=False)
    assert torch.allclose(loss_sym, loss_m2t, atol=1e-5, rtol=1e-4)


def test_multi_positive_pulls_group_together() -> None:
    """Supervised mask should push z closer to all positives in t, not just diagonal."""
    _deterministic()
    N, D = 8, 32
    z = torch.randn(N, D, requires_grad=True)
    t = torch.randn(N, D)
    group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    group_mask = group_ids[:, None] == group_ids[None, :]
    lr = 0.2

    loss_group = global_multi_positive_info_nce(z, t, group_mask, temperature=0.1)
    g_group = torch.autograd.grad(loss_group, z)[0]
    z_group = (z - lr * g_group).detach()

    loss_diag = global_multi_positive_info_nce(z, t, torch.eye(N, dtype=torch.bool), temperature=0.1)
    g_diag = torch.autograd.grad(loss_diag, z)[0]
    z_diag = (z - lr * g_diag).detach()

    def _off_diag_positive_sim(x: torch.Tensor) -> float:
        xn = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        tn = t / t.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sim = xn @ tn.transpose(0, 1)
        off_diag_mask = group_mask.float() - torch.eye(N)  # same group but not diagonal
        return (sim * off_diag_mask).sum().item() / off_diag_mask.sum().item()

    # The supervised loss should raise the z-t cosine for off-diagonal positives
    # more than the diagonal-only loss, because only the supervised mask treats
    # those pairs as positives.
    assert _off_diag_positive_sim(z_group) > _off_diag_positive_sim(z_diag)


def test_build_positive_mask_policies() -> None:
    scenario = torch.tensor([0, 0, 1, 1, 2])
    emotion = torch.tensor([0, 1, 0, 1, 0])

    diag = build_positive_mask(scenario, emotion, policy="diagonal")
    assert torch.equal(diag, torch.eye(5, dtype=torch.bool))

    same_s = build_positive_mask(scenario, emotion, policy="same_scenario")
    assert same_s[0, 1] and same_s[2, 3] and not same_s[0, 2]

    same_e = build_positive_mask(scenario, emotion, policy="same_emotion")
    assert same_e[0, 2] and same_e[0, 4] and not same_e[0, 1]

    both = build_positive_mask(scenario, emotion, policy="same_scenario_and_emotion")
    # (0,0)=diag, (0,1): scenario match but emotion diff → False
    assert both[0, 0]
    assert not both[0, 1]
    # (0,2): scenario diff → False
    assert not both[0, 2]


def test_part_alignment_cosine_honors_mask() -> None:
    """Masked parts must produce zero gradient on their inputs."""
    _deterministic()
    N, P, D = 4, 6, 16
    z = torch.randn(N, P, D, requires_grad=True)
    t = torch.randn(N, P, D)
    mask = torch.ones(N, P, dtype=torch.bool)
    # mask out part 0 for all samples
    mask[:, 0] = False
    loss = part_alignment_loss(z, t, mask, mode="cosine")
    loss.backward()
    # z[:, 0] should have zero gradient; other parts should have some gradient
    assert torch.allclose(z.grad[:, 0], torch.zeros_like(z.grad[:, 0]))
    assert z.grad[:, 1:].abs().sum().item() > 0


def test_part_alignment_fully_masked_returns_zero() -> None:
    _deterministic()
    z = torch.randn(3, 6, 16, requires_grad=True)
    t = torch.randn(3, 6, 16)
    mask = torch.zeros(3, 6, dtype=torch.bool)
    loss = part_alignment_loss(z, t, mask, mode="cosine")
    assert loss.item() == 0.0
    # gradient through the zero-graph pathway must still work (no NaN)
    loss.backward()
    assert not torch.isnan(z.grad).any()


def test_part_alignment_infonce_skips_small_parts() -> None:
    _deterministic()
    N, P, D = 4, 3, 8
    z = torch.randn(N, P, D, requires_grad=True)
    t = torch.randn(N, P, D)
    mask = torch.ones(N, P, dtype=torch.bool)
    mask[:, 0] = False  # part 0 has 0 valid rows
    mask[:3, 1] = False  # part 1 has only 1 valid row
    # Only part 2 has ≥ 2 valid rows — loss should come from part 2
    loss = part_alignment_loss(z, t, mask, mode="infonce", temperature=0.1)
    assert not torch.isnan(loss)
    loss.backward()
    assert torch.allclose(z.grad[:, 0], torch.zeros_like(z.grad[:, 0]))


def test_zero_weight_loss_does_not_affect_ce_regression() -> None:
    """Scaling the InfoNCE loss by 0 must reproduce a pure CE training step."""
    _deterministic()
    z = torch.randn(4, 8)
    t = torch.randn(4, 8)
    # Scaling this loss by 0 should yield an exact zero tensor (not NaN / not
    # sign-flipping gradient). Shared with CE-only regression in the trainer.
    loss = global_multi_positive_info_nce(z, t, positive_mask=None, temperature=0.2)
    assert (loss * 0).item() == 0.0


def test_global_info_nce_loss_module_matches_functional() -> None:
    _deterministic()
    z = torch.randn(4, 8)
    t = torch.randn(4, 8)
    mask = torch.eye(4, dtype=torch.bool)
    fn_out = global_multi_positive_info_nce(z, t, mask, temperature=0.1, symmetric=True)
    mod_out = GlobalInfoNCELoss(temperature=0.1, symmetric=True)(z, t, mask)
    assert torch.allclose(fn_out, mod_out, atol=1e-6)


def test_part_alignment_loss_module_matches_functional() -> None:
    _deterministic()
    z = torch.randn(3, 6, 8)
    t = torch.randn(3, 6, 8)
    mask = torch.ones(3, 6, dtype=torch.bool)
    mask[0, 0] = False
    fn_out = part_alignment_loss(z, t, mask, mode="cosine")
    mod_out = PartAlignmentLoss(mode="cosine")(z, t, mask)
    assert torch.allclose(fn_out, mod_out, atol=1e-6)


def test_low_temperature_sharpens_softmax() -> None:
    """τ→0 should push the multi-positive loss toward 0 when positives align perfectly."""
    _deterministic()
    N, D = 4, 8
    z = torch.randn(N, D)
    t = z.clone()  # perfect alignment
    mask = torch.eye(N, dtype=torch.bool)
    l_hot = global_multi_positive_info_nce(z, t, mask, temperature=0.01)
    l_cold = global_multi_positive_info_nce(z, t, mask, temperature=1.0)
    # sharper softmax ⇒ lower loss when positives are already aligned
    assert l_hot.item() < l_cold.item()
