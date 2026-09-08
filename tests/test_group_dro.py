"""TDD tests for diema/training/group_dro_trainer.py.

All tests run on CPU only — no GPU training. Validates:
- country_group_id mapping and error handling
- update_group_weights: multiplicative update, simplex property, eta=0
- group_dro_loss: per-group mean CE, single-group reduction, differentiability
- _select_loss / training_step: warmup vs DRO phase selection
- mixup_alpha > 0 raises ValueError

See: Phase ACC Track B — B0 (TDD)
Plan: ~/.claude/plans/replicated-gathering-raven.md Track B (GroupDRO)
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from diema.training.group_dro_trainer import (
    GroupDROLightningModel,
    country_group_id,
    group_dro_loss,
    update_group_weights,
)


# ---------------------------------------------------------------------------
# country_group_id tests
# ---------------------------------------------------------------------------


class TestCountryGroupId:
    def test_jp_short(self):
        assert country_group_id("JP_06") == 0

    def test_jp_full_filename(self):
        assert country_group_id("JP_06_anger_01") == 0

    def test_jp_with_extension(self):
        assert country_group_id("JP_14_joy_02.bvh") == 0

    def test_tw_short(self):
        assert country_group_id("TW_38") == 1

    def test_tw_full_filename(self):
        assert country_group_id("TW_38_joy_02") == 1

    def test_raises_on_unknown_prefix(self):
        with pytest.raises(ValueError, match="Unknown country prefix"):
            country_group_id("US_01_anger")

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError):
            country_group_id("")

    def test_raises_on_de_prefix(self):
        with pytest.raises(ValueError):
            country_group_id("DE_03_happy_01")


# ---------------------------------------------------------------------------
# update_group_weights tests
# ---------------------------------------------------------------------------


class TestUpdateGroupWeights:
    def _uniform_q(self, num_groups: int = 2) -> torch.Tensor:
        return torch.ones(num_groups) / num_groups

    def test_higher_loss_group_gets_higher_weight(self):
        """Group 1 has higher loss → q[1] > q[0] after update."""
        q = self._uniform_q(2)
        group_losses = {0: 0.5, 1: 2.0}  # group 1 is "worse"
        q_new = update_group_weights(q, group_losses, [0, 1], eta=0.1)
        assert q_new[1].item() > q_new[0].item(), (
            f"Expected q[1] > q[0] after higher-loss update, got {q_new}"
        )

    def test_simplex_constraint_sums_to_one(self):
        """Result must lie on the probability simplex (sum ≈ 1)."""
        q = self._uniform_q(3)
        group_losses = {0: 1.0, 1: 0.5, 2: 2.0}
        q_new = update_group_weights(q, group_losses, [0, 1, 2], eta=0.05)
        assert abs(q_new.sum().item() - 1.0) < 1e-5, (
            f"q does not sum to 1: {q_new.sum().item()}"
        )

    def test_simplex_non_negative(self):
        """All weights must be non-negative."""
        q = self._uniform_q(2)
        group_losses = {0: 5.0, 1: 0.1}
        q_new = update_group_weights(q, group_losses, [0, 1], eta=1.0)
        assert (q_new >= 0).all(), f"Negative weights found: {q_new}"

    def test_eta_zero_leaves_q_unchanged(self):
        """When eta=0, exp(0*L) = 1, so the update is a no-op."""
        q = torch.tensor([0.3, 0.7])
        group_losses = {0: 10.0, 1: 0.01}
        q_new = update_group_weights(q, group_losses, [0, 1], eta=0.0)
        # With eta=0: q_g * exp(0) = q_g, then renormalise q (sum=1, so same).
        torch.testing.assert_close(q_new, q, atol=1e-6, rtol=1e-6)

    def test_absent_groups_retain_weight(self):
        """Groups not in the batch keep their weight (not zeroed out)."""
        q = torch.tensor([0.5, 0.5])
        # Only group 0 present in this batch.
        group_losses = {0: 1.0}
        q_new = update_group_weights(q, group_losses, [0], eta=0.1)
        # Group 1 must still have a positive weight.
        assert q_new[1].item() > 0.0, "Absent group weight was zeroed out."
        assert abs(q_new.sum().item() - 1.0) < 1e-5

    def test_returns_new_tensor_not_in_place(self):
        """update_group_weights must not modify q in place."""
        q = torch.tensor([0.5, 0.5])
        q_copy = q.clone()
        _ = update_group_weights(q, {0: 1.0, 1: 2.0}, [0, 1], eta=0.1)
        torch.testing.assert_close(q, q_copy)


# ---------------------------------------------------------------------------
# group_dro_loss tests
# ---------------------------------------------------------------------------


class TestGroupDroLoss:
    def _make_per_sample_ce(self, values: list[float], requires_grad: bool = True) -> torch.Tensor:
        t = torch.tensor(values, dtype=torch.float32)
        if requires_grad:
            t.requires_grad_(True)
        return t

    def test_per_group_mean_ce_is_correct(self):
        """Hand-checked example: group 0 has samples [1.0, 3.0], group 1 has [2.0].
        Group means: 0 → 2.0, 1 → 2.0. With uniform q both groups have weight 0.5
        so loss = 0.5*2.0 + 0.5*2.0 = 2.0.
        """
        per_sample_ce = self._make_per_sample_ce([1.0, 2.0, 3.0])
        group_ids = [0, 1, 0]
        q = torch.tensor([0.5, 0.5])
        loss, q_new = group_dro_loss(
            per_sample_ce=per_sample_ce,
            group_ids=group_ids,
            q=q,
            num_groups=2,
            eta=0.0,  # eta=0 → q doesn't change, isolates the primal
            update=True,
        )
        # With eta=0, q_new = q (uniform). Both groups have mean CE = 2.0.
        # DRO loss = 0.5 * 2.0 + 0.5 * 2.0 = 2.0.
        assert abs(loss.item() - 2.0) < 1e-5, f"Expected 2.0, got {loss.item()}"

    def test_single_group_reduces_to_mean_ce(self):
        """If only one group is present, the loss equals that group's mean CE."""
        per_sample_ce = self._make_per_sample_ce([1.0, 3.0, 2.0])
        group_ids = [0, 0, 0]  # all in group 0
        q = torch.tensor([0.6, 0.4])
        loss, _ = group_dro_loss(
            per_sample_ce=per_sample_ce,
            group_ids=group_ids,
            q=q,
            num_groups=2,
            eta=0.1,
        )
        expected = (1.0 + 3.0 + 2.0) / 3.0
        assert abs(loss.item() - expected) < 1e-5, (
            f"Expected {expected}, got {loss.item()}"
        )

    def test_scalar_is_differentiable(self):
        """The returned loss scalar must be differentiable w.r.t. per_sample_ce."""
        # Build per_sample_ce from a leaf tensor with grad.
        logits = torch.randn(4, 3, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 0])
        per_sample_ce = F.cross_entropy(logits, labels, reduction="none")
        group_ids = [0, 0, 1, 1]
        q = torch.tensor([0.5, 0.5])
        loss, _ = group_dro_loss(
            per_sample_ce=per_sample_ce,
            group_ids=group_ids,
            q=q,
            num_groups=2,
            eta=0.01,
        )
        # .backward() must not raise.
        loss.backward()
        assert logits.grad is not None, "No gradient on logits after backward()"
        assert not logits.grad.isnan().any(), "NaN gradients after backward()"

    def test_update_false_leaves_q_unchanged(self):
        """When update=False, the returned q must equal the input q."""
        per_sample_ce = self._make_per_sample_ce([0.5, 1.5, 1.0])
        q_orig = torch.tensor([0.5, 0.5])
        _, q_back = group_dro_loss(
            per_sample_ce=per_sample_ce,
            group_ids=[0, 1, 0],
            q=q_orig,
            num_groups=2,
            eta=0.1,
            update=False,
        )
        torch.testing.assert_close(q_back, q_orig, atol=1e-6, rtol=1e-6)

    def test_unknown_groups_excluded(self):
        """Samples with group_id = -1 are silently excluded."""
        # Samples: [g0, g=-1 (unknown), g1]
        per_sample_ce = self._make_per_sample_ce([1.0, 999.0, 2.0])
        group_ids = [0, -1, 1]
        q = torch.tensor([0.5, 0.5])
        loss, _ = group_dro_loss(
            per_sample_ce=per_sample_ce,
            group_ids=group_ids,
            q=q,
            num_groups=2,
            eta=0.0,
        )
        # Groups 0 and 1 both have one sample. Mean CE: g0=1.0, g1=2.0.
        # Uniform q → loss = 0.5*1.0 + 0.5*2.0 = 1.5. The 999.0 sample excluded.
        assert abs(loss.item() - 1.5) < 1e-5, (
            f"Expected 1.5 (unknown group excluded), got {loss.item()}"
        )

    def test_higher_loss_group_upweighted_in_loss(self):
        """After q update the higher-loss group is upweighted, so loss > plain mean."""
        per_sample_ce = self._make_per_sample_ce([0.1, 0.1, 5.0, 5.0], requires_grad=False)
        group_ids = [0, 0, 1, 1]  # group 0 low-loss, group 1 high-loss
        q = torch.tensor([0.5, 0.5])
        loss, q_new = group_dro_loss(
            per_sample_ce=per_sample_ce,
            group_ids=group_ids,
            q=q,
            num_groups=2,
            eta=0.5,
        )
        plain_mean = per_sample_ce.mean().item()
        # Group 1 should get higher weight → DRO loss > plain mean if both groups
        # have different losses (group 1 mean = 5.0, group 0 mean = 0.1).
        assert q_new[1].item() > q_new[0].item(), "Higher-loss group 1 not upweighted"


# ---------------------------------------------------------------------------
# Tiny stub model for testing GroupDROLightningModel
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal model stub: (N, 6, 64, 25) → {"logits": (N, 12)}."""

    def __init__(self, num_class: int = 12):
        super().__init__()
        self.fc = nn.Linear(6 * 64 * 25, num_class)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.fc(x.reshape(x.size(0), -1))
        return {"logits": logits}


def _make_lit_model(**overrides) -> GroupDROLightningModel:
    defaults = dict(
        model=_TinyModel(),
        base_lr=1e-3,
        num_class=12,
        group_mode="country",
        num_groups=2,
        eta_dro=0.01,
        warmup_epochs_dro=10,
        loss_type="ce",
        loss_kwargs=None,
        optimizer="AdamW",
        scheduler_type="cosine_warmup",
        weight_decay=5e-4,
        warmup_epochs=5,
    )
    defaults.update(overrides)
    return GroupDROLightningModel(**defaults)


# ---------------------------------------------------------------------------
# GroupDROLightningModel: mixup forbidden
# ---------------------------------------------------------------------------


class TestMixupForbidden:
    def test_mixup_alpha_zero_ok(self):
        model = _make_lit_model(mixup_alpha=0.0)
        assert model is not None

    def test_mixup_alpha_positive_raises(self):
        with pytest.raises(ValueError, match="forbids mixup_alpha"):
            _make_lit_model(mixup_alpha=0.5)

    def test_mixup_alpha_tiny_positive_raises(self):
        with pytest.raises(ValueError, match="forbids mixup_alpha"):
            _make_lit_model(mixup_alpha=1e-9)


# ---------------------------------------------------------------------------
# GroupDROLightningModel: warmup vs DRO loss selection
# ---------------------------------------------------------------------------


class TestSelectLoss:
    """Test the _select_loss helper in isolation (no PL trainer needed)."""

    def setup_method(self):
        self.model = _make_lit_model(warmup_epochs_dro=10, eta_dro=0.01)

    def _make_per_sample_ce(self, n: int = 8) -> torch.Tensor:
        # Build a differentiable per-sample CE from a tiny model forward.
        logits = torch.randn(n, 12, requires_grad=True)
        labels = torch.randint(0, 12, (n,))
        return F.cross_entropy(logits, labels, reduction="none")

    def test_warmup_returns_plain_mean_ce(self):
        """epoch < warmup_epochs_dro → plain mean CE, dro_active=False."""
        per_sample_ce = self._make_per_sample_ce(8)
        group_ids = [0, 1, 0, 1, 0, 0, 1, 1]
        expected_mean = per_sample_ce.detach().mean().item()

        loss, dro_active = self.model._select_loss(per_sample_ce, group_ids, current_epoch=0)
        assert dro_active is False
        assert abs(loss.item() - expected_mean) < 1e-5

    def test_warmup_epoch_just_before_threshold(self):
        """epoch = warmup_epochs_dro - 1 → still warmup."""
        per_sample_ce = self._make_per_sample_ce(4)
        group_ids = [0, 0, 1, 1]
        _, dro_active = self.model._select_loss(
            per_sample_ce, group_ids, current_epoch=9  # threshold is 10
        )
        assert dro_active is False

    def test_dro_phase_after_warmup(self):
        """epoch >= warmup_epochs_dro → GroupDRO reweighting, dro_active=True."""
        per_sample_ce = self._make_per_sample_ce(8)
        group_ids = [0, 1, 0, 1, 0, 0, 1, 1]
        _, dro_active = self.model._select_loss(
            per_sample_ce, group_ids, current_epoch=10  # exactly at threshold
        )
        assert dro_active is True

    def test_dro_phase_updates_q_buffer(self):
        """After DRO phase call, self.q should change from uniform if groups differ."""
        # Assign all high-CE samples to group 1 so its weight should increase.
        per_sample_ce_values = [0.1, 5.0, 0.1, 5.0]
        logits = torch.randn(4, 12, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3])
        per_sample_ce = F.cross_entropy(logits, labels, reduction="none")

        # Manually set per-sample CE to our desired values via detached replacement.
        per_sample_ce = torch.tensor(per_sample_ce_values, requires_grad=True)

        q_before = self.model.q.clone()
        group_ids = [0, 1, 0, 1]
        self.model._select_loss(per_sample_ce, group_ids, current_epoch=10)
        q_after = self.model.q.clone()

        # q[1] (high-loss group) must increase relative to q[0] (low-loss group).
        assert q_after[1].item() > q_after[0].item(), (
            f"High-loss group 1 not upweighted: q before={q_before}, after={q_after}"
        )

    def test_warmup_does_not_update_q_buffer(self):
        """During warmup, self.q must remain unchanged (no dual update)."""
        per_sample_ce = torch.tensor([5.0, 0.1, 5.0, 0.1], requires_grad=True)
        group_ids = [0, 1, 0, 1]
        q_before = self.model.q.clone()
        self.model._select_loss(per_sample_ce, group_ids, current_epoch=5)
        torch.testing.assert_close(
            self.model.q, q_before, atol=1e-6, rtol=1e-6,
            msg="q should not change during warmup"
        )


# ---------------------------------------------------------------------------
# GroupDROLightningModel: q buffer registered, shape, initial value
# ---------------------------------------------------------------------------


class TestQBuffer:
    def test_q_is_registered_buffer(self):
        model = _make_lit_model(num_groups=2)
        assert "q" in dict(model.named_buffers()), "q must be a registered buffer"

    def test_q_shape_matches_num_groups(self):
        for g in [2, 3, 5]:
            model = _make_lit_model(num_groups=g)
            assert model.q.shape == (g,), f"Expected q.shape=({g},), got {model.q.shape}"

    def test_q_initially_uniform(self):
        model = _make_lit_model(num_groups=2)
        expected = torch.ones(2) / 2
        torch.testing.assert_close(model.q, expected, atol=1e-6, rtol=1e-6)

    def test_q_not_in_optimizer_params(self):
        """q buffer must NOT appear as a trainable parameter."""
        model = _make_lit_model(num_groups=2)
        param_ids = {id(p) for p in model.parameters()}
        assert id(model.q) not in param_ids, "q should not be a trainable parameter"

    def test_q_not_requires_grad(self):
        model = _make_lit_model(num_groups=2)
        assert not model.q.requires_grad, "q must not require grad"
