"""Tests for SupCon loss + model + trainer.

Covered:
  - SupConLoss: basic properties, performer-aware weighting, edge cases
  - SupConConv1DTr_Model: forward shape, z_sc on unit sphere
  - SupConLightningModel: training_step runs end-to-end
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from diema.training.losses import SupConLoss


# ---------------------------------------------------------------------------
# SupConLoss
# ---------------------------------------------------------------------------


class TestSupConLoss:
    def test_loss_positive_with_positives(self):
        """Loss should be positive (> 0) when there are positive pairs."""
        torch.manual_seed(0)
        loss_fn = SupConLoss(temperature=0.1)
        z = torch.randn(8, 32)
        z = z / z.norm(dim=1, keepdim=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(z, labels)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_loss_decreases_for_aligned_pairs(self):
        """When same-class features are identical, loss should be lower
        than when they are random."""
        torch.manual_seed(0)
        loss_fn = SupConLoss(temperature=0.1)
        labels = torch.tensor([0, 0, 1, 1])

        # Random features
        z_random = torch.randn(4, 16)
        z_random = z_random / z_random.norm(dim=1, keepdim=True)
        loss_random = loss_fn(z_random, labels)

        # Aligned: same-class features are identical
        v0 = torch.randn(1, 16)
        v1 = torch.randn(1, 16)
        z_aligned = torch.cat([v0, v0, v1, v1], dim=0)
        z_aligned = z_aligned / z_aligned.norm(dim=1, keepdim=True)
        loss_aligned = loss_fn(z_aligned, labels)

        assert loss_aligned.item() < loss_random.item()

    def test_performer_aware_weighting(self):
        """Cross-performer positives should be weighted more heavily."""
        torch.manual_seed(0)
        loss_fn = SupConLoss(temperature=0.1, w_strong=2.0, w_weak=0.5)
        z = torch.randn(4, 16)
        z = z / z.norm(dim=1, keepdim=True)
        labels = torch.tensor([0, 0, 0, 1])
        perf_ids = torch.tensor([0, 1, 0, 2])

        loss_with_perf = loss_fn(z, labels, performer_ids=perf_ids)
        loss_no_perf = loss_fn(z, labels, performer_ids=None)

        # Both should be finite
        assert torch.isfinite(loss_with_perf)
        assert torch.isfinite(loss_no_perf)
        # They should differ due to weighting
        assert not torch.allclose(loss_with_perf, loss_no_perf, atol=1e-6)

    def test_no_positives_returns_zero(self):
        """When every sample has a unique label, there are no positives."""
        loss_fn = SupConLoss(temperature=0.1)
        z = torch.randn(4, 8)
        z = z / z.norm(dim=1, keepdim=True)
        labels = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(z, labels)
        assert loss.item() == 0.0

    def test_single_sample_returns_zero(self):
        loss_fn = SupConLoss(temperature=0.1)
        z = torch.randn(1, 8)
        z = z / z.norm(dim=1, keepdim=True)
        loss = loss_fn(z, torch.tensor([0]))
        assert loss.item() == 0.0

    def test_gradient_flows(self):
        loss_fn = SupConLoss(temperature=0.1)
        z = torch.randn(6, 16, requires_grad=True)
        z_norm = z / z.norm(dim=1, keepdim=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loss = loss_fn(z_norm, labels)
        loss.backward()
        assert z.grad is not None and z.grad.abs().sum() > 0

    def test_rejects_wrong_ndim(self):
        loss_fn = SupConLoss(temperature=0.1)
        with pytest.raises(ValueError, match="N, D"):
            loss_fn(torch.randn(4, 8, 2), torch.tensor([0, 0, 1, 1]))

    def test_rejects_bad_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            SupConLoss(temperature=0.0)


# ---------------------------------------------------------------------------
# SupConConv1DTr_Model
# ---------------------------------------------------------------------------


class TestSupConModel:
    def test_forward_shape_and_normalization(self):
        torch.manual_seed(0)
        from diema.models import build_model
        from diema.models.conv1d_transformer.supcon_conv1d_tr_model import (
            SupConConv1DTr_Model,
        )
        from diema.models.skeleton_graph import DIEMA_INWARD_EDGES

        backbone = build_model(SimpleNamespace(
            model=SimpleNamespace(
                name="conv1d_transformer", num_class=12, in_channels=6,
                dim=48, num_blocks=1, kernel_size=7,
                num_heads=4, mlp_ratio=2.0, drop_rate=0.1,
                late_dropout=0.0, late_dropout_start_step=999999,
                pool_type="gap",
            ),
            skeleton=SimpleNamespace(num_nodes=25, inward_edges=DIEMA_INWARD_EDGES),
        ))
        model = SupConConv1DTr_Model(backbone=backbone, proj_dim=64)

        x = torch.randn(2, 6, 64, 25)
        out = model(x)

        assert out["logits"].shape == (2, 12)
        assert out["features"].shape == (2, 96)  # dim * 2
        assert out["z_sc"].shape == (2, 64)
        # z_sc should be on the unit sphere
        norms = out["z_sc"].norm(dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_feature_dim_property(self):
        from diema.models import build_model
        from diema.models.conv1d_transformer.supcon_conv1d_tr_model import (
            SupConConv1DTr_Model,
        )
        from diema.models.skeleton_graph import DIEMA_INWARD_EDGES

        backbone = build_model(SimpleNamespace(
            model=SimpleNamespace(
                name="conv1d_transformer", num_class=12, in_channels=6,
                dim=48, num_blocks=1, kernel_size=7, num_heads=4,
                mlp_ratio=2.0, drop_rate=0.1,
                late_dropout=0.0, late_dropout_start_step=999999,
                pool_type="gap",
            ),
            skeleton=SimpleNamespace(num_nodes=25, inward_edges=DIEMA_INWARD_EDGES),
        ))
        model = SupConConv1DTr_Model(backbone=backbone, proj_dim=64)
        assert model.feature_dim == 96


# ---------------------------------------------------------------------------
# SupConLightningModel training_step
# ---------------------------------------------------------------------------


class TestSupConTrainer:
    def test_training_step_end_to_end(self):
        torch.manual_seed(0)
        from diema.models import build_model
        from diema.models.conv1d_transformer.supcon_conv1d_tr_model import (
            SupConConv1DTr_Model,
        )
        from diema.models.skeleton_graph import DIEMA_INWARD_EDGES
        from diema.training.supcon_trainer import SupConLightningModel

        backbone = build_model(SimpleNamespace(
            model=SimpleNamespace(
                name="conv1d_transformer", num_class=12, in_channels=6,
                dim=48, num_blocks=1, kernel_size=7, num_heads=4,
                mlp_ratio=2.0, drop_rate=0.1,
                late_dropout=0.0, late_dropout_start_step=999999,
                pool_type="gap",
            ),
            skeleton=SimpleNamespace(num_nodes=25, inward_edges=DIEMA_INWARD_EDGES),
        ))
        model = SupConConv1DTr_Model(backbone=backbone, proj_dim=64)
        fn_map = {"s0": 0, "s1": 1, "s2": 2, "s3": 3}
        lit = SupConLightningModel(
            model=model, base_lr=1e-3, num_class=12,
            filename_to_perf_idx=fn_map, num_performer=4,
            supcon_lambda=0.05, supcon_warmup_epochs=0,
            loss_type="ce", optimizer="AdamW", scheduler_type="cosine",
        )
        # Stub trainer-dependent attributes
        fake_opt = SimpleNamespace(param_groups=[{"lr": 1e-3}])
        lit.optimizers = lambda: fake_opt
        lit.log = lambda *a, **k: None
        lit._current_fx_name = "training_step"
        lit.trainer = SimpleNamespace(current_epoch=5)

        inputs = torch.randn(4, 6, 64, 25)
        labels = torch.tensor([0, 0, 1, 1])
        filenames = ["s0", "s1", "s2", "s3"]
        batch = (inputs, labels, filenames)

        loss = lit.training_step(batch, 0)
        assert torch.isfinite(loss)
        assert loss.dim() == 0
        loss.backward()
        backbone_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.backbone.parameters()
        )
        assert backbone_has_grad

    def test_warmup_skips_supcon(self):
        """During warmup epochs, SupCon loss should not be applied."""
        torch.manual_seed(0)
        from diema.models import build_model
        from diema.models.conv1d_transformer.supcon_conv1d_tr_model import (
            SupConConv1DTr_Model,
        )
        from diema.models.skeleton_graph import DIEMA_INWARD_EDGES
        from diema.training.supcon_trainer import SupConLightningModel

        backbone = build_model(SimpleNamespace(
            model=SimpleNamespace(
                name="conv1d_transformer", num_class=12, in_channels=6,
                dim=48, num_blocks=1, kernel_size=7, num_heads=4,
                mlp_ratio=2.0, drop_rate=0.1,
                late_dropout=0.0, late_dropout_start_step=999999,
                pool_type="gap",
            ),
            skeleton=SimpleNamespace(num_nodes=25, inward_edges=DIEMA_INWARD_EDGES),
        ))
        model = SupConConv1DTr_Model(backbone=backbone, proj_dim=64)
        lit = SupConLightningModel(
            model=model, base_lr=1e-3, num_class=12,
            filename_to_perf_idx={}, num_performer=0,
            supcon_lambda=0.05, supcon_warmup_epochs=10,
            loss_type="ce", optimizer="AdamW", scheduler_type="cosine",
        )
        fake_opt = SimpleNamespace(param_groups=[{"lr": 1e-3}])
        lit.optimizers = lambda: fake_opt
        logged = {}
        lit.log = lambda name, val, **k: logged.update({name: val})
        lit._current_fx_name = "training_step"
        lit.trainer = SimpleNamespace(current_epoch=3)

        inputs = torch.randn(4, 6, 64, 25)
        labels = torch.tensor([0, 0, 1, 1])
        batch = (inputs, labels, ["a", "b", "c", "d"])

        loss = lit.training_step(batch, 0)
        assert torch.isfinite(loss)
        # SupCon loss should NOT be logged during warmup
        assert "train_loss_supcon" not in logged
