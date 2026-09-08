"""Tests for CenterLoss and CenterLossLightningModel."""

from types import SimpleNamespace

import torch
import pytest

from diema.training.losses import CenterLoss, exp_lambda_warmup


class TestCenterLoss:
    def test_positive_loss(self):
        loss_fn = CenterLoss(num_classes=4, feat_dim=8)
        features = torch.randn(6, 8)
        labels = torch.tensor([0, 1, 2, 3, 0, 1])
        loss = loss_fn(features, labels)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_zero_loss_when_features_equal_centers(self):
        loss_fn = CenterLoss(num_classes=3, feat_dim=4)
        # Set features to match centers exactly
        with torch.no_grad():
            loss_fn.centers.copy_(torch.eye(3, 4))
        features = torch.eye(3, 4)
        labels = torch.tensor([0, 1, 2])
        loss = loss_fn(features, labels)
        assert loss.item() < 1e-6

    def test_gradient_flows_to_features_and_centers(self):
        loss_fn = CenterLoss(num_classes=4, feat_dim=8)
        features = torch.randn(4, 8, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(features, labels)
        loss.backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0
        assert loss_fn.centers.grad is not None


class TestExpLambdaWarmup:
    def test_zero_at_start(self):
        assert exp_lambda_warmup(0, 1.0, tau=20.0) < 0.01

    def test_reaches_target(self):
        val = exp_lambda_warmup(100, 1.0, tau=20.0)
        assert val > 0.99

    def test_monotonic_increase(self):
        vals = [exp_lambda_warmup(e, 0.01, tau=20.0) for e in range(80)]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i-1]


class TestCenterLossTrainer:
    def test_training_step_end_to_end(self):
        torch.manual_seed(0)
        from diema.models import build_model
        from diema.models.skeleton_graph import DIEMA_INWARD_EDGES
        from diema.training.center_loss_trainer import CenterLossLightningModel

        model = build_model(SimpleNamespace(
            model=SimpleNamespace(
                name="conv1d_transformer", num_class=12, in_channels=6,
                dim=48, num_blocks=1, kernel_size=7, num_heads=4,
                mlp_ratio=2.0, drop_rate=0.1,
                late_dropout=0.0, late_dropout_start_step=999999,
                pool_type="gap",
            ),
            skeleton=SimpleNamespace(num_nodes=25, inward_edges=DIEMA_INWARD_EDGES),
        ))
        lit = CenterLossLightningModel(
            model=model, base_lr=1e-3, num_class=12,
            feat_dim=model.feature_dim,
            lambda_center=0.001, lambda_tau=20.0, center_lr=0.5,
            loss_type="ce", optimizer="AdamW", scheduler_type="cosine",
        )
        fake_opt = SimpleNamespace(param_groups=[{"lr": 1e-3}])
        lit.optimizers = lambda: fake_opt
        logged = {}
        lit.log = lambda name, val, **k: logged.update({name: val})

        inputs = torch.randn(4, 6, 64, 25)
        labels = torch.tensor([0, 1, 2, 3])
        batch = (inputs, labels, ["a", "b", "c", "d"])

        loss = lit.training_step(batch, 0)
        assert torch.isfinite(loss)
        assert "train_loss_center" in logged
        loss.backward()
