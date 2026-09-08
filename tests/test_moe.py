"""Tests for MoE model, losses, and trainer."""

from __future__ import annotations
from types import SimpleNamespace

import pytest
import torch

from diema.training.losses import (
    moe_load_balancing_loss,
    moe_entropy_regularization,
    moe_expert_diversity_loss,
)


class TestMoELosses:
    def test_load_balance_uniform_is_one(self):
        """Uniform routing → f_k=1/K, p_k=1/K → K*(1/K^2)*K = 1.0."""
        rw = torch.ones(8, 4) / 4
        loss = moe_load_balancing_loss(rw)
        assert torch.allclose(loss, torch.tensor(1.0), atol=0.05)

    def test_load_balance_skewed_is_higher(self):
        """All samples routed to expert 0 → higher loss."""
        rw = torch.zeros(8, 4)
        rw[:, 0] = 1.0
        loss = moe_load_balancing_loss(rw)
        assert loss.item() > 1.5

    def test_entropy_uniform_is_max(self):
        import math
        rw = torch.ones(8, 4) / 4
        ent = moe_entropy_regularization(rw)
        assert torch.allclose(ent, torch.tensor(math.log(4)), atol=0.01)

    def test_entropy_peaked_is_low(self):
        rw = torch.zeros(8, 4)
        rw[:, 0] = 1.0
        ent = moe_entropy_regularization(rw)
        assert ent.item() < 0.01

    def test_expert_diversity_identical_is_positive(self):
        """Identical expert outputs → high cosine similarity → positive loss."""
        v = torch.randn(4, 12)
        experts = [v, v, v, v]
        loss = moe_expert_diversity_loss(experts)
        assert loss.item() > 0.9  # cos(v, v) ≈ 1

    def test_expert_diversity_orthogonal_is_low(self):
        experts = [torch.eye(4, 12)[i:i+1].expand(4, -1) for i in range(4)]
        loss = moe_expert_diversity_loss(experts)
        assert loss.item() < 0.1

    def test_expert_diversity_single_expert_is_zero(self):
        loss = moe_expert_diversity_loss([torch.randn(4, 12)])
        assert loss.item() == 0.0

    def test_gradient_flow_all_losses(self):
        rw = torch.randn(4, 4, requires_grad=True).softmax(dim=1)
        experts = [torch.randn(4, 12, requires_grad=True) for _ in range(4)]
        loss = (
            moe_load_balancing_loss(rw)
            + moe_entropy_regularization(rw)
            + moe_expert_diversity_loss(experts)
        )
        loss.backward()
        assert all(e.grad is not None for e in experts)


class TestMoEModel:
    def test_forward_shape(self):
        torch.manual_seed(0)
        from diema.models import build_model
        from diema.models.conv1d_transformer.moe_head_model import FeatureAdapterMoE_Model
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
        model = FeatureAdapterMoE_Model(
            backbone=backbone,
            num_experts=2, adapter_bottleneck=32, router_hidden=16,
        )
        x = torch.randn(2, 6, 64, 25)
        out = model(x)

        assert out["logits"].shape == (2, 12)
        assert out["features"].shape == (2, 96)
        assert out["adapted_features"].shape == (2, 96)
        assert out["routing_weights"].shape == (2, 2)
        # Routing weights should sum to 1
        assert torch.allclose(
            out["routing_weights"].sum(dim=1),
            torch.ones(2), atol=1e-5,
        )


class TestMoETrainer:
    def test_training_step_end_to_end(self):
        torch.manual_seed(0)
        from diema.models import build_model
        from diema.models.conv1d_transformer.moe_head_model import FeatureAdapterMoE_Model
        from diema.models.skeleton_graph import DIEMA_INWARD_EDGES
        from diema.training.moe_trainer import MoELightningModel

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
        model = FeatureAdapterMoE_Model(
            backbone=backbone,
            num_experts=2, adapter_bottleneck=32, router_hidden=16,
        )
        lit = MoELightningModel(
            model=model, base_lr=1e-3, num_class=12,
            lambda_lb=0.01, lambda_ent=0.01, lambda_div=0.005,
            loss_type="ce", optimizer="AdamW", scheduler_type="cosine",
        )
        fake_opt = SimpleNamespace(param_groups=[{"lr": 1e-3}])
        lit.optimizers = lambda: fake_opt
        lit.log = lambda *a, **k: None

        inputs = torch.randn(4, 6, 64, 25)
        labels = torch.tensor([0, 1, 2, 3])
        batch = (inputs, labels, ["a", "b", "c", "d"])

        loss = lit.training_step(batch, 0)
        assert torch.isfinite(loss)
        assert loss.dim() == 0
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad
