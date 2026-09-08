""" tests: PartAlignConv1DTr_Model forward-shape + backbone wrap."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diema.models.base import BaseModel  # noqa: E402
from diema.models.conv1d_transformer.partalign_conv1d_tr_model import (  # noqa: E402
    PartAlignConv1DTr_Model,
)


class _FakeBackbone(BaseModel):
    """Minimal BaseModel stub that emits deterministic features and logits."""

    def __init__(self, num_class: int = 12, feat_dim: int = 64, num_nodes: int = 25):
        super().__init__()
        self._num_class = num_class
        self._feat_dim = feat_dim
        self._num_nodes = num_nodes
        self.in_proj = torch.nn.Linear(num_nodes * 6 * 64, feat_dim)
        self.cls_head = torch.nn.Linear(feat_dim, num_class)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (N, C, T, V) — flatten and project into features for test stubs.
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


def test_forward_shapes():
    backbone = _FakeBackbone(num_class=12, feat_dim=64, num_nodes=25)
    model = PartAlignConv1DTr_Model(backbone, text_dim=384, num_parts=6)
    model.train()
    x = torch.randn(4, 6, 64, 25)  # (N, C, T, V) with T=64 matching stub
    out = model(x)
    assert out["logits"].shape == (4, 12)
    assert out["features"].shape == (4, 64)
    assert out["z_parts"].shape == (4, 6, 384)


def test_z_parts_are_l2_normalized():
    backbone = _FakeBackbone()
    model = PartAlignConv1DTr_Model(backbone, text_dim=64, num_parts=6)
    model.eval()  # disable BN side effects for determinism
    x = torch.randn(3, 6, 64, 25)
    with torch.no_grad():
        z = model(x)["z_parts"]
    norms = z.norm(dim=2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_properties_propagate_from_backbone():
    backbone = _FakeBackbone(num_class=7, feat_dim=128, num_nodes=20)
    model = PartAlignConv1DTr_Model(backbone, text_dim=256, num_parts=4)
    assert model.feature_dim == 128
    assert model.output_dim == 7
    assert model.num_nodes == 20
    assert model.text_dim == 256
    assert model.num_parts == 4


def test_custom_proj_hidden():
    backbone = _FakeBackbone(feat_dim=64)
    model = PartAlignConv1DTr_Model(
        backbone, text_dim=32, num_parts=6, proj_hidden=48,
    )
    # Shape of first Linear weight should match (48, 64).
    first = model.part_head[0]
    assert isinstance(first, torch.nn.Linear)
    assert first.out_features == 48


def test_gradient_flows_through_z_parts():
    backbone = _FakeBackbone(feat_dim=64)
    model = PartAlignConv1DTr_Model(backbone, text_dim=16, num_parts=3)
    model.train()
    x = torch.randn(5, 6, 64, 25, requires_grad=False)
    out = model(x)
    # Synthetic target for part loss: arbitrary but not all-zero so grads exist.
    target = torch.randn_like(out["z_parts"])
    loss = (out["z_parts"] - target).pow(2).mean()
    loss.backward()
    # The final linear in part_head should have non-zero gradient.
    last_linear = model.part_head[-1]
    assert isinstance(last_linear, torch.nn.Linear)
    assert last_linear.weight.grad is not None
    assert last_linear.weight.grad.abs().sum().item() > 0


def test_batch_size_one_no_bn_error():
    """BatchNorm1d needs N>=2 in train(); ensure we raise a meaningful error path
    or succeed if BN tracks running stats. eval() path must always succeed."""
    backbone = _FakeBackbone()
    model = PartAlignConv1DTr_Model(backbone, text_dim=16, num_parts=3)
    model.eval()
    x = torch.randn(1, 6, 64, 25)
    out = model(x)
    assert out["z_parts"].shape == (1, 3, 16)
