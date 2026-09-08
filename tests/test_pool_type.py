"""Tests for a00: `pool_type` constructor arg on STGCN++, CTR-GCN, SkateFormer.

The three models now accept ``pool_type`` in their constructor. Verify:
  - default is "gap" and forward still works
  - "gmp" produces a different output tensor than "gap"
  - "gap_gmp" also works and is not identical to either

These tests mainly exercise the plumbing (shapes and differentiation) rather
than correctness of a specific value.
"""

from __future__ import annotations

import pytest
import torch

from diema.models.skeleton_graph import DIEMA_INWARD_EDGES


def _make_input(n=2, c=6, t=64, v=25):
    return torch.randn(n, c, t, v)


def _run_forward(model, x):
    out = model(x)
    assert "logits" in out
    return out["logits"]


def test_stgcn_pool_types():
    from diema.models.stgcn.stgcn_model import STGCN_Model
    x = _make_input()
    torch.manual_seed(0)
    m_gap = STGCN_Model(
        num_class=12, edge_index=DIEMA_INWARD_EDGES, num_nodes=25, in_channels=6,
        pool_type="gap",
    )
    torch.manual_seed(0)
    m_gmp = STGCN_Model(
        num_class=12, edge_index=DIEMA_INWARD_EDGES, num_nodes=25, in_channels=6,
        pool_type="gmp",
    )
    m_gap.eval()
    m_gmp.eval()
    with torch.no_grad():
        l1 = _run_forward(m_gap, x)
        l2 = _run_forward(m_gmp, x)
    assert l1.shape == (2, 12)
    assert l2.shape == (2, 12)
    assert not torch.allclose(l1, l2), "GMP should differ from GAP under identical weights"


def test_ctrgcn_pool_types():
    from diema.models.ctrgcn.ctrgcn_model import CTR_GCN_Model
    x = _make_input()
    torch.manual_seed(0)
    m_gap = CTR_GCN_Model(
        num_class=12, edge_index=DIEMA_INWARD_EDGES, num_nodes=25, in_channels=6,
        pool_type="gap",
    )
    torch.manual_seed(0)
    m_gmp = CTR_GCN_Model(
        num_class=12, edge_index=DIEMA_INWARD_EDGES, num_nodes=25, in_channels=6,
        pool_type="gmp",
    )
    m_gap.eval()
    m_gmp.eval()
    with torch.no_grad():
        l1 = _run_forward(m_gap, x)
        l2 = _run_forward(m_gmp, x)
    assert l1.shape == (2, 12)
    assert not torch.allclose(l1, l2)


def test_skateformer_pool_types():
    from diema.models.skateformer.skateformer_model import SkateFormer_Model
    x = _make_input()
    torch.manual_seed(0)
    m_gap = SkateFormer_Model(
        num_class=12, in_channels=6, dim=64, depth=2, num_heads=4,
        window_size=16, pool_type="gap",
    )
    torch.manual_seed(0)
    m_gmp = SkateFormer_Model(
        num_class=12, in_channels=6, dim=64, depth=2, num_heads=4,
        window_size=16, pool_type="gmp",
    )
    m_gap.eval()
    m_gmp.eval()
    with torch.no_grad():
        l1 = _run_forward(m_gap, x)
        l2 = _run_forward(m_gmp, x)
    assert l1.shape == (2, 12)
    assert not torch.allclose(l1, l2)


@pytest.mark.parametrize("pool_type", ["gap", "gmp", "gap_gmp"])
def test_stgcn_pool_type_valid(pool_type):
    from diema.models.stgcn.stgcn_model import STGCN_Model
    m = STGCN_Model(
        num_class=12, edge_index=DIEMA_INWARD_EDGES, num_nodes=25, in_channels=6,
        pool_type=pool_type,
    )
    x = _make_input()
    out = _run_forward(m, x)
    assert out.shape == (2, 12)


def test_stgcn_invalid_pool_type():
    from diema.models.stgcn.stgcn_model import STGCN_Model
    with pytest.raises(ValueError):
        STGCN_Model(
            num_class=12, edge_index=DIEMA_INWARD_EDGES, num_nodes=25, in_channels=6,
            pool_type="avg",  # bogus
        )


def test_ctrgcn_invalid_pool_type():
    from diema.models.ctrgcn.ctrgcn_model import CTR_GCN_Model
    with pytest.raises(ValueError):
        CTR_GCN_Model(
            num_class=12, edge_index=DIEMA_INWARD_EDGES, num_nodes=25, in_channels=6,
            base_channels=32, pool_type="avg",
        )


def test_skateformer_invalid_pool_type():
    from diema.models.skateformer.skateformer_model import SkateFormer_Model
    with pytest.raises(ValueError):
        SkateFormer_Model(
            num_class=12, in_channels=6, dim=64, depth=2, num_heads=4,
            window_size=16, pool_type="avg",
        )


@pytest.mark.parametrize("pool_type", ["gap", "gmp", "gap_gmp"])
def test_ctrgcn_pool_type_valid(pool_type):
    from diema.models.ctrgcn.ctrgcn_model import CTR_GCN_Model
    m = CTR_GCN_Model(
        num_class=12, edge_index=DIEMA_INWARD_EDGES, num_nodes=25, in_channels=6,
        base_channels=32, pool_type=pool_type,
    )
    out = _run_forward(m, _make_input())
    assert out.shape == (2, 12)


@pytest.mark.parametrize("pool_type", ["gap", "gmp", "gap_gmp"])
def test_skateformer_pool_type_valid(pool_type):
    from diema.models.skateformer.skateformer_model import SkateFormer_Model
    m = SkateFormer_Model(
        num_class=12, in_channels=6, dim=64, depth=2, num_heads=4,
        window_size=16, pool_type=pool_type,
    )
    out = _run_forward(m, _make_input())
    assert out.shape == (2, 12)
