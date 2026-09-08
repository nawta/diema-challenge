""" exp075 tests: JointMaskedConv1DTransformer_Model forward + masking.

Verifies the wrapper:
1. Builds with valid keep_joints and rejects invalid ones (empty, dup, OOR).
2. Produces the canonical Conv1DTransformer output shape.
3. Zero-mask is *exact* on the V axis: input values at non-kept indices have
   no effect on the masked tensor that flows into the inner model.
4. ``from_config`` round-trips an OmegaConf-style config namespace.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diema.models.conv1d_transformer.joint_masked_conv1d_transformer_model import (  # noqa: E402
    JointMaskedConv1DTransformer_Model,
)


def _tiny(keep_joints):
    """Build a tiny model so tests stay fast on CPU."""
    return JointMaskedConv1DTransformer_Model(
        num_class=12,
        keep_joints=keep_joints,
        num_nodes=25,
        in_channels=6,
        clip_length=64,
        dim=32,
        num_blocks=1,
        kernel_size=5,
        num_heads=2,
        mlp_ratio=2.0,
        drop_rate=0.0,
        late_dropout=0.0,
        late_dropout_start_step=0,
        pool_type="gap",
    )


def test_forward_shape_top4():
    torch.manual_seed(0)
    model = _tiny([5, 6, 7, 8])
    model.eval()
    x = torch.randn(2, 6, 64, 25)
    out = model(x)
    assert "logits" in out
    assert out["logits"].shape == (2, 12)


def test_keep_joints_validation():
    with pytest.raises(ValueError, match="non-empty"):
        _tiny([])
    with pytest.raises(ValueError, match="duplicates"):
        _tiny([5, 5, 6, 7])
    with pytest.raises(ValueError, match="out of range"):
        _tiny([0, 1, 25])  # 25 is OOR for num_nodes=25
    with pytest.raises(ValueError, match="out of range"):
        _tiny([-1, 1, 2])


def test_zero_mask_isolates_kept_joints():
    """Outputs must depend ONLY on values at kept joints — perturbing
    non-kept joints must yield identical logits."""
    torch.manual_seed(0)
    keep = [5, 6, 7, 8]
    model = _tiny(keep)
    model.eval()
    x1 = torch.randn(2, 6, 64, 25)

    # Perturb every non-kept joint (set to extreme values).
    x2 = x1.clone()
    non_kept = [j for j in range(25) if j not in keep]
    for j in non_kept:
        x2[:, :, :, j] = 1000.0  # arbitrary, very different from x1

    with torch.no_grad():
        y1 = model(x1)["logits"]
        y2 = model(x2)["logits"]
    assert torch.allclose(y1, y2, atol=1e-6), (
        f"max diff = {(y1 - y2).abs().max().item():.2e} — "
        "non-kept joints leaked into output"
    )


def test_input_shape_validation():
    model = _tiny([5, 6, 7, 8])
    model.eval()
    # Wrong V
    with pytest.raises(ValueError, match="V"):
        model(torch.randn(2, 6, 64, 24))
    # Wrong ndim
    with pytest.raises(ValueError, match="ndim"):
        model(torch.randn(2, 6, 64))


def test_from_config_namespace():
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            num_class=12,
            keep_joints=[5, 6, 7, 8, 10, 14, 17, 18, 20, 22],
            in_channels=6,
            clip_length=64,
            dim=32,
            num_blocks=1,
            kernel_size=5,
            num_heads=2,
            mlp_ratio=2.0,
            drop_rate=0.0,
            late_dropout=0.0,
            late_dropout_start_step=0,
            pool_type="gap",
        ),
        skeleton=SimpleNamespace(num_nodes=25),
    )
    model = JointMaskedConv1DTransformer_Model.from_config(cfg)
    assert model.num_kept == 10
    assert model.keep_joints == [5, 6, 7, 8, 10, 14, 17, 18, 20, 22]
    # Mask buffer is correct
    expected = torch.zeros(25)
    expected[[5, 6, 7, 8, 10, 14, 17, 18, 20, 22]] = 1.0
    assert torch.equal(model.node_mask, expected)


def test_node_mask_buffer_not_persistent():
    """node_mask must be registered as non-persistent so checkpoints don't
    bake in a stale mask if keep_joints changes between resume runs."""
    model = _tiny([5, 6, 7, 8])
    state = model.state_dict()
    # The non-persistent buffer must not appear in state_dict.
    assert "node_mask" not in state
