"""Shape and basic-property tests for the four new models.

Each model must satisfy the BaseModel contract:
  - Input: (N, C=6, T=64, V=25)
  - Output: dict with "logits" key of shape (N, num_class)

Also verifies:
  - Registered in the global model registry
  - from_config() builds the model from a SimpleNamespace cfg
  - Gradient flows through the model (loss.backward() works)
  - Parameter count is non-zero and reasonable
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from diema.models import build_model, list_models
from diema.models.skeleton_graph import DIEMA_INWARD_EDGES


def _small_cfg(name: str, **overrides) -> SimpleNamespace:
    base_kwargs = {
        "stgcn": dict(num_class=12, in_channels=6, plusplus=True, pool_type="gap"),
        "ctrgcn": dict(num_class=12, in_channels=6, base_channels=32, pool_type="gap"),
        "skateformer": dict(
            num_class=12, in_channels=6, dim=64, depth=2, num_heads=4, window_size=16,
        ),
        "protogcn": dict(num_class=12, in_channels=6, base_channels=32),
        "conv1d_transformer": dict(
            num_class=12, in_channels=6, dim=96, num_blocks=2, kernel_size=15, drop_rate=0.1,
        ),
        "region_aware_conv1d_transformer": dict(
            num_class=12, in_channels=6, clip_length=64,
            dim=64, num_conv_blocks=2, kernel_size=7,
            num_cross_blocks=1, num_heads=4, mlp_ratio=2.0,
            drop_rate=0.1, use_per_part_gate=True,
        ),
        "keypoint_pool_mlp": dict(
            num_class=12, in_channels=6, hidden_dim=96, num_blocks=2, dropout=0.3,
            lateral_connection=True,
        ),
        "conformer": dict(
            num_class=12, in_channels=6, num_layers=2, hidden_dim=96, mlp_dim=192,
            num_heads=4, conv_kernel_size=15, dropout=0.1, drop_path=0.1,
        ),
        "squeezeformer": dict(
            num_class=12, in_channels=6, num_layers=2, d_model=96, num_heads=4, mlp_ratio=2,
            conv_kernel_size=15, dropout=0.1, drop_path=0.1,
        ),
    }[name]
    base_kwargs.update(overrides)
    return SimpleNamespace(
        model=SimpleNamespace(name=name, **base_kwargs),
        skeleton=SimpleNamespace(num_nodes=25, inward_edges=DIEMA_INWARD_EDGES),
    )


NEW_MODELS = [
    "conv1d_transformer",
    "region_aware_conv1d_transformer",
    "keypoint_pool_mlp",
    "conformer",
    "squeezeformer",
]


def test_all_new_models_registered():
    registered = list_models()
    for name in NEW_MODELS:
        assert name in registered, f"{name} missing from registry: {registered}"


@pytest.mark.parametrize("name", NEW_MODELS)
def test_new_model_forward_shape(name):
    cfg = _small_cfg(name)
    m = build_model(cfg)
    m.eval()
    x = torch.randn(3, 6, 64, 25)
    with torch.no_grad():
        out = m(x)
    assert "logits" in out
    assert out["logits"].shape == (3, 12)


@pytest.mark.parametrize("name", NEW_MODELS)
def test_new_model_gradient_flows(name):
    cfg = _small_cfg(name)
    m = build_model(cfg)
    m.train()
    x = torch.randn(2, 6, 64, 25)
    targets = torch.tensor([0, 1])
    out = m(x)
    loss = F.cross_entropy(out["logits"], targets)
    loss.backward()
    # Check that at least one parameter received a non-zero gradient
    any_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in m.parameters()
    )
    assert any_grad, f"no parameter received a gradient for {name}"


@pytest.mark.parametrize("name", NEW_MODELS)
def test_new_model_output_dim(name):
    cfg = _small_cfg(name)
    m = build_model(cfg)
    assert m.output_dim == 12


@pytest.mark.parametrize("name", NEW_MODELS)
def test_new_model_param_count_sane(name):
    cfg = _small_cfg(name)
    m = build_model(cfg)
    n_params = sum(p.numel() for p in m.parameters())
    # Should have at least 10K params (sanity floor) and at most 100M (ceiling)
    assert 10_000 < n_params < 100_000_000, f"{name}: {n_params} params"


def test_conv1d_transformer_late_dropout_not_persistent():
    """LateDropout._step should not be a persistent buffer — checkpoints
    should not freeze the step counter."""
    from diema.models.conv1d_transformer.conv1d_transformer_model import LateDropout
    ld = LateDropout(p=0.8, start_step=100)
    state_dict = ld.state_dict()
    # `_step` should NOT appear in state_dict
    assert "_step" not in state_dict, f"_step found in state_dict: {list(state_dict.keys())}"


def test_conv1d_transformer_late_dropout_activation():
    """LateDropout should only drop after start_step training-mode forwards."""
    from diema.models.conv1d_transformer.conv1d_transformer_model import LateDropout
    ld = LateDropout(p=1.0, start_step=3)  # p=1.0 → all zeros once active
    x = torch.ones(4, 8)

    # Before start_step: should pass through
    ld.train()
    y0 = ld(x)
    assert torch.allclose(y0, x)
    y1 = ld(x)
    assert torch.allclose(y1, x)
    y2 = ld(x)  # step == start_step, so active now
    # y2 should be all zeros (p=1.0 dropout)
    assert torch.allclose(y2, torch.zeros_like(x))

    # Eval mode: should be identity
    ld.eval()
    y_eval = ld(x)
    assert torch.allclose(y_eval, x)


def test_keypoint_pool_mlp_bone_root_is_zero():
    """Root node (index 1 = Hips) has parent=self → bone = joint - joint = 0."""
    import torch as _torch
    cfg = _small_cfg("keypoint_pool_mlp")
    m = build_model(cfg)
    parents = m.parent_indices.tolist()
    # Hips is at CTV index 1 and has NO inward edge (it's the BVH root),
    # so its parent defaults to itself in the from-edges builder.
    assert parents[1] == 1, f"Hips parent should be self (1), got {parents[1]}"
    # Virtual root (index 0) has an inward edge (0, 1), so parent[0] = 1.
    assert parents[0] == 1, f"virtual_root parent should be 1, got {parents[0]}"


def test_squeezeformer_rope_cos_sin_non_persistent():
    """RoPE cos/sin buffers must not pollute checkpoints."""
    from diema.models.squeezeformer.rope_attention import RotaryEmbedding
    rope = RotaryEmbedding(dim=48, max_seq_len=256)
    state_dict = rope.state_dict()
    # Both cos and sin buffers should be non-persistent
    keys = list(state_dict.keys())
    for k in keys:
        assert not (k.endswith("cos") or k.endswith("sin")), f"{k} should be non-persistent"


def test_conformer_no_back_to_back_norm():
    """After the Opus review fix, Conformer_Model should not have both
    `post_norm` AND `final_norm` at the encoder stack boundary."""
    cfg = _small_cfg("conformer")
    m = build_model(cfg)
    # The model itself should no longer have a post_norm attribute (it was
    # removed in favor of the block's internal final_norm).
    assert not hasattr(m, "post_norm"), (
        "Conformer_Model.post_norm should have been removed "
        "(ConformerBlock already ends with final_norm)"
    )
