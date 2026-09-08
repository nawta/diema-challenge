"""Fidelity tests for the clean-room ResNet3dSlowOnly port (exp086).

The strongest correctness signal for a clean-room reimplementation is that
the official pretrained checkpoint loads with ZERO missing / unexpected /
shape-mismatched backbone tensors.
"""

from __future__ import annotations

import os

import pytest
import torch

from diema.models.posec3d.resnet3d_slowonly_port import (
    Bottleneck3d,
    ResNet3dSlowOnly,
    build_posec3d_backbone,
    load_backbone_checkpoint,
)

CKPT = "data/pyskl/checkpoints/poseconv3d_ntu60_xsub_joint.pth"


def test_build_and_param_count():
    m = build_posec3d_backbone()
    n = len([k for k in m.state_dict() if "num_batches_tracked" not in k])
    assert n == 215  # matches checkpoint backbone tensor count
    assert m.feat_dim == 512


def test_forward_shape():
    m = build_posec3d_backbone().eval()
    with torch.no_grad():
        y = m(torch.randn(2, 17, 48, 56, 56))
    assert y.shape[:2] == (2, 512)
    assert y.ndim == 5  # (N, 512, T', H', W')


def test_bottleneck_expansion():
    assert Bottleneck3d.expansion == 4


@pytest.mark.skipif(not os.path.exists(CKPT), reason="PoseC3D ckpt not present")
def test_checkpoint_loads_exactly():
    m = build_posec3d_backbone()
    rep = load_backbone_checkpoint(m, CKPT)
    assert rep["missing"] == 0
    assert rep["unexpected"] == 0
    assert rep["mismatch"] == 0
    assert rep["loaded"] == 258  # 215 params + 43 num_batches buffers


@pytest.mark.skipif(not os.path.exists(CKPT), reason="PoseC3D ckpt not present")
def test_loaded_model_deterministic_forward():
    m = build_posec3d_backbone()
    load_backbone_checkpoint(m, CKPT)
    m.eval()
    x = torch.randn(1, 17, 48, 64, 64)
    with torch.no_grad():
        a = m(x).mean(dim=(2, 3, 4))
        b = m(x).mean(dim=(2, 3, 4))
    torch.testing.assert_close(a, b)
    assert a.shape == (1, 512)
