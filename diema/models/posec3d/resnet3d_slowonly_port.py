"""Clean-room pure-PyTorch port of PYSKL ResNet3dSlowOnly (PoseC3D backbone).

PYSKL's ResNet3d depends on mmcv-full==1.5.0, which has no prebuilt wheel
for this machine's new CUDA/Blackwell + modern torch. Rather than install a
fragile source build, this module re-implements exactly the architecture
defined by configs/posec3d/slowonly_r50_ntu60_xsub/joint.py so that the
pretrained checkpoint data/pyskl/checkpoints/poseconv3d_ntu60_xsub_joint.pth
loads with ZERO missing / ZERO unexpected backbone keys and identical
tensor shapes (validated in tests/test_resnet3d_slowonly_port.py).

Config (slowonly_r50_ntu60_xsub/joint.py):
    in_channels=17, base_channels=32, num_stages=3, stage_blocks=(4,6,3),
    conv1_kernel=(1,7,7), conv1_stride=(1,1), pool1_stride=(1,1),
    inflate=(0,1,1), inflate_style='3x1x1',
    spatial_strides=(2,2,2), temporal_strides=(1,1,2), out_indices=(2,)

ConvModule = Conv3d(bias=False) + BN3d + ReLU, with submodule names
'.conv' / '.bn' (matches mmcv ConvModule key layout, hence the checkpoint).

Reference (not imported): tmp/pyskl/pyskl/models/cnns/resnet3d.py
                          tmp/pyskl/pyskl/models/cnns/resnet3d_slowonly.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvModule(nn.Module):
    """conv(bias=False) → bn → optional relu. Names: .conv, .bn."""

    def __init__(self, in_c, out_c, kernel, stride=1, padding=0, with_act=True):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_c)
        self.with_act = with_act
        if with_act:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.with_act:
            x = self.act(x)
        return x


class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None,
                 inflate=True):
        super().__init__()
        # inflate_style '3x1x1': conv1 temporal kernel, conv2 spatial kernel
        if inflate:
            c1_k, c1_p = (3, 1, 1), (1, 0, 0)
        else:
            c1_k, c1_p = 1, 0
        c2_k, c2_p = (1, 3, 3), (0, 1, 1)

        self.conv1 = ConvModule(inplanes, planes, c1_k, stride=1, padding=c1_p)
        self.conv2 = ConvModule(
            planes, planes, c2_k,
            stride=(stride[0], stride[1], stride[1]), padding=c2_p)
        self.conv3 = ConvModule(
            planes, planes * self.expansion, 1, with_act=False)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv3(self.conv2(self.conv1(x)))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_res_layer(inplanes, planes, blocks, stride, inflate_flags):
    """inflate_flags: per-block bool list (len == blocks)."""
    downsample = None
    if stride[1] != 1 or inplanes != planes * Bottleneck3d.expansion:
        downsample = ConvModule(
            inplanes, planes * Bottleneck3d.expansion, 1,
            stride=(stride[0], stride[1], stride[1]), with_act=False)
    layers = [Bottleneck3d(inplanes, planes, stride=stride,
                           downsample=downsample, inflate=inflate_flags[0])]
    inplanes = planes * Bottleneck3d.expansion
    for i in range(1, blocks):
        layers.append(Bottleneck3d(inplanes, planes, stride=(1, 1),
                                   inflate=inflate_flags[i]))
    return nn.Sequential(*layers)


class ResNet3dSlowOnly(nn.Module):
    """PoseC3D SlowOnly backbone for slowonly_r50_ntu60_xsub/joint.py.

    forward(x): x is (N, 17, T, H, W) heatmap volume → (N, 512, T', H', W')
    (output of layer3 = out_indices=(2,)).
    """

    def __init__(self, in_channels=17, base_channels=32,
                 stage_blocks=(4, 6, 3), conv1_kernel=(1, 7, 7),
                 conv1_stride=(1, 1), pool1_stride=(1, 1),
                 inflate=(0, 1, 1), spatial_strides=(2, 2, 2),
                 temporal_strides=(1, 1, 2)):
        super().__init__()
        pad = tuple((k - 1) // 2 for k in conv1_kernel)
        self.conv1 = ConvModule(
            in_channels, base_channels, conv1_kernel,
            stride=(conv1_stride[0], conv1_stride[1], conv1_stride[1]),
            padding=pad)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(pool1_stride[0], pool1_stride[1], pool1_stride[1]),
            padding=(0, 1, 1))

        inplanes = base_channels
        self.res_layers = []
        for i, num_blocks in enumerate(stage_blocks):
            planes = base_channels * 2 ** i
            stride = (temporal_strides[i], spatial_strides[i])
            inflate_flags = [bool(inflate[i])] * num_blocks
            layer = _make_res_layer(inplanes, planes, num_blocks, stride,
                                    inflate_flags)
            inplanes = planes * Bottleneck3d.expansion
            name = f"layer{i + 1}"
            self.add_module(name, layer)
            self.res_layers.append(name)
        self.feat_dim = base_channels * 2 ** (len(stage_blocks) - 1) * Bottleneck3d.expansion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        for name in self.res_layers:
            x = getattr(self, name)(x)
        return x  # (N, 512, T', H', W')


def build_posec3d_backbone() -> ResNet3dSlowOnly:
    return ResNet3dSlowOnly()


def load_backbone_checkpoint(model: ResNet3dSlowOnly, ckpt_path: str) -> dict:
    """Load 'backbone.*' weights from a PYSKL Recognizer3D checkpoint.

    Returns a report dict with missing/unexpected/loaded counts. Raises if
    any backbone tensor is missing or shape-mismatched (fidelity guarantee).
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    bb = {
        k[len("backbone."):]: v
        for k, v in sd.items()
        if k.startswith("backbone.")
    }
    model_sd = model.state_dict()
    missing = [k for k in model_sd if k not in bb]
    unexpected = [k for k in bb if k not in model_sd]
    mismatch = [
        k for k in model_sd
        if k in bb and tuple(bb[k].shape) != tuple(model_sd[k].shape)
    ]
    if missing or mismatch:
        raise RuntimeError(
            f"port/ckpt mismatch — missing={missing[:5]} "
            f"({len(missing)}), shape_mismatch={mismatch[:5]} ({len(mismatch)})"
        )
    model.load_state_dict(bb, strict=True)
    return {
        "loaded": len(bb),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "mismatch": len(mismatch),
    }
