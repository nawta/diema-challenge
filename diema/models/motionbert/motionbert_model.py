"""MotionBERTModel — backbone finetune for DIEM-A 12-class emotion.

Wraps Walter0807/MotionBERT-Lite (DSTformer) as a BaseModel:
  Input  (BaseModel contract): (N, C=3, T, V=25) joint_pos stream
  Project in forward()         → (N, T, 17, 3) MotionBERT-format (H36M-17, root-relative, scale [-1, 1], conf channel)
  Backbone                     → (N, T, 17, dim_rep=512)
  Pool over (T, J)             → (N, 512)
  Head Linear                  → (N, num_class=12)

The torch-side projector replicates `diema/features/bvh_to_h36m17.py`
(numpy version used for exp081 frozen-feature extraction) but operates
on the dataloader's (N, 3, T, 25) tensor without numpy fallback.

Pretrained loading:
  state = torch.load(pretrained_path)["model_pos"] (or similar)
  self.backbone.load_state_dict(state, strict=False)

Freeze/unfreeze schedule (plan):
  a00: freeze backbone permanently, train head only (linear probe equivalent)
  a01: freeze backbone for `unfreeze_epoch` epochs, then unfreeze (full finetune)

See: docs/analysis/exp081_motionbert_transfer.md (frozen-feature baseline)
Related: diema/features/bvh_to_h36m17.py (numpy projector reference)
         tmp/MotionBERT/lib/model/DSTformer.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from diema.models.base import BaseModel
from diema.models.registry import register_model

_MB_SRC = Path(__file__).resolve().parents[3] / "tmp" / "MotionBERT"
if str(_MB_SRC) not in sys.path:
    sys.path.insert(0, str(_MB_SRC))


# BVH-24 → H36M-17 mapping (mirrors diema.features.bvh_to_h36m17.H36M_TO_BVH).
# "synth_root", "synth_belly", "synth_nose" handled separately.
_H36M_BVH_DIRECT: dict[int, int] = {
    1:  16,    # rhip   ← RightUpLeg
    2:  17,    # rkne   ← RightLeg
    3:  18,    # rank   ← RightFoot
    4:  20,    # lhip   ← LeftUpLeg
    5:  21,    # lkne   ← LeftLeg
    6:  22,    # lank   ← LeftFoot
    8:  5,     # neck   ← BVH Neck (base)
    10: 7,     # head   ← BVH Head
    11: 13,    # lsho   ← LeftArm pivot
    12: 14,    # lelb   ← LeftForeArm
    13: 15,    # lwri   ← LeftHand
    14: 9,     # rsho   ← RightArm pivot
    15: 10,    # relb   ← RightForeArm
    16: 11,    # rwri   ← RightHand
}


def _project_to_h36m17(
    x_ctv25: torch.Tensor,
    drop_axis: str = "y",
    flip_vertical: bool = True,
    nose_offset_ratio: float = 0.05,
) -> torch.Tensor:
    """Convert dataloader (N, 3, T, V=25) joint_pos → MotionBERT (N, T, 17, 3).

    The CTV-25 layout (per diema/features/multi_stream.py):
      - node 0 stores world Hip position (root_pos)
      - nodes 1-24 store joint_pos[i-1] − Hips (root-relative)
    """
    if x_ctv25.ndim != 4 or x_ctv25.shape[1] != 3 or x_ctv25.shape[3] != 25:
        raise ValueError(f"expected (N, 3, T, 25) joint_pos, got {x_ctv25.shape}")
    N, _, T, _ = x_ctv25.shape

    root_pos = x_ctv25[..., 0:1]  # (N, 3, T, 1) — world Hip per frame
    joint_relative = x_ctv25[..., 1:]  # (N, 3, T, 24) — relative offsets
    world_bvh24 = joint_relative + root_pos  # (N, 3, T, 24) — absolute world

    if drop_axis == "y":
        xy = world_bvh24[:, [0, 2], :, :]  # keep X (lateral), Z (vertical)
    elif drop_axis == "z":
        xy = world_bvh24[:, [0, 1], :, :]
    elif drop_axis == "x":
        xy = world_bvh24[:, [1, 2], :, :]
    else:
        raise ValueError(f"drop_axis must be x|y|z, got {drop_axis!r}")
    if flip_vertical:
        xy = xy.clone()
        xy[:, 1, :, :] = -xy[:, 1, :, :]

    xy = xy.permute(0, 2, 3, 1).contiguous()  # (N, T, 24, 2)

    out_xy = torch.zeros(N, T, 17, 2, device=xy.device, dtype=xy.dtype)
    out_conf = torch.ones(N, T, 17, device=xy.device, dtype=xy.dtype)

    for h36m_idx, bvh_idx in _H36M_BVH_DIRECT.items():
        out_xy[:, :, h36m_idx, :] = xy[:, :, bvh_idx, :]

    out_xy[:, :, 0, :] = 0.5 * (xy[:, :, 16, :] + xy[:, :, 20, :])
    out_xy[:, :, 7, :] = 0.5 * (out_xy[:, :, 0, :] + out_xy[:, :, 8, :])
    out_xy[:, :, 9, :] = xy[:, :, 7, :]  # head as placeholder
    out_conf[:, :, 9] = 0.5

    root_h36m = out_xy[:, :, 0:1, :]
    out_xy = out_xy - root_h36m

    abs_max = out_xy.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
    out_xy = out_xy / abs_max

    out_xy[:, :, 9, 1] = out_xy[:, :, 10, 1] + nose_offset_ratio
    out_xy[:, :, 9, 0] = out_xy[:, :, 10, 0]

    out = torch.cat([out_xy, out_conf.unsqueeze(-1)], dim=-1)  # (N, T, 17, 3)
    return out


class MotionBERTModel(BaseModel):
    """DSTformer backbone + classifier head for DIEM-A emotion."""

    def __init__(
        self,
        num_class: int = 12,
        pretrained_path: str | None = None,
        dim_feat: int = 256,
        dim_rep: int = 512,
        depth: int = 5,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        num_joints: int = 17,
        maxlen: int = 243,
        head_dropout: float = 0.3,
        freeze_backbone: bool = False,
        drop_axis: str = "y",
        flip_vertical: bool = True,
        head_type: str = "pooled",
    ) -> None:
        super().__init__()
        from functools import partial

        from lib.model.DSTformer import DSTformer  # noqa: E402

        self._num_class = int(num_class)
        self.drop_axis = drop_axis
        self.flip_vertical = flip_vertical

        self.backbone = DSTformer(
            dim_in=3, dim_out=3,
            dim_feat=dim_feat, dim_rep=dim_rep,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            num_joints=num_joints, maxlen=maxlen,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            att_fuse=True,
        )

        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                for key in ("model_pos", "state_dict", "model"):
                    if key in state:
                        state = state[key]
                        break
            state = {k.removeprefix("module."): v for k, v in state.items()}
            incompat = self.backbone.load_state_dict(state, strict=False)
            # head + ts_attn often differ between pretrain checkpoints; warn but accept
            missing_critical = [
                k for k in incompat.missing_keys
                if not (k.startswith("head") or k.startswith("ts_attn"))
            ]
            if missing_critical:
                print(f"[MotionBERTModel] WARN missing critical: {len(missing_critical)} keys (first 5: {missing_critical[:5]})")
            if incompat.unexpected_keys:
                print(f"[MotionBERTModel] WARN unexpected: {len(incompat.unexpected_keys)} keys (first 5: {incompat.unexpected_keys[:5]})")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.head_type = head_type
        self.num_joints = num_joints
        self.head_dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        if head_type == "pooled":
            feat_dim = dim_rep
        elif head_type == "per_joint_flat":
            feat_dim = num_joints * dim_rep
        else:
            raise ValueError(f"head_type must be pooled|per_joint_flat, got {head_type!r}")
        # BatchNorm before head — emulates sklearn StandardScaler that was
        # used in exp081 frozen-feature linear probe (which got 24.43% vs
        # this Lightning path's 16.49% before this fix).
        self.head_bn = nn.BatchNorm1d(feat_dim)
        self.head = nn.Linear(feat_dim, self._num_class)

    @property
    def output_dim(self) -> int:
        return self._num_class

    @property
    def feature_dim(self) -> int:
        return self.head.in_features

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(trainable)

    def forward(self, x: torch.Tensor) -> dict:
        self._validate_input(x)
        x_mb = _project_to_h36m17(x, drop_axis=self.drop_axis, flip_vertical=self.flip_vertical)
        rep = self.backbone(x_mb, return_rep=True)  # (N, T, 17, 512)
        if self.head_type == "pooled":
            features = rep.mean(dim=(1, 2))  # (N, 512)
        else:  # per_joint_flat
            features = rep.mean(dim=1)         # (N, 17, 512)
            features = features.flatten(1)     # (N, 17*512)
        features_bn = self.head_bn(features)
        features_d = self.head_dropout(features_bn)
        logits = self.head(features_d)
        return {"logits": logits, "features": features_bn}

    @classmethod
    def from_config(cls, config):
        return cls(
            num_class=config.model.num_class,
            pretrained_path=getattr(config.model, "pretrained_path", None),
            dim_feat=getattr(config.model, "dim_feat", 256),
            dim_rep=getattr(config.model, "dim_rep", 512),
            depth=getattr(config.model, "depth", 5),
            num_heads=getattr(config.model, "num_heads", 8),
            mlp_ratio=getattr(config.model, "mlp_ratio", 4),
            num_joints=getattr(config.model, "num_joints", 17),
            maxlen=getattr(config.model, "maxlen", 243),
            head_dropout=getattr(config.model, "head_dropout", 0.3),
            freeze_backbone=getattr(config.model, "freeze_backbone", False),
            drop_axis=getattr(config.model, "drop_axis", "y"),
            flip_vertical=getattr(config.model, "flip_vertical", True),
            head_type=getattr(config.model, "head_type", "pooled"),
        )


register_model("motionbert", MotionBERTModel)
