"""Region-Aware Conv1D + Transformer (exp034).

Part-aware extension of Conv1DTransformer_Model (exp020 a01).
The input skeleton is split into 6 anatomical parts (torso / head / r_arm /
l_arm / r_leg / l_leg), each processed by a per-branch temporal Conv1D
encoder. A shared cross-part Transformer then lets every branch attend to
every other branch, and a learnable per-part gate scales each branch's
contribution before the classifier.

    - SGN-style joint-id and frame-id embeddings are added to the per-joint
      projection of each branch, then the joint dimension is summed away to
      yield a temporal sequence per branch.
    - Per-branch stack is 3 × Conv1DBlock (no per-branch transformer).
    - Cross-part transformer operates on GAP-pooled per-part features,
      treating parts as a length-6 token sequence.
    - Per-part gating is computed from the gated feature itself and applied
      before concatenation.

Architecture (dim=128 default, 6 parts)::

    (N, 6, 64, 25)
      -> gather per-part joints -> 6 × (N, 6, 64, V_p)
      -> per-branch Conv2d(6 -> dim, k=1) -> (N, dim, 64, V_p)
      -> + joint_id_emb(V_p) + frame_id_emb(64)
      -> mean over V_p -> (N, dim, 64)
      -> 3 × Conv1DBlock(dim, k=7)
      -> temporal GAP -> (N, dim)
      -> stack parts -> (N, 6, dim) -> (N, dim, 6)
      -> 2 × BNSwishTransformerBlock(dim) on part-token axis
      -> (N, 6, dim) -> per-part gate (sigmoid MLP) -> scale
      -> concat -> Linear(6*dim, num_class)

Compatible with diema's BaseModel contract:
    Input : (N, C, T, V=25)
    Output: {"logits": (N, num_class)}

Related: diema/models/conv1d_transformer/conv1d_transformer_model.py (backbone),
         diema/models/skeleton_graph.py (DIEMA_BODY_PARTS / build_part_indices)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diema.models.base import BaseModel
from diema.models.conv1d_transformer.conv1d_transformer_model import (
    BNSwishTransformerBlock,
    Conv1DBlock,
    LateDropout,
    Swish,
)
from diema.models.skeleton_graph import DIEMA_BODY_PARTS, build_part_indices


class _PartBranch(nn.Module):
    """Per-body-part temporal encoder.

    Projects 6D rotation features to ``dim`` per joint, adds SGN-style joint
    and frame embeddings, collapses the joint axis by mean pooling, and
    runs a stack of Conv1DBlocks along the temporal dimension.
    """

    def __init__(
        self,
        in_channels: int,
        dim: int,
        num_conv_blocks: int,
        kernel_size: int,
        drop_rate: float,
    ):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels

        # Per-joint projection: C_in -> dim via 1x1 Conv2d over (T, V).
        self.joint_proj = nn.Conv2d(in_channels, dim, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(dim)

        self.conv_blocks = nn.ModuleList([
            Conv1DBlock(dim, kernel_size=kernel_size, drop_rate=drop_rate)
            for _ in range(num_conv_blocks)
        ])

    def forward(
        self,
        x_part: torch.Tensor,
        joint_emb: torch.Tensor,
        frame_emb: torch.Tensor,
    ) -> torch.Tensor:
        # x_part: (N, C_in, T, V_p)
        # joint_emb: (dim, V_p) — part-specific, pre-indexed by the caller
        # frame_emb: (dim, T)   — shared, pre-indexed by the caller
        h = self.joint_proj(x_part)             # (N, dim, T, V_p)

        # Broadcast-add joint-id and frame-id embeddings BEFORE BatchNorm.
        # Putting BN after the embedding add lets BN normalize the shifted
        # activations, matching the "Conv -> add semantic embedding -> BN"
        # order used in SGN-style designs. (Early versions of this file
        # had BN before the add, which left the embedding contribution
        # un-normalized.)
        h = h + joint_emb.unsqueeze(0).unsqueeze(2)   # (1, dim, 1, V_p)
        h = h + frame_emb.unsqueeze(0).unsqueeze(3)   # (1, dim, T, 1)
        h = self.proj_bn(h)

        # Collapse joints by mean -> temporal sequence (N, dim, T).
        h = h.mean(dim=3)

        for blk in self.conv_blocks:
            h = blk(h)
        return h


class RegionAwareConv1DTransformer_Model(BaseModel):
    """Region-aware Conv1D + cross-part Transformer classifier.

    Args:
        num_class: number of output classes (DIEM-A: 12).
        in_channels: input channels per joint (6 for 6D rotation).
        num_nodes: skeleton joint count (must be 25 for DIEM-A).
        clip_length: temporal length at the model input (default 64).
        dim: branch / transformer width. Default 128.
        num_conv_blocks: Conv1DBlock depth per branch. Default 3.
        kernel_size: depthwise Conv1D kernel size. Default 7 (shorter than
            exp020 because each branch is only 4-5 joints, so local context
            is already restricted).
        num_cross_blocks: number of cross-part BNSwishTransformerBlocks.
        num_heads: attention heads of the cross-part transformer.
        mlp_ratio: FFN expansion ratio of the cross-part transformer.
        drop_rate: dropout inside Conv1DBlock and cross-part transformer.
        late_dropout: probability of the final LateDropout.
        late_dropout_start_step: steps before LateDropout activates.
        use_per_part_gate: if True, apply sigmoid gates to each branch gap
            feature before concatenation.
        body_parts: optional override of the canonical ``DIEMA_BODY_PARTS``.
    """

    def __init__(
        self,
        num_class: int,
        in_channels: int = 6,
        num_nodes: int = 25,
        clip_length: int = 64,
        dim: int = 128,
        num_conv_blocks: int = 3,
        kernel_size: int = 7,
        num_cross_blocks: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.3,
        late_dropout: float = 0.8,
        late_dropout_start_step: int = 1000,
        use_per_part_gate: bool = True,
        body_parts: dict[str, list[int]] | None = None,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_class = num_class
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.clip_length = clip_length
        self.dim = dim
        self.use_per_part_gate = use_per_part_gate

        parts = DIEMA_BODY_PARTS if body_parts is None else body_parts
        self.part_names: list[str] = list(parts.keys())
        part_idx_tensors = build_part_indices(parts)

        # Register each part's node indices as a non-persistent buffer so it
        # rides along with the module's device / dtype moves but is not
        # snapshotted into checkpoints.
        for name, idx in part_idx_tensors.items():
            self.register_buffer(f"part_idx_{name}", idx, persistent=False)

        # Per-branch encoder (one per body part).
        self.branches = nn.ModuleDict({
            name: _PartBranch(
                in_channels=in_channels,
                dim=dim,
                num_conv_blocks=num_conv_blocks,
                kernel_size=kernel_size,
                drop_rate=drop_rate,
            )
            for name in self.part_names
        })

        # Shared joint / frame id embeddings (SGN-style).
        self.joint_embedding = nn.Embedding(num_nodes, dim)
        self.frame_embedding = nn.Embedding(clip_length, dim)
        self.register_buffer(
            "frame_ids",
            torch.arange(clip_length, dtype=torch.long),
            persistent=False,
        )

        # Cross-part transformer on a length-N_parts token sequence.
        self.cross_blocks = nn.ModuleList([
            BNSwishTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop=drop_rate,
            )
            for _ in range(num_cross_blocks)
        ])

        # Per-part gating: MLP(dim -> dim//2 -> 1) -> sigmoid, one per part.
        if use_per_part_gate:
            self.part_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, dim // 2),
                    Swish(),
                    nn.Linear(dim // 2, 1),
                )
                for _ in self.part_names
            ])
        else:
            self.part_gates = None

        # Head.
        num_parts = len(self.part_names)
        self.late_drop = LateDropout(p=late_dropout, start_step=late_dropout_start_step)
        self.fc = nn.Linear(num_parts * dim, num_class)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.MultiheadAttention):
                # nn.MultiheadAttention stores Q/K/V projections as a raw
                # Parameter (`in_proj_weight` / `in_proj_bias`), not as an
                # nn.Linear submodule, so the isinstance(nn.Linear) branch
                # above does not reach them. Initialize them explicitly
                # to match the Xavier convention used by transformer papers.
                if m.in_proj_weight is not None:
                    nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                # out_proj is an nn.Linear, so the Linear branch already
                # handled it on this same iteration of `self.modules()`.
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / self.num_class))
        nn.init.zeros_(self.fc.bias)

    def _get_part_indices(self, name: str) -> torch.Tensor:
        return getattr(self, f"part_idx_{name}")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_input(x)
        N, C, T, V = x.size()
        if C != self.in_channels or V != self.num_nodes:
            raise ValueError(
                f"RegionAwareConv1DTransformer_Model expected "
                f"(N, {self.in_channels}, T, {self.num_nodes}), got {tuple(x.shape)}"
            )
        if T != self.clip_length:
            raise ValueError(
                f"RegionAwareConv1DTransformer_Model was built for T={self.clip_length}, "
                f"got T={T}. Rebuild the model with matching clip_length."
            )

        frame_emb = self.frame_embedding(self.frame_ids).transpose(0, 1)  # (dim, T)

        per_part_feats: list[torch.Tensor] = []
        for name in self.part_names:
            idx = self._get_part_indices(name)                # (V_p,)
            x_part = x.index_select(dim=3, index=idx)          # (N, C, T, V_p)
            joint_emb_part = self.joint_embedding(idx).transpose(0, 1)  # (dim, V_p)
            h = self.branches[name](x_part, joint_emb_part, frame_emb)  # (N, dim, T)
            feat = h.mean(dim=2)                               # GAP over T -> (N, dim)
            per_part_feats.append(feat)

        # Stack to (N, num_parts, dim) then (N, dim, num_parts) for transformer.
        part_tokens = torch.stack(per_part_feats, dim=1)        # (N, P, dim)
        tokens_cf = part_tokens.transpose(1, 2)                 # (N, dim, P)
        for blk in self.cross_blocks:
            tokens_cf = blk(tokens_cf)
        part_tokens = tokens_cf.transpose(1, 2)                 # (N, P, dim)

        if self.part_gates is not None:
            gated = []
            for i, gate_mlp in enumerate(self.part_gates):
                g = torch.sigmoid(gate_mlp(part_tokens[:, i, :]))  # (N, 1)
                gated.append(part_tokens[:, i, :] * g)
            part_tokens = torch.stack(gated, dim=1)

        flat = part_tokens.flatten(1)                            # (N, P*dim)
        # Expose the pre-dropout / pre-FC feature vector so downstream
        # wrappers (e.g. DualHeadWrapper) can tap into the
        # backbone embedding without copying or re-running the model.
        features = flat
        flat = self.late_drop(flat)
        logits = self.fc(flat)
        return {"logits": logits, "features": features}

    @property
    def feature_dim(self) -> int:
        """Dimension of the pre-FC feature vector produced by ``forward``.

        Equals ``num_parts * dim`` for the region-aware layout. Used by
         dual-head wrappers to size their projection heads.
        """
        return len(self.part_names) * self.dim

    @property
    def output_dim(self) -> int:
        return self.num_class

    @classmethod
    def from_config(cls, config) -> "RegionAwareConv1DTransformer_Model":
        return cls(
            num_class=config.model.num_class,
            in_channels=getattr(config.model, "in_channels", 6),
            num_nodes=getattr(config.skeleton, "num_nodes", 25),
            clip_length=getattr(config.model, "clip_length", 64),
            dim=getattr(config.model, "dim", 128),
            num_conv_blocks=getattr(config.model, "num_conv_blocks", 3),
            kernel_size=getattr(config.model, "kernel_size", 7),
            num_cross_blocks=getattr(config.model, "num_cross_blocks", 2),
            num_heads=getattr(config.model, "num_heads", 4),
            mlp_ratio=getattr(config.model, "mlp_ratio", 2.0),
            drop_rate=getattr(config.model, "drop_rate", 0.3),
            late_dropout=getattr(config.model, "late_dropout", 0.8),
            late_dropout_start_step=getattr(config.model, "late_dropout_start_step", 1000),
            use_per_part_gate=getattr(config.model, "use_per_part_gate", True),
            body_parts=getattr(config.model, "body_parts", None),
        )


from diema.models.registry import register_model  # noqa: E402

register_model(
    "region_aware_conv1d_transformer", RegionAwareConv1DTransformer_Model
)
