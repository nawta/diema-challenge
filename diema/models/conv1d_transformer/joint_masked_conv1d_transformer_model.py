"""Conv1D + Transformer with input joint-masking (exp075).

Wraps the canonical ``Conv1DTransformer_Model`` so that only a *subset* of
CTV joints is exposed to the model (other joints are zero-masked). Input
shape is preserved as ``(N, C, T, V=25)``.

Mechanism (NOT mean-pooling): the inner ``Conv1DTransformer_Model``
flattens the joint axis into per-frame features via
``permute(0, 3, 1, 2).contiguous().view(N, V*C, T)`` (so masked joints
become 6 consecutive zero channels per frame). The first layer is a
``BatchNorm1d(V*C)`` which, for a perpetually-zero channel, produces
the channel's bias parameter ``beta`` as output (since both batch and
running ``mean=0, var=0+eps`` ⇒ output ≈ 0/√eps · gamma + beta = beta).
The downstream stem (``Conv1d(V*C → dim, k=1)``) then sees masked
channels as a constant per frame and absorbs them as a per-output
bias-equivalent term (no temporal/per-sample information leaks; verified
empirically by ``test_zero_mask_isolates_kept_joints`` with
``allclose atol=1e-6``).

Per plan §11.5 option A, this is the **zero-mask** variant
(option B would route a V=N tensor through a model with a V=N adapter
in the dataloader; deferred). The wrapper is a one-line input multiply
plus the existing Conv1D+Tr forward.

Used by exp075 to quantify "how much DIEM-A emotion classification can
be done from just N out of 25 joints?" — joint-masking-driven artifact
production for paper figures.

Checkpoint compatibility: ``node_mask`` is registered with
``persistent=False`` so it is **not** saved into ``state_dict``. When
resuming a checkpoint trained for a particular ``keep_joints``, the
caller MUST construct the wrapper with the SAME ``keep_joints`` list —
otherwise the wrapper silently applies a different mask to the loaded
weights, producing incorrect predictions without any error.

See: §5.3 / §11.5
Related: diema/models/conv1d_transformer/conv1d_transformer_model.py
         (the underlying Conv1D+Transformer),
         diema/models/skeleton_graph.py (DIEMA_JOINT_NAMES).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diema.models.base import BaseModel
from diema.models.conv1d_transformer.conv1d_transformer_model import (
    Conv1DTransformer_Model,
)


class JointMaskedConv1DTransformer_Model(BaseModel):
    """Conv1D+Transformer with a fixed binary joint-mask applied to the
    input tensor along the V axis.

    Args:
        num_class: number of output classes (DIEM-A: 12).
        keep_joints: list of CTV indices to keep (other indices zero-masked).
        num_nodes: total CTV nodes (default 25 for DIEM-A).
        in_channels: input feature channels (default 6 for 6D rotation).
        clip_length: temporal length (default 64).
        dim: Conv1D+Tr internal width (default 128 for "tiny").
        num_blocks: number of Conv1D+Tr block pairs (default 2 for "tiny").
        kernel_size: depthwise Conv1D kernel.
        num_heads: transformer attention heads.
        mlp_ratio: FFN expansion ratio.
        drop_rate: dropout inside Conv1DBlock and transformer.
        late_dropout: probability of the final LateDropout.
        late_dropout_start_step: steps before LateDropout activates.
        pool_type: temporal pooling ('gap' / 'gmp' / 'gap_gmp').
    """

    def __init__(
        self,
        num_class: int,
        keep_joints: list[int],
        num_nodes: int = 25,
        in_channels: int = 6,
        clip_length: int = 64,
        dim: int = 128,
        num_blocks: int = 2,
        kernel_size: int = 17,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.3,
        late_dropout: float = 0.8,
        late_dropout_start_step: int = 1000,
        pool_type: str = "gap",
    ):
        super().__init__()
        if not keep_joints:
            raise ValueError("keep_joints must be non-empty")
        if len(set(keep_joints)) != len(keep_joints):
            raise ValueError(f"keep_joints must be unique, got duplicates: {keep_joints}")
        for j in keep_joints:
            if not (0 <= j < num_nodes):
                raise ValueError(
                    f"keep_joints index {j} out of range [0, {num_nodes})"
                )
        self.num_nodes = num_nodes
        self.keep_joints = list(keep_joints)
        self.num_kept = len(self.keep_joints)

        # Build a (V,) binary mask (1 for kept joints, 0 elsewhere).
        mask = torch.zeros(num_nodes, dtype=torch.float32)
        mask[self.keep_joints] = 1.0
        self.register_buffer("node_mask", mask, persistent=False)

        # clip_length is accepted in our __init__ for future symmetry but
        # the inner Conv1DTransformer_Model treats T dynamically and does
        # not need it as a constructor arg.
        self._clip_length = clip_length

        # Inner Conv1D+Transformer (full-V input expected).
        self.base_model = Conv1DTransformer_Model(
            num_class=num_class,
            in_channels=in_channels,
            num_nodes=num_nodes,
            dim=dim,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            late_dropout=late_dropout,
            late_dropout_start_step=late_dropout_start_step,
            pool_type=pool_type,
        )

    def forward(self, x: torch.Tensor) -> dict:
        # x: (N, C, T, V). Zero-mask non-kept joints along V.
        if x.dim() != 4:
            raise ValueError(f"expected input ndim=4 (N,C,T,V), got shape {x.shape}")
        if x.shape[3] != self.num_nodes:
            raise ValueError(
                f"expected V={self.num_nodes}, got V={x.shape[3]}"
            )
        x_masked = x * self.node_mask.view(1, 1, 1, -1)
        return self.base_model(x_masked)

    @property
    def output_dim(self) -> int:
        return self.base_model.output_dim

    @classmethod
    def from_config(cls, config) -> "JointMaskedConv1DTransformer_Model":
        return cls(
            num_class=config.model.num_class,
            keep_joints=list(config.model.keep_joints),
            num_nodes=getattr(config.skeleton, "num_nodes", 25),
            in_channels=getattr(config.model, "in_channels", 6),
            clip_length=getattr(config.model, "clip_length", 64),
            dim=getattr(config.model, "dim", 128),
            num_blocks=getattr(config.model, "num_blocks", 2),
            kernel_size=getattr(config.model, "kernel_size", 17),
            num_heads=getattr(config.model, "num_heads", 4),
            mlp_ratio=getattr(config.model, "mlp_ratio", 2.0),
            drop_rate=getattr(config.model, "drop_rate", 0.3),
            late_dropout=getattr(config.model, "late_dropout", 0.8),
            late_dropout_start_step=getattr(config.model, "late_dropout_start_step", 1000),
            pool_type=getattr(config.model, "pool_type", "gap"),
        )


from diema.models.registry import register_model  # noqa: E402

register_model(
    "joint_masked_conv1d_transformer", JointMaskedConv1DTransformer_Model,
)
