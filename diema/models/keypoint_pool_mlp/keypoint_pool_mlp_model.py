"""Structured Keypoint Pooling MLP (DONJYARAHOI) — ASL Signs 12th place.

Architecture:
  - Pure MLP, 2-stream (joint + bone), bidirectional lateral connections
  - Per-frame encoder: flatten joints → Linear → N x (MLPBlock + lateral swap)
  - GAP over time → concat joint/bone features → FC head → logits

Reference:
  - Writeup: https://www.kaggle.com/competitions/asl-signs/writeups/donjyarahoi-12th-place-solution-mlp-based-structur
  - Two-Stream paper: https://arxiv.org/abs/2211.01367
  - Structured Keypoint Pooling: https://arxiv.org/abs/2303.15270
  - (ASL Signs 12th direct port)

DIEM-A Adaptation:
  The original solution uses 107-joint 2D coordinates + motion lags (107+142 feats
  x 3 temporal variants) from the ASL Signs MediaPipe pipeline. For DIEM-A we stay
  within the existing rotation_6d pipeline and derive "bone-like" features inside
  the model instead of plumbing a second stream through MotionDataset:
    - Joint stream: raw (N, 6, T, V) rotation_6d features as-is.
    - Bone stream: per-edge delta `joint[v] - joint[parent[v]]` in rotation_6d
      space (with the root having parent=self → zero vector).
  This preserves the "two independent MLP stacks + lateral swaps" structure while
  using the same data loader as exp003/exp006. Note that the original uses
  Euclidean bone vectors (3D xyz); our 6-dim "bone" is a difference of rotation
  representations, which is not geometrically a true bone but provides a
  complementary structural signal for the lateral interaction.

Compatible with diema BaseModel:
  Input: (N, C, T, V), Output: {"logits": (N, num_class)}

Related: diema/models/ctrgcn/, diema/models/protogcn/, diema/models/skeleton_graph.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from diema.models.base import BaseModel


class MLPBlock(nn.Module):
    """Residual 2-layer MLP block with GELU + Dropout (applied per (N, T, D) token).

    Uses a single dropout after the activation (post-activation dropout) —
    the previous "double dropout" after both fc1 and fc2 was flagged as too
    aggressive (effective drop rate ≈ 2p - p^2).
    """

    def __init__(self, hidden_dim: int, expansion: int = 4, dropout: float = 0.3):
        super().__init__()
        inner = hidden_dim * expansion
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, inner)
        self.fc2 = nn.Linear(inner, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return residual + x


class KeypointPoolMLP_Model(BaseModel):
    """MLP-only 2-stream (joint + bone) model with bidirectional lateral connections.

    Args:
        num_class: number of output classes.
        edge_index: list of (child, parent) tuples (CTV node indices). Used to
            derive `parent_indices` so the bone stream can compute per-joint
            deltas `joint[v] - joint[parent[v]]` inside the forward pass.
        num_nodes: number of skeleton nodes (default 25 for DIEM-A).
        in_channels: input feature channels per joint (default 6 for rotation_6d).
        hidden_dim: MLP width.
        num_blocks: number of MLP blocks per stream (with a lateral swap after each).
        dropout: dropout rate used in every MLP block and the classifier head.
        lateral_connection: whether to add the bidirectional joint<->bone swap
            between every MLP block (bidirectional lateral connection from the
            Two-Stream Network paper). Set to False to run two independent MLP
            towers for ablation.
    """

    def __init__(
        self,
        num_class: int,
        edge_index: list,
        num_nodes: int = 25,
        in_channels: int = 6,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.3,
        lateral_connection: bool = True,
    ):
        super().__init__()
        self.num_class = num_class
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.lateral_connection = lateral_connection

        # Build parent indices from inward_edges. Nodes without an incoming edge
        # (orphans / the virtual root) get parent=self → zero bone vector.
        parents = list(range(num_nodes))  # default: self → bone = 0
        for child, parent in edge_index:
            if 0 <= child < num_nodes and 0 <= parent < num_nodes:
                parents[child] = parent
        self.register_buffer(
            "parent_indices",
            torch.tensor(parents, dtype=torch.long),
            persistent=False,
        )

        flat_dim = num_nodes * in_channels  # e.g. 25 * 6 = 150

        # BatchNorm on the raw (N, V*C, T) layout — same pattern as ctrgcn/protogcn.
        self.joint_bn = nn.BatchNorm1d(flat_dim)
        self.bone_bn = nn.BatchNorm1d(flat_dim)

        # Per-stream input projection (raw (N, T, V*C) token → hidden_dim).
        self.joint_proj = nn.Linear(flat_dim, hidden_dim)
        self.bone_proj = nn.Linear(flat_dim, hidden_dim)

        # MLP blocks (one stack per stream) + optional lateral projections.
        self.joint_blocks = nn.ModuleList(
            [MLPBlock(hidden_dim, expansion=4, dropout=dropout) for _ in range(num_blocks)]
        )
        self.bone_blocks = nn.ModuleList(
            [MLPBlock(hidden_dim, expansion=4, dropout=dropout) for _ in range(num_blocks)]
        )
        if lateral_connection:
            self.lateral_jb = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_blocks)]
            )
            self.lateral_bj = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_blocks)]
            )
        else:
            self.lateral_jb = None
            self.lateral_bj = None

        # Classifier head: concat joint + bone GAP features.
        self.head_norm = nn.LayerNorm(2 * hidden_dim)
        self.head_fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.head_drop = nn.Dropout(dropout)
        self.head_fc2 = nn.Linear(hidden_dim, num_class)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _compute_bone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Derive bone-like features: joint[v] - joint[parent[v]].

        Args:
            x: (N, C, T, V) joint features (rotation_6d in DIEM-A).

        Returns:
            (N, C, T, V) tensor of per-joint deltas (zero vector at root).
        """
        # Gather parent features along the V axis.
        # x is (N, C, T, V); expand parent_indices to index along V.
        parents = self.parent_indices.to(x.device)  # (V,)
        parent_feats = x.index_select(dim=3, index=parents)  # (N, C, T, V)
        return x - parent_feats

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_input(x)
        N, C, T, V = x.size()
        assert V == self.num_nodes, f"Expected V={self.num_nodes}, got {V}"
        assert C == self.in_channels, f"Expected C={self.in_channels}, got {C}"

        # Bone features must be computed on the pre-BN tensor so that the delta
        # is well-defined (differences of BN'd features would also be fine but
        # the skeleton semantics are cleaner on raw input).
        bone_x = self._compute_bone_features(x)  # (N, C, T, V)

        # Data BN: (N, C, T, V) → (N, V*C, T) → BN → back to (N, T, V*C).
        def _bn_flatten(tensor: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
            t = tensor.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)  # (N, V*C, T)
            t = bn(t)
            t = t.view(N, V, C, T).permute(0, 3, 1, 2).contiguous()  # (N, T, V, C)
            return t.view(N, T, V * C)  # (N, T, V*C)

        joint_tok = _bn_flatten(x, self.joint_bn)     # (N, T, V*C)
        bone_tok = _bn_flatten(bone_x, self.bone_bn)  # (N, T, V*C)

        # Project to hidden_dim. Tokens are (N, T, hidden_dim).
        joint_h = self.joint_proj(joint_tok)
        bone_h = self.bone_proj(bone_tok)

        # Stacked MLP blocks with bidirectional lateral connections.
        for i in range(self.num_blocks):
            joint_h = self.joint_blocks[i](joint_h)
            bone_h = self.bone_blocks[i](bone_h)
            if self.lateral_connection:
                joint_new = joint_h + self.lateral_jb[i](bone_h)
                bone_new = bone_h + self.lateral_bj[i](joint_h)
                joint_h, bone_h = joint_new, bone_new

        # GAP over time (dim=1 after our (N, T, D) layout).
        joint_pooled = joint_h.mean(dim=1)  # (N, hidden_dim)
        bone_pooled = bone_h.mean(dim=1)    # (N, hidden_dim)

        fused = torch.cat([joint_pooled, bone_pooled], dim=1)  # (N, 2*hidden_dim)
        fused = self.head_norm(fused)
        fused = self.head_fc1(fused)
        fused = F.gelu(fused)
        fused = self.head_drop(fused)
        logits = self.head_fc2(fused)
        return {"logits": logits}

    @property
    def output_dim(self) -> int:
        return self.num_class

    @classmethod
    def from_config(cls, config) -> "KeypointPoolMLP_Model":
        return cls(
            num_class=config.model.num_class,
            edge_index=config.skeleton.inward_edges,
            num_nodes=config.skeleton.num_nodes,
            in_channels=config.model.in_channels,
            hidden_dim=getattr(config.model, "hidden_dim", 256),
            num_blocks=getattr(config.model, "num_blocks", 4),
            dropout=getattr(config.model, "dropout", 0.3),
            lateral_connection=getattr(config.model, "lateral_connection", True),
        )


from diema.models.registry import register_model  # noqa: E402

register_model("keypoint_pool_mlp", KeypointPoolMLP_Model)
