"""CTR-GCN with PC-edges-only subset adjacency (exp074 option A).

Wraps the canonical ``CTR_GCN_Model`` so it operates on a *subset* of the
DIEM-A 25-node skeleton. The subset is specified via ``node_subset``
(CTV indices into the canonical 25-node space). Forward pass slices the
input tensor along V to len(node_subset), and the adjacency is built
from the existing parent-child edges *restricted to* the subset (with
re-indexed node IDs 0..len-1).

Plan §11.4 calls this "option A (PC-edges only subset)" — i.e., use
existing ``DIEMA_INWARD_EDGES`` for the upper-head + arms +
lower-context joints rather than constructing the full UbH-Graph 5
edge subsets (which is option B).

Used by exp074 alongside the Region-Aware Conv1D+Transformer 3-part
variant (option B from §5.3 of the same plan), to span both the
GCN-style and Conv-Transformer-style architectural inductive biases.

See: §5.3 / §11.4
Related: diema/models/ctrgcn/ctrgcn_model.py (full 25-node CTR-GCN),
         diema/models/skeleton_graph.py (DIEMA_UPPER_BODY_3PARTS,
         DIEMA_INWARD_EDGES).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diema.models.base import BaseModel
from diema.models.ctrgcn.ctrgcn_model import CTR_GCN_Model


def filter_edges_to_subset(
    edge_index: list[tuple[int, int]],
    node_subset: list[int],
) -> list[tuple[int, int]]:
    """Filter ``(child, parent)`` edges to only those where BOTH endpoints
    are in ``node_subset``, then re-index node IDs to 0..len(subset)-1.

    Args:
        edge_index: list of (child, parent) tuples in the original index
            space (e.g., DIEMA_INWARD_EDGES with CTV indices 0-24).
        node_subset: list of CTV indices to keep, in the desired re-index
            order. Position i in the list maps to new index i.

    Returns:
        List of edges with re-indexed nodes; edges whose endpoints fall
        outside ``node_subset`` are dropped.
    """
    pos: dict[int, int] = {orig: new for new, orig in enumerate(node_subset)}
    out: list[tuple[int, int]] = []
    for child, parent in edge_index:
        if child in pos and parent in pos:
            out.append((pos[child], pos[parent]))
    return out


class CTRGCNSubset_Model(BaseModel):
    """CTR-GCN restricted to a subset of CTV nodes.

    Args:
        num_class: number of output classes (DIEM-A: 12).
        full_edge_index: list of (child, parent) tuples in the *original*
            CTV index space (e.g., DIEMA_INWARD_EDGES).
        node_subset: list of original CTV indices to keep. Length defines
            the new num_nodes for the inner CTR-GCN. Order matters — the
            i-th entry becomes the new node index i.
        full_num_nodes: original num_nodes (default 25). Used only for
            input-shape validation.
        in_channels: input feature channels (default: 6 for 6D rotation).
        base_channels: starting channel count (default: 32 — half of the
            full-25-node CTR-GCN baseline of 64, to keep params <0.5M).
        adj_strategy: adjacency strategy ("aagcn" / "spatial_config").
        dropout: classifier head dropout.
        adaptive: whether the inner CTR-GCN's adjacency is learnable.
        kernel_size: temporal kernel size in MS-TCN.
        pool_type: temporal pooling strategy.
    """

    def __init__(
        self,
        num_class: int,
        full_edge_index: list,
        node_subset: list[int],
        full_num_nodes: int = 25,
        in_channels: int = 6,
        base_channels: int = 32,
        adj_strategy: str = "aagcn",
        dropout: float = 0.5,
        adaptive: bool = True,
        kernel_size: int = 5,
        pool_type: str = "gap",
    ):
        super().__init__()
        if len(node_subset) < 2:
            raise ValueError(
                f"node_subset must have at least 2 nodes, got {len(node_subset)}"
            )
        if len(set(node_subset)) != len(node_subset):
            raise ValueError(
                f"node_subset must contain unique indices, got duplicates: {node_subset}"
            )
        if adj_strategy == "spatial_config":
            # `spatial_config` in skeleton_graph.build_adjacency hardcodes
            # `centroid=1` (Hips) which is typically NOT in a sparsified
            # subset. With Hips out, centripetal/centrifugal subsets lose
            # the centroid → leg-end nodes get INF hop distance and fall
            # out of all subsets, leaving only self-loops. Fail fast rather
            # than silently produce a degenerate adjacency.
            raise NotImplementedError(
                "adj_strategy='spatial_config' is not supported by "
                "CTRGCNSubset_Model: it uses a hardcoded centroid (Hips=1) "
                "that is typically excluded from upper-body subsets, causing "
                "leg nodes to lose all spatial mixing. Use 'aagcn' instead."
            )
        self.full_num_nodes = full_num_nodes
        self.node_subset = list(node_subset)
        self.subset_num_nodes = len(self.node_subset)

        # Build the subset adjacency by filtering+re-indexing the original edges.
        subset_edges = filter_edges_to_subset(full_edge_index, self.node_subset)
        if not subset_edges:
            raise ValueError(
                "filtered edge set is empty; check node_subset has connected joints"
            )

        # Register subset indices as a buffer so .to(device) follows the model.
        self.register_buffer(
            "_subset_idx",
            torch.tensor(self.node_subset, dtype=torch.long),
            persistent=False,
        )

        # Inner CTR-GCN over the subset.
        self.ctrgcn = CTR_GCN_Model(
            num_class=num_class,
            edge_index=subset_edges,
            num_nodes=self.subset_num_nodes,
            in_channels=in_channels,
            base_channels=base_channels,
            adj_strategy=adj_strategy,
            dropout=dropout,
            adaptive=adaptive,
            kernel_size=kernel_size,
            pool_type=pool_type,
            use_arcface_head=False,
        )

    def forward(self, x: torch.Tensor) -> dict:
        # x: (N, C, T, V_full=25) → slice to (N, C, T, V_subset)
        if x.dim() != 4:
            raise ValueError(f"expected input ndim=4 (N,C,T,V), got shape {x.shape}")
        if x.shape[3] != self.full_num_nodes:
            raise ValueError(
                f"expected V={self.full_num_nodes}, got V={x.shape[3]}"
            )
        x_sub = x.index_select(dim=3, index=self._subset_idx)  # (N, C, T, V_subset)
        return self.ctrgcn(x_sub)

    @property
    def output_dim(self) -> int:
        return self.ctrgcn.output_dim

    @property
    def feature_dim(self) -> int:
        # CTR_GCN_Model itself does not expose a feature_dim today (CenterLoss
        # paths are typically routed through Conv1D+Tr models in this repo).
        # Rather than silently return 0 — which would cause CenterLoss to
        # construct a zero-dim Embedding and silently corrupt training — raise
        # so callers get a clear error if they try to wire CenterLoss into
        # CTRGCNSubset_Model. Fix path: add `feature_dim` to CTR_GCN_Model
        # (e.g., return base_channels * 4 = the final-block channel count).
        if hasattr(self.ctrgcn, "feature_dim"):
            return self.ctrgcn.feature_dim
        raise NotImplementedError(
            "CTRGCNSubset_Model.feature_dim requires the wrapped CTR_GCN_Model "
            "to expose a feature_dim property; CenterLoss / DualHead paths are "
            "not currently supported for this wrapper."
        )

    @classmethod
    def from_config(cls, config) -> "CTRGCNSubset_Model":
        return cls(
            num_class=config.model.num_class,
            full_edge_index=config.skeleton.inward_edges,
            node_subset=config.model.node_subset,
            full_num_nodes=getattr(config.skeleton, "num_nodes", 25),
            in_channels=getattr(config.model, "in_channels", 6),
            base_channels=getattr(config.model, "base_channels", 32),
            adj_strategy=getattr(config.model, "adj_strategy", "aagcn"),
            dropout=getattr(config.model, "dropout", 0.5),
            adaptive=getattr(config.model, "adaptive", True),
            kernel_size=getattr(config.model, "kernel_size", 5),
            pool_type=getattr(config.model, "pool_type", "gap"),
        )


from diema.models.registry import register_model  # noqa: E402

register_model("ctrgcn_subset", CTRGCNSubset_Model)
