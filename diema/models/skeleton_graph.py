"""Skeleton graph utilities for DIEM-A 25-node STGCN/CTR-GCN/SkateFormer models.

Provides:
  - DiemaSkeleton: canonical 25-node skeleton with joint names, edges, body parts
  - build_adjacency: multiple adjacency strategies (aagcn, distance, spatial_config)
  - get_hop_distance: BFS-based hop distance matrix

CTV node 0 is a virtual root (root_pos), nodes 1-24 are BVH joints.
The canonical joint name table is the single source of truth for index→name mapping.

See:, docs/decisions.md
Related: diema/data/preprocess.py (CTV layout), diema/models/stgcn/adj_matrix.py
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


# Canonical CTV node names (V=25). Single source of truth for index→name mapping.
# Node 0 is the virtual root containing root_pos. Nodes 1-24 follow BVH joint order.
DIEMA_JOINT_NAMES: list[str] = [
    "virtual_root",   # 0
    "Hips",           # 1
    "Spine",          # 2
    "Spine1",         # 3
    "Spine2",         # 4
    "Spine3",         # 5
    "Neck",           # 6
    "Neck1",          # 7
    "Head",           # 8
    "RightShoulder",  # 9
    "RightArm",       # 10
    "RightForeArm",   # 11
    "RightHand",      # 12
    "LeftShoulder",   # 13
    "LeftArm",        # 14
    "LeftForeArm",    # 15
    "LeftHand",       # 16
    "RightUpLeg",     # 17
    "RightLeg",       # 18
    "RightFoot",      # 19
    "RightToeBase",   # 20
    "LeftUpLeg",      # 21
    "LeftLeg",        # 22
    "LeftFoot",       # 23
    "LeftToeBase",    # 24
]


# Inward edges (child → parent) in CTV node index space. From benchmark config.
DIEMA_INWARD_EDGES: list[tuple[int, int]] = [
    (0, 1),                                       # virtual_root → Hips
    (2, 1), (3, 2), (4, 3), (5, 4),               # spine chain
    (6, 5), (7, 6), (8, 7),                       # neck/head
    (9, 5), (10, 9), (11, 10), (12, 11),          # right arm
    (13, 5), (14, 13), (15, 14), (16, 15),        # left arm
    (17, 1), (18, 17), (19, 18), (20, 19),        # right leg
    (21, 1), (22, 21), (23, 22), (24, 23),        # left leg
]


# Body-part groups in CTV node index space (6 groups, 25 nodes total)
DIEMA_BODY_PARTS: dict[str, list[int]] = {
    "torso":  [0, 1, 2, 3, 4],          # virtual_root + Hips + Spine, Spine1, Spine2
    "head":   [5, 6, 7, 8],             # Spine3, Neck, Neck1, Head
    "r_arm":  [9, 10, 11, 12],          # R Shoulder → R Hand
    "l_arm":  [13, 14, 15, 16],         # L Shoulder → L Hand
    "r_leg":  [17, 18, 19, 20],         # R UpLeg → R ToeBase
    "l_leg":  [21, 22, 23, 24],         # L UpLeg → L ToeBase
}


# exp074: upper-body 3-part split, designed around joint
# masking findings on exp034 regionaware_convtr_a00 (10-fold mean f1_drop):
#
# Rank table (top-25 from docs/analysis/joint_masking/exp034_*_joint_importance.csv):
#
#   Rank | joint           | f1_drop pp | included | part
#   -----+-----------------+-----------+----------+--------------
#     1  | Spine3 (5)      |   25.88   | ✓        | head_chain
#     2  | Neck1 (7)       |   25.76   | ✓        | head_chain
#     3  | Neck (6)        |   25.72   | ✓        | head_chain
#     4  | Head (8)        |   25.70   | ✓        | head_chain
#   ----- 4 dominant tier (-25.7 ± 0.2 pp), all in head_chain ------
#     5  | RightLeg (18)   |   10.23   | ✓        | lower_context
#     6  | LeftLeg (22)    |    8.58   | ✓        | lower_context
#     7  | RightUpLeg (17) |    8.36   | ✓        | lower_context
#     8  | RightToeBase (20)|   7.94   | ✗ excluded for L/R symmetry
#     9  | RightArm (10)   |    7.65   | ✓        | arms
#    10  | LeftArm (14)    |    7.60   | ✓        | arms
#    11  | RightFoot (19)  |    7.31   | ✗ excluded for L/R symmetry
#    12  | LeftToeBase (24)|    6.81   | ✗ excluded for L/R symmetry
#    13  | LeftUpLeg (21)  |    6.77   | ✓ (below rank-10 cut, kept for symmetry)
#    14  | RightForeArm    |    6.43   | ✓        | arms
#    15  | LeftFoot (23)   |    6.25   | ✗ excluded for L/R symmetry
#    16  | LeftForeArm     |    5.54   | ✓        | arms
#    17  | RightShoulder   |    5.04   | ✓        | arms
#    18  | RightHand       |    5.00   | ✓        | arms
#    19  | LeftHand        |    3.86   | ✓        | arms
#    20  | LeftShoulder    |    3.75   | ✓        | arms
#   ----- 21–25: near-noise tier ---------------------------------
#    21  | Hips (1)        |    1.90   | ✗
#    22  | Spine2 (4)      |    1.75   | ✗
#    23  | Spine1 (3)      |    1.72   | ✗
#    24  | Spine (2)       |    1.72   | ✗
#    25  | virtual_root (0)|    0.80   | ✗
#
# Selection notes:
# - head_chain (4 nodes): pure-importance — the four ranks #1-#4 tied at
#   -25.7 ± 0.2 pp.
# - lower_context (4 nodes): R/L UpLeg + R/L Leg = 4 leg joints with L/R
#   symmetry. Note: legs (ranks #5-#7) are actually MORE important per joint
#   than arms (ranks #9-#20), but we keep this part small (4 nodes) and
#   symmetric. RightToeBase (rank #8, 7.94 pp) is excluded for L/R symmetry
#   (LeftToeBase = rank #12). LeftUpLeg (rank #13, 6.77 pp) is *included*
#   even though it falls below rank-10, also for symmetry.
# - arms (8 nodes): all 8 R/L arm joints. Per-joint importance (5-7 pp) is
#   below legs but the larger node count gives the part a comparable total
#   contribution.
# - Excluded 9 nodes: 5 mid-spine cluster (#21-#25, ranks #21-#25 at
#   1.7-0.8 pp = the actual near-noise tier) + 4 R/L Foot+ToeBase (mid-rank
#   6-8 pp, dropped for L/R symmetry of the lower_context part).
#
# **Important**: the "near-noise tier" is the mid-spine cluster, NOT
# feet/toes. Feet/toes are mid-importance (6-8 pp) and are excluded for
# part-symmetry, not for being near-noise.
DIEMA_UPPER_BODY_3PARTS: dict[str, list[int]] = {
    "head_chain":     [5, 6, 7, 8],                          # Spine3, Neck, Neck1, Head
    "arms":           [9, 10, 11, 12, 13, 14, 15, 16],       # R/L Shoulder→Hand
    "lower_context":  [17, 18, 21, 22],                      # R/L UpLeg, R/L Leg (no foot/toe)
}


# Left-right joint pairs.
#
# Two index conventions are needed:
# - **BVH indices** (0-23, no virtual root): used by `pybvh_ml.mirror_quaternions`
#   which operates on the raw 24-joint quaternion array.
# - **CTV indices** (0-24, with virtual root at 0): used by SkateFormer body-part
#   partitioning and any reasoning on the 25-node packed tensor.
#
# Mapping: CTV index N corresponds to BVH index N-1 (for N >= 1).

DIEMA_LR_PAIRS_BVH: list[tuple[int, int]] = [
    (12, 8), (13, 9), (14, 10), (15, 11),     # arms (L↔R Shoulder/Arm/ForeArm/Hand)
    (20, 16), (21, 17), (22, 18), (23, 19),   # legs (L↔R UpLeg/Leg/Foot/ToeBase)
]

DIEMA_LR_PAIRS_CTV: list[tuple[int, int]] = [
    (13, 9), (14, 10), (15, 11), (16, 12),    # arms in CTV
    (21, 17), (22, 18), (23, 19), (24, 20),   # legs in CTV
]

# Backward-compat alias: default is BVH (matches benchmark + pybvh_ml augmentation API).
DIEMA_LR_PAIRS = DIEMA_LR_PAIRS_BVH


@dataclass
class DiemaSkeleton:
    """Canonical 25-node DIEM-A skeleton definition.

    Attributes:
        num_nodes: 25 (1 virtual root + 24 BVH joints)
        joint_names: canonical name per CTV node index
        inward_edges: (child, parent) tuples in CTV index space
        body_parts: 6 anatomical groups
        lr_pairs: left-right pairs for mirror augmentation
        parent_indices: parent of each node (-1 for root)
    """
    num_nodes: int = 25
    joint_names: list[str] = field(default_factory=lambda: list(DIEMA_JOINT_NAMES))
    inward_edges: list[tuple[int, int]] = field(default_factory=lambda: list(DIEMA_INWARD_EDGES))
    body_parts: dict[str, list[int]] = field(default_factory=lambda: dict(DIEMA_BODY_PARTS))
    lr_pairs: list[tuple[int, int]] = field(default_factory=lambda: list(DIEMA_LR_PAIRS_CTV))

    @property
    def parent_indices(self) -> list[int]:
        """Parent index for each node (-1 if root)."""
        parents = [-1] * self.num_nodes
        for child, parent in self.inward_edges:
            parents[child] = parent
        return parents

    def get_body_part_of_node(self, node: int) -> str:
        for part_name, nodes in self.body_parts.items():
            if node in nodes:
                return part_name
        raise ValueError(f"Node {node} not in any body part")

    def __post_init__(self):
        # Validate consistency
        assert len(self.joint_names) == self.num_nodes, \
            f"joint_names length {len(self.joint_names)} != num_nodes {self.num_nodes}"
        all_part_nodes = sorted(n for nodes in self.body_parts.values() for n in nodes)
        assert all_part_nodes == list(range(self.num_nodes)), \
            f"body_parts must cover all {self.num_nodes} nodes, got {all_part_nodes}"


def get_hop_distance(num_nodes: int, edges: list[tuple[int, int]], max_hops: int | None = None) -> np.ndarray:
    """Compute pairwise hop distance matrix via BFS.

    Args:
        num_nodes: number of nodes
        edges: list of (a, b) tuples (treated as undirected)
        max_hops: cap distances at this value (None = no cap)

    Returns:
        (num_nodes, num_nodes) int array. Unreachable pairs are np.iinfo(int).max
    """
    # Build adjacency list
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    INF = np.iinfo(np.int32).max
    dist = np.full((num_nodes, num_nodes), INF, dtype=np.int32)

    for src in range(num_nodes):
        dist[src, src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[src, v] == INF:
                    dist[src, v] = dist[src, u] + 1
                    if max_hops is None or dist[src, v] < max_hops:
                        queue.append(v)

    if max_hops is not None:
        dist = np.where(dist > max_hops, INF, dist)

    return dist


def build_adjacency(
    edges: list[tuple[int, int]],
    num_nodes: int,
    strategy: str = "aagcn",
) -> torch.Tensor:
    """Build a graph adjacency matrix for spatial GCN layers.

    Args:
        edges: (child, parent) tuples (interpreted as inward links)
        num_nodes: number of nodes
        strategy: one of
            - "aagcn":  3-subset (self, inward, outward), L1-normalized columns. Same as STGCN++ benchmark.
            - "distance":  hop-based, single-subset, normalized
            - "spatial_config":  3-subset (root, centripetal, centrifugal), CTR-GCN style

    Returns:
        Tensor of shape (K, num_nodes, num_nodes) where K is the number of subsets.
    """
    if strategy == "aagcn":
        return _build_aagcn(edges, num_nodes)
    elif strategy == "distance":
        return _build_distance(edges, num_nodes)
    elif strategy == "spatial_config":
        return _build_spatial_config(edges, num_nodes)
    else:
        raise ValueError(f"Unknown adjacency strategy: {strategy}")


def _build_aagcn(edges: list[tuple[int, int]], num_nodes: int) -> torch.Tensor:
    """3-subset adjacency: [self, inward, outward], L1-normalized by columns."""
    self_mat = torch.eye(num_nodes, dtype=torch.float32)
    inward = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for a, b in edges:
        inward[a, b] = 1.0
    inward_norm = F.normalize(inward, p=1, dim=0)
    outward = inward.transpose(0, 1)
    outward_norm = F.normalize(outward, p=1, dim=0)
    return torch.stack([self_mat, inward_norm, outward_norm], dim=0)


def _build_distance(edges: list[tuple[int, int]], num_nodes: int) -> torch.Tensor:
    """Single-subset adjacency from hop distances (1-hop reachability + identity)."""
    dist = get_hop_distance(num_nodes, edges, max_hops=1)
    A = (dist <= 1).astype(np.float32)
    A_t = torch.from_numpy(A)
    A_norm = F.normalize(A_t, p=1, dim=0)
    return A_norm.unsqueeze(0)


def _build_spatial_config(edges: list[tuple[int, int]], num_nodes: int) -> torch.Tensor:
    """3-subset CTR-GCN-style: [root_loop, centripetal, centrifugal] using hop distance.

    centripetal: edge points toward node closer to skeleton centroid (parent)
    centrifugal: edge points toward node farther from centroid (child)
    """
    # Use node 1 (Hips) as the centroid reference (closest to body center)
    centroid = 1
    dist = get_hop_distance(num_nodes, edges, max_hops=None)
    dist_to_center = dist[centroid]

    self_mat = torch.eye(num_nodes, dtype=torch.float32)
    centripetal = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    centrifugal = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

    # For each undirected edge, classify by hop distance to centroid
    edge_set = set()
    for a, b in edges:
        edge_set.add((a, b))
        edge_set.add((b, a))
    for a, b in edge_set:
        if dist_to_center[a] > dist_to_center[b]:
            centripetal[a, b] = 1.0   # a is farther → moving inward
        elif dist_to_center[a] < dist_to_center[b]:
            centrifugal[a, b] = 1.0   # a is closer → moving outward

    centripetal_norm = F.normalize(centripetal, p=1, dim=0)
    centrifugal_norm = F.normalize(centrifugal, p=1, dim=0)
    return torch.stack([self_mat, centripetal_norm, centrifugal_norm], dim=0)


def build_part_indices(
    body_parts: dict[str, list[int]] | None = None,
) -> dict[str, torch.Tensor]:
    """Return per-body-part long tensors of CTV node indices.

    Used by Region-Aware models to gather per-part joint slices
    from a `(N, C, T, V=25)` tensor. Tensor ordering matches the insertion
    order of `body_parts` so downstream code can rely on stable part names.

    Args:
        body_parts: mapping of part name → list of CTV node indices.
            Defaults to the canonical `DIEMA_BODY_PARTS` (6 parts, 25 nodes).

    Returns:
        dict of `{part_name: LongTensor(|part|)}` with each tensor holding the
        CTV node indices belonging to that part.
    """
    parts = DIEMA_BODY_PARTS if body_parts is None else body_parts
    return {name: torch.tensor(nodes, dtype=torch.long) for name, nodes in parts.items()}


# Module-level singleton for convenience
diema_skeleton = DiemaSkeleton()
