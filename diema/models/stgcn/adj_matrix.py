"""Graph adjacency matrix for AAGCN-style spatial convolution.

Constructs a 3-layer normalized adjacency matrix:
  [self-links, inward-links, outward-links]
Each layer is L1-normalized by columns.

Ported from: an internal baseline
"""

import torch
import torch.nn.functional as F


class GraphAAGCN:
    """Adjacency matrix for Adaptive Graph Convolutional Network.

    Produces A of shape (3, num_nodes, num_nodes):
      - A[0]: self-links (identity matrix)
      - A[1]: inward-links (normalized child→parent edges)
      - A[2]: outward-links (normalized parent→child edges)

    Args:
        edge_index: list of (child, parent) tuples
        num_nodes: number of joints in the skeleton
    """

    def __init__(self, edge_index: list, num_nodes: int):
        self.num_nodes = num_nodes
        self.edge_index = torch.tensor(edge_index)
        self.A = self._get_spatial_graph(num_nodes)

    def _get_spatial_graph(self, num_nodes: int) -> torch.Tensor:
        self_mat = torch.eye(num_nodes)
        inward_mat = _to_dense_adj(self.edge_index, num_nodes)
        inward_mat_norm = F.normalize(inward_mat, dim=0, p=1)
        outward_mat = inward_mat.transpose(0, 1)
        outward_mat_norm = F.normalize(outward_mat, dim=0, p=1)
        return torch.stack((self_mat, inward_mat_norm, outward_mat_norm))


def _to_dense_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Convert edge list to dense adjacency matrix."""
    if edge_index.numel() > 0:
        effective = int(edge_index.max()) + 1
        assert effective <= num_nodes

    adj = torch.zeros((num_nodes, num_nodes))
    for i, j in edge_index:
        adj[i, j] = 1
    return adj
