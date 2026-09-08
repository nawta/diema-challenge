"""Prototype matching module for ProtoGCN.

Maintains a learnable bank of motion prototypes per body part, computes
similarity between the encoder features and prototypes, and aggregates
per-body-part responses into a final feature vector for classification.

Design (simplified faithful adaptation):
  - Per body part p (1 to P), bank shape (num_proto, feat_dim)
  - For each input feature (B, feat_dim) at part p, similarity
    sims[b, k] = cos(feat[b], proto[p, k])
  - Output per part: weighted sum prototypes[p].T @ softmax(sims, dim=k) → (B, feat_dim)
  - Concatenate across parts → (B, P * feat_dim)

This gives the classifier a body-part-aware, prototype-anchored representation
that highlights the most discriminative micro-motion signatures.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    """Learnable prototype bank for one body part.

    Args:
        num_proto: number of prototypes (K)
        feat_dim: feature dimension per prototype (D)
    """

    def __init__(self, num_proto: int, feat_dim: int):
        super().__init__()
        self.num_proto = num_proto
        self.feat_dim = feat_dim
        # Initialize on unit sphere
        proto = torch.randn(num_proto, feat_dim)
        proto = F.normalize(proto, dim=-1)
        self.prototypes = nn.Parameter(proto, requires_grad=True)
        self.tau = nn.Parameter(torch.tensor(0.1))  # temperature

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, D) feature vector for one body part
        Returns:
            (B, D) reconstructed feature as a soft mixture of prototypes
        """
        proto = F.normalize(self.prototypes, dim=-1)              # (K, D)
        feat_n = F.normalize(feat, dim=-1)                         # (B, D)
        sims = feat_n @ proto.t()                                  # (B, K)
        weights = F.softmax(sims / self.tau.clamp(min=1e-3), dim=-1)
        recon = weights @ proto                                    # (B, D)
        return recon


class PerBodyPartPrototype(nn.Module):
    """Wraps multiple PrototypeBanks (one per body part) and aggregates them.

    Args:
        body_parts: list of node-index lists. Lengths can vary.
        feat_dim: encoder feature dimension per node (e.g. 256 for STGCN++ final)
        num_proto: number of prototypes per body part
    """

    def __init__(
        self,
        body_parts: Sequence[Sequence[int]],
        feat_dim: int,
        num_proto: int = 16,
    ):
        super().__init__()
        self.body_parts = [list(p) for p in body_parts]
        self.feat_dim = feat_dim
        self.banks = nn.ModuleList([
            PrototypeBank(num_proto=num_proto, feat_dim=feat_dim)
            for _ in self.body_parts
        ])

    def forward(self, feat_per_node: torch.Tensor) -> torch.Tensor:
        """Aggregate per-body-part prototype responses.

        Args:
            feat_per_node: (B, D, V) or (B, D, T, V). We pool over T and
                average over the part nodes to get one (B, D) per part.

        Returns:
            (B, P * D) concatenated body-part features.
        """
        if feat_per_node.dim() == 4:
            # (B, D, T, V) → average over T
            feat_per_node = feat_per_node.mean(dim=2)
        # feat_per_node: (B, D, V)

        outputs = []
        for part_nodes, bank in zip(self.body_parts, self.banks):
            idx = torch.tensor(part_nodes, dtype=torch.long, device=feat_per_node.device)
            part_feats = feat_per_node.index_select(2, idx)   # (B, D, |part|)
            pooled = part_feats.mean(dim=2)                   # (B, D)
            recon = bank(pooled)                              # (B, D) prototype mixture
            outputs.append(recon)

        return torch.cat(outputs, dim=1)                      # (B, P*D)
