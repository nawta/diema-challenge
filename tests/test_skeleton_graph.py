"""Tests for diema/models/skeleton_graph.py."""

import numpy as np
import pytest
import torch

from diema.models.skeleton_graph import (
    DIEMA_BODY_PARTS,
    DIEMA_INWARD_EDGES,
    DIEMA_JOINT_NAMES,
    DIEMA_LR_PAIRS,
    DiemaSkeleton,
    build_adjacency,
    build_part_indices,
    diema_skeleton,
    get_hop_distance,
)


class TestDiemaSkeleton:
    def test_default_construction(self):
        sk = DiemaSkeleton()
        assert sk.num_nodes == 25
        assert len(sk.joint_names) == 25
        assert sk.joint_names[0] == "virtual_root"
        assert sk.joint_names[1] == "Hips"
        assert sk.joint_names[8] == "Head"

    def test_body_parts_cover_all_nodes(self):
        sk = DiemaSkeleton()
        all_nodes = sorted(n for nodes in sk.body_parts.values() for n in nodes)
        assert all_nodes == list(range(25))

    def test_body_parts_no_overlap(self):
        sk = DiemaSkeleton()
        all_nodes = []
        for nodes in sk.body_parts.values():
            all_nodes.extend(nodes)
        assert len(all_nodes) == len(set(all_nodes))

    def test_body_part_count(self):
        sk = DiemaSkeleton()
        assert len(sk.body_parts) == 6
        assert "torso" in sk.body_parts
        assert "head" in sk.body_parts
        assert "r_arm" in sk.body_parts
        assert "l_arm" in sk.body_parts
        assert "r_leg" in sk.body_parts
        assert "l_leg" in sk.body_parts

    def test_parent_indices(self):
        sk = DiemaSkeleton()
        parents = sk.parent_indices
        assert len(parents) == 25
        # Hips (1) is the parent of virtual_root (0) per inward_edges
        assert parents[0] == 1
        # Spine (2) → Hips (1)
        assert parents[2] == 1

    def test_lr_pairs(self):
        sk = DiemaSkeleton()
        # Each pair must reference valid nodes
        for a, b in sk.lr_pairs:
            assert 0 <= a < 25
            assert 0 <= b < 25

    def test_get_body_part_of_node(self):
        sk = DiemaSkeleton()
        assert sk.get_body_part_of_node(0) == "torso"
        assert sk.get_body_part_of_node(8) == "head"
        assert sk.get_body_part_of_node(12) == "r_arm"
        assert sk.get_body_part_of_node(16) == "l_arm"
        assert sk.get_body_part_of_node(20) == "r_leg"
        assert sk.get_body_part_of_node(24) == "l_leg"


class TestHopDistance:
    def test_self_distance_zero(self):
        dist = get_hop_distance(25, DIEMA_INWARD_EDGES)
        for i in range(25):
            assert dist[i, i] == 0

    def test_symmetric(self):
        dist = get_hop_distance(25, DIEMA_INWARD_EDGES)
        assert (dist == dist.T).all()

    def test_one_hop_for_edges(self):
        dist = get_hop_distance(25, DIEMA_INWARD_EDGES)
        for a, b in DIEMA_INWARD_EDGES:
            assert dist[a, b] == 1

    def test_max_hops_cap(self):
        dist = get_hop_distance(25, DIEMA_INWARD_EDGES, max_hops=2)
        # All finite values must be <= 2
        finite = dist[dist < np.iinfo(np.int32).max]
        assert (finite <= 2).all()


class TestBuildAdjacency:
    def test_aagcn_shape(self):
        A = build_adjacency(DIEMA_INWARD_EDGES, 25, strategy="aagcn")
        assert A.shape == (3, 25, 25)
        assert A.dtype == torch.float32

    def test_aagcn_self_loop(self):
        A = build_adjacency(DIEMA_INWARD_EDGES, 25, strategy="aagcn")
        # First subset is identity
        assert torch.allclose(A[0], torch.eye(25))

    def test_aagcn_inward_outward_transpose(self):
        A = build_adjacency(DIEMA_INWARD_EDGES, 25, strategy="aagcn")
        # Before normalization, outward = inward.T
        # After column-norm, the relation only holds if columns sum match.
        # We just check that both have nonzero entries on edges
        for a, b in DIEMA_INWARD_EDGES:
            assert A[1, a, b] > 0  # inward
            assert A[2, b, a] > 0  # outward

    def test_distance_shape(self):
        A = build_adjacency(DIEMA_INWARD_EDGES, 25, strategy="distance")
        assert A.shape == (1, 25, 25)

    def test_spatial_config_shape(self):
        A = build_adjacency(DIEMA_INWARD_EDGES, 25, strategy="spatial_config")
        assert A.shape == (3, 25, 25)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            build_adjacency(DIEMA_INWARD_EDGES, 25, strategy="bogus")


class TestBuildPartIndices:
    def test_default_parts(self):
        idx = build_part_indices()
        assert list(idx.keys()) == list(DIEMA_BODY_PARTS.keys())
        for name, tensor in idx.items():
            assert tensor.dtype == torch.long
            assert tensor.tolist() == DIEMA_BODY_PARTS[name]

    def test_covers_all_nodes(self):
        idx = build_part_indices()
        all_nodes = sorted(int(v) for t in idx.values() for v in t)
        assert all_nodes == list(range(25))

    def test_custom_parts(self):
        custom = {"upper": [0, 1, 2], "lower": [3, 4]}
        idx = build_part_indices(custom)
        assert idx["upper"].tolist() == [0, 1, 2]
        assert idx["lower"].tolist() == [3, 4]


class TestSkeletonInvariants:
    def test_inward_edges_count(self):
        # 24 edges connect the 25 nodes (tree topology)
        assert len(DIEMA_INWARD_EDGES) == 24

    def test_singleton_module_consistent(self):
        assert diema_skeleton.num_nodes == 25
        assert diema_skeleton.joint_names == DIEMA_JOINT_NAMES

    def test_lr_pairs_within_arms_and_legs(self):
        sk = DiemaSkeleton()
        for left, right in sk.lr_pairs:
            l_part = sk.get_body_part_of_node(left)
            r_part = sk.get_body_part_of_node(right)
            assert l_part.startswith("l_") and r_part.startswith("r_"), \
                f"({left},{right}) parts: {l_part}, {r_part}"
            assert l_part[2:] == r_part[2:]
