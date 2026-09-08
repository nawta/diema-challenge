"""Skeleton normalization utilities.

See: diema_challenge_implementation_spec.md §6.1 — root centering
Related: diema/data/preprocess.py (applies center_root during preprocessing)
"""

import numpy as np


def center_root(root_positions: np.ndarray) -> np.ndarray:
    """Center root positions by subtracting the mean position.

    Applied during preprocessing to remove global translation.
    After centering, the skeleton is relative to the average root position.

    Args:
        root_positions: array of shape (F, 3) — root position per frame

    Returns:
        Centered root positions of shape (F, 3)
    """
    mean_pos = root_positions.mean(axis=0, keepdims=True)
    return root_positions - mean_pos


def compute_bone_lengths(joint_positions: np.ndarray, parent_indices: list[int]) -> np.ndarray:
    """Compute bone lengths from joint positions.

    Args:
        joint_positions: array of shape (J, 3) — T-pose joint positions
        parent_indices: list of parent joint indices. Root has parent -1.

    Returns:
        Bone lengths array of shape (J,). Root bone length is 0.
    """
    num_joints = len(parent_indices)
    lengths = np.zeros(num_joints)
    for j in range(num_joints):
        p = parent_indices[j]
        if p >= 0:
            lengths[j] = np.linalg.norm(joint_positions[j] - joint_positions[p])
    return lengths
