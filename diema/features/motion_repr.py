"""Rotation representation conversions for motion capture data.

Wraps pybvh-ml's conversion utilities for quaternion↔6D↔Euler transforms.
The 6D continuous rotation representation (Zhou et al., 2019) is the
default for training as it avoids gimbal lock and double-cover ambiguity.

See: diema_challenge_implementation_spec.md §6.1 — target representation
Related: diema/data/dataset.py (uses convert_arrays at load time)
"""

import numpy as np
import pybvh_ml


SUPPORTED_REPRESENTATIONS = ("quaternion", "6d", "euler", "axisangle", "rotmat")

# Channels per joint for each representation
REPR_CHANNELS: dict[str, int] = {
    "quaternion": 4,
    "6d": 6,
    "euler": 3,
    "axisangle": 3,
    "rotmat": 9,
}


def convert_representation(
    joint_data: np.ndarray,
    source_repr: str,
    target_repr: str,
    euler_orders=None,
) -> np.ndarray:
    """Convert joint rotation data between representations.

    Args:
        joint_data: array of shape (F, J, C_source) where C_source depends on source_repr
        source_repr: source representation name
        target_repr: target representation name
        euler_orders: per-joint Euler orders (required when involving Euler)

    Returns:
        Array of shape (F, J, C_target)
    """
    if source_repr == target_repr:
        return joint_data

    if source_repr not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(f"Unsupported source representation: {source_repr}")
    if target_repr not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(f"Unsupported target representation: {target_repr}")

    return pybvh_ml.convert_arrays(
        joint_data, source_repr, target_repr,
        euler_orders=euler_orders,
    )


def get_num_channels(target_repr: str) -> int:
    """Get the number of channels per joint for a given representation.

    Args:
        target_repr: representation name

    Returns:
        Number of channels (e.g., 6 for '6d', 4 for 'quaternion')
    """
    if target_repr not in REPR_CHANNELS:
        raise ValueError(f"Unknown representation: {target_repr}")
    return REPR_CHANNELS[target_repr]
