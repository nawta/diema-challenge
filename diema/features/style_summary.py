"""Lightweight style descriptor for MoE routing.

Computes a 21-dimensional style summary from raw rotation_6d input
``(N, 6, T, 25)`` **without gradient flow** (``torch.no_grad()``).
The summary captures per-body-part speed statistics and asymmetry,
which correlate with performer identity but NOT emotion class.

The style summary is used as a routing signal for the Style-Conditioned
MoE head — style is never a *prediction target*, only a *routing signal*.

Related: diema/models/conv1d_transformer/moe_head_model.py,
         diema/models/skeleton_graph.py::DIEMA_BODY_PARTS
"""

from __future__ import annotations

import torch

from diema.models.skeleton_graph import DIEMA_BODY_PARTS, build_part_indices

# Cache part indices on first call
_PART_INDICES: dict[str, torch.Tensor] | None = None
STYLE_DIM = 21


def _get_part_indices(device: torch.device) -> dict[str, torch.Tensor]:
    global _PART_INDICES
    if _PART_INDICES is None or next(iter(_PART_INDICES.values())).device != device:
        _PART_INDICES = {
            k: v.to(device) for k, v in build_part_indices().items()
        }
    return _PART_INDICES


@torch.no_grad()
def compute_style_summary(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute a 21-dim style descriptor from raw skeleton input.

    Components (in order):
        0-5:   per-part mean speed (6 body parts)
        6-11:  per-part speed std (6 body parts)
        12-14: root angular velocity mean (3 dims: rotation_6d channels 0-2 of virtual root)
        15-17: root angular velocity std  (3 dims)
        18:    arm/leg energy ratio
        19:    L-R arm asymmetry
        20:    L-R leg asymmetry

    Args:
        x: input tensor ``(N, C, T, V)`` where C=6 (rotation_6d),
            T=clip_length, V=25.
        eps: numerical floor for divisions.

    Returns:
        Style descriptor ``(N, 21)``, detached (no gradient).
    """
    if x.ndim != 4:
        raise ValueError(f"Expected (N, C, T, V), got {tuple(x.shape)}")
    N, C, T, V = x.shape

    parts = _get_part_indices(x.device)
    part_names = sorted(parts.keys())  # canonical order

    # Frame-to-frame differences: (N, C, T-1, V)
    if T < 2:
        return torch.zeros(N, STYLE_DIM, device=x.device, dtype=x.dtype)
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]  # (N, C, T-1, V)

    # Per-joint speed: L2 norm over C dim → (N, T-1, V)
    speed = dx.norm(dim=1)  # (N, T-1, V)

    # Per-part mean speed and std
    part_mean_speeds = []
    part_std_speeds = []
    part_mean_dict = {}
    for name in part_names:
        idx = parts[name]  # (|part|,)
        part_speed = speed[:, :, idx]  # (N, T-1, |part|)
        # Mean over time and joints within the part
        mean_s = part_speed.mean(dim=(1, 2))  # (N,)
        # unbiased=False to avoid NaN when T-1=1 (single frame diff)
        std_s = part_speed.std(dim=(1, 2), unbiased=False).clamp_min(eps)  # (N,)
        part_mean_speeds.append(mean_s)
        part_std_speeds.append(std_s)
        part_mean_dict[name] = mean_s

    # Root velocity: node 0, channels 0:3 (translational component)
    root_dx = dx[:, :3, :, 0]  # (N, 3, T-1) — first 3 channels of virtual root
    root_mean = root_dx.mean(dim=2)  # (N, 3)
    root_std = root_dx.std(dim=2, unbiased=False).clamp_min(eps)  # (N, 3)

    # Arm/leg energy ratio
    arm_energy = part_mean_dict["r_arm"] + part_mean_dict["l_arm"]
    leg_energy = part_mean_dict["r_leg"] + part_mean_dict["l_leg"]
    arm_leg_ratio = arm_energy / (leg_energy + eps)  # (N,)

    # L-R asymmetry: |left - right| / (max(left, right) + eps)
    lr_arm_asym = (
        (part_mean_dict["l_arm"] - part_mean_dict["r_arm"]).abs()
        / (torch.max(part_mean_dict["l_arm"], part_mean_dict["r_arm"]) + eps)
    )
    lr_leg_asym = (
        (part_mean_dict["l_leg"] - part_mean_dict["r_leg"]).abs()
        / (torch.max(part_mean_dict["l_leg"], part_mean_dict["r_leg"]) + eps)
    )

    # Concatenate: 6 + 6 + 3 + 3 + 1 + 1 + 1 = 21
    summary = torch.stack(
        part_mean_speeds + part_std_speeds
        + [root_mean[:, 0], root_mean[:, 1], root_mean[:, 2]]
        + [root_std[:, 0], root_std[:, 1], root_std[:, 2]]
        + [arm_leg_ratio, lr_arm_asym, lr_leg_asym],
        dim=1,
    )  # (N, 21)
    return summary.detach()
