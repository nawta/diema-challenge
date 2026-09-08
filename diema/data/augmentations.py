"""Custom augmentations applied to (C, T, V) skeleton tensors.

These augmentations are run **after** the existing pybvh_ml quaternion-space
augmentation pipeline and operate on the already-packed CTV tensor. They are
called from :class:`diema.data.dataset.MotionDataset` per sample (train mode).

 (ASL Signs 8th, Darragh):
    - sequence_cutout_per_body_part: zero-mask N segments of length L on
      each body-part with probability p.
    - time_shift_delta_features: stack lag-difference channels.

 (Domain Generalization Package):
    - body_scale_jitter: multiply positional channels by a per-sample
      uniform scale, simulating body-size variation.

Related: diema/data/dataset.py, diema/models/skeleton_graph.py

"""

from __future__ import annotations

import numpy as np

from diema.models.skeleton_graph import DIEMA_BODY_PARTS


def sequence_cutout_per_body_part(
    data_ctv: np.ndarray,
    rng: np.random.Generator,
    body_parts: dict[str, list[int]] | None = None,
    p: float = 0.4,
    num_segments: int = 5,
    segment_ratio: float = 0.15,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Zero/NaN out ``num_segments`` time windows for each body part.

    For every body part independently:
      1. Draw u ~ Uniform(0, 1). If ``u >= p``, skip this part.
      2. Sample ``num_segments`` random start positions in [0, T - L].
      3. Set ``data_ctv[:, start:start+L, nodes]`` to ``fill_value`` for each
         start, where ``L = ceil(segment_ratio * T)`` and ``nodes`` are the
         CTV node indices that belong to this part.

    The default settings (p=0.4, 5 segments, 0.15*L window) reproduce the
    Darragh 8th-place ASL Signs writeup. Reference:
    https://www.kaggle.com/competitions/asl-signs/writeups/darragh-8th-place-solution-close-but-no-cigar

    Reproducibility note: the ``rng.random()`` draws are consumed in the
    iteration order of ``body_parts.values()``. Since Python 3.7 dict
    iteration preserves insertion order, this is deterministic as long as
    you always pass the same dict literal. If you pass different dict orders
    across runs, the RNG state downstream will diverge — flag this if you
    build ``body_parts`` dynamically.

    Args:
        data_ctv: (C, T, V) ndarray, modified in place AND returned.
        rng: numpy Generator for reproducibility.
        body_parts: name → list of CTV node indices. Defaults to
            DIEMA_BODY_PARTS (6 anatomical groups).
        p: per-part probability of applying any segments.
        num_segments: number of disjoint windows to mask per chosen part.
            Must be >= 0 (0 disables masking).
        segment_ratio: window length as a fraction of T. Must be > 0.
        fill_value: value used to fill the masked windows (default 0.0).

    Returns:
        The modified ndarray (same object, for chaining convenience).
    """
    if body_parts is None:
        body_parts = DIEMA_BODY_PARTS
    if data_ctv.ndim != 3:
        raise ValueError(f"data_ctv must be (C, T, V), got shape {data_ctv.shape}")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    if num_segments < 0:
        raise ValueError(f"num_segments must be >= 0, got {num_segments}")
    if segment_ratio <= 0.0:
        raise ValueError(f"segment_ratio must be > 0, got {segment_ratio}")
    if num_segments == 0 or p == 0.0:
        return data_ctv
    _, T, _ = data_ctv.shape
    L = max(int(round(segment_ratio * T)), 1)
    L = min(L, T)
    max_start = max(T - L, 0)

    for nodes in body_parts.values():
        if rng.random() >= p:
            continue
        for _ in range(num_segments):
            start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
            end = start + L
            for n in nodes:
                data_ctv[:, start:end, n] = fill_value
    return data_ctv


def body_scale_jitter(
    data_ctv: np.ndarray,
    rng: np.random.Generator,
    scale_range: tuple[float, float] = (0.9, 1.1),
    pos_channels: tuple[int, ...] = (0, 1, 2),
    pos_nodes: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Multiply positional channels by a per-sample uniform scale.

    A single random scale ``s ~ Uniform(scale_range)`` is sampled and
    applied to the positional channels of the specified nodes. The scale
    is isotropic (same factor across x/y/z) so the skeleton geometry is
    preserved up to a global rescale.

    Intended for position-based streams (``joint_pos`` / ``bone`` /
    ``joint_motion`` / ``bone_motion``). The caller should pass
    ``pos_nodes=tuple(range(1, 25))`` — i.e., joints 1..24 — so that the
    *global translation* stored at virtual_root (CTV node 0) is not
    touched. Scaling root translation would be "root translation jitter"
    which is a different augmentation from body-scale jitter.

    ``rotation_6d`` stream: the caller should **skip this function
    entirely**. Rotations are scale-invariant, and virtual_root only
    carries global translation in that stream — neither changes body
    size. See ``diema/data/dataset.py`` section 4d for the current
    dataset-level guard.

    Args:
        data_ctv: (C, T, V) ndarray, modified in place AND returned.
        rng: numpy Generator for reproducibility.
        scale_range: (lo, hi) tuple. ``lo`` must be > 0 and ``hi >= lo``.
        pos_channels: indices of channels that hold positional/vector data.
            Default (0, 1, 2) which is the standard x/y/z convention.
        pos_nodes: tuple of CTV node indices to scale. ``None`` scales all
            nodes (which is almost never what you want — see the stream
            guidance above).

    Returns:
        The modified ndarray (same object, for chaining convenience).
    """
    if data_ctv.ndim != 3:
        raise ValueError(f"data_ctv must be (C, T, V), got shape {data_ctv.shape}")
    lo, hi = scale_range
    if not (0.0 < lo and lo <= hi):
        raise ValueError(f"scale_range must satisfy 0 < lo <= hi, got {scale_range}")
    C = data_ctv.shape[0]
    for c in pos_channels:
        if not (0 <= c < C):
            raise ValueError(f"pos_channel {c} out of range for C={C}")
    s = float(rng.uniform(lo, hi))
    if pos_nodes is None:
        for c in pos_channels:
            data_ctv[c, :, :] *= s
    else:
        for c in pos_channels:
            for n in pos_nodes:
                data_ctv[c, :, n] *= s
    return data_ctv


def time_shift_delta_features(
    data_ctv: np.ndarray,
    lags: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16),
    fill_mode: str = "edge",
) -> np.ndarray:
    """Stack lag-difference features along the channel axis.

    For each lag ``k`` in ``lags``, computes ``x[t] - x[t-k]`` for
    ``t >= k``. Returns a ``(C * (1 + len(lags)), T, V)`` tensor where the
    first ``C`` channels are the original signal and each subsequent block
    of ``C`` channels is one lag difference. When ``k >= T`` the entire lag
    block is filled with ``0`` (there are no valid differences at all).

     (Darragh 8th place ASL Signs): "time shift delta features"
    using lags = (1, 2, 3, 4, 6, 8, 12, 16). Darragh emphasises that head
    frames must be filled with a non-zero value so the first frames are
    not degenerate.

    Fill modes (applied only to the head, frames ``[0, k)``):
      - ``"edge"`` (default): copy the first valid delta ``delta[k]`` into
        all head frames. Equivalent to "nearest-neighbour left fill" rather
        than recursive pandas-style forward fill, but shares the same
        motivation — give the head frames a sensible, non-zero value.
      - ``"zero"``: leave the head zeros untouched.

    Args:
        data_ctv: (C, T, V) ndarray
        lags: tuple of positive integer frame offsets
        fill_mode: 'edge' (first-valid-delta copy) or 'zero'

    Returns:
        (C * (1 + len(lags)), T, V) ndarray
    """
    if data_ctv.ndim != 3:
        raise ValueError(f"data_ctv must be (C, T, V), got shape {data_ctv.shape}")
    if fill_mode not in ("edge", "zero"):
        raise ValueError(f"fill_mode must be 'edge' or 'zero', got {fill_mode}")
    C, T, V = data_ctv.shape
    out_blocks = [data_ctv]
    for k in lags:
        if k <= 0:
            raise ValueError(f"lag must be positive, got {k}")
        delta = np.zeros_like(data_ctv)
        if k < T:
            delta[:, k:, :] = data_ctv[:, k:, :] - data_ctv[:, :-k, :]
            if fill_mode == "edge":
                # Copy the first valid delta into the head frames [0, k).
                delta[:, :k, :] = delta[:, k:k + 1, :]
        # else: k >= T → no valid diffs, block stays all zeros
        out_blocks.append(delta)
    return np.concatenate(out_blocks, axis=0)
