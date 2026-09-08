"""Multi-crop policies for exp068.

Pure functions that compute the crop indices for K-fold multi-crop inference.
The four policies are:

    phase_shift            — K different segment-based subsamplings
                            (each spans the full sequence, training-aligned).
    uniform_endpoints      — K contiguous 64-frame windows with start
                            positions evenly spaced over the sequence.
    center_quartiles       — K contiguous windows at fixed quantile phases.
    motion_energy_softmax  — K contiguous windows weighted by
                            softmax(window-summed motion energy / tau).

The first policy (phase_shift) keeps the input distribution close to the
training distribution (segment-based uniform subsampling); the latter three
introduce some test-time distribution shift but cover larger fractions of
the original sequence.

See: §5.1 / §11.2
Related: tools/infer_test_multicrop.py (CLI driver),
         pybvh_ml.uniform_temporal_sample (training-time sampler).
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Segment-based sampling (mirrors pybvh_ml.uniform_temporal_sample for
# the long-sequence regime, with a fixed numerical phase ∈ [0, 1)).
# ---------------------------------------------------------------------------

def segment_indices_with_phase(
    num_frames: int, clip_length: int, phase: float,
) -> np.ndarray:
    """Pick ``clip_length`` segment-based indices using a deterministic phase.

    Mirrors ``pybvh_ml.uniform_temporal_sample`` for long sequences: divide
    [0, num_frames) into ``clip_length`` equal segments using integer
    boundaries, then within each segment pick offset = ``floor(phase *
    seg_size)``. Phase ∈ [0, 1) parameterises the sub-segment offset; the
    training-time sampler uses uniform-random phases per segment, while
    test-time uses a fixed RNG-seeded pattern.

    Behaviour for short sequences mirrors ``uniform_temporal_sample``:

    - ``num_frames < clip_length``: sequential indices starting at
      ``floor(phase * (clip_length - num_frames + 1))``; caller applies
      ``% num_frames`` for wrap-around.
    - ``clip_length <= num_frames < 2 * clip_length``: one of
      ``num_frames - clip_length`` "gap insertions" of contiguous indices,
      indexed by ``floor(phase * (n_gaps + 1))``.
    - ``num_frames >= 2 * clip_length``: per-segment offset
      ``floor(phase * seg_size)``.

    Args:
        num_frames: total frames in the source sequence.
        clip_length: number of frame indices to return.
        phase: float in [0, 1) selecting the sub-segment offset.

    Returns:
        ndarray of shape (clip_length,), dtype int.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if clip_length < 1:
        raise ValueError(f"clip_length must be >= 1, got {clip_length}")
    if not (0.0 <= phase < 1.0):
        raise ValueError(f"phase must be in [0, 1), got {phase}")

    # Short sequence: sequential indices with phase-controlled start.
    if num_frames < clip_length:
        max_start = clip_length - num_frames + 1  # number of valid starts
        start = int(math.floor(phase * max_start))
        return np.arange(start, start + clip_length, dtype=np.intp)

    # Mid-density sequence (clip_length <= N < 2*clip_length).
    if num_frames < 2 * clip_length:
        n_gaps = num_frames - clip_length
        # Choose `n_gaps` distinct positions out of `clip_length + 1`
        # using the phase as a bijective ordering. Reproducible.
        rng = np.random.default_rng(int(round(phase * 1_000_000)))
        gap_positions = rng.choice(clip_length + 1, size=n_gaps, replace=False)
        offset = np.zeros(clip_length + 1, dtype=np.intp)
        offset[gap_positions] = 1
        offset = np.cumsum(offset)
        basic = np.arange(clip_length, dtype=np.intp)
        return basic + offset[:clip_length]

    # Long sequence: integer segment boundaries.
    boundaries = np.array(
        [i * num_frames // clip_length for i in range(clip_length + 1)],
        dtype=np.intp,
    )
    seg_sizes = np.diff(boundaries)
    seg_starts = boundaries[:clip_length]
    # Per-segment offset is floor(phase * seg_size); identical phase across
    # all segments. clipped to [0, seg_size - 1].
    offsets = np.minimum(
        np.floor(phase * seg_sizes).astype(np.intp),
        seg_sizes - 1,
    )
    return seg_starts + offsets


def crop_indices_phase_shift(
    num_frames: int, clip_length: int, num_crops: int,
) -> np.ndarray:
    """K segment-based subsamplings with phases evenly spaced over [0, 1).

    Each subsampling spans the full sequence and is a Monte-Carlo realisation
    of the training-time sampler ``pybvh_ml.uniform_temporal_sample(..., mode='train')``
    in expectation: the training sampler picks a *random* offset per segment,
    while phase_shift uses a *single fixed phase* across all segments. Averaging
    K phases approximates the ensemble of training-time random samplings.

    Note: this is NOT identical to the single-crop ``mode='test'`` baseline,
    which uses ``rng.integers`` with seed=0 (different per-segment offsets).
    The K=infinity limit of phase_shift converges to the *expectation* over
    training-time random samplings, not to the seed=0 realisation. This makes
    phase_shift a denoised expectation estimator rather than a same-distribution
    multi-view ensemble.

    Returns shape ``(num_crops, clip_length)``. For sequences with
    ``num_frames < clip_length``, indices are wrapped via ``% num_frames``
    so callers can index directly.
    """
    if num_crops < 1:
        raise ValueError(f"num_crops must be >= 1, got {num_crops}")
    phases = (np.arange(num_crops) + 0.5) / num_crops  # [1/(2K), 3/(2K), ...]
    out = np.zeros((num_crops, clip_length), dtype=np.intp)
    for k, p in enumerate(phases):
        idx = segment_indices_with_phase(num_frames, clip_length, float(p))
        # Wrap-around for short sequences (N < clip_length) — keep indices
        # in [0, num_frames) so callers don't need a separate modulo.
        out[k] = idx % max(num_frames, 1)
    return out


# ---------------------------------------------------------------------------
# Window-based crops
# ---------------------------------------------------------------------------

def _window_to_indices(start: int, clip_length: int, num_frames: int) -> np.ndarray:
    """Build the index array for a contiguous window starting at ``start``."""
    raw = np.arange(start, start + clip_length, dtype=np.intp)
    # Wrap-around for sequences shorter than clip_length (rare: only ~0.05%
    # of test data per dataset stats); never produces out-of-bounds indices.
    return raw % max(num_frames, 1)


def crop_indices_uniform_endpoints(
    num_frames: int, clip_length: int, num_crops: int,
) -> np.ndarray:
    """K contiguous windows with starts evenly spaced over [0, max_start].

    For ``num_frames <= clip_length``, all crops collapse to start=0
    (single window). For ``num_crops == 1`` and a non-trivial sequence,
    the single crop is centred. Returns shape ``(num_crops, clip_length)``.
    """
    if num_crops < 1:
        raise ValueError(f"num_crops must be >= 1, got {num_crops}")
    max_start = max(num_frames - clip_length, 0)
    if max_start == 0:
        starts = np.zeros(num_crops, dtype=np.intp)
    elif num_crops == 1:
        starts = np.array([max_start // 2], dtype=np.intp)
    else:
        starts = np.linspace(0, max_start, num_crops).round().astype(np.intp)
    out = np.stack([
        _window_to_indices(int(s), clip_length, num_frames) for s in starts
    ], axis=0)
    return out


def crop_indices_center_quartiles(
    num_frames: int, clip_length: int, num_crops: int,
) -> np.ndarray:
    """K contiguous windows at fixed quantile phases of [0, max_start].

    For K=3 phases = (0.25, 0.5, 0.75); K=5 = (0.10, 0.30, 0.50, 0.70, 0.90);
    K=7 = ~equispaced quantiles excluding endpoints. Generic K uses
    ``(k+0.5)/K`` for k=0..K-1 (mid-quantiles, never touching edges).
    Returns shape ``(num_crops, clip_length)``.
    """
    if num_crops < 1:
        raise ValueError(f"num_crops must be >= 1, got {num_crops}")
    max_start = max(num_frames - clip_length, 0)
    phases = (np.arange(num_crops) + 0.5) / num_crops  # mid-quantiles
    if max_start == 0:
        starts = np.zeros(num_crops, dtype=np.intp)
    else:
        starts = np.round(phases * max_start).astype(np.intp)
    out = np.stack([
        _window_to_indices(int(s), clip_length, num_frames) for s in starts
    ], axis=0)
    return out


# ---------------------------------------------------------------------------
# Motion-energy softmax
# ---------------------------------------------------------------------------

def per_frame_motion_energy(joint_positions: np.ndarray) -> np.ndarray:
    """Per-frame motion energy = ``||joint_pos[t+1] - joint_pos[t]||_F``.

    Args:
        joint_positions: (T, J, 3) joint positions (Cartesian).

    Returns:
        (T,) energy per frame; energy[t] is the change FROM frame t to t+1
        for t < T-1, and energy[T-1] = energy[T-2] (edge-padded so the
        result has the same length as the input).
    """
    if joint_positions.ndim != 3 or joint_positions.shape[2] != 3:
        raise ValueError(
            f"expected joint_positions shape (T, J, 3), got {joint_positions.shape}"
        )
    T = joint_positions.shape[0]
    if T < 2:
        # No frame-to-frame motion to compute → return zeros.
        return np.zeros(T, dtype=np.float64)
    diff = np.diff(joint_positions, axis=0)  # (T-1, J, 3)
    energy = np.linalg.norm(
        diff.reshape(diff.shape[0], -1), axis=1,
    )  # (T-1,)
    # Edge-pad so output has length T.
    return np.concatenate([energy, energy[-1:]]).astype(np.float64)


def window_summed_energy(
    energy: np.ndarray, clip_length: int,
) -> np.ndarray:
    """Sum of frame-energy over windows of length ``clip_length`` starting at each position.

    Args:
        energy: (T,) per-frame energy.
        clip_length: window length.

    Returns:
        (max_start + 1,) where max_start = max(T - clip_length, 0).
        For T <= clip_length, returns a single-element array with the total energy.
    """
    T = energy.shape[0]
    if T <= clip_length:
        return np.array([energy.sum()], dtype=np.float64)
    cum = np.concatenate([[0.0], np.cumsum(energy)])  # (T+1,)
    return cum[clip_length:] - cum[: T - clip_length + 1]


def crop_starts_motion_energy_softmax(
    num_frames: int, clip_length: int, num_crops: int,
    energy: np.ndarray, tau: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """K window starts evenly spaced, with weights = softmax(energy/τ).

    The K starts are placed uniformly over [0, max_start] (same as
    ``uniform_endpoints``); the weights replace the equal 1/K weighting
    with ``softmax(window_summed_energy[starts] / tau)``. This is the
    smooth version of the plan's "motion-energy top windows" (hard top-N)
    described in §5.1, with the smoothness controlled by τ.

    Args:
        num_frames: total frames.
        clip_length: window length.
        num_crops: number of windows.
        energy: (num_frames,) per-frame motion energy.
        tau: softmax temperature; large τ → equal weights; small τ → top-K.

    Returns:
        (starts, weights) where ``starts.shape == (num_crops,)`` and
        ``weights.shape == (num_crops,)``, ``weights.sum() == 1``.
    """
    if num_crops < 1:
        raise ValueError(f"num_crops must be >= 1, got {num_crops}")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if energy.shape[0] != num_frames:
        raise ValueError(
            f"energy.shape[0] ({energy.shape[0]}) != num_frames ({num_frames})"
        )
    max_start = max(num_frames - clip_length, 0)
    if max_start == 0:
        starts = np.zeros(num_crops, dtype=np.intp)
        weights = np.full(num_crops, 1.0 / num_crops)
        return starts, weights

    starts = np.linspace(0, max_start, num_crops).round().astype(np.intp)
    win_e = window_summed_energy(energy, clip_length)
    e_at_starts = win_e[starts]
    e = e_at_starts / tau
    e = e - e.max()  # numerical stability
    w = np.exp(e)
    w = w / w.sum()
    return starts, w.astype(np.float64)


def crop_indices_motion_energy_softmax(
    num_frames: int, clip_length: int, num_crops: int,
    energy: np.ndarray, tau: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Wraps ``crop_starts_motion_energy_softmax`` with index expansion.

    Returns ``(indices, weights)`` where ``indices.shape ==
    (num_crops, clip_length)`` and ``weights.shape == (num_crops,)``.
    """
    starts, weights = crop_starts_motion_energy_softmax(
        num_frames, clip_length, num_crops, energy, tau=tau,
    )
    indices = np.stack([
        _window_to_indices(int(s), clip_length, num_frames) for s in starts
    ], axis=0)
    return indices, weights


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_logits(
    per_crop_logits: np.ndarray,
    aggregation: str,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate per-crop logits into per-sample logits.

    Args:
        per_crop_logits: shape ``(N, K, num_classes)`` where N=samples, K=crops.
        aggregation: 'logit_mean' or 'prob_mean'.
        weights: optional ``(K,)`` weights per crop. Defaults to uniform.

    Returns:
        (N, num_classes) aggregated logits. For ``prob_mean``, the returned
        "logits" are ``log p_avg`` (so that downstream argmax / softmax
        agree with the mean-probability decision).
    """
    if per_crop_logits.ndim != 3:
        raise ValueError(
            f"expected per_crop_logits.ndim == 3, got {per_crop_logits.ndim}"
        )
    N, K, _ = per_crop_logits.shape
    if weights is None:
        weights = np.full(K, 1.0 / K, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (K,):
            raise ValueError(
                f"weights.shape ({weights.shape}) must equal ({K},)"
            )
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(
                f"weights must sum to 1, got {weights.sum()}"
            )
        if (weights < 0).any():
            raise ValueError("weights must be non-negative")

    if aggregation == "logit_mean":
        # Weighted mean over the K axis.
        return (per_crop_logits * weights[None, :, None]).sum(axis=1)
    if aggregation == "prob_mean":
        # Weighted mean of softmax probabilities, returned as log p.
        z = per_crop_logits
        m = z.max(axis=-1, keepdims=True)  # (N, K, 1)
        p = np.exp(z - m)
        p = p / p.sum(axis=-1, keepdims=True)  # (N, K, num_classes)
        p_avg = (p * weights[None, :, None]).sum(axis=1)
        # log p_avg with floor to keep argmax stable. 1e-12 ≈ float32 epsilon
        # range, well above the 1e-30 underflow floor while not perturbing
        # likelihoods of any class with non-trivial mass.
        return np.log(np.clip(p_avg, 1e-12, None))
    raise ValueError(f"unknown aggregation '{aggregation}'")
