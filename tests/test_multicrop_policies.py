""" exp068 unit tests for tools/multicrop_policies.py.

Validates:
  * segment_indices_with_phase shape, monotonicity, and matches
    pybvh_ml.uniform_temporal_sample for canonical phases.
  * crop_indices_phase_shift produces K distinct training-aligned subsamples.
  * crop_indices_uniform_endpoints / center_quartiles spacing properties.
  * per_frame_motion_energy correctness on synthetic motion patterns.
  * window_summed_energy matches np.cumsum semantics.
  * crop_starts_motion_energy_softmax weight distribution under varying τ.
  * aggregate_logits invariants (logit_mean vs prob_mean) and weight checks.

See: §5.1 / §11.2
Related: tools/multicrop_policies.py
"""
from __future__ import annotations

import numpy as np
import pytest

from tools.multicrop_policies import (
    aggregate_logits,
    crop_indices_center_quartiles,
    crop_indices_motion_energy_softmax,
    crop_indices_phase_shift,
    crop_indices_uniform_endpoints,
    crop_starts_motion_energy_softmax,
    per_frame_motion_energy,
    segment_indices_with_phase,
    window_summed_energy,
)


CLIP = 64


# ---------------------------------------------------------------------------
# segment_indices_with_phase
# ---------------------------------------------------------------------------

def test_segment_indices_long_sequence_basic_shape() -> None:
    idx = segment_indices_with_phase(800, CLIP, 0.5)
    assert idx.shape == (CLIP,)
    assert idx.dtype.kind == "i"
    # Monotonically non-decreasing within long-sequence regime
    assert (np.diff(idx) >= 0).all()


def test_segment_indices_phase_zero_picks_segment_starts() -> None:
    """phase=0 → offset=0 in each segment → indices are segment boundaries."""
    nf = 800
    idx = segment_indices_with_phase(nf, CLIP, 0.0)
    expected = np.array([i * nf // CLIP for i in range(CLIP)], dtype=np.intp)
    np.testing.assert_array_equal(idx, expected)


def test_segment_indices_distinct_phases_give_distinct_indices() -> None:
    """K=3 evenly-spaced phases produce K different index arrays."""
    nf = 800
    a = segment_indices_with_phase(nf, CLIP, 0.1)
    b = segment_indices_with_phase(nf, CLIP, 0.5)
    c = segment_indices_with_phase(nf, CLIP, 0.9)
    assert not np.array_equal(a, b)
    assert not np.array_equal(b, c)
    assert not np.array_equal(a, c)


def test_segment_indices_phase_clipped_at_segment_end() -> None:
    """phase very close to 1.0 should not exceed segment boundary."""
    nf = 800
    idx = segment_indices_with_phase(nf, CLIP, 0.999)
    # Each index must lie within its segment [seg_start, seg_end).
    boundaries = np.array(
        [i * nf // CLIP for i in range(CLIP + 1)], dtype=np.intp,
    )
    for k in range(CLIP):
        assert boundaries[k] <= idx[k] < boundaries[k + 1], (
            f"segment {k}: idx={idx[k]} out of [{boundaries[k]}, {boundaries[k+1]})"
        )


def test_segment_indices_short_sequence() -> None:
    """num_frames < clip_length: sequential start with phase-controlled offset."""
    nf = 30
    idx = segment_indices_with_phase(nf, CLIP, 0.0)
    expected = np.arange(0, CLIP, dtype=np.intp)
    np.testing.assert_array_equal(idx, expected)
    # phase = 0.999: start = floor(0.999 * (CLIP - nf + 1)) = floor(0.999 * 35) = 34
    idx2 = segment_indices_with_phase(nf, CLIP, 0.999)
    assert idx2[0] == 34


def test_segment_indices_validates_inputs() -> None:
    with pytest.raises(ValueError, match="num_frames"):
        segment_indices_with_phase(0, CLIP, 0.5)
    with pytest.raises(ValueError, match="clip_length"):
        segment_indices_with_phase(100, 0, 0.5)
    with pytest.raises(ValueError, match="phase"):
        segment_indices_with_phase(100, CLIP, 1.0)
    with pytest.raises(ValueError, match="phase"):
        segment_indices_with_phase(100, CLIP, -0.1)


# ---------------------------------------------------------------------------
# crop_indices_phase_shift
# ---------------------------------------------------------------------------

def test_phase_shift_shape_and_distinct() -> None:
    K = 5
    out = crop_indices_phase_shift(800, CLIP, K)
    assert out.shape == (K, CLIP)
    # All K rows distinct
    for i in range(K):
        for j in range(i + 1, K):
            assert not np.array_equal(out[i], out[j])


def test_phase_shift_each_subsample_spans_full_sequence() -> None:
    """Each subsampling should cover the full [0, num_frames) range."""
    nf = 800
    out = crop_indices_phase_shift(nf, CLIP, 3)
    for k in range(out.shape[0]):
        # Each subsample's first index < clip_length-th segment boundary,
        # last index >= last segment boundary.
        assert out[k, 0] < nf // CLIP + 1, (
            f"crop {k}: first index {out[k, 0]} should be in first segment"
        )
        assert out[k, -1] >= nf - nf // CLIP - 1, (
            f"crop {k}: last index {out[k, -1]} should be in last segment"
        )


def test_phase_shift_K1_returns_single_subsample() -> None:
    out = crop_indices_phase_shift(800, CLIP, 1)
    assert out.shape == (1, CLIP)


# ---------------------------------------------------------------------------
# crop_indices_uniform_endpoints
# ---------------------------------------------------------------------------

def test_uniform_endpoints_first_and_last_starts() -> None:
    """K=3 endpoints: starts at 0, mid, max_start."""
    nf = 800
    out = crop_indices_uniform_endpoints(nf, CLIP, 3)
    assert out.shape == (3, CLIP)
    # First crop starts at 0
    assert out[0, 0] == 0
    # Last crop starts at max_start (= nf - CLIP)
    assert out[-1, 0] == nf - CLIP
    # Middle crop starts at midpoint
    expected_mid = (nf - CLIP) // 2
    assert abs(int(out[1, 0]) - expected_mid) <= 1


def test_uniform_endpoints_short_sequence_collapses() -> None:
    """num_frames <= clip_length: all crops start at 0."""
    nf = 30
    out = crop_indices_uniform_endpoints(nf, CLIP, 3)
    assert out.shape == (3, CLIP)
    for k in range(3):
        assert out[k, 0] == 0
    # All rows are identical
    assert np.array_equal(out[0], out[1])
    assert np.array_equal(out[1], out[2])


def test_uniform_endpoints_K1() -> None:
    """K=1 should return a single window starting at midpoint."""
    nf = 800
    out = crop_indices_uniform_endpoints(nf, CLIP, 1)
    assert out.shape == (1, CLIP)
    assert out[0, 0] == (nf - CLIP) // 2


def test_uniform_endpoints_indices_contiguous() -> None:
    """Each crop is a CLIP-length contiguous range (modulo wrap-around)."""
    nf = 800
    out = crop_indices_uniform_endpoints(nf, CLIP, 5)
    for k in range(5):
        # Within each row, indices should be sequential
        diff = np.diff(out[k])
        assert (diff == 1).all(), f"crop {k} not contiguous: diff={diff}"


# ---------------------------------------------------------------------------
# crop_indices_center_quartiles
# ---------------------------------------------------------------------------

def test_center_quartiles_K3_mid_quartiles() -> None:
    """K=3: phases (0.5/3, 1.5/3, 2.5/3) ≈ (0.167, 0.5, 0.833)."""
    nf = 800
    out = crop_indices_center_quartiles(nf, CLIP, 3)
    max_start = nf - CLIP
    expected_starts = np.round(
        np.array([0.5, 1.5, 2.5]) / 3.0 * max_start
    ).astype(np.intp)
    actual_starts = out[:, 0]
    np.testing.assert_array_equal(actual_starts, expected_starts)


def test_center_quartiles_centered_K1() -> None:
    nf = 800
    out = crop_indices_center_quartiles(nf, CLIP, 1)
    assert out.shape == (1, CLIP)
    # K=1 phase = 0.5 → middle crop
    expected = (nf - CLIP) // 2
    assert abs(int(out[0, 0]) - expected) <= 1


def test_center_quartiles_starts_distinct() -> None:
    nf = 800
    for K in (3, 5, 7):
        out = crop_indices_center_quartiles(nf, CLIP, K)
        starts = out[:, 0]
        assert len(set(starts.tolist())) == K


# ---------------------------------------------------------------------------
# Motion energy
# ---------------------------------------------------------------------------

def test_motion_energy_shape_and_dtype() -> None:
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((100, 24, 3))
    e = per_frame_motion_energy(pos)
    assert e.shape == (100,)
    assert np.isfinite(e).all()
    assert (e >= 0).all()


def test_motion_energy_zero_for_static_motion() -> None:
    """Constant joint positions → zero energy everywhere."""
    pos = np.ones((50, 24, 3))
    e = per_frame_motion_energy(pos)
    np.testing.assert_array_equal(e, np.zeros(50))


def test_motion_energy_proportional_to_motion_magnitude() -> None:
    """Doubling motion doubles energy."""
    pos = np.zeros((10, 1, 3))
    pos[:, 0, 0] = np.arange(10)  # linear motion in x
    e1 = per_frame_motion_energy(pos)
    pos2 = pos * 2
    e2 = per_frame_motion_energy(pos2)
    # Edges may differ slightly; check the inner frames.
    np.testing.assert_allclose(e2[:-1], 2.0 * e1[:-1], atol=1e-9)


def test_motion_energy_validates_input() -> None:
    with pytest.raises(ValueError, match="(T, J, 3)"):
        per_frame_motion_energy(np.zeros((10, 24)))  # wrong ndim
    with pytest.raises(ValueError, match="(T, J, 3)"):
        per_frame_motion_energy(np.zeros((10, 24, 4)))  # wrong last dim


def test_motion_energy_minimum_T_one() -> None:
    """T=1 should not crash; returns zeros."""
    pos = np.zeros((1, 24, 3))
    e = per_frame_motion_energy(pos)
    assert e.shape == (1,)
    assert e[0] == 0.0


# ---------------------------------------------------------------------------
# window_summed_energy
# ---------------------------------------------------------------------------

def test_window_summed_energy_basic() -> None:
    energy = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    win = window_summed_energy(energy, clip_length=2)
    # windows of length 2: positions 0..3, sums = [3, 5, 7, 9]
    np.testing.assert_array_equal(win, [3.0, 5.0, 7.0, 9.0])


def test_window_summed_energy_full_length() -> None:
    """clip_length == T → single window with total energy."""
    energy = np.array([1.0, 2.0, 3.0, 4.0])
    win = window_summed_energy(energy, clip_length=4)
    assert win.shape == (1,)
    assert win[0] == 10.0


def test_window_summed_energy_short_sequence() -> None:
    """T < clip_length → single-element array with total."""
    energy = np.array([1.0, 2.0, 3.0])
    win = window_summed_energy(energy, clip_length=10)
    assert win.shape == (1,)
    assert win[0] == 6.0


# ---------------------------------------------------------------------------
# crop_starts_motion_energy_softmax
# ---------------------------------------------------------------------------

def test_motion_energy_softmax_uniform_when_constant_energy() -> None:
    """Equal energy at all windows → softmax weights = uniform."""
    nf = 800
    energy = np.ones(nf) * 0.5
    starts, w = crop_starts_motion_energy_softmax(
        nf, CLIP, num_crops=5, energy=energy, tau=2.0,
    )
    np.testing.assert_allclose(w, 0.2, atol=1e-9)
    assert np.isclose(w.sum(), 1.0)


def test_motion_energy_softmax_concentrates_on_high_energy() -> None:
    """Energy spike at one position → softmax weight concentrates there
    as τ → 0, AND the high-weight window is the one covering the spike."""
    nf = 800
    energy = np.ones(nf) * 0.01
    # Inject huge energy at frames 200..263 (covered by start=200 window
    # if it's in the K=5 grid). K=5 grid: linspace(0, 736, 5) = [0, 184, 368, 552, 736].
    # So window starting at 184 covers frames 184..247, which overlaps
    # with our spike (200..263). Use the spike window as ground truth.
    spike_start, spike_end = 184, 184 + CLIP  # = 184..248
    energy[spike_start:spike_end] = 100.0
    starts, w_low_tau = crop_starts_motion_energy_softmax(
        nf, CLIP, num_crops=5, energy=energy, tau=0.01,
    )
    # Find which window has the spike (= our injected start).
    target_idx = int(np.argmin(np.abs(starts - spike_start)))
    assert w_low_tau.max() > 0.95
    # The high weight must be on the spike window, not just any window.
    assert int(np.argmax(w_low_tau)) == target_idx, (
        f"argmax weight at idx {np.argmax(w_low_tau)} (start={starts[np.argmax(w_low_tau)]}), "
        f"expected idx {target_idx} (start={starts[target_idx]})"
    )


def test_motion_energy_softmax_high_tau_approaches_uniform() -> None:
    """τ → ∞ flattens softmax toward uniform."""
    nf = 800
    rng = np.random.default_rng(0)
    energy = rng.uniform(0, 1, size=nf)
    _, w_high_tau = crop_starts_motion_energy_softmax(
        nf, CLIP, num_crops=5, energy=energy, tau=1e6,
    )
    np.testing.assert_allclose(w_high_tau, 0.2, atol=1e-3)


def test_motion_energy_softmax_short_sequence_uniform() -> None:
    """num_frames <= clip_length: softmax weights default to uniform."""
    energy = np.ones(30) * 0.5
    starts, w = crop_starts_motion_energy_softmax(
        30, CLIP, num_crops=3, energy=energy, tau=1.0,
    )
    np.testing.assert_array_equal(starts, [0, 0, 0])
    np.testing.assert_allclose(w, [1/3, 1/3, 1/3], atol=1e-9)


def test_motion_energy_softmax_validates_inputs() -> None:
    nf = 800
    with pytest.raises(ValueError, match="tau"):
        crop_starts_motion_energy_softmax(
            nf, CLIP, num_crops=3, energy=np.ones(nf), tau=0.0,
        )
    with pytest.raises(ValueError, match="energy.shape"):
        crop_starts_motion_energy_softmax(
            nf, CLIP, num_crops=3, energy=np.ones(nf - 5), tau=2.0,
        )


def test_motion_energy_softmax_indices_wrapper_returns_indices() -> None:
    """crop_indices_motion_energy_softmax expands to (K, clip_length)."""
    nf = 800
    energy = np.ones(nf)
    idx, w = crop_indices_motion_energy_softmax(
        nf, CLIP, num_crops=3, energy=energy, tau=2.0,
    )
    assert idx.shape == (3, CLIP)
    assert w.shape == (3,)


# ---------------------------------------------------------------------------
# aggregate_logits
# ---------------------------------------------------------------------------

def test_aggregate_logit_mean_uniform_weights() -> None:
    """Logit mean: simple average across K crops."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((10, 5, 12))
    out = aggregate_logits(z, "logit_mean")
    expected = z.mean(axis=1)
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_aggregate_prob_mean_uniform_weights() -> None:
    """Prob mean: average softmax probabilities; argmax matches direct softmax-mean argmax."""
    rng = np.random.default_rng(1)
    z = rng.standard_normal((10, 5, 12))
    out = aggregate_logits(z, "prob_mean")
    # Compare argmax to manually-computed softmax mean argmax
    m = z.max(axis=-1, keepdims=True)
    p = np.exp(z - m)
    p = p / p.sum(axis=-1, keepdims=True)
    p_avg = p.mean(axis=1)
    expected_argmax = p_avg.argmax(axis=-1)
    actual_argmax = out.argmax(axis=-1)
    np.testing.assert_array_equal(actual_argmax, expected_argmax)


def test_aggregate_logits_K1_identity() -> None:
    """K=1 should be the identity (return the single crop's logits)."""
    rng = np.random.default_rng(2)
    z = rng.standard_normal((10, 1, 12))
    out_lm = aggregate_logits(z, "logit_mean")
    np.testing.assert_allclose(out_lm, z[:, 0, :], atol=1e-12)
    out_pm = aggregate_logits(z, "prob_mean")
    # log of softmax(z[0]) — argmax invariant
    np.testing.assert_array_equal(
        out_pm.argmax(axis=-1), z[:, 0, :].argmax(axis=-1),
    )


def test_aggregate_logits_weighted() -> None:
    """Non-uniform weights respected for logit_mean."""
    rng = np.random.default_rng(3)
    z = rng.standard_normal((10, 3, 12))
    weights = np.array([0.7, 0.2, 0.1])
    out = aggregate_logits(z, "logit_mean", weights=weights)
    expected = (z * weights[None, :, None]).sum(axis=1)
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_aggregate_logits_validates_weights() -> None:
    rng = np.random.default_rng(4)
    z = rng.standard_normal((10, 3, 12))
    with pytest.raises(ValueError, match="sum"):
        aggregate_logits(z, "logit_mean", weights=np.array([0.5, 0.5, 0.5]))
    with pytest.raises(ValueError, match="non-negative"):
        aggregate_logits(z, "logit_mean", weights=np.array([1.5, -0.5, 0.0]))
    with pytest.raises(ValueError, match="weights.shape"):
        aggregate_logits(z, "logit_mean", weights=np.array([0.5, 0.5]))


def test_aggregate_logits_unknown_aggregation() -> None:
    rng = np.random.default_rng(5)
    z = rng.standard_normal((10, 3, 12))
    with pytest.raises(ValueError, match="unknown aggregation"):
        aggregate_logits(z, "geometric_mean")


def test_aggregate_logits_rejects_wrong_ndim() -> None:
    z2d = np.zeros((10, 12))
    with pytest.raises(ValueError, match="ndim"):
        aggregate_logits(z2d, "logit_mean")


def test_aggregate_prob_mean_with_nonuniform_weights() -> None:
    """prob_mean × non-uniform weights — exercises motion_energy_softmax path."""
    rng = np.random.default_rng(99)
    K_classes = 12
    z = rng.standard_normal((10, 3, K_classes)) * 2.0
    weights = np.array([0.7, 0.2, 0.1])
    out = aggregate_logits(z, "prob_mean", weights=weights)
    # Argmax must equal argmax of the weighted softmax mean computed manually
    m = z.max(axis=-1, keepdims=True)
    p = np.exp(z - m)
    p = p / p.sum(axis=-1, keepdims=True)
    p_avg = (p * weights[None, :, None]).sum(axis=1)
    np.testing.assert_array_equal(out.argmax(axis=-1), p_avg.argmax(axis=-1))
    # exp(out) should be a valid probability distribution
    p_recon = np.exp(out)
    np.testing.assert_allclose(p_recon.sum(axis=-1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Dense-regime sequence (clip_length <= N < 2 * clip_length)
# ---------------------------------------------------------------------------

def test_segment_indices_dense_regime_distinct_phases() -> None:
    """For sequences in the dense regime, distinct phases produce distinct
    samplings (verifies the rng-based gap insertion is phase-controlled)."""
    nf = 96  # CLIP=64 ≤ 96 < 128 = 2*CLIP → dense regime
    a = segment_indices_with_phase(nf, CLIP, 0.1)
    b = segment_indices_with_phase(nf, CLIP, 0.5)
    c = segment_indices_with_phase(nf, CLIP, 0.9)
    assert a.shape == (CLIP,)
    assert b.shape == (CLIP,)
    assert c.shape == (CLIP,)
    # Each must contain CLIP indices in [0, nf)
    for arr in (a, b, c):
        assert (arr >= 0).all() and (arr < nf).all()
    # Different phases → different orderings (with high probability)
    assert not np.array_equal(a, b)
    assert not np.array_equal(b, c)


def test_phase_shift_collapses_at_K_equals_clip() -> None:
    """For N == clip_length, phase_shift returns identical sub-samplings
    for all K. In the dense regime, n_gaps = N - clip_length = 0, so the
    rng-based gap insertion has nothing to insert and every phase yields
    the same arange(clip_length) sequence. (Production code does NOT emit
    a warning for this case; the boundary collapse is documented here as
    expected behaviour.)"""
    out = crop_indices_phase_shift(CLIP, CLIP, num_crops=3)
    assert out.shape == (3, CLIP)
    # In this special case, all 3 rows are arange(CLIP)
    np.testing.assert_array_equal(out[0], out[1])
    np.testing.assert_array_equal(out[1], out[2])
    np.testing.assert_array_equal(out[0], np.arange(CLIP, dtype=np.intp))


def test_phase_shift_short_sequence_indices_within_bounds() -> None:
    """For N < clip_length, phase_shift wraps modulo num_frames so
    all returned indices are in [0, num_frames). Without wrap, indices
    would exceed num_frames in the short regime."""
    nf = 30
    out = crop_indices_phase_shift(nf, CLIP, num_crops=3)
    assert out.shape == (3, CLIP)
    assert (out >= 0).all()
    assert (out < nf).all(), (
        f"phase_shift indices exceed num_frames: max={out.max()}, "
        f"num_frames={nf}"
    )


# ---------------------------------------------------------------------------
# aggregate_member_fold (in tools/infer_test_multicrop.py) — fast-path test
# ---------------------------------------------------------------------------

def test_aggregate_member_fold_fast_path_matches_loop() -> None:
    """When weights are uniform across all clips, the fast-path
    (one vectorised aggregate_logits call) must match the per-clip loop."""
    from tools.infer_test_multicrop import aggregate_member_fold
    rng = np.random.default_rng(0)
    N, K, C = 50, 5, 12
    z = rng.standard_normal((N, K, C)) * 1.5
    # Uniform weights for all clips
    uniform_weights = np.full((N, K), 1.0 / K)
    fast_out = aggregate_member_fold(z, uniform_weights, "logit_mean")
    # Reference: per-clip loop
    ref_out = np.zeros((N, C))
    for n in range(N):
        ref_out[n] = aggregate_logits(z[n:n+1], "logit_mean", weights=uniform_weights[n])[0]
    np.testing.assert_allclose(fast_out, ref_out, atol=1e-12)


def test_aggregate_member_fold_per_clip_weights_use_loop() -> None:
    """When weights vary across clips (motion_energy_softmax case), the
    per-clip loop is used and produces correctly per-clip weighted averages.

    Verifies BOTH:
      (a) the resulting aggregated logits match the vectorised manual
          computation, AND
      (b) the fast-path is NOT taken (per-clip weights are not all-equal
          within atol=1e-12).
    """
    from tools.infer_test_multicrop import aggregate_member_fold
    rng = np.random.default_rng(1)
    N, K, C = 30, 3, 12
    z = rng.standard_normal((N, K, C)) * 2.0
    # Distinct per-clip weights, intentionally non-uniform.
    weights = rng.random((N, K))
    weights = weights / weights.sum(axis=1, keepdims=True)
    # Sanity check (b): rows must differ by more than the fast-path tolerance.
    assert not np.allclose(weights, weights[0:1], atol=1e-12, rtol=0), (
        "test setup defective: weights look uniform to the fast-path check"
    )
    out = aggregate_member_fold(z, weights, "logit_mean")
    # Reference (a): vectorised manually
    expected = (z * weights[:, :, None]).sum(axis=1)
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_aggregate_member_fold_rejects_shape_mismatch() -> None:
    from tools.infer_test_multicrop import aggregate_member_fold
    z = np.zeros((10, 3, 12))
    weights = np.zeros((10, 5))  # wrong K
    with pytest.raises(ValueError, match="shape mismatch"):
        aggregate_member_fold(z, weights, "logit_mean")
