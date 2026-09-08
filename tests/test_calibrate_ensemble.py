""" exp067 unit tests for tools/calibrate_ensemble.py.

Synthetic-data tests for each calibration primitive:
  * softmax / log-sum-exp numerical stability
  * temperature scaling: NLL fit recovers a known T
  * uniform-prior Bayes adjust: matches the analytic formula
  * L2 class-bias: respects regularisation, agrees with autograd at numerical
    finite differences
  * direction-consistency: simple sanity cases
  * paired bootstrap: zero-diff when inputs are identical, positive when
    method B strictly improves over method A on a synthetic toy

See: §5.1 / §11.1
Related: tools/calibrate_ensemble.py
"""
from __future__ import annotations

import numpy as np
import pytest

from tools.calibrate_ensemble import (
    NUM_CLASSES,
    PRIORITY_CLASSES,
    direction_consistency,
    fit_class_bias_l2,
    fit_temperature,
    go_no_go_decision,
    inner_lambda_search,
    log_sum_exp,
    macro_f1,
    nll_loss,
    paired_bootstrap_f1_diff,
    softmax,
    uniform_prior_bias,
    FoldOOF,
)


# ---------------------------------------------------------------------------
# Numerical primitives
# ---------------------------------------------------------------------------

def test_log_sum_exp_matches_scipy() -> None:
    from scipy.special import logsumexp as scipy_lse
    rng = np.random.default_rng(0)
    z = rng.standard_normal((50, 12)) * 5.0
    ours = log_sum_exp(z, axis=-1)
    theirs = scipy_lse(z, axis=-1)
    np.testing.assert_allclose(ours, theirs, atol=1e-10)


def test_log_sum_exp_extreme_values() -> None:
    # Large positive logits should not overflow
    z = np.array([[1000.0, 1001.0, 999.0]])
    out = log_sum_exp(z)
    assert np.isfinite(out).all()
    # Exact: 1001 + log(e^-1 + 1 + e^-2) ≈ 1001.404
    expected = 1001.0 + np.log(np.exp(-1.0) + 1.0 + np.exp(-2.0))
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_softmax_sums_to_one() -> None:
    rng = np.random.default_rng(1)
    z = rng.standard_normal((100, 12))
    p = softmax(z, axis=-1)
    np.testing.assert_allclose(p.sum(axis=-1), 1.0, atol=1e-10)


def test_softmax_extreme_values_safe() -> None:
    z = np.array([[1e6, -1e6, 0.0]])
    p = softmax(z)
    assert np.isfinite(p).all()
    assert abs(p[0, 0] - 1.0) < 1e-9
    assert abs(p[0, 1]) < 1e-9


def test_nll_loss_matches_definition() -> None:
    rng = np.random.default_rng(2)
    N, K = 30, 12
    z = rng.standard_normal((N, K))
    y = rng.integers(0, K, size=N)
    # Compute the same thing via softmax → log → index.
    p = softmax(z)
    expected = float(-np.log(p[np.arange(N), y]).mean())
    np.testing.assert_allclose(nll_loss(z, y), expected, atol=1e-10)


def test_nll_loss_extreme_logits_no_overflow() -> None:
    """Confirm log-sum-exp keeps NLL finite for huge logits."""
    z = np.array([[1000.0, -1000.0, 0.0]])
    # correct answer: prob ~ 1, NLL ~ 0
    assert nll_loss(z, np.array([0])) == pytest.approx(0.0, abs=1e-6)
    # wrong answer: NLL is large but finite
    nll_wrong = nll_loss(z, np.array([1]))
    assert np.isfinite(nll_wrong)
    assert nll_wrong > 100.0  # ≈ 2000


def test_nll_loss_perfect_predictor_zero() -> None:
    """An infinitely confident, correct predictor has NLL ~ 0."""
    K = NUM_CLASSES
    N = 50
    rng = np.random.default_rng(123)
    y = rng.integers(0, K, size=N).astype(np.int64)
    # 100x boost on the correct class for each row
    z = np.zeros((N, K))
    z[np.arange(N), y] = 100.0
    assert nll_loss(z, y) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def test_fit_temperature_recovers_known_T() -> None:
    """Generate logits z = T_true * scores where scores produce label freq.
    Calibrating with z should recover T ~ T_true."""
    rng = np.random.default_rng(7)
    N, K = 8000, NUM_CLASSES
    T_true = 2.5
    # well-calibrated "true" logits (large enough to see meaningful softmax)
    score = rng.standard_normal((N, K)) * 1.5
    # sample labels from the true distribution.
    p_true = softmax(score)
    y = np.array([rng.choice(K, p=p_true[i]) for i in range(N)], dtype=np.int64)
    # observed logits are scaled (over-confident if T_true>1 inverted: actually
    # if observed = score, and we want to "soften" a too-sharp model, we use T>1).
    # Generate observed = score / T_true (so model is over-confident relative to truth)
    z_obs = score / T_true
    T_fit = fit_temperature(z_obs, y, init=1.0)
    # With N=8000 the MLE for 1/T_true=0.4 should sit within ~5%.
    np.testing.assert_allclose(T_fit, 1.0 / T_true, rtol=0.05)


def test_fit_temperature_preserves_argmax_invariance() -> None:
    """T-only scaling MUST leave argmax (and hence Macro-F1) unchanged.

    The mathematical identity ``argmax(α z) == argmax(z) for α > 0`` is
    what makes Macro-F1 invariant. We therefore test the **specific
    contract of fit_temperature** here: it must return T strictly > 0
    (so that ``z / T`` preserves argmax) regardless of input distribution.
    """
    rng = np.random.default_rng(42)
    z = rng.standard_normal((500, NUM_CLASSES)) * 2.5
    y = rng.integers(0, NUM_CLASSES, size=500).astype(np.int64)
    T = fit_temperature(z, y, init=1.0)
    # Contract: T must be positive. If a refactor allows T<=0 (e.g., wrong
    # bounds, or dropping the bounds entirely), z/T flips signs and
    # argmax breaks; this is the regression we guard against.
    assert T > 0, f"fit_temperature returned non-positive T={T}"
    base_preds = z.argmax(axis=1)
    cal_preds = (z / T).argmax(axis=1)
    np.testing.assert_array_equal(base_preds, cal_preds)
    # Stress: T_fit on flat / sharp / sparse-label inputs should still be > 0
    for cfg in [
        rng.standard_normal((100, NUM_CLASSES)) * 0.01,  # nearly flat
        rng.standard_normal((100, NUM_CLASSES)) * 50.0,  # very sharp
    ]:
        y_local = rng.integers(0, NUM_CLASSES, size=100).astype(np.int64)
        T_local = fit_temperature(cfg, y_local, init=1.0)
        assert T_local > 0, f"got non-positive T={T_local}"


def test_fit_temperature_no_change_when_calibrated() -> None:
    """If labels are sampled from softmax(z), temperature should be ~1."""
    rng = np.random.default_rng(8)
    N, K = 5000, NUM_CLASSES
    z = rng.standard_normal((N, K)) * 1.0
    p = softmax(z)
    y = np.array([rng.choice(K, p=p[i]) for i in range(N)], dtype=np.int64)
    T = fit_temperature(z, y, init=1.0)
    assert 0.85 < T < 1.20, f"expected T~1, got {T}"


def test_fit_temperature_sharpens_for_underconfident_model() -> None:
    """If observed logits are overly flat (very low magnitude) but labels
    are deterministic, fit should yield T<<1 to sharpen them."""
    rng = np.random.default_rng(9)
    N, K = 1000, NUM_CLASSES
    # sharp truth
    sharp_z = rng.standard_normal((N, K)) * 5.0
    p_true = softmax(sharp_z)
    y = np.array([rng.choice(K, p=p_true[i]) for i in range(N)], dtype=np.int64)
    # observed logits are nearly flat (over-soft)
    z_obs = sharp_z * 0.1
    T = fit_temperature(z_obs, y, init=1.0)
    assert T < 1.0, f"expected sharpening (T<1), got {T}"


# ---------------------------------------------------------------------------
# Uniform-prior Bayes adjust
# ---------------------------------------------------------------------------

def test_uniform_prior_bias_balanced_input_returns_log_uniform() -> None:
    """If model is perfectly uniform, b should be log(1/K) for every class."""
    K = NUM_CLASSES
    z = np.zeros((1000, K))  # all zeros => uniform softmax
    b = uniform_prior_bias(z)
    expected = np.full(K, np.log(1.0 / K))
    np.testing.assert_allclose(b, expected, atol=1e-9)


def test_uniform_prior_bias_increases_with_class_marginal() -> None:
    """A class the model over-predicts should get a larger (less negative) bias
    so that subtracting it reduces that class's logit."""
    K = NUM_CLASSES
    rng = np.random.default_rng(0)
    z = rng.standard_normal((5000, K))
    # boost class 5 strongly
    z[:, 5] += 2.0
    b = uniform_prior_bias(z)
    # b[5] should be the largest (least negative) bias.
    assert int(np.argmax(b)) == 5, f"expected class 5 biggest, got {np.argmax(b)}"
    # subtracting b will then reduce class 5 logit the most.
    z_cal = z - b
    p_cal = softmax(z_cal).mean(axis=0)
    # Marginal predicted prob of each class should be much closer to 1/K after.
    assert np.std(p_cal) < np.std(softmax(z).mean(axis=0))


# ---------------------------------------------------------------------------
# L2 class-bias
# ---------------------------------------------------------------------------

def test_fit_class_bias_l2_zero_with_perfect_balance() -> None:
    """If the train set already has logits whose softmax-mean = label freq,
    the optimal bias under any L2 should be ~0 (no shift needed)."""
    rng = np.random.default_rng(3)
    N, K = 600, NUM_CLASSES
    # build z so that argmax matches label and labels are uniform freq
    y = np.tile(np.arange(K), N // K).astype(np.int64)
    rng.shuffle(y)
    z = rng.standard_normal((N, K)) * 0.5
    z[np.arange(N), y] += 4.0  # large boost on the true class
    b = fit_class_bias_l2(z, y, lam=0.001)
    # b should be small (model already correctly uses each class)
    assert np.linalg.norm(b) < 0.5


def test_fit_class_bias_l2_lambda_zero_overfits() -> None:
    """Without regularisation, the bias should drive predicted marginal toward
    label frequency: mean softmax(z - b)_y == freq(y)."""
    rng = np.random.default_rng(4)
    N, K = 500, NUM_CLASSES
    z = rng.standard_normal((N, K)) * 2.0
    # imbalanced labels
    y = np.array([0] * 100 + [1] * 50 + [2] * 30 + list(range(3, K)) * (320 // (K - 3)),
                 dtype=np.int64)
    y = y[:N]
    rng.shuffle(y)
    b = fit_class_bias_l2(z, y, lam=0.0)
    p = softmax(z - b)
    p_marginal = p.mean(axis=0)
    label_freq = np.bincount(y, minlength=K) / N
    # very tight (the unconstrained MLE matches label freq exactly).
    np.testing.assert_allclose(p_marginal, label_freq, atol=5e-3)


def test_fit_class_bias_l2_l2_pulls_to_zero() -> None:
    """Large lambda should drive the bias near zero."""
    rng = np.random.default_rng(5)
    N, K = 300, NUM_CLASSES
    z = rng.standard_normal((N, K)) * 2.0
    y = rng.integers(0, K, size=N).astype(np.int64)
    b_strong = fit_class_bias_l2(z, y, lam=100.0)
    b_weak = fit_class_bias_l2(z, y, lam=0.001)
    assert np.linalg.norm(b_strong) < np.linalg.norm(b_weak)
    # very large reg should be tiny
    b_huge = fit_class_bias_l2(z, y, lam=1e6)
    assert np.linalg.norm(b_huge) < 1e-2


def test_fit_class_bias_l2_gradient_matches_finite_difference() -> None:
    """Verify our analytic gradient against numerical differences."""
    rng = np.random.default_rng(6)
    N, K = 80, NUM_CLASSES
    z = rng.standard_normal((N, K))
    y = rng.integers(0, K, size=N).astype(np.int64)
    lam = 0.7
    b0 = rng.standard_normal(K) * 0.3

    # Manually rebuild loss & gradient
    onehot = np.zeros((N, K))
    onehot[np.arange(N), y] = 1.0
    label_freq = onehot.mean(axis=0)

    def loss(b):
        scaled = z - b
        log_probs = scaled - log_sum_exp(scaled, axis=-1, keepdims=True)
        return float(-log_probs[np.arange(N), y].mean() + lam * (b * b).sum())

    p = softmax(z - b0)
    analytic_grad = label_freq - p.mean(axis=0) + 2.0 * lam * b0

    # Finite difference
    eps = 1e-5
    fd = np.zeros(K)
    for k in range(K):
        bp = b0.copy(); bp[k] += eps
        bm = b0.copy(); bm[k] -= eps
        fd[k] = (loss(bp) - loss(bm)) / (2 * eps)
    np.testing.assert_allclose(analytic_grad, fd, atol=5e-4)


# ---------------------------------------------------------------------------
# Inner CV
# ---------------------------------------------------------------------------

def test_inner_lambda_search_picks_some_lambda() -> None:
    """Smoke test: inner CV returns a lambda from the candidate list."""
    rng = np.random.default_rng(11)
    K = NUM_CLASSES
    n_folds = 5
    folds = []
    for f in range(n_folds):
        N_f = 200
        z = rng.standard_normal((N_f, K))
        y = rng.integers(0, K, size=N_f).astype(np.int64)
        folds.append(FoldOOF(
            fold=f,
            logits=z,
            labels=y,
            filenames=[f"clip_{f}_{i}" for i in range(N_f)],
            performers=[f"perf_{f}"] * N_f,
        ))
    lams = [0.001, 0.01, 0.1, 1.0]
    best, scores = inner_lambda_search(folds, lams, np.random.default_rng(0))
    assert best in lams
    assert set(scores.keys()) == set(lams)


def test_inner_lambda_search_prefers_large_lambda_for_pure_noise() -> None:
    """If labels are uncorrelated with logits (pure noise), the inner CV
    should prefer larger lambdas (closer to no calibration)."""
    rng = np.random.default_rng(13)
    K = NUM_CLASSES
    n_folds = 4
    folds = []
    for f in range(n_folds):
        N_f = 400
        z = rng.standard_normal((N_f, K)) * 0.5
        # pure noise labels: uncorrelated with z
        y = rng.integers(0, K, size=N_f).astype(np.int64)
        folds.append(FoldOOF(
            fold=f, logits=z, labels=y,
            filenames=[f"c_{f}_{i}" for i in range(N_f)],
            performers=[f"p_{f}"] * N_f,
        ))
    lams = [0.001, 0.1, 10.0, 1000.0]
    best, scores = inner_lambda_search(folds, lams, np.random.default_rng(0))
    # The smallest lambda has the most freedom to overfit train noise → worse val NLL
    # The largest lambda forces b≈0, which is the right answer for pure noise.
    assert scores[1000.0] <= scores[0.001] + 0.02


def test_inner_lambda_search_rejects_empty_lambdas() -> None:
    """Passing an empty lambda list should raise immediately."""
    rng = np.random.default_rng(0)
    K = NUM_CLASSES
    folds = []
    for f in range(3):
        z = rng.standard_normal((50, K))
        y = rng.integers(0, K, size=50).astype(np.int64)
        folds.append(FoldOOF(
            fold=f, logits=z, labels=y,
            filenames=[f"c_{f}_{i}" for i in range(50)],
            performers=[f"p_{f}"] * 50,
        ))
    with pytest.raises(ValueError, match="lambdas"):
        inner_lambda_search(folds, [], np.random.default_rng(0))


def test_inner_lambda_search_sample_weighted_for_unequal_folds() -> None:
    """Verify the aggregated NLL is exactly the sample-weighted mean
    (= NLL on the union of held-out samples) and NOT the unweighted
    fold-mean. We construct folds with strongly differing sizes and NLLs
    and check that the returned scores match the sample-weighted formula
    closely, while differing from the unweighted formula by >10x.
    """
    rng = np.random.default_rng(31)
    K = NUM_CLASSES
    # Two large "easy" folds (1000 samples, NLL ≈ 0) and one tiny "hard"
    # fold (100 samples, NLL ≈ ln(K)). Inner LPO holds each out once, so:
    #   when held=easy_i: train includes hard, val=easy → val NLL ≈ 0
    #   when held=hard:   train includes both easies, val=hard → val NLL ≈ ln(K)
    N_easy, N_hard = 1000, 100
    y_easy = rng.integers(0, K, size=N_easy).astype(np.int64)
    z_easy = rng.standard_normal((N_easy, K)) * 0.1
    z_easy[np.arange(N_easy), y_easy] += 30.0  # huge boost on true class

    z_hard = rng.standard_normal((N_hard, K)) * 0.5
    y_hard = rng.integers(0, K, size=N_hard).astype(np.int64)

    fold_easy_a = FoldOOF(
        fold=0, logits=z_easy, labels=y_easy,
        filenames=[f"a_{i}" for i in range(N_easy)],
        performers=["p_a"] * N_easy,
    )
    fold_easy_b = FoldOOF(
        fold=1, logits=z_easy.copy(), labels=y_easy.copy(),
        filenames=[f"b_{i}" for i in range(N_easy)],
        performers=["p_b"] * N_easy,
    )
    fold_hard = FoldOOF(
        fold=2, logits=z_hard, labels=y_hard,
        filenames=[f"h_{i}" for i in range(N_hard)],
        performers=["p_hard"] * N_hard,
    )
    folds = [fold_easy_a, fold_easy_b, fold_hard]
    lams = [10.0]  # large lambda so b ≈ 0; nll is then dominated by raw logits
    _, scores = inner_lambda_search(folds, lams, np.random.default_rng(0))

    # Recreate the per-fold NLLs by hand using the same large-lambda regime
    # (b ≈ 0 → cal ≈ raw logits).
    nll_easy = nll_loss(z_easy, y_easy)  # ≈ 0
    nll_hard = nll_loss(z_hard, y_hard)  # ≈ ln(K)
    # Inner LPO produces 3 (val_nll, n_val) pairs:
    weighted_expected = (
        nll_easy * N_easy + nll_easy * N_easy + nll_hard * N_hard
    ) / (N_easy + N_easy + N_hard)
    unweighted_expected = (nll_easy + nll_easy + nll_hard) / 3.0

    # Sample-weighted: ~ (0 * 2000 + ln(K) * 100) / 2100 ≈ ln(K)/21 ≈ 0.118
    # Unweighted:       ~ (0 + 0 + ln(K)) / 3                ≈ 0.828
    # The two answers must differ by ~7x; verify our implementation lands
    # very close to the weighted prediction and far from the unweighted one.
    actual = scores[10.0]
    assert abs(actual - weighted_expected) < 0.05, (
        f"scores[10.0]={actual:.4f} differs from sample-weighted "
        f"expected={weighted_expected:.4f}"
    )
    # Must be unambiguously closer to weighted than unweighted.
    assert abs(actual - weighted_expected) < abs(actual - unweighted_expected) / 5.0, (
        f"scores[10.0]={actual:.4f} too close to unweighted "
        f"({unweighted_expected:.4f}); weighted={weighted_expected:.4f}"
    )


def test_inner_lambda_search_rejects_single_fold() -> None:
    """LPO inner CV needs >= 2 folds (need to leave one out)."""
    rng = np.random.default_rng(0)
    K = NUM_CLASSES
    only_fold = FoldOOF(
        fold=0, logits=rng.standard_normal((50, K)),
        labels=rng.integers(0, K, size=50).astype(np.int64),
        filenames=[f"c_{i}" for i in range(50)],
        performers=["p"] * 50,
    )
    with pytest.raises(ValueError, match=">= 2"):
        inner_lambda_search([only_fold], [0.1], rng)


# ---------------------------------------------------------------------------
# Direction consistency
# ---------------------------------------------------------------------------

def test_direction_consistency_all_agree() -> None:
    full_b = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    per_fold = [full_b.copy() for _ in range(10)]
    out = direction_consistency(per_fold, full_b)
    assert out["overall_agreement_rate"] == 1.0
    assert out["num_classes_above_75pct"] == 12


def test_direction_consistency_all_disagree() -> None:
    full_b = np.array([1.0] * 12)
    per_fold = [-1.0 * np.ones(12) for _ in range(10)]
    out = direction_consistency(per_fold, full_b)
    assert out["overall_agreement_rate"] == 0.0
    assert out["num_classes_above_75pct"] == 0


def test_direction_consistency_zero_treated_as_match() -> None:
    full_b = np.zeros(12)
    per_fold = [1e-5 * np.ones(12) for _ in range(10)]  # below eps default 1e-3
    out = direction_consistency(per_fold, full_b, eps=1e-3)
    assert out["overall_agreement_rate"] == 1.0


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_zero_when_identical() -> None:
    rng = np.random.default_rng(20)
    N = 500
    labels = rng.integers(0, NUM_CLASSES, size=N).astype(np.int64)
    preds = rng.integers(0, NUM_CLASSES, size=N).astype(np.int64)
    out = paired_bootstrap_f1_diff(preds, preds, labels, n_iter=200, seed=0)
    assert abs(out["mean_diff_pp"]) < 1e-10
    assert out["std_diff_pp"] < 1e-9


def test_bootstrap_positive_when_strict_improvement() -> None:
    """If preds_b is always correct and preds_a is always wrong, diff > 0."""
    rng = np.random.default_rng(21)
    N = 500
    labels = rng.integers(0, NUM_CLASSES, size=N).astype(np.int64)
    preds_a = (labels + 1) % NUM_CLASSES  # always wrong
    preds_b = labels.copy()  # always correct
    out = paired_bootstrap_f1_diff(preds_a, preds_b, labels, n_iter=200, seed=0)
    # F1(b) = 1.0, F1(a) = 0 → diff ~ 100 pp
    assert out["mean_diff_pp"] > 80.0
    assert out["ci_2.5_pp"] > 50.0


def test_bootstrap_ci_brackets_mean() -> None:
    rng = np.random.default_rng(22)
    N = 500
    labels = rng.integers(0, NUM_CLASSES, size=N).astype(np.int64)
    preds_a = rng.integers(0, NUM_CLASSES, size=N).astype(np.int64)
    preds_b = rng.integers(0, NUM_CLASSES, size=N).astype(np.int64)
    out = paired_bootstrap_f1_diff(preds_a, preds_b, labels, n_iter=200, seed=0)
    assert out["ci_2.5_pp"] <= out["mean_diff_pp"] <= out["ci_97.5_pp"]


def test_bootstrap_n_iter_one_no_nan_std() -> None:
    """Pathological smoke case: n_iter=1 should not produce NaN std."""
    rng = np.random.default_rng(40)
    N = 100
    labels = rng.integers(0, NUM_CLASSES, size=N).astype(np.int64)
    preds_a = rng.integers(0, NUM_CLASSES, size=N).astype(np.int64)
    preds_b = rng.integers(0, NUM_CLASSES, size=N).astype(np.int64)
    out = paired_bootstrap_f1_diff(preds_a, preds_b, labels, n_iter=1, seed=0)
    assert np.isfinite(out["mean_diff_pp"])
    assert np.isfinite(out["std_diff_pp"])
    assert out["std_diff_pp"] == 0.0


def test_bootstrap_rejects_zero_iter() -> None:
    """n_iter=0 makes no statistical sense and should raise."""
    labels = np.array([0, 1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="n_iter"):
        paired_bootstrap_f1_diff(labels, labels, labels, n_iter=0, seed=0)


def test_bootstrap_uses_paired_indices() -> None:
    """Verify pairing: when ``preds_b`` is a deterministic shift of
    ``preds_a`` (so per-clip diff is constant), the bootstrap diff has
    zero variance — but only if the SAME indices are used to evaluate
    both methods. Independent (unpaired) bootstraps would produce
    significant diff variance even with constant per-clip diffs.
    """
    rng = np.random.default_rng(50)
    N = 300
    labels = rng.integers(0, NUM_CLASSES, size=N).astype(np.int64)
    # B is exactly correct everywhere; A is exactly wrong everywhere.
    preds_a = (labels + 1) % NUM_CLASSES
    preds_b = labels.copy()
    out = paired_bootstrap_f1_diff(preds_a, preds_b, labels, n_iter=200, seed=7)
    # In *every* paired resample, F1(B)=1.0 and F1(A)=0.0, so diff = +100pp
    # exactly. CI width MUST be zero. (Unpaired would give large width.)
    assert out["mean_diff_pp"] == pytest.approx(100.0, abs=1e-9)
    assert out["std_diff_pp"] == pytest.approx(0.0, abs=1e-9)
    assert out["ci_2.5_pp"] == pytest.approx(100.0, abs=1e-9)
    assert out["ci_97.5_pp"] == pytest.approx(100.0, abs=1e-9)


def test_bootstrap_seed_determinism() -> None:
    """Two callers with the same seed must produce byte-identical results."""
    rng = np.random.default_rng(60)
    N = 300
    labels = rng.integers(0, NUM_CLASSES, size=N).astype(np.int64)
    preds_a = rng.integers(0, NUM_CLASSES, size=N).astype(np.int64)
    preds_b = rng.integers(0, NUM_CLASSES, size=N).astype(np.int64)
    out1 = paired_bootstrap_f1_diff(preds_a, preds_b, labels, n_iter=50, seed=7)
    out2 = paired_bootstrap_f1_diff(preds_a, preds_b, labels, n_iter=50, seed=7)
    assert out1["mean_diff_pp"] == out2["mean_diff_pp"]
    assert out1["ci_2.5_pp"] == out2["ci_2.5_pp"]
    assert out1["ci_97.5_pp"] == out2["ci_97.5_pp"]


# ---------------------------------------------------------------------------
# macro_f1 sanity
# ---------------------------------------------------------------------------

def test_macro_f1_perfect() -> None:
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
    preds = labels.copy()
    assert macro_f1(preds, labels) == pytest.approx(1.0)


def test_macro_f1_all_wrong() -> None:
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
    preds = (labels + 1) % NUM_CLASSES
    f1 = macro_f1(preds, labels)
    assert f1 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# go_no_go_decision boundary cases
# ---------------------------------------------------------------------------

def _make_metrics_dict(overall_f1: float, per_class_f1: dict[str, float]) -> dict:
    return {
        "overall_f1_seed_mean": overall_f1,
        "per_class_f1_seed_mean": per_class_f1,
    }


def _baseline_per_class() -> dict[str, float]:
    # All classes baseline at 30%; priority classes set explicitly so deltas matter.
    base = {
        "anger": 0.30, "contempt": 0.30, "disgust": 0.30, "fear": 0.30,
        "joy": 0.30, "sadness": 0.30, "surprise": 0.30, "jealousy": 0.30,
        "shame": 0.30, "guilt": 0.30, "gratitude": 0.30, "pride": 0.30,
    }
    return base


def test_go_no_go_passes_when_all_criteria_met() -> None:
    base_pcf1 = _baseline_per_class()
    cal_pcf1 = dict(base_pcf1)
    # +0.30pp overall (above the 0.20pp threshold)
    cal_pcf1["guilt"] += 0.05
    cal_pcf1["sadness"] += 0.05
    cal_pcf1["pride"] += 0.05
    method_metrics = {
        "baseline": _make_metrics_dict(0.30, base_pcf1),
        "good_method": _make_metrics_dict(0.3030, cal_pcf1),
    }
    bootstrap = {"good_method": {"ci_2.5_pp": -0.05, "ci_97.5_pp": +0.5}}
    direction = {}
    out = go_no_go_decision(method_metrics, bootstrap, direction)
    assert out["good_method"]["verdict"] == "GO"


def test_go_no_go_fails_when_delta_below_threshold() -> None:
    """+0.10pp F1 (below the +0.20pp threshold) → criterion 1 fails → NO-GO."""
    base_pcf1 = _baseline_per_class()
    cal_pcf1 = dict(base_pcf1)
    cal_pcf1["guilt"] += 0.05
    cal_pcf1["sadness"] += 0.05
    method_metrics = {
        "baseline": _make_metrics_dict(0.30, base_pcf1),
        "weak": _make_metrics_dict(0.3010, cal_pcf1),
    }
    bootstrap = {"weak": {"ci_2.5_pp": +0.05, "ci_97.5_pp": +0.20}}
    out = go_no_go_decision(method_metrics, bootstrap, {})
    assert out["weak"]["verdict"] == "NO-GO"
    assert out["weak"]["criterion_1_plus_0.20pt"] is False


def test_go_no_go_fails_when_priority_classes_dont_improve() -> None:
    base_pcf1 = _baseline_per_class()
    cal_pcf1 = dict(base_pcf1)
    # Big overall improvement, but only ONE priority class improves
    cal_pcf1["guilt"] += 0.05
    # sadness, pride unchanged
    method_metrics = {
        "baseline": _make_metrics_dict(0.30, base_pcf1),
        "uneven": _make_metrics_dict(0.3050, cal_pcf1),
    }
    bootstrap = {"uneven": {"ci_2.5_pp": -0.02, "ci_97.5_pp": +0.40}}
    out = go_no_go_decision(method_metrics, bootstrap, {})
    assert out["uneven"]["verdict"] == "NO-GO"
    assert out["uneven"]["criterion_2_2plus_priority_improve"] is False


def test_go_no_go_fails_when_ci_lower_too_negative() -> None:
    base_pcf1 = _baseline_per_class()
    cal_pcf1 = dict(base_pcf1)
    for c in PRIORITY_CLASSES:
        cal_pcf1[c] += 0.05
    method_metrics = {
        "baseline": _make_metrics_dict(0.30, base_pcf1),
        "risky": _make_metrics_dict(0.3030, cal_pcf1),
    }
    # CI lower bound = -0.30pp (below -0.10pp threshold)
    bootstrap = {"risky": {"ci_2.5_pp": -0.30, "ci_97.5_pp": +0.50}}
    out = go_no_go_decision(method_metrics, bootstrap, {})
    assert out["risky"]["verdict"] == "NO-GO"
    assert out["risky"]["criterion_3_ci_lower_above_minus_0.10pp"] is False


def test_go_no_go_fails_when_direction_consistency_low() -> None:
    base_pcf1 = _baseline_per_class()
    cal_pcf1 = dict(base_pcf1)
    for c in PRIORITY_CLASSES:
        cal_pcf1[c] += 0.05
    method_metrics = {
        "baseline": _make_metrics_dict(0.30, base_pcf1),
        "unstable": _make_metrics_dict(0.3030, cal_pcf1),
    }
    bootstrap = {"unstable": {"ci_2.5_pp": -0.05, "ci_97.5_pp": +0.50}}
    direction = {"unstable": {"num_classes_above_75pct": 5}}  # below 9
    out = go_no_go_decision(method_metrics, bootstrap, direction)
    assert out["unstable"]["verdict"] == "NO-GO"
    assert out["unstable"]["criterion_4_direction_consistency_75pct_9classes"] is False


def test_go_no_go_boundary_delta_exactly_0_20pp() -> None:
    """ΔF1 == +0.20pp exactly should PASS criterion 1 (>=)."""
    base_pcf1 = _baseline_per_class()
    cal_pcf1 = dict(base_pcf1)
    for c in PRIORITY_CLASSES:
        cal_pcf1[c] += 0.05
    method_metrics = {
        "baseline": _make_metrics_dict(0.30, base_pcf1),
        "edge": _make_metrics_dict(0.3020, cal_pcf1),
    }
    bootstrap = {"edge": {"ci_2.5_pp": -0.05, "ci_97.5_pp": +0.50}}
    out = go_no_go_decision(method_metrics, bootstrap, {})
    assert out["edge"]["criterion_1_plus_0.20pt"] is True
    assert out["edge"]["verdict"] == "GO"


def test_go_no_go_boundary_ci_exactly_minus_0_10pp() -> None:
    """CI lower bound == -0.10pp exactly should FAIL criterion 3 (strict >)."""
    base_pcf1 = _baseline_per_class()
    cal_pcf1 = dict(base_pcf1)
    for c in PRIORITY_CLASSES:
        cal_pcf1[c] += 0.05
    method_metrics = {
        "baseline": _make_metrics_dict(0.30, base_pcf1),
        "edge": _make_metrics_dict(0.3030, cal_pcf1),
    }
    bootstrap = {"edge": {"ci_2.5_pp": -0.10, "ci_97.5_pp": +0.40}}
    out = go_no_go_decision(method_metrics, bootstrap, {})
    assert out["edge"]["criterion_3_ci_lower_above_minus_0.10pp"] is False
    assert out["edge"]["verdict"] == "NO-GO"
