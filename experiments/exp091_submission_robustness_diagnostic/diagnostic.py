"""exp091 — submission-robustness diagnostic (zero-training, OOF-only).

Design doc: "" (4-model debate consensus: Codex GPT-5.5/5.4
+ Opus x2, 2 rounds). The orthogonal-ensemble well is dry (6 hard negatives +
ceiling analysis rho=0.526). This experiment NEVER touches the submission
file; its only output is a pre-registered first-match-wins decision over
{ship locked 36.94%, ship a strictly-better parameter-free variant,
unlock a constrained exp092}.

Related:
  - experiments/exp090_ensemble_patterns/sweep.py  (OOF-logits data layout)
  - docs/analysis/exp090_ensemble_patterns_verdict.md  (locked submission)
  - docs/analysis/explainability_report.md  §8  (performer variance crux)

Key bug this fixes: exp090 results.json `D_loo_11way` was computed on the
softmax-mean base (~0.358), NOT the locked logit-mean (0.36935). All
member-attribution here is recomputed on the logit-mean base.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import f1_score

from diema.data.parser import IDX_TO_EMOTION, emotion_to_label

PROD = [
    "exp002_a04_smooth", "exp003_ctrgcn_a00", "exp004_skateformer_a01",
    "exp006_protogcn_a00", "exp020_conv1d_transformer_a01",
    "exp023_keypoint_pool_mlp_a00", "exp034_regionaware_convtr_a00",
]
MEMBERS11 = PROD + ["MB", "C3D", "MAMP60xsub", "MAMP120xset"]

# locked / reference OOF Macro-F1 (exp090 results.json, byte-stable constants)
LOCKED_F1 = 0.3693539601421651      # 11-way logit-mean == geometric-mean
BASELINE_7WAY = 0.33723676836425226
SOFTMAX_11WAY = 0.357812826188972
RANK_11WAY = 0.36470194410290513

EMO = {v: k for k, v in IDX_TO_EMOTION.items()}
# confusion pairs from exp087/explainability_report; (a,b,directed)
CONFUSION_PAIRS = [
    (EMO["jealousy"], EMO["contempt"], False),
    (EMO["guilt"], EMO["sadness"], True),   # directed guilt -> sadness
    (EMO["fear"], EMO["surprise"], False),
]

# pre-registered gate constants (F1 fraction units)
GAIN_DELTA = 0.0020          # +0.20pp OOF improvement to ship a variant
PERCLASS_REG = 0.0030        # per-class F1 drop counted as a >0.30pp regression
MAX_PERCLASS_REG = 2         # at most this many classes may regress
MIN_FOLDS_NONNEG = 7         # variant must be non-negative on >= this many folds
FOLD_CONC_LIMIT = 0.50       # one fold may not own > this share of the gain
RESCUER_MIN_FOLDS = 6        # rescuer must be stable on >= this many of 10 folds


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def combine(prob_list, method):
    """Parameter-free ensemble combination. P: list of (N,12) prob arrays."""
    P = np.stack(prob_list, axis=0)  # (K, N, 12)
    if method == "softmax_mean":
        return P.mean(axis=0)
    if method in ("logit_mean", "geometric_mean"):
        # softmax of mean log-prob == geometric mean == production raw-logit conv
        return softmax(np.log(np.clip(P, 1e-12, 1.0)).mean(axis=0))
    if method == "rank_mean":
        R = np.stack([rankdata(p, axis=-1) for p in P], axis=0).mean(axis=0)
        return R / R.sum(-1, keepdims=True)
    if method == "median":
        m = np.median(P, axis=0)
        return m / m.sum(-1, keepdims=True)
    if method == "trimmed_mean":
        if P.shape[0] < 3:
            raise ValueError("trimmed_mean needs >=3 members")
        s = np.sort(P, axis=0)[1:-1]  # drop per-cell min and max
        m = s.mean(axis=0)
        return m / m.sum(-1, keepdims=True)
    raise ValueError(method)


def macro_f1(probs, y):
    return float(f1_score(y, probs.argmax(-1), average="macro"))


def per_class_f1(probs, y):
    return f1_score(y, probs.argmax(-1), average=None, labels=list(range(12)))


def _align(logits, fns, ref):
    m = {str(f): i for i, f in enumerate(fns)}
    return softmax(np.stack([logits[m[str(f)]] for f in ref]))


def load_sources(splits, artifacts_root="output/artifacts"):
    """Return (src dict member->(N,12) prob, y, performer_ids, fold_ids).

    ref = concat of all 10 LPO val folds (exp090 convention). performer id =
    first two underscore tokens of the filename (e.g. 'JP_06'); the LPO unit.
    """
    ref = np.array([str(fn) for fid in range(10)
                    for fn, _ in splits[fid]["val"]])
    y = np.array([emotion_to_label(f) for f in ref], dtype=np.int64)
    performer = np.array(["_".join(s.split("_")[:2]) for s in ref])
    fold_ids = np.concatenate(
        [np.full(len(splits[fid]["val"]), fid, np.int64) for fid in range(10)])
    root = Path(artifacts_root)
    src = {}
    for nm in PROD:
        fl = [np.load(root / f"{nm}/fold_{fid:02d}/oof_logits.npy")
              for fid in range(10)]
        src[nm] = softmax(np.concatenate(fl, 0))
    pred = Path("output/predictions")
    spec = {
        "MB": ("motionbert_lite_per_joint/exp081_motionbert_oof_logits.npy",
               "motionbert_lite_per_joint/exp081_motionbert_oof_filenames.npy"),
        "C3D": ("c3d_contact_only/oof_logits.npy",
                "c3d_contact_only/oof_filenames.npy"),
        "MAMP60xsub": ("mamp_ntu60xsub/oof_logits.npy",
                       "mamp_ntu60xsub/oof_filenames.npy"),
        "MAMP120xset": ("mamp_ntu120xset/oof_logits.npy",
                        "mamp_ntu120xset/oof_filenames.npy"),
    }
    for k, (lg, fn) in spec.items():
        src[k] = _align(np.load(pred / lg),
                        np.load(pred / fn, allow_pickle=True), ref)
    return src, y, performer, fold_ids


def locked_probs(src):
    return combine([src[m] for m in MEMBERS11], "logit_mean")


# ── step 1: reproducibility lock ──────────────────────────────────────────
def reproducibility(src, y):
    lm = combine([src[m] for m in MEMBERS11], "logit_mean")
    gm = combine([src[m] for m in MEMBERS11], "geometric_mean")
    f1_lm, f1_gm = macro_f1(lm, y), macro_f1(gm, y)
    f1_sm = macro_f1(combine([src[m] for m in MEMBERS11], "softmax_mean"), y)
    f1_7 = macro_f1(combine([src[m] for m in PROD], "softmax_mean"), y)
    f1_rk = macro_f1(combine([src[m] for m in MEMBERS11], "rank_mean"), y)
    ok = (np.allclose(lm, gm) and abs(f1_lm - LOCKED_F1) < 1e-6
          and abs(f1_lm - f1_gm) < 1e-9 and abs(f1_7 - BASELINE_7WAY) < 1e-6)
    return {
        "logit_mean_f1": f1_lm, "geometric_mean_f1": f1_gm,
        "softmax_mean_f1": f1_sm, "rank_mean_f1": f1_rk, "sevenway_f1": f1_7,
        "logit_eq_geometric": bool(np.allclose(lm, gm)),
        "matches_locked_constant": bool(abs(f1_lm - LOCKED_F1) < 1e-6),
        "reproducible": bool(ok),
    }


# ── step 2: leave-one-out on the LOGIT-MEAN base (the bug fix) ─────────────
def loo_logitmean(src, y):
    full = macro_f1(locked_probs(src), y)
    loo = {}
    for drop in MEMBERS11:
        keep = [src[m] for m in MEMBERS11 if m != drop]
        f = macro_f1(combine(keep, "logit_mean"), y)
        loo[drop] = {"f1": f, "delta_pp": (f - full) * 100.0}
    return {"full_logitmean_f1": full, "loo": loo,
            "most_critical": min(loo, key=lambda k: loo[k]["f1"])}


# ── step 3: per-fold decomposition of the +1.15pp logit-vs-softmax gain ────
def fold_decomposition(src, y, fold_ids):
    lm = locked_probs(src)
    sm = combine([src[m] for m in MEMBERS11], "softmax_mean")
    global_delta = (macro_f1(lm, y) - macro_f1(sm, y))
    per_fold = []
    for fid in range(10):
        idx = fold_ids == fid
        d = macro_f1(lm[idx], y[idx]) - macro_f1(sm[idx], y[idx])
        per_fold.append({"fold": fid, "delta_pp": d * 100.0,
                         "n": int(idx.sum())})
    pos = [f["delta_pp"] for f in per_fold if f["delta_pp"] > 0]
    sum_pos = sum(pos) if pos else 0.0
    max_share = (max(pos) / sum_pos) if sum_pos > 0 else 1.0
    return {
        "global_delta_pp": global_delta * 100.0,
        "per_fold": per_fold,
        "n_folds_positive": int(sum(1 for f in per_fold if f["delta_pp"] > 0)),
        "max_single_fold_share": float(max_share),
        "concentrated": bool(max_share > FOLD_CONC_LIMIT),
    }


# ── step 4: per-performer stratified bootstrap CI ─────────────────────────
def performer_bootstrap(src, y, performer, n_boot=2000, seed=42):
    probs = locked_probs(src)
    rng = np.random.default_rng(seed)
    perfs = np.unique(performer)
    by_perf = {p: np.where(performer == p)[0] for p in perfs}
    point = macro_f1(probs, y)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.choice(perfs, size=len(perfs), replace=True)
        idx = np.concatenate([by_perf[p] for p in pick])
        stats[b] = macro_f1(probs[idx], y[idx])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    # single-performer leave-one-out swing (no <=2 performers dominate)
    swings = {}
    for p in perfs:
        keep = performer != p
        swings[p] = (macro_f1(probs[keep], y[keep]) - point) * 100.0
    top2 = sorted(swings.values(), key=abs, reverse=True)[:2]
    return {
        "point_f1": point, "ci_lo": float(lo), "ci_hi": float(hi),
        "ci_lo_gt_7way": bool(lo > BASELINE_7WAY),
        "top2_perf_abs_swing_pp": [abs(x) for x in top2],
        "max_perf_swing_pp": float(max(swings.values(), key=abs)),
    }


# ── step 5: confusion-pair attribution (on the logit-mean base) ───────────
def _pair_errors(pred, y, a, b, directed):
    if directed:  # count true==a predicted==b only
        return int(np.sum((y == a) & (pred == b)))
    return int(np.sum(((y == a) & (pred == b)) | ((y == b) & (pred == a))))


def confusion_pair_attribution(src, y, fold_ids):
    full_pred = locked_probs(src).argmax(-1)
    out = []
    for a, b, directed in CONFUSION_PAIRS:
        base = _pair_errors(full_pred, y, a, b, directed)
        deltas = {}
        for drop in MEMBERS11:
            keep = [src[m] for m in MEMBERS11 if m != drop]
            p = combine(keep, "logit_mean").argmax(-1)
            deltas[drop] = _pair_errors(p, y, a, b, directed) - base
        rescuer = max(deltas, key=lambda k: deltas[k])  # removing it hurts most
        # fold consistency: in how many folds does removing rescuer raise error
        keep = [src[m] for m in MEMBERS11 if m != rescuer]
        rp = combine(keep, "logit_mean").argmax(-1)
        consistent = 0
        for fid in range(10):
            m = fold_ids == fid
            if (_pair_errors(rp[m], y[m], a, b, directed)
                    > _pair_errors(full_pred[m], y[m], a, b, directed)):
                consistent += 1
        out.append({
            "pair": f"{IDX_TO_EMOTION[a]}{'->' if directed else '<->'}"
                    f"{IDX_TO_EMOTION[b]}",
            "base_errors": base, "rescuer": rescuer,
            "rescuer_error_increase": deltas[rescuer],
            "rescuer_fold_consistency": consistent,
            "stable": bool(consistent >= RESCUER_MIN_FOLDS
                           and deltas[rescuer] > 0),
        })
    return {"pairs": out,
            "any_stable_rescuer": bool(any(p["stable"] for p in out))}


# ── step 6: parameter-free aggregation sweep ──────────────────────────────
def parameter_free_sweep(src, y):
    """Only argmax-affecting parameter-free rules. Temperature is omitted
    (rank-invariant under argmax => no-op on hard-label Macro-F1);
    train-fold class-prior is omitted (a fitted 12-vector => LPO-overfit,
    hard-negative #3). Per the Round-2 Opus critique of Codex-P3."""
    locked = locked_probs(src)
    base_pc = per_class_f1(locked, y)
    res = {}
    for method in ("median", "trimmed_mean"):
        pr = combine([src[m] for m in MEMBERS11], method)
        f = macro_f1(pr, y)
        reg = int(np.sum((base_pc - per_class_f1(pr, y)) > PERCLASS_REG))
        res[method] = {"f1": f, "delta_pp": (f - LOCKED_F1) * 100.0,
                       "classes_regressed_gt_0_30pp": reg}
    return res


# ── decision: single pre-registered first-match-wins rule ─────────────────
def decide(repro, fold_dec, boot, sweep, attribution, src, y, fold_ids):
    # D0 guard — robustness fail-safe -> ship locked (NOT rank-mean: 0.3647
    # is strictly worse than the locked 0.3694)
    if (not repro["reproducible"] or not boot["ci_lo_gt_7way"]
            or fold_dec["concentrated"]):
        reason = []
        if not repro["reproducible"]:
            reason.append("reproducibility_failed")
        if not boot["ci_lo_gt_7way"]:
            reason.append(f"ci_lo({boot['ci_lo']:.4f})<=7way")
        if fold_dec["concentrated"]:
            reason.append(f"fold_gain_concentrated({fold_dec['max_single_fold_share']:.2f})")
        return {"action": "SHIP_LOCKED", "branch": "D0_guard",
                "reasons": reason, "submission": "ensemble_11way_logitmean.csv"}

    # D1 — a strictly-better parameter-free variant (locked computed lazily,
    # only once a candidate clears the +0.20pp gate)
    locked = None
    for name, r in sorted(sweep.items(), key=lambda kv: -kv[1]["f1"]):
        if r["delta_pp"] < GAIN_DELTA * 100.0:
            continue
        if locked is None:
            locked = locked_probs(src)
        vp = combine([src[m] for m in MEMBERS11], name)
        nonneg = 0
        for fid in range(10):
            m = fold_ids == fid
            if macro_f1(vp[m], y[m]) >= macro_f1(locked[m], y[m]):
                nonneg += 1
        if (nonneg >= MIN_FOLDS_NONNEG
                and r["classes_regressed_gt_0_30pp"] <= MAX_PERCLASS_REG):
            return {"action": "SHIP_VARIANT", "branch": "D1_param_free",
                    "variant": name, "variant_f1": r["f1"],
                    "folds_nonneg": nonneg}

    # D2 — unlock constrained exp092 (P1 frozen nested-LPO re-rank)
    if attribution["any_stable_rescuer"]:
        stable = [p for p in attribution["pairs"] if p["stable"]]
        return {"action": "UNLOCK_EXP092", "branch": "D2_stable_rescuer",
                "stable_pairs": stable,
                "note": "exp092 = P1 frozen parameter-free re-rank, "
                        "nested-LPO scalar, |inner-outer|<0.15pp; submission "
                        "stays locked until exp092 fully passes"}

    # D3 — default, dominant outcome
    return {"action": "SHIP_LOCKED", "branch": "D3_default",
            "submission": "ensemble_11way_logitmean.csv",
            "locked_f1": LOCKED_F1}


def run(splits_pkl="data/diema_challenge/processed/split_lpo_10fold.pkl",
        artifacts_root="output/artifacts",
        out_json="experiments/exp091_submission_robustness_diagnostic/results.json"):
    t0 = time.time()
    with open(splits_pkl, "rb") as f:
        splits = pickle.load(f)
    src, y, performer, fold_ids = load_sources(splits, artifacts_root)
    repro = reproducibility(src, y)
    loo = loo_logitmean(src, y)
    fold_dec = fold_decomposition(src, y, fold_ids)
    boot = performer_bootstrap(src, y, performer)
    attribution = confusion_pair_attribution(src, y, fold_ids)
    sweep = parameter_free_sweep(src, y)
    decision = decide(repro, fold_dec, boot, sweep, attribution,
                      src, y, fold_ids)
    out = {
        "step1_reproducibility": repro,
        "step2_loo_logitmean": loo,
        "step3_fold_decomposition": fold_dec,
        "step4_performer_bootstrap": boot,
        "step5_confusion_pair_attribution": attribution,
        "step6_parameter_free_sweep": sweep,
        "decision": decision,
        "elapsed_sec": time.time() - t0,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    r = run()
    d = r["decision"]
    print(f"[exp091] reproducible={r['step1_reproducibility']['reproducible']} "
          f"locked={r['step1_reproducibility']['logit_mean_f1']*100:.2f}%")
    print(f"[exp091] CI95=[{r['step4_performer_bootstrap']['ci_lo']*100:.2f}, "
          f"{r['step4_performer_bootstrap']['ci_hi']*100:.2f}]  "
          f"fold_concentration={r['step3_fold_decomposition']['max_single_fold_share']:.2f}")
    print(f"[exp091] DECISION: {d['action']}  ({d['branch']})")
    print(f"[exp091] saved {r['elapsed_sec']:.0f}s")
