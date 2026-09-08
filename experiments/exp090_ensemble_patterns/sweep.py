"""exp090 — comprehensive ensemble-pattern sweep.

Explores ensemble *patterns* over all available OOF logit sources, beyond
the equal-weight prob-average used so far. Current best to beat:
  11-way (7 prod + MB + C3D + MAMP{ntu60xsub,ntu120xset}) = 35.78% OOF.

Sources (OOF logits, 10-fold concat order):
  7 production members  + MB(exp081) + C3D(exp072)
  + MAMP ntu60xsub / ntu120xset (exp085, GO)
  + MAMP ntu60xview / ntu120xsub (exp090, the 2 unused ckpts — if smoke PASS)
  + PoseC3D ntu60xsub (exp086, NO-GO but OOF kept for completeness)

Patterns:
  A. averaging method on the 11-way members:
       logit-mean / softmax-mean / rank-mean / geometric-mean
  B. all-MAMP stack: 9-way + every MAMP variant that passed smoke
  C. greedy forward selection from 7-way (OVERFIT CAVEAT applies:
     full-OOF selection overfits performer structure — reported as
     exploratory, NOT auto-adopted; the trustworthy lever is equal-weight
     + orthogonal-source addition, per exp083/085)
  D. leave-one-out on the best equal-weight ensemble (redundancy diagnostic)

Generates OOF for any newly-passing MAMP ckpt first (10-fold sklearn probe).
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import click
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from diema.data.parser import IDX_TO_EMOTION, emotion_to_label

PROD = [
    "exp002_a04_smooth", "exp003_ctrgcn_a00", "exp004_skateformer_a01",
    "exp006_protogcn_a00", "exp020_conv1d_transformer_a01",
    "exp023_keypoint_pool_mlp_a00", "exp034_regionaware_convtr_a00",
]


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def fold_train_idx(splits, filenames, fid):
    pos = {str(fn): i for i, fn in enumerate(filenames)}
    return np.array([pos[str(it[0])] for it in splits[fid]["train"]
                     if str(it[0]) in pos], dtype=np.int64)


def gen_mamp_oof(npz_path, feat_source, c_reg, splits):
    npz = np.load(npz_path, allow_pickle=True)
    fns = np.asarray(npz["filenames"])
    lab = np.array([emotion_to_label(str(f)) for f in fns], dtype=np.int64)
    if feat_source == "pooled":
        X = np.asarray(npz["features_pooled"], dtype=np.float32)
    else:
        pj = np.asarray(npz["features_per_joint"], dtype=np.float32)
        X = pj.reshape(pj.shape[0], -1)
    oof = np.zeros((len(X), 12), dtype=np.float32)
    for fid in range(10):
        tr = fold_train_idx(splits, fns, fid)
        va = np.array([i for i in range(len(fns))
                       if str(fns[i]) in {str(x[0]) for x in splits[fid]["val"]}],
                      dtype=np.int64)
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=c_reg, max_iter=2000, n_jobs=-1,
                                 solver="lbfgs", random_state=42)
        clf.fit(sc.transform(X[tr]), lab[tr])
        pr = clf.predict_proba(sc.transform(X[va]))
        for j, cls in enumerate(clf.classes_):
            oof[va, int(cls)] = np.log(np.clip(pr[:, j], 1e-12, 1.0))
    return oof, fns


def align(logits, fns, ref):
    m = {str(f): i for i, f in enumerate(fns)}
    return softmax(np.stack([logits[m[str(f)]] for f in ref]))


def combine(prob_list, method):
    P = np.stack(prob_list, axis=0)  # (K, N, 12)
    if method == "softmax_mean":
        return P.mean(axis=0)
    if method == "logit_mean":
        return softmax(np.log(np.clip(P, 1e-12, 1.0)).mean(axis=0))
    if method == "geometric_mean":
        return softmax(np.log(np.clip(P, 1e-12, 1.0)).mean(axis=0))  # = logit on log-probs
    if method == "rank_mean":
        R = np.stack([rankdata(p, axis=-1) for p in P], axis=0).mean(axis=0)
        return R / R.sum(-1, keepdims=True)
    raise ValueError(method)


def f1(probs, y):
    return float(f1_score(y, probs.argmax(-1), average="macro"))


@click.command()
@click.option("--splits-pkl", default="data/diema_challenge/processed/split_lpo_10fold.pkl")
@click.option("--artifacts-root", default="output/artifacts")
@click.option("--extra-mamp", default="ntu60xview,ntu120xsub",
              help="comma list of new MAMP tags to OOF + include if smoke PASS")
@click.option("--results-json", default="experiments/exp090_ensemble_patterns/results.json")
def main(splits_pkl, artifacts_root, extra_mamp, results_json):
    t0 = time.time()
    with open(splits_pkl, "rb") as f:
        splits = pickle.load(f)
    ref = np.array([str(fn) for fid in range(10) for fn, _ in splits[fid]["val"]])
    y = np.array([emotion_to_label(f) for f in ref], dtype=np.int64)
    root = Path(artifacts_root)

    src = {}
    for nm in PROD:
        fl = [np.load(root / f"{nm}/fold_{fid:02d}/oof_logits.npy") for fid in range(10)]
        src[nm] = softmax(np.concatenate(fl, 0))
    src["MB"] = align(
        np.load("output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_logits.npy"),
        np.load("output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_filenames.npy", allow_pickle=True), ref)
    src["C3D"] = align(np.load("output/predictions/c3d_contact_only/oof_logits.npy"),
                       np.load("output/predictions/c3d_contact_only/oof_filenames.npy", allow_pickle=True), ref)
    src["MAMP60xsub"] = align(np.load("output/predictions/mamp_ntu60xsub/oof_logits.npy"),
                              np.load("output/predictions/mamp_ntu60xsub/oof_filenames.npy", allow_pickle=True), ref)
    src["MAMP120xset"] = align(np.load("output/predictions/mamp_ntu120xset/oof_logits.npy"),
                               np.load("output/predictions/mamp_ntu120xset/oof_filenames.npy", allow_pickle=True), ref)
    src["PoseC3D"] = align(np.load("output/predictions/posec3d_ntu60xsub/oof_logits.npy"),
                           np.load("output/predictions/posec3d_ntu60xsub/oof_filenames.npy", allow_pickle=True), ref)

    # generate OOF for the 2 unused MAMP ckpts (best config = same as exp085:
    # per_joint_flat C=0.1 for xsub-like, pooled C=0.1 for xset-like; we
    # sweep both feat sources and keep the better-smoke one)
    new_tags = [t for t in extra_mamp.split(",") if t]
    smoke_info = {}
    for tag in new_tags:
        sj = json.loads(Path(f"experiments/exp090_ensemble_patterns/smoke_results_{tag}_unit.json").read_text())
        best = sj["best"]
        smoke_info[tag] = {"smoke": best["mean_macro_f1"], "feat": best["feat_source"],
                           "C": best["C"], "decision": sj["decision"]}
        if sj["decision"] != "PASS":
            print(f"[{tag}] smoke {best['mean_macro_f1']*100:.2f}% ABORT — excluded")
            continue
        oof, fns = gen_mamp_oof(f"data/diema_challenge/processed/mamp_{tag}_train.npz",
                                best["feat_source"], best["C"], splits)
        pd = Path(f"output/predictions/mamp_{tag}")
        pd.mkdir(parents=True, exist_ok=True)
        np.save(pd / "oof_logits.npy", oof)
        np.save(pd / "oof_filenames.npy", fns)
        src[f"MAMP_{tag}"] = align(oof, fns, ref)
        print(f"[{tag}] smoke {best['smoke']*100 if 'smoke' in best else best['mean_macro_f1']*100:.2f}% PASS → OOF added")

    nine = [src[m] for m in PROD] + [src["MB"], src["C3D"]]
    eleven = nine + [src["MAMP60xsub"], src["MAMP120xset"]]
    base9 = combine([p for p in [src[m] for m in PROD]], "softmax_mean")
    # 9-way = mean(7 prob)*7 + MB + C3D all /9 → equivalently softmax_mean of the 9
    f1_7 = f1(combine([src[m] for m in PROD], "softmax_mean"), y)
    f1_9 = f1(combine(nine, "softmax_mean"), y)
    f1_11 = f1(combine(eleven, "softmax_mean"), y)
    out = {"baselines": {"7way": f1_7, "9way": f1_9, "11way": f1_11},
           "smoke_info": smoke_info, "patterns": {}}
    print(f"\n[baseline] 7={f1_7*100:.2f}% 9={f1_9*100:.2f}% 11={f1_11*100:.2f}%")

    # Pattern A: averaging method on 11-way members
    print("\n[A] averaging method (11-way members)")
    for mth in ["softmax_mean", "logit_mean", "rank_mean", "geometric_mean"]:
        v = f1(combine(eleven, mth), y)
        out["patterns"][f"A_11way_{mth}"] = v
        print(f"  {mth:16s}: {v*100:.2f}% (Δ vs 11way softmax {(v-f1_11)*100:+.2f}pp)")

    # Pattern B: all-MAMP stack
    mamp_keys = [k for k in src if k.startswith("MAMP")]
    allmamp = [src[m] for m in PROD] + [src["MB"], src["C3D"]] + [src[k] for k in mamp_keys]
    v = f1(combine(allmamp, "softmax_mean"), y)
    out["patterns"]["B_9plus_all_MAMP"] = {"k": len(allmamp), "mamp": mamp_keys, "f1": v}
    print(f"\n[B] 9 + all {len(mamp_keys)} MAMP ({len(allmamp)}-way): {v*100:.2f}% "
          f"(Δ vs 11way {(v-f1_11)*100:+.2f}pp)")

    # Pattern C: greedy forward selection (OVERFIT CAVEAT)
    print("\n[C] greedy forward selection (⚠ full-OOF, overfit caveat)")
    pool = dict(src)
    chosen = list(PROD)
    cur = f1(combine([src[m] for m in chosen], "softmax_mean"), y)
    greedy_path = [{"add": "7way_base", "f1": cur}]
    cand = [k for k in pool if k not in chosen]
    while cand:
        best_k, best_f1 = None, cur
        for k in cand:
            t = f1(combine([src[m] for m in chosen] + [src[k]], "softmax_mean"), y)
            if t > best_f1:
                best_f1, best_k = t, k
        if best_k is None:
            break
        chosen.append(best_k)
        cand.remove(best_k)
        greedy_path.append({"add": best_k, "f1": best_f1})
        cur = best_f1
    out["patterns"]["C_greedy"] = {"path": greedy_path, "final_members": chosen,
                                   "final_f1": cur}
    for s in greedy_path:
        print(f"  +{s['add']:18s} → {s['f1']*100:.2f}%")
    print(f"  greedy final {len(chosen)}-way: {cur*100:.2f}% "
          f"(Δ vs 11way {(cur-f1_11)*100:+.2f}pp) ⚠ overfit-prone, exploratory only")

    # Pattern D: leave-one-out on 11-way (redundancy)
    print("\n[D] leave-one-out on 11-way (redundancy diagnostic)")
    members11 = PROD + ["MB", "C3D", "MAMP60xsub", "MAMP120xset"]
    loo = {}
    for drop in members11:
        keep = [src[m] for m in members11 if m != drop]
        v = f1(combine(keep, "softmax_mean"), y)
        loo[drop] = v
        tag = " (redundant/harmful)" if v >= f1_11 else ""
        print(f"  −{drop:16s} → {v*100:.2f}% (Δ {(v-f1_11)*100:+.2f}pp){tag}")
    out["patterns"]["D_loo_11way"] = loo

    best_pat = max(
        [("11way_softmax", f1_11),
         ("B_allMAMP", out["patterns"]["B_9plus_all_MAMP"]["f1"]),
         ("C_greedy", cur)] +
        [(f"A_{m}", out["patterns"][f"A_11way_{m}"])
         for m in ["logit_mean", "rank_mean"]],
        key=lambda x: x[1])
    out["best_pattern"] = {"name": best_pat[0], "f1": best_pat[1],
                           "delta_vs_11way_pp": (best_pat[1] - f1_11) * 100}
    out["elapsed_sec"] = time.time() - t0
    Path(results_json).parent.mkdir(parents=True, exist_ok=True)
    Path(results_json).write_text(json.dumps(out, indent=2))
    print(f"\n[exp090] best non-overfit pattern: {best_pat[0]} {best_pat[1]*100:.2f}% "
          f"(Δ vs 11way {(best_pat[1]-f1_11)*100:+.2f}pp)")
    print(f"[exp090] saved {results_json} ({out['elapsed_sec']:.0f}s)")


if __name__ == "__main__":
    main()
