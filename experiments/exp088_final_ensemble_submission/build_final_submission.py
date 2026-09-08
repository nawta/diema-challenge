"""exp088 — final best-ensemble test-set submission (9-way + MAMP).

exp085 established the best OOF ensembles:
  - 10-way (9 + MAMP ntu120xset) = 35.40% OOF, zero per-class regression
  - 11-way (9 + both MAMP)        = 35.78% OOF, highest absolute
  (exp086 PoseC3D excluded — redundant with MAMP.)

This converts those OOF lifts into actual submission CSVs by:
  1. per-fold sklearn LogReg on MAMP train features (best exp085 config),
     applied to MAMP test features, averaged over 10 folds
     (identical protocol to exp084's MB/C3D test-logit generation)
  2. softmax-probability averaging consistent with exp085 OOF measurement:
       9-way   = (mean(softmax(7 members))*7 + sm(MB) + sm(C3D)) / 9
       10-way  = (9-way*9 + sm(MAMP_ntu120)) / 10
       11-way  = (9-way*9 + sm(MAMP_ntu60) + sm(MAMP_ntu120)) / 11
  3. format-check both vs the 7-way reference CSV.

Reuses exp084 7-way/MB/C3D test logits + exp085 best probe configs.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from diema.data.parser import IDX_TO_EMOTION, emotion_to_label

MEMBERS = [
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


def mamp_test_logits(train_npz, test_npz, feat_source, c_reg, splits):
    """Per-fold LogReg on MAMP features → mean test logits (exp084 protocol)."""
    tr = np.load(train_npz, allow_pickle=True)
    te = np.load(test_npz, allow_pickle=True)
    tr_fns = np.asarray(tr["filenames"])
    tr_lab = np.array([emotion_to_label(str(f)) for f in tr_fns], dtype=np.int64)
    if feat_source == "pooled":
        Xtr = np.asarray(tr["features_pooled"], dtype=np.float32)
        Xte = np.asarray(te["features_pooled"], dtype=np.float32)
    else:
        Xtr = np.asarray(tr["features_per_joint"], dtype=np.float32).reshape(len(tr_fns), -1)
        Xte = np.asarray(te["features_per_joint"], dtype=np.float32).reshape(
            len(te["filenames"]), -1)
    te_fns = [str(x) for x in te["filenames"]]
    NUM = 12
    folds = np.zeros((10, len(te_fns), NUM), dtype=np.float32)
    for fid in range(10):
        idx = fold_train_idx(splits, tr_fns, fid)
        sc = StandardScaler().fit(Xtr[idx])
        clf = LogisticRegression(C=c_reg, max_iter=2000, n_jobs=-1,
                                 solver="lbfgs", random_state=42)
        clf.fit(sc.transform(Xtr[idx]), tr_lab[idx])
        pr = clf.predict_proba(sc.transform(Xte))
        full = np.full((len(te_fns), NUM), 1e-12, dtype=np.float32)
        for j, cls in enumerate(clf.classes_):
            full[:, int(cls)] = pr[:, j]
        folds[fid] = np.log(np.clip(full, 1e-12, 1.0))
    return folds.mean(axis=0), te_fns


def align(logits, fns, ref):
    m = {str(f): i for i, f in enumerate(fns)}
    out = np.zeros((len(ref), logits.shape[1]), dtype=np.float32)
    for i, f in enumerate(ref):
        out[i] = logits[m[str(f)]]
    return softmax(out)


def write_csv(probs, names, path):
    pred = probs.argmax(1)
    df = pd.DataFrame({
        "sample_name": names,
        "predicted_label": pred,
        "predicted_emotion": [IDX_TO_EMOTION[int(p)] for p in pred],
    })
    for c in range(12):
        df[f"prob_{IDX_TO_EMOTION[c]}"] = probs[:, c]
    df.to_csv(path, index=False)
    return pred


@click.command()
@click.option("--splits-pkl", default="data/diema_challenge/processed/split_lpo_10fold.pkl")
@click.option("--seven-way-dir", default="output/predictions/ensemble_7way")
@click.option("--mb-test", default="output/predictions/motionbert_lite_per_joint/exp081_motionbert_test_logits.npy")
@click.option("--mb-test-fns", default="output/predictions/motionbert_lite_per_joint/exp081_motionbert_test_filenames.npy")
@click.option("--c3d-test", default="output/predictions/c3d_contact_only/test_logits.npy")
@click.option("--c3d-test-fns", default="output/predictions/c3d_contact_only/test_filenames.npy")
@click.option("--reference-csv", default="output/submissions/ensemble_7way.csv")
@click.option("--results-json", default="experiments/exp088_final_ensemble_submission/results.json")
def main(splits_pkl, seven_way_dir, mb_test, mb_test_fns, c3d_test, c3d_test_fns,
         reference_csv, results_json):
    t0 = time.time()
    with open(splits_pkl, "rb") as f:
        splits = pickle.load(f)
    ref = pd.read_csv(reference_csv)
    names = ref["sample_name"].astype(str).tolist()
    N = len(names)
    print(f"[ref] {N} test samples")

    # 7-way (logit-space mean → softmax, production convention) then to prob avg:
    seven_logits = np.stack(
        [np.load(Path(seven_way_dir) / f"{m}_test_logits.npy") for m in MEMBERS],
        axis=0)  # (7, N, 12)
    seven_prob_avg = softmax(seven_logits, -1).mean(axis=0)  # (N,12) prob-avg

    mb = align(np.load(mb_test), np.load(mb_test_fns, allow_pickle=True), names)
    c3 = align(np.load(c3d_test), np.load(c3d_test_fns, allow_pickle=True), names)
    nine = (seven_prob_avg * 7 + mb + c3) / 9

    # MAMP test logits (best exp085 configs)
    print("[mamp] ntu60xsub per_joint_flat C=0.1 ...")
    m60_log, m60_fns = mamp_test_logits(
        "data/diema_challenge/processed/mamp_ntu60xsub_train.npz",
        "data/diema_challenge/processed/mamp_ntu60xsub_test.npz",
        "per_joint_flat", 0.1, splits)
    print("[mamp] ntu120xset pooled C=0.1 ...")
    m120_log, m120_fns = mamp_test_logits(
        "data/diema_challenge/processed/mamp_ntu120xset_train.npz",
        "data/diema_challenge/processed/mamp_ntu120xset_test.npz",
        "pooled", 0.1, splits)
    np.save("output/predictions/mamp_ntu60xsub/test_logits.npy", m60_log)
    np.save("output/predictions/mamp_ntu60xsub/test_filenames.npy", np.array(m60_fns))
    np.save("output/predictions/mamp_ntu120xset/test_logits.npy", m120_log)
    np.save("output/predictions/mamp_ntu120xset/test_filenames.npy", np.array(m120_fns))
    m60 = align(m60_log, m60_fns, names)
    m120 = align(m120_log, m120_fns, names)

    ten = (seven_prob_avg * 7 + mb + c3 + m120) / 10          # 9 + MAMP ntu120
    eleven = (seven_prob_avg * 7 + mb + c3 + m60 + m120) / 11  # 9 + both MAMP

    subs = {
        "ensemble_9way": nine,
        "ensemble_10way_mamp_ntu120": ten,
        "ensemble_11way_mamp_both": eleven,
    }
    preds = {}
    for name, probs in subs.items():
        p = Path("output/submissions") / f"{name}.csv"
        preds[name] = write_csv(probs, names, p)
        print(f"  wrote {p}")

    ref_pred = ref["predicted_label"].to_numpy()
    diag = {"n": N, "elapsed_sec": time.time() - t0}
    for name in subs:
        d = int((preds[name] != ref_pred).sum())
        dist = np.bincount(preds[name], minlength=12).tolist()
        diag[name] = {"diff_vs_7way": d, "diff_pct": round(d / N * 100, 2),
                      "class_dist": dist}
        print(f"[{name}] differs from 7-way: {d}/{N} ({d/N*100:.2f}%)")
    diag["ref_class_dist"] = np.bincount(ref_pred, minlength=12).tolist()

    Path(results_json).parent.mkdir(parents=True, exist_ok=True)
    with open(results_json, "w") as f:
        json.dump(diag, f, indent=2)
    print(f"\n[exp088] saved {results_json} ({diag['elapsed_sec']:.0f}s)")


if __name__ == "__main__":
    main()
