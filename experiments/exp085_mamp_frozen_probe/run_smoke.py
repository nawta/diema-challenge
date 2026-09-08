"""exp085 MAMP frozen-probe smoke — 3-fold LPO linear probe.

 gate (pre-registered, + prior_work doc):
  3-fold LPO mean Macro-F1 ≥ 24.0% on 12-class DIEM-A.
  (anchor: exp081 frozen MotionBERT smoke 24.82%)
  Below: archive MAMP branch, document as representation-gap.
  Pass: proceed to full 10-fold + 9/10-way ensemble eval.

Sweeps {pooled, per_joint_flat} × C ∈ {0.1, 1.0, 10.0} like exp081, picks
the best config to promote. Mirrors exp081_motionbert_transfer/smoke_probe.py.

Inputs:
  --features-npz from tools/extract_mamp_features_for_diema.py
    features_pooled    : (7992, 256)
    features_per_joint : (7992, 25, 256)
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import click
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from diema.data.parser import emotion_to_label


def get_fold_indices(splits, filenames, fold_idx):
    fold = splits[fold_idx]
    name_to_pos = {str(fn): i for i, fn in enumerate(filenames)}
    train_idx = np.array(
        [name_to_pos[str(it[0])] for it in fold["train"] if str(it[0]) in name_to_pos],
        dtype=np.int64,
    )
    val_idx = np.array(
        [name_to_pos[str(it[0])] for it in fold["val"] if str(it[0]) in name_to_pos],
        dtype=np.int64,
    )
    return train_idx, val_idx


def health_check(features: np.ndarray) -> dict:
    per_dim_std = features.std(axis=0)
    return {
        "shape": list(features.shape),
        "nan_count": int(np.isnan(features).sum()),
        "inf_count": int(np.isinf(features).sum()),
        "dead_dims": int((per_dim_std < 1e-4).sum()),
        "mean_var_per_dim": float(features.var(axis=0).mean()),
    }


def run_one(features, labels, filenames, splits, fold_ids, c_reg, max_iter):
    per_fold = []
    for fid in fold_ids:
        tr, va = get_fold_indices(splits, filenames, fid)
        scaler = StandardScaler().fit(features[tr])
        clf = LogisticRegression(
            C=c_reg, max_iter=max_iter, n_jobs=-1, solver="lbfgs", random_state=42
        )
        clf.fit(scaler.transform(features[tr]), labels[tr])
        y_pred = clf.predict(scaler.transform(features[va]))
        per_fold.append({
            "fold": fid,
            "macro_f1": float(f1_score(labels[va], y_pred, average="macro")),
            "acc": float(accuracy_score(labels[va], y_pred)),
        })
    f1s = [r["macro_f1"] for r in per_fold]
    return per_fold, float(np.mean(f1s)), float(np.std(f1s))


@click.command()
@click.option("--features-npz", required=True)
@click.option("--splits-pkl", default="data/diema_challenge/processed/split_lpo_10fold.pkl")
@click.option("--folds", default="0,1,2")
@click.option("--max-iter", default=2000, type=int)
@click.option("--tag", default="ntu60xsub_unit")
@click.option("--output-dir", default="experiments/exp085_mamp_frozen_probe")
def main(features_npz, splits_pkl, folds, max_iter, tag, output_dir):
    t0 = time.time()
    npz = np.load(features_npz, allow_pickle=True)
    filenames = np.asarray(npz["filenames"])
    labels = np.array([emotion_to_label(str(fn)) for fn in filenames], dtype=np.int64)
    pooled = np.asarray(npz["features_pooled"], dtype=np.float32)
    per_joint = np.asarray(npz["features_per_joint"], dtype=np.float32)
    per_joint_flat = per_joint.reshape(per_joint.shape[0], -1)
    print(f"[smoke:{tag}] pooled {pooled.shape}, per_joint_flat {per_joint_flat.shape}")

    h_pool = health_check(pooled)
    print(f"[smoke] pooled health: {h_pool}")
    if h_pool["nan_count"] or h_pool["inf_count"]:
        print("[smoke] ABORT: NaN/Inf"); sys.exit(2)

    with open(splits_pkl, "rb") as f:
        splits = pickle.load(f)
    fold_ids = [int(x) for x in folds.split(",")]

    grid = []
    for src_name, X in [("pooled", pooled), ("per_joint_flat", per_joint_flat)]:
        for c_reg in (0.1, 1.0, 10.0):
            per_fold, mean_f1, std_f1 = run_one(
                X, labels, filenames, splits, fold_ids, c_reg, max_iter
            )
            grid.append({
                "feat_source": src_name, "C": c_reg,
                "per_fold": per_fold, "mean_macro_f1": mean_f1, "std_macro_f1": std_f1,
            })
            print(f"[smoke] {src_name:15s} C={c_reg:<5} "
                  f"mean Macro-F1 {mean_f1*100:.2f}% ± {std_f1*100:.2f}%")

    best = max(grid, key=lambda g: g["mean_macro_f1"])
    decision = "PASS" if best["mean_macro_f1"] >= 0.24 else "ABORT"
    print(f"\n[smoke:{tag}] BEST {best['feat_source']} C={best['C']} "
          f"→ {best['mean_macro_f1']*100:.2f}% | gate 24.0% → {decision}")

    out = {
        "tag": tag,
        "features_npz": features_npz,
        "gate_threshold": 0.24,
        "grid": grid,
        "best": {k: best[k] for k in ("feat_source", "C", "mean_macro_f1", "std_macro_f1")},
        "decision": decision,
        "elapsed_sec": time.time() - t0,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / f"smoke_results_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[smoke:{tag}] saved smoke_results_{tag}.json")


if __name__ == "__main__":
    main()
