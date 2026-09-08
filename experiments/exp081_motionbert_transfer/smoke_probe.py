"""exp081 smoke probe — 3-fold LPO linear probe on frozen MotionBERT-Lite features.

Decision gate (pre-registered):
  3-fold LPO mean Macro-F1 ≥ 20.0% on 12-class DIEM-A.
  Below: abort exp081, document as representation-gap failure.
  Pass: proceed to full 10-fold + 8-way ensemble eval.

Inputs:
  --features-npz: from extract_motionbert_features_for_diema.py
    features_pooled: (7992, 512) — mean-pooled (T, J)
    features_per_joint: (7992, 17, 512) — mean-pooled T only

  data/diema_challenge/processed/split_lpo_10fold.pkl
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


def get_fold_indices(splits: list[dict], filenames: np.ndarray, fold_idx: int) -> tuple[np.ndarray, np.ndarray]:
    fold = splits[fold_idx]
    name_to_pos = {str(fn): i for i, fn in enumerate(filenames)}
    train_idx = np.array([name_to_pos[str(item[0])] for item in fold["train"] if str(item[0]) in name_to_pos], dtype=np.int64)
    val_idx = np.array([name_to_pos[str(item[0])] for item in fold["val"] if str(item[0]) in name_to_pos], dtype=np.int64)
    return train_idx, val_idx


def health_check(features: np.ndarray) -> dict:
    per_dim_std = features.std(axis=0)
    return {
        "shape": list(features.shape),
        "nan_count": int(np.isnan(features).sum()),
        "inf_count": int(np.isinf(features).sum()),
        "dead_dims": int((per_dim_std < 1e-4).sum()),
        "dead_ratio": float((per_dim_std < 1e-4).sum() / features.shape[1]),
        "mean_var_per_dim": float(features.var(axis=0).mean()),
        "min_per_dim_std": float(per_dim_std.min()),
        "max_per_dim_std": float(per_dim_std.max()),
    }


@click.command()
@click.option("--features-npz", default="data/diema_challenge/processed/motionbert_features_b00_train.npz")
@click.option("--splits-pkl", default="data/diema_challenge/processed/split_lpo_10fold.pkl")
@click.option("--folds", default="0,1,2")
@click.option("--feat-source", default="pooled", type=click.Choice(["pooled", "per_joint_flat"]))
@click.option("--c-reg", "c_reg", default=1.0, type=float)
@click.option("--max-iter", default=2000, type=int)
@click.option("--output", default="experiments/exp081_motionbert_transfer/smoke_results.json")
def main(features_npz: str, splits_pkl: str, folds: str, feat_source: str, c_reg: float, max_iter: int, output: str) -> None:
    t0 = time.time()
    npz = np.load(features_npz, allow_pickle=True)
    filenames = np.asarray(npz["filenames"])
    labels = np.array([emotion_to_label(str(fn)) for fn in filenames], dtype=np.int64)
    if feat_source == "pooled":
        features = np.asarray(npz["features_pooled"], dtype=np.float32)
    elif feat_source == "per_joint_flat":
        per_joint = np.asarray(npz["features_per_joint"], dtype=np.float32)  # (N, 17, 512)
        features = per_joint.reshape(per_joint.shape[0], -1)
    print(f"[smoke] features {features.shape} ({feat_source}), labels {labels.shape}", flush=True)

    h = health_check(features)
    print(f"[smoke] health: {h}", flush=True)
    if h["nan_count"] or h["inf_count"]:
        print("[smoke] ABORT: NaN/Inf in features", flush=True)
        sys.exit(2)

    with open(splits_pkl, "rb") as f:
        splits = pickle.load(f)

    fold_ids = [int(x) for x in folds.split(",")]
    per_fold = []
    for fid in fold_ids:
        train_idx, val_idx = get_fold_indices(splits, filenames, fid)
        X_tr, X_va = features[train_idx], features[val_idx]
        y_tr, y_va = labels[train_idx], labels[val_idx]
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)
        clf = LogisticRegression(C=c_reg, max_iter=max_iter, n_jobs=-1, solver="lbfgs", random_state=42)
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_va_s)
        macro_f1 = float(f1_score(y_va, y_pred, average="macro"))
        acc = float(accuracy_score(y_va, y_pred))
        per_fold.append({"fold": fid, "macro_f1": macro_f1, "acc": acc, "n_train": len(train_idx), "n_val": len(val_idx)})
        print(f"[smoke] fold {fid}: Macro-F1 {macro_f1:.4f}, Acc {acc:.4f}, n_train {len(train_idx)} n_val {len(val_idx)}", flush=True)

    f1s = [r["macro_f1"] for r in per_fold]
    mean_f1 = float(np.mean(f1s))
    std_f1 = float(np.std(f1s))
    decision = "PASS" if mean_f1 >= 0.20 else "ABORT"

    results = {
        "features_npz": features_npz,
        "feat_source": feat_source,
        "C": c_reg,
        "folds": fold_ids,
        "health": h,
        "per_fold": per_fold,
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
        "gate_threshold": 0.20,
        "decision": decision,
        "elapsed_sec": time.time() - t0,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[smoke] mean Macro-F1 {mean_f1:.4f} ± {std_f1:.4f}, decision {decision}", flush=True)


if __name__ == "__main__":
    main()
