"""exp077 smoke probe — 3-fold LPO linear probe on AIDE-extracted features.

Decision gate (pre-registered):
  3-fold LPO mean Macro-F1 ≥ 20.0% on 12-class DIEM-A.
  Below: abort exp077, document as Codex skeptic's representation-mismatch
  failure mode.
  Pass: proceed to full 10-fold + 8-way ensemble eval.

Inputs:
  data/diema_challenge/processed/aide_features_b00_train.npz
    features_concat: (7992, 1024) AIDE penultimate features
    filenames: (7992,) clip filenames matching split metadata

  data/diema_challenge/processed/split_lpo_10fold.pkl
    list[10] of dict('train', 'val', 'test') with [(filename, clip_idx), ...]

  data/diema_challenge/raw/bvh/train/  (for label parsing from filename)

Outputs:
  results.json — per-fold Macro-F1, Acc, Macro-F1 mean ± std, decision

Usage:
  conda run -n acii2026 python experiments/exp077_aide_feature_transfer/smoke_probe.py \
      --folds 0,1,2

Related: diema/features/bvh_to_halpe14.py (projector)
         diema/data/splits.py (LPO split generation)
"""

from __future__ import annotations

import json
import pickle
import re
import sys
import time
from pathlib import Path

import click
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

from diema.data.parser import emotion_to_label  # canonical 12-class mapping


def parse_emotion_label(filename: str) -> int:
    return emotion_to_label(filename)


def load_features_and_labels(features_npz: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npz = np.load(features_npz, allow_pickle=True)
    features = np.asarray(npz["features_concat"], dtype=np.float32)
    filenames = np.asarray(npz["filenames"])
    labels = np.array([parse_emotion_label(str(fn)) for fn in filenames], dtype=np.int64)
    return features, labels, filenames


def get_fold_indices(splits: list[dict], filenames: np.ndarray, fold_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return train_indices, val_indices for given fold based on filename matching."""
    fold = splits[fold_idx]
    name_to_pos = {str(fn): i for i, fn in enumerate(filenames)}
    train_idx = np.array([name_to_pos[str(item[0])] for item in fold["train"] if str(item[0]) in name_to_pos], dtype=np.int64)
    val_idx = np.array([name_to_pos[str(item[0])] for item in fold["val"] if str(item[0]) in name_to_pos], dtype=np.int64)
    return train_idx, val_idx


def feature_health_check(features: np.ndarray) -> dict:
    """Codex skeptic's A1 gate: feature variance, dead-feature ratio, NaN."""
    nan_count = int(np.isnan(features).sum())
    inf_count = int(np.isinf(features).sum())
    per_dim_std = features.std(axis=0)
    dead_dims = int((per_dim_std < 1e-4).sum())
    mean_var = float(features.var(axis=0).mean())
    return {
        "shape": list(features.shape),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "dead_dims": dead_dims,
        "dead_ratio": dead_dims / features.shape[1],
        "mean_var_per_dim": mean_var,
        "min_per_dim_std": float(per_dim_std.min()),
        "max_per_dim_std": float(per_dim_std.max()),
    }


@click.command()
@click.option("--features-npz", default="data/diema_challenge/processed/aide_features_b00_train.npz", show_default=True)
@click.option("--splits-pkl", default="data/diema_challenge/processed/split_lpo_10fold.pkl", show_default=True)
@click.option("--folds", default="0,1,2", show_default=True, help="comma-sep fold ids for smoke")
@click.option("--C", "C_reg", default=1.0, show_default=True, help="LogReg C")
@click.option("--max-iter", default=2000, show_default=True)
@click.option("--output", default="experiments/exp077_aide_feature_transfer/smoke_results.json", show_default=True)
def main(features_npz: str, splits_pkl: str, folds: str, C_reg: float, max_iter: int, output: str) -> None:
    t0 = time.time()
    features, labels, filenames = load_features_and_labels(features_npz)
    print(f"[smoke] features {features.shape}, labels {labels.shape}", flush=True)

    health = feature_health_check(features)
    print(f"[smoke] health: {health}", flush=True)
    if health["nan_count"] > 0 or health["inf_count"] > 0:
        print("[smoke] ABORT: NaN/Inf in features", flush=True)
        sys.exit(2)
    if health["dead_ratio"] > 0.3:
        print(f"[smoke] WARN: {health['dead_ratio']:.1%} dead dims (gate < 30%)", flush=True)
    if health["mean_var_per_dim"] < 0.01:
        print(f"[smoke] WARN: mean variance {health['mean_var_per_dim']:.6f} (gate ≥ 0.01)", flush=True)

    with open(splits_pkl, "rb") as f:
        splits = pickle.load(f)

    fold_ids = [int(x) for x in folds.split(",")]
    per_fold = []
    for fid in fold_ids:
        train_idx, val_idx = get_fold_indices(splits, filenames, fid)
        X_train = features[train_idx]
        y_train = labels[train_idx]
        X_val = features[val_idx]
        y_val = labels[val_idx]
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s = scaler.transform(X_val)
        clf = LogisticRegression(
            C=C_reg, max_iter=max_iter, n_jobs=-1, multi_class="multinomial",
            solver="lbfgs", random_state=42,
        )
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_val_s)
        macro_f1 = float(f1_score(y_val, y_pred, average="macro"))
        acc = float(accuracy_score(y_val, y_pred))
        per_class = [float(x) for x in f1_score(y_val, y_pred, average=None, labels=list(range(12)))]
        per_fold.append({
            "fold": fid, "n_train": len(train_idx), "n_val": len(val_idx),
            "macro_f1": macro_f1, "acc": acc, "per_class_f1": per_class,
        })
        print(f"[smoke] fold {fid}: Macro-F1 {macro_f1:.4f}, Acc {acc:.4f}, n_train {len(train_idx)} n_val {len(val_idx)}", flush=True)

    f1s = [r["macro_f1"] for r in per_fold]
    accs = [r["acc"] for r in per_fold]
    f1_mean = float(np.mean(f1s))
    f1_std = float(np.std(f1s))
    decision = "PASS" if f1_mean >= 0.20 else "ABORT"

    results = {
        "features_npz": features_npz,
        "splits_pkl": splits_pkl,
        "folds": fold_ids,
        "C": C_reg,
        "max_iter": max_iter,
        "health": health,
        "per_fold": per_fold,
        "mean_macro_f1": f1_mean,
        "std_macro_f1": f1_std,
        "mean_acc": float(np.mean(accs)),
        "gate_threshold": 0.20,
        "decision": decision,
        "elapsed_sec": time.time() - t0,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[smoke] mean Macro-F1 {f1_mean:.4f} ± {f1_std:.4f}, decision {decision}", flush=True)
    print(f"[smoke] saved {output}", flush=True)
    sys.exit(0 if decision == "PASS" else 1)


if __name__ == "__main__":
    main()
