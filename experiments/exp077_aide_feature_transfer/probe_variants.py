"""exp077 probe variants — explore whether AIDE features get above the 20% gate
under any of: MLP head, per-stream feature subsets, or different probe Cs.

Cheap follow-ups to the failed linear-probe smoke (15.99% < 20% gate). Goal:
verify that the gate failure isn't a probe-architecture artifact before
declaring NO-GO and pivoting.

Outputs:
  results_variants.json — mean Macro-F1 per variant
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_probe import get_fold_indices, parse_emotion_label  # noqa: E402


@click.command()
@click.option("--features-npz", default="data/diema_challenge/processed/aide_features_b00_train.npz")
@click.option("--splits-pkl", default="data/diema_challenge/processed/split_lpo_10fold.pkl")
@click.option("--folds", default="0,1,2")
@click.option("--output", default="experiments/exp077_aide_feature_transfer/results_variants.json")
def main(features_npz: str, splits_pkl: str, folds: str, output: str) -> None:
    t0 = time.time()
    npz = np.load(features_npz, allow_pickle=True)
    filenames = np.asarray(npz["filenames"])
    labels = np.array([parse_emotion_label(str(fn)) for fn in filenames], dtype=np.int64)
    with open(splits_pkl, "rb") as f:
        splits = pickle.load(f)
    fold_ids = [int(x) for x in folds.split(",")]

    variants = {
        "concat_linear_C0.1":   {"feats": "features_concat",         "head": "linear", "C": 0.1},
        "concat_linear_C1.0":   {"feats": "features_concat",         "head": "linear", "C": 1.0},
        "concat_linear_C10":    {"feats": "features_concat",         "head": "linear", "C": 10.0},
        "concat_mlp_256":       {"feats": "features_concat",         "head": "mlp", "hidden": (256,)},
        "concat_mlp_512_128":   {"feats": "features_concat",         "head": "mlp", "hidden": (512, 128)},
        "joint_root_1_linear":  {"feats": "features_joint_root_1",   "head": "linear", "C": 1.0},
        "joint_root_14_linear": {"feats": "features_joint_root_14",  "head": "linear", "C": 1.0},
        "bone_root_1_linear":   {"feats": "features_bone_root_1",    "head": "linear", "C": 1.0},
        "bone_root_14_linear":  {"feats": "features_bone_root_14",   "head": "linear", "C": 1.0},
    }
    results: dict[str, dict] = {}

    for name, cfg in variants.items():
        X = np.asarray(npz[cfg["feats"]], dtype=np.float32)
        per_fold = []
        for fid in fold_ids:
            train_idx, val_idx = get_fold_indices(splits, filenames, fid)
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            scaler = StandardScaler().fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_val_s = scaler.transform(X_val)
            if cfg["head"] == "linear":
                clf = LogisticRegression(C=cfg["C"], max_iter=2000, n_jobs=-1, solver="lbfgs", random_state=42)
            else:
                clf = MLPClassifier(
                    hidden_layer_sizes=cfg["hidden"], max_iter=200,
                    random_state=42, early_stopping=True, n_iter_no_change=10,
                    alpha=1e-3,
                )
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_val_s)
            per_fold.append({
                "fold": fid,
                "macro_f1": float(f1_score(y_val, y_pred, average="macro")),
                "acc": float(accuracy_score(y_val, y_pred)),
            })
        f1s = [r["macro_f1"] for r in per_fold]
        mean_f1 = float(np.mean(f1s))
        std_f1 = float(np.std(f1s))
        results[name] = {
            "cfg": cfg,
            "per_fold": per_fold,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
        }
        print(f"[variant] {name:30s} {mean_f1:.4f} ± {std_f1:.4f}", flush=True)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[variants] saved {output} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
