"""exp081 full 10-fold — best smoke config promoted (per_joint_flat C=1.0).

Generates OOF logits per fold for ensemble integration:
  output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_logits.npy
  output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_labels.npy
  output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_filenames.npy
"""

from __future__ import annotations

import json
import pickle
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
    train = np.array([name_to_pos[str(it[0])] for it in fold["train"] if str(it[0]) in name_to_pos], dtype=np.int64)
    val = np.array([name_to_pos[str(it[0])] for it in fold["val"] if str(it[0]) in name_to_pos], dtype=np.int64)
    return train, val


@click.command()
@click.option("--features-npz", default="data/diema_challenge/processed/motionbert_features_b00_train.npz")
@click.option("--splits-pkl", default="data/diema_challenge/processed/split_lpo_10fold.pkl")
@click.option("--c-reg", "c_reg", default=1.0, type=float)
@click.option("--output-dir", default="output/predictions/motionbert_lite_per_joint")
@click.option("--results-json", default="experiments/exp081_motionbert_transfer/full_10fold_results.json")
def main(features_npz, splits_pkl, c_reg, output_dir, results_json):
    t0 = time.time()
    npz = np.load(features_npz, allow_pickle=True)
    filenames = np.asarray(npz["filenames"])
    labels = np.array([emotion_to_label(str(fn)) for fn in filenames], dtype=np.int64)
    per_joint = np.asarray(npz["features_per_joint"], dtype=np.float32)
    features = per_joint.reshape(per_joint.shape[0], -1)
    print(f"[10fold] features {features.shape}, labels {labels.shape}")

    with open(splits_pkl, "rb") as f:
        splits = pickle.load(f)

    NUM_CLASS = 12
    oof_logits = np.zeros((len(features), NUM_CLASS), dtype=np.float32)
    oof_labels = np.full((len(features),), -1, dtype=np.int64)
    fold_results = []

    for fid in range(10):
        train_idx, val_idx = get_fold_indices(splits, filenames, fid)
        X_tr, X_va = features[train_idx], features[val_idx]
        y_tr, y_va = labels[train_idx], labels[val_idx]
        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(C=c_reg, max_iter=2000, n_jobs=-1, solver="lbfgs", random_state=42)
        clf.fit(scaler.transform(X_tr), y_tr)
        # OOF predict on val
        proba = clf.predict_proba(scaler.transform(X_va))  # (n_val, n_classes_seen)
        # In case sklearn drops absent classes, fill into full 12-class output
        for j, cls in enumerate(clf.classes_):
            oof_logits[val_idx, int(cls)] = np.log(np.clip(proba[:, j], 1e-12, 1.0)).astype(np.float32)
        oof_labels[val_idx] = y_va
        y_pred = clf.predict(scaler.transform(X_va))
        macro_f1 = float(f1_score(y_va, y_pred, average="macro"))
        acc = float(accuracy_score(y_va, y_pred))
        fold_results.append({"fold": fid, "macro_f1": macro_f1, "acc": acc, "n_train": len(train_idx), "n_val": len(val_idx)})
        print(f"  fold {fid}: Macro-F1 {macro_f1:.4f}, Acc {acc:.4f}")

    # Aggregate
    f1s = [r["macro_f1"] for r in fold_results]
    accs = [r["acc"] for r in fold_results]
    mean_f1, std_f1 = float(np.mean(f1s)), float(np.std(f1s))
    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))

    # Save OOF
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "exp081_motionbert_oof_logits.npy", oof_logits)
    np.save(out_dir / "exp081_motionbert_oof_labels.npy", oof_labels)
    np.save(out_dir / "exp081_motionbert_oof_filenames.npy", filenames)

    results = {
        "C": c_reg, "feat_source": "per_joint_flat",
        "per_fold": fold_results,
        "mean_macro_f1": mean_f1, "std_macro_f1": std_f1,
        "mean_acc": mean_acc, "std_acc": std_acc,
        "elapsed_sec": time.time() - t0,
    }
    Path(results_json).parent.mkdir(parents=True, exist_ok=True)
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[10fold] mean Macro-F1 {mean_f1:.4f} ± {std_f1:.4f}, mean Acc {mean_acc:.4f}, time {time.time()-t0:.1f}s")
    print(f"[10fold] OOF saved to {out_dir}/")


if __name__ == "__main__":
    main()
