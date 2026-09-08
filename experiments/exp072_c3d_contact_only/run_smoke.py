"""exp072 C3D contact-only smoke + 10-fold + ensemble add.

Pipeline:
  1. Load existing 51-D C3D cache (/output/c3d_stats_cache.npz)
  2. Filter to ~22 contact/dynamics-only features (drop path_length,
     rms_spread, foot-heights, duration — these caused exp056 country
     leakage 69.8%)
  3. Filter to train split (drop test clips for OOF computation)
  4. 3-fold LPO linear probe smoke gate ≥ 20% Macro-F1
  5. Country probe diagnostic: train classifier to predict country from
     features; target accuracy ≤ 60% (vs exp056's 69.8%)
  6. 10-fold conditional: if smoke passes, generate OOF logits for
     ensemble integration
  7. 8-way ensemble eval: 7-way production + C3D = 8-way; gate F1
     +0.20pp AND no per-class regression > 0.30pp

Outputs:
  experiments/exp072_c3d_contact_only/smoke_results.json
  experiments/exp072_c3d_contact_only/country_probe.json
  experiments/exp072_c3d_contact_only/full_10fold_results.json (if gate pass)
  output/predictions/c3d_contact_only/oof_logits.npy (if gate pass)
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from diema.data.parser import IDX_TO_EMOTION, emotion_to_label

# Contact/dynamics-only feature subset (22-D)
# Dropping: path_length × 8, rms_spread × 8, foot heights × 8, marker_dropout, duration, n_frames
# Keeping: global speeds 2-6, head speeds 9-10, torso 13-14, l_arm 17-18, r_arm 21-22,
#          pelvis 25-26, l_leg 29-30, r_leg 33-34, asym 35-38, root_sway 47
CONTACT_FEATURES_22D = [
    2, 3, 4, 5, 6,            # global_mean_speed, max_speed, mean_accel, max_accel, mean_jerk
    9, 10,                    # head speeds
    13, 14,                   # torso speeds
    17, 18,                   # l_arm speeds
    21, 22,                   # r_arm speeds
    25, 26,                   # pelvis speeds
    29, 30,                   # l_leg speeds
    33, 34,                   # r_leg speeds
    35, 36, 37, 38,           # asymmetries (4)
    47,                       # root_sway_xy_rms
]


def country_from_filename(fn: str) -> str:
    """e.g. 'JP_06_anger_1_H' → 'JP'."""
    m = re.match(r"^([A-Z]{2})_", fn)
    return m.group(1) if m else "??"


def health_check(features: np.ndarray) -> dict:
    per_dim_std = features.std(axis=0)
    return {
        "shape": list(features.shape),
        "nan_count": int(np.isnan(features).sum()),
        "inf_count": int(np.isinf(features).sum()),
        "dead_dims": int((per_dim_std < 1e-6).sum()),
        "mean_var_per_dim": float(features.var(axis=0).mean()),
    }


def get_fold_indices(splits, filenames, fold_idx):
    fold = splits[fold_idx]
    name_to_pos = {str(fn): i for i, fn in enumerate(filenames)}
    train = np.array([name_to_pos[str(it[0])] for it in fold["train"] if str(it[0]) in name_to_pos], dtype=np.int64)
    val = np.array([name_to_pos[str(it[0])] for it in fold["val"] if str(it[0]) in name_to_pos], dtype=np.int64)
    return train, val


@click.command()
@click.option("--cache-path", default="output/c3d_stats_cache.npz")
@click.option("--splits-pkl", default="data/diema_challenge/processed/split_lpo_10fold.pkl")
@click.option("--motion-npz", default="data/diema_challenge/processed/motion_train_quat.npz")
@click.option("--folds-smoke", default="0,1,2")
@click.option("--c-reg", "c_reg", default=1.0, type=float)
@click.option("--output-dir", default="experiments/exp072_c3d_contact_only")
@click.option("--full-10fold/--smoke-only", default=False)
def main(cache_path, splits_pkl, motion_npz, folds_smoke, c_reg, output_dir, full_10fold):
    t0 = time.time()
    cache = np.load(cache_path, allow_pickle=True)
    all_features = np.asarray(cache["features"], dtype=np.float32)
    all_filenames = np.asarray([str(x) for x in cache["filenames"]])
    feature_names = list(cache["feature_names"])
    print(f"[exp072] cache shape: {all_features.shape}, {len(all_filenames)} clips")

    # Filter to train split (only clips that are in motion_train_quat.npz / have labels)
    motion_npz_data = np.load(motion_npz, allow_pickle=True)
    train_filenames = set(str(fn) for fn in motion_npz_data["filenames"])
    train_mask = np.array([fn in train_filenames for fn in all_filenames])
    features_train = all_features[train_mask]
    filenames_train = all_filenames[train_mask]
    print(f"[exp072] train split: {features_train.shape[0]} clips")

    # Contact-only subset
    features_22 = features_train[:, CONTACT_FEATURES_22D]
    print(f"[exp072] contact-only features: {features_22.shape}")
    print(f"[exp072] kept feature names:")
    for i, idx in enumerate(CONTACT_FEATURES_22D):
        print(f"  {i:2d}: {feature_names[idx]}")

    # Labels
    labels = np.array([emotion_to_label(fn) for fn in filenames_train], dtype=np.int64)
    countries = np.array([country_from_filename(fn) for fn in filenames_train])

    # A1: feature health
    health = health_check(features_22)
    print(f"\n[A1 health] {health}")
    if health["nan_count"] or health["inf_count"]:
        print("ABORT: NaN/Inf in C3D features"); sys.exit(2)

    # Splits
    with open(splits_pkl, "rb") as f:
        splits = pickle.load(f)

    # A2: 3-fold smoke
    fold_ids = [int(x) for x in folds_smoke.split(",")]
    per_fold_smoke = []
    for fid in fold_ids:
        train_idx, val_idx = get_fold_indices(splits, filenames_train, fid)
        X_tr, X_va = features_22[train_idx], features_22[val_idx]
        y_tr, y_va = labels[train_idx], labels[val_idx]
        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(C=c_reg, max_iter=2000, n_jobs=-1, solver="lbfgs", random_state=42)
        clf.fit(scaler.transform(X_tr), y_tr)
        y_pred = clf.predict(scaler.transform(X_va))
        macro_f1 = float(f1_score(y_va, y_pred, average="macro"))
        acc = float(accuracy_score(y_va, y_pred))
        per_fold_smoke.append({"fold": fid, "macro_f1": macro_f1, "acc": acc, "n_train": len(train_idx), "n_val": len(val_idx)})
        print(f"[A2 smoke] fold {fid}: Macro-F1 {macro_f1*100:.2f}%, Acc {acc*100:.2f}%")

    smoke_mean = float(np.mean([r["macro_f1"] for r in per_fold_smoke]))
    smoke_std = float(np.std([r["macro_f1"] for r in per_fold_smoke]))
    smoke_decision = "PASS" if smoke_mean >= 0.20 else "ABORT"
    print(f"\n[A2] mean Macro-F1 {smoke_mean*100:.2f}% ± {smoke_std*100:.2f}%, decision {smoke_decision}")

    # Country probe diagnostic
    print("\n[country probe] training classifier to predict country from C3D features...")
    country_results = []
    for fid in fold_ids:
        train_idx, val_idx = get_fold_indices(splits, filenames_train, fid)
        scaler = StandardScaler().fit(features_22[train_idx])
        clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1, solver="lbfgs", random_state=42)
        clf.fit(scaler.transform(features_22[train_idx]), countries[train_idx])
        y_pred = clf.predict(scaler.transform(features_22[val_idx]))
        cacc = float(accuracy_score(countries[val_idx], y_pred))
        country_results.append({"fold": fid, "country_acc": cacc})
        print(f"  fold {fid}: country acc {cacc*100:.2f}%")
    country_mean = float(np.mean([r["country_acc"] for r in country_results]))
    country_gate = "PASS" if country_mean <= 0.60 else "FAIL"
    print(f"[country probe] mean acc {country_mean*100:.2f}% (target ≤ 60.00%, vs exp056 69.8%); gate {country_gate}")

    smoke_results = {
        "kept_features": [feature_names[i] for i in CONTACT_FEATURES_22D],
        "n_features": len(CONTACT_FEATURES_22D),
        "n_clips": int(features_22.shape[0]),
        "health": health,
        "smoke_per_fold": per_fold_smoke,
        "smoke_mean_macro_f1": smoke_mean,
        "smoke_std_macro_f1": smoke_std,
        "smoke_gate_threshold": 0.20,
        "smoke_decision": smoke_decision,
        "country_probe_per_fold": country_results,
        "country_probe_mean_acc": country_mean,
        "country_probe_target": 0.60,
        "country_probe_gate": country_gate,
        "elapsed_sec": time.time() - t0,
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "smoke_results.json", "w") as f:
        json.dump(smoke_results, f, indent=2)
    print(f"\n[exp072] smoke results saved to {out_dir / 'smoke_results.json'}")

    if not full_10fold:
        if smoke_decision == "ABORT":
            print("[exp072] smoke ABORT — skipping full 10-fold (use --full-10fold to override)")
        else:
            print("[exp072] --full-10fold not set — stopping after smoke")
        sys.exit(0 if smoke_decision == "PASS" else 1)
    if smoke_decision == "ABORT":
        print("[exp072] WARN: continuing full 10-fold despite smoke ABORT (--full-10fold override)")

    # Full 10-fold (conditional)
    print("\n[B1] running full 10-fold LPO for OOF logits...")
    NUM_CLASS = 12
    oof_logits = np.zeros((len(features_22), NUM_CLASS), dtype=np.float32)
    full_results = []
    for fid in range(10):
        train_idx, val_idx = get_fold_indices(splits, filenames_train, fid)
        X_tr, X_va = features_22[train_idx], features_22[val_idx]
        y_tr, y_va = labels[train_idx], labels[val_idx]
        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(C=c_reg, max_iter=2000, n_jobs=-1, solver="lbfgs", random_state=42)
        clf.fit(scaler.transform(X_tr), y_tr)
        proba = clf.predict_proba(scaler.transform(X_va))
        for j, cls in enumerate(clf.classes_):
            oof_logits[val_idx, int(cls)] = np.log(np.clip(proba[:, j], 1e-12, 1.0)).astype(np.float32)
        y_pred = clf.predict(scaler.transform(X_va))
        macro_f1 = float(f1_score(y_va, y_pred, average="macro"))
        acc = float(accuracy_score(y_va, y_pred))
        full_results.append({"fold": fid, "macro_f1": macro_f1, "acc": acc})
        print(f"  fold {fid}: Macro-F1 {macro_f1*100:.2f}%, Acc {acc*100:.2f}%")
    full_mean = float(np.mean([r["macro_f1"] for r in full_results]))
    full_std = float(np.std([r["macro_f1"] for r in full_results]))
    print(f"\n[B1] 10-fold mean Macro-F1 {full_mean*100:.2f}% ± {full_std*100:.2f}%")

    # Save OOF
    pred_dir = Path("output/predictions/c3d_contact_only")
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "oof_logits.npy", oof_logits)
    np.save(pred_dir / "oof_filenames.npy", filenames_train)

    full_results_dict = {
        "C": c_reg,
        "per_fold": full_results,
        "mean_macro_f1": full_mean,
        "std_macro_f1": full_std,
    }
    with open(out_dir / "full_10fold_results.json", "w") as f:
        json.dump(full_results_dict, f, indent=2)
    print(f"[exp072] saved OOF + full results")


if __name__ == "__main__":
    main()
