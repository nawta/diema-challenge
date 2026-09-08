"""exp084 — generate MB and C3D test-set logits for 9-way submission.

Reuses:
  - tools/extract_motionbert_features_for_diema.py (MB feature forward)
  - diema/features/c3d_contact_features (via output/c3d_stats_cache.npz +
    CONTACT_FEATURES_22D subset from exp072)
  - train OOF protocol from experiments/exp081/full_10fold.py and
    experiments/exp072/run_smoke.py

Outputs:
  output/predictions/motionbert_lite_per_joint/exp081_motionbert_test_logits.npy
  output/predictions/motionbert_lite_per_joint/exp081_motionbert_test_filenames.npy
  output/predictions/c3d_contact_only/test_logits.npy
  output/predictions/c3d_contact_only/test_filenames.npy

Per-fold approach (consistent with exp081/072 OOF protocol):
  1. Fit StandardScaler + LogisticRegression on each fold's TRAIN split
  2. Apply per-fold probe to TEST features
  3. Average the 10 fold predictions (logit-space) → test logits

This matches the OOF protocol semantically: each test prediction is the mean
of 10 cross-validated probes, just like each train prediction is the OOF probe
from its held-out fold.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import time
from pathlib import Path

import click
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diema.data.parser import emotion_to_label

# C3D contact-only feature indices (from exp072_c3d_contact_only/run_smoke.py L45-56)
CONTACT_FEATURES_22D = [
    2, 3, 4, 5, 6,            # global speeds
    9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30, 33, 34,  # body-part speeds
    35, 36, 37, 38,           # asymmetries
    47,                       # root_sway_xy_rms
]


def get_fold_indices(splits, filenames, fold_idx):
    fold = splits[fold_idx]
    name_to_pos = {str(fn): i for i, fn in enumerate(filenames)}
    train = np.array(
        [name_to_pos[str(it[0])] for it in fold["train"] if str(it[0]) in name_to_pos],
        dtype=np.int64,
    )
    return train


def extract_mb_test_features(motion_test_npz: str, checkpoint: str, output_npz: str, device: str) -> None:
    """Call extract_motionbert_features_for_diema.py on test split.

    Reuses the existing script (no duplication). It expects motion_test_quat.npz
    in the same schema as motion_train_quat.npz.
    """
    if Path(output_npz).exists():
        print(f"[mb] test features already exist at {output_npz}, skipping extraction")
        return
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "extract_motionbert_features_for_diema.py"),
        "--motion-npz", motion_test_npz,
        "--checkpoint", checkpoint,
        "--output", output_npz,
        "--device", device,
    ]
    print("[mb] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def fit_and_predict_mb(
    train_features_npz: str,
    test_features_npz: str,
    splits_pkl: str,
    c_reg: float,
) -> tuple[np.ndarray, list[str]]:
    """Per-fold LogReg on MB per_joint_flat features; return mean test logits."""
    train_npz = np.load(train_features_npz, allow_pickle=True)
    test_npz = np.load(test_features_npz, allow_pickle=True)
    train_filenames = np.asarray(train_npz["filenames"])
    train_labels = np.array(
        [emotion_to_label(str(fn)) for fn in train_filenames], dtype=np.int64
    )
    train_features = np.asarray(train_npz["features_per_joint"], dtype=np.float32).reshape(
        train_filenames.shape[0], -1
    )
    test_filenames = list(np.asarray(test_npz["filenames"]))
    test_features = np.asarray(test_npz["features_per_joint"], dtype=np.float32).reshape(
        len(test_filenames), -1
    )
    print(f"[mb] train features {train_features.shape}, test features {test_features.shape}")

    with open(splits_pkl, "rb") as f:
        splits = pickle.load(f)

    NUM_CLASS = 12
    test_logits_folds = np.zeros((10, len(test_filenames), NUM_CLASS), dtype=np.float32)
    for fid in range(10):
        train_idx = get_fold_indices(splits, train_filenames, fid)
        X_tr = train_features[train_idx]
        y_tr = train_labels[train_idx]
        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(
            C=c_reg, max_iter=2000, n_jobs=-1, solver="lbfgs", random_state=42
        )
        clf.fit(scaler.transform(X_tr), y_tr)
        proba = clf.predict_proba(scaler.transform(test_features))
        # Pad to 12 classes (sklearn drops absent labels)
        full = np.full((len(test_filenames), NUM_CLASS), 1e-12, dtype=np.float32)
        for j, cls in enumerate(clf.classes_):
            full[:, int(cls)] = proba[:, j]
        test_logits_folds[fid] = np.log(np.clip(full, 1e-12, 1.0))
        print(f"  [mb fold {fid}] n_train {len(train_idx)}, applied LogReg → test logits")
    mean_test_logits = test_logits_folds.mean(axis=0)
    return mean_test_logits, test_filenames


def fit_and_predict_c3d(
    cache_path: str,
    motion_train_npz: str,
    splits_pkl: str,
    c_reg: float,
) -> tuple[np.ndarray, list[str]]:
    """Per-fold LogReg on 22-D C3D contact features; return mean test logits."""
    cache = np.load(cache_path, allow_pickle=True)
    all_features = np.asarray(cache["features"], dtype=np.float32)
    all_filenames = np.asarray([str(x) for x in cache["filenames"]])

    motion_data = np.load(motion_train_npz, allow_pickle=True)
    train_filenames_set = set(str(fn) for fn in motion_data["filenames"])
    is_train = np.array([fn in train_filenames_set for fn in all_filenames])
    is_test = ~is_train

    train_features_22 = all_features[is_train][:, CONTACT_FEATURES_22D]
    train_filenames = all_filenames[is_train]
    train_labels = np.array(
        [emotion_to_label(fn) for fn in train_filenames], dtype=np.int64
    )
    test_features_22 = all_features[is_test][:, CONTACT_FEATURES_22D]
    test_filenames = list(all_filenames[is_test])
    print(f"[c3d] train features {train_features_22.shape}, test features {test_features_22.shape}")

    with open(splits_pkl, "rb") as f:
        splits = pickle.load(f)

    NUM_CLASS = 12
    test_logits_folds = np.zeros((10, len(test_filenames), NUM_CLASS), dtype=np.float32)
    for fid in range(10):
        train_idx = get_fold_indices(splits, train_filenames, fid)
        X_tr = train_features_22[train_idx]
        y_tr = train_labels[train_idx]
        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(
            C=c_reg, max_iter=2000, n_jobs=-1, solver="lbfgs", random_state=42
        )
        clf.fit(scaler.transform(X_tr), y_tr)
        proba = clf.predict_proba(scaler.transform(test_features_22))
        full = np.full((len(test_filenames), NUM_CLASS), 1e-12, dtype=np.float32)
        for j, cls in enumerate(clf.classes_):
            full[:, int(cls)] = proba[:, j]
        test_logits_folds[fid] = np.log(np.clip(full, 1e-12, 1.0))
        print(f"  [c3d fold {fid}] n_train {len(train_idx)}, applied LogReg → test logits")
    mean_test_logits = test_logits_folds.mean(axis=0)
    return mean_test_logits, test_filenames


@click.command()
@click.option("--motion-train-npz", default="data/diema_challenge/processed/motion_train_quat.npz")
@click.option("--motion-test-npz", default="data/diema_challenge/processed/motion_test_quat.npz")
@click.option("--mb-train-features", default="data/diema_challenge/processed/motionbert_features_b00_train.npz")
@click.option("--mb-test-features", default="data/diema_challenge/processed/motionbert_features_b00_test.npz")
@click.option("--mb-checkpoint", default="data/motionbert/checkpoints/MB_pretrain_lite.bin")
@click.option("--c3d-cache", default="output/c3d_stats_cache.npz")
@click.option("--splits-pkl", default="data/diema_challenge/processed/split_lpo_10fold.pkl")
@click.option("--mb-out", default="output/predictions/motionbert_lite_per_joint/exp081_motionbert_test_logits.npy")
@click.option("--mb-fns-out", default="output/predictions/motionbert_lite_per_joint/exp081_motionbert_test_filenames.npy")
@click.option("--c3d-out", default="output/predictions/c3d_contact_only/test_logits.npy")
@click.option("--c3d-fns-out", default="output/predictions/c3d_contact_only/test_filenames.npy")
@click.option("--mb-c-reg", default=1.0, type=float)
@click.option("--c3d-c-reg", default=1.0, type=float)
@click.option("--device", default="cuda:0")
@click.option("--skip-mb", is_flag=True, help="skip MB extraction (debug)")
@click.option("--skip-c3d", is_flag=True, help="skip C3D extraction (debug)")
def main(
    motion_train_npz: str,
    motion_test_npz: str,
    mb_train_features: str,
    mb_test_features: str,
    mb_checkpoint: str,
    c3d_cache: str,
    splits_pkl: str,
    mb_out: str,
    mb_fns_out: str,
    c3d_out: str,
    c3d_fns_out: str,
    mb_c_reg: float,
    c3d_c_reg: float,
    device: str,
    skip_mb: bool,
    skip_c3d: bool,
) -> None:
    t0 = time.time()
    if not skip_mb:
        print("\n=== MotionBERT branch ===")
        extract_mb_test_features(motion_test_npz, mb_checkpoint, mb_test_features, device)
        mb_logits, mb_fns = fit_and_predict_mb(
            mb_train_features, mb_test_features, splits_pkl, mb_c_reg
        )
        Path(mb_out).parent.mkdir(parents=True, exist_ok=True)
        np.save(mb_out, mb_logits)
        np.save(mb_fns_out, np.array(mb_fns))
        print(f"[mb] saved {mb_out} shape {mb_logits.shape}")

    if not skip_c3d:
        print("\n=== C3D contact branch ===")
        c3d_logits, c3d_fns = fit_and_predict_c3d(
            c3d_cache, motion_train_npz, splits_pkl, c3d_c_reg
        )
        Path(c3d_out).parent.mkdir(parents=True, exist_ok=True)
        np.save(c3d_out, c3d_logits)
        np.save(c3d_fns_out, np.array(c3d_fns))
        print(f"[c3d] saved {c3d_out} shape {c3d_logits.shape}")

    print(f"\n[exp084.gen] elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
