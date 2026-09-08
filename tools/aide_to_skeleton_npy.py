""" exp076 — AIDE annotation JSON → UbH-GCN.npy format converter.

Converts AIDE per-clip annotation JSON files (containing Halpe-136
keypoints from AlphaPose) into the (N, C=3, T=16, V=14, M=1) tensor
format expected by UbH-GCN's `feeders/feeder_aide.py`.

AIDE layout (per `ydk122024/AIDE/dataset.py:load_frames`):
    AIDE_Dataset/
      0001/                       # clip directory (frames discarded for skeleton-only path)
      ...
      annotation/0001.json        # per-clip JSON with pose_list + emotion_label
    training.csv, validation.csv, testing.csv

JSON structure (per clip):
    {
      "pose_list": [
        {"result": [{"keypoints": [x0, y0, s0, x1, y1, s1, ...],  # Halpe-136 × 3
                     "bbox": [x, y, w, h], "face_bbox": [x, y, w, h]}]},
        ... # N_frames entries
      ],
      "emotion_label": "Anxiety" | "Peace" | "Weariness" | "Happiness" | "Anger",
      "driver_behavior_label": ...,
      "scene_centric_context_label": ...,
      "vehicle_based_context_label": ...,
    }

UbH-GCN expects 14 upper-body joints from Halpe-26 body subset:

    Halpe-26 body indices (0-indexed):
      0 Nose, 1 LEye, 2 REye, 3 LEar, 4 REar,
      5 LShoulder, 6 RShoulder, 7 LElbow, 8 RElbow,
      9 LWrist, 10 RWrist, 11 LHip, 12 RHip,
      13 LKnee, 14 RKnee, 15 LAnkle, 16 RAnkle,
      17 Head, 18 Neck, 19 Hip (center),
      20 LBigToe, 21 RBigToe, 22 LSmallToe, 23 RSmallToe,
      24 LHeel, 25 RHeel.

The exact 14-joint subset for UbH-GCN is *inferred* from
`tmp/UbH-GCN-source/feeders/bone_pairs.py:aide_pairs_upper_head`. The
public repo does not contain the explicit Halpe→UbH-GCN mapping table,
so we derive it from:
  (a) the 14-node adjacency graph (showing chains: torso→arms, torso→head),
  (b) the AIDE paper Figure 1 (cockpit view of upper body),
  (c) face joints (4 of them) suggest LEye/REye/LEar/REar are kept.

Inferred mapping (1-indexed UbH-GCN joint ID → Halpe-26 body 0-indexed):
    UbH-GCN 1  = Halpe Head (17)         — face anchor
    UbH-GCN 2  = Halpe LEye (1)
    UbH-GCN 3  = Halpe REye (2)
    UbH-GCN 4  = Halpe LEar (3)
    UbH-GCN 5  = Halpe REar (4)
    UbH-GCN 6  = Halpe RShoulder (6)
    UbH-GCN 7  = Halpe LShoulder (5)
    UbH-GCN 8  = Halpe RElbow (8)
    UbH-GCN 9  = Halpe LElbow (7)
    UbH-GCN 10 = Halpe RWrist (10)
    UbH-GCN 11 = Halpe LWrist (9)
    UbH-GCN 12 = Halpe Nose (0)            — connects to 1 (Head); chin proxy
    UbH-GCN 13 = Halpe Neck (18)            — shoulder-center hub
    UbH-GCN 14 = Halpe Hip-center (19)     — root anchor

This is a **best-guess inference**; will be re-validated after we
inspect one real AIDE annotation JSON. See for the
verification protocol.

Frame sampling (matches AIDE dataset.py `load_frames`):
    indices = [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44]
    → exactly 16 frames per 3-second clip (sampled deterministically).

Output:
    train_data.npy  shape (N_train, 3, 16, 14, 1) float32
    train_label.pkl list[int] in [0..4] for the 5 emotion classes
    eval_data.npy   shape (N_eval, ...)
    eval_label.pkl
    test_data.npy   shape (N_test, ...)
    test_label.pkl

See: §5.3 / §11.4
Related: tmp/UbH-GCN-source/feeders/feeder_aide.py (target consumer)
         experiments/exp076_ubh_gcn_aide_reproduction
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

AIDE_EMOTION_LABELS = (
    "Anxiety", "Peace", "Weariness", "Happiness", "Anger",
)

# Inferred Halpe-26 body index → UbH-GCN 14-joint index (0-indexed for numpy).
# UbH-GCN joints are 1-indexed in the paper; we use 0-indexed internally and
# subtract 1 from the paper joint ids.
#
# This mapping is the BEST-GUESS inference from
# `tmp/UbH-GCN-source/feeders/bone_pairs.py:aide_pairs_upper_head` and the
# Halpe-26 body keypoint layout. Will be re-validated after we inspect a
# real AIDE annotation JSON (especially: joints 12, 13, 14 — chin, neck,
# hip-center anchors).
UBH_GCN_TO_HALPE = {
    0: 17,   # UbH-GCN 1 (head)            ← Halpe Head (17)
    1: 1,    # UbH-GCN 2 (LEye)            ← Halpe LEye (1)
    2: 2,    # UbH-GCN 3 (REye)            ← Halpe REye (2)
    3: 3,    # UbH-GCN 4 (LEar)            ← Halpe LEar (3)
    4: 4,    # UbH-GCN 5 (REar)            ← Halpe REar (4)
    5: 6,    # UbH-GCN 6 (RShoulder)       ← Halpe RShoulder (6)
    6: 5,    # UbH-GCN 7 (LShoulder)       ← Halpe LShoulder (5)
    7: 8,    # UbH-GCN 8 (RElbow)          ← Halpe RElbow (8)
    8: 7,    # UbH-GCN 9 (LElbow)          ← Halpe LElbow (7)
    9: 10,   # UbH-GCN 10 (RWrist)         ← Halpe RWrist (10)
    10: 9,   # UbH-GCN 11 (LWrist)         ← Halpe LWrist (9)
    11: 0,   # UbH-GCN 12 (chin)           ← Halpe Nose (0)  [PROVISIONAL]
    12: 18,  # UbH-GCN 13 (neck hub)       ← Halpe Neck (18)
    13: 19,  # UbH-GCN 14 (hip-center root)← Halpe Hip-center (19)
}

# AIDE samples per 3-sec clip at frame indices below (per dataset.py).
FRAME_INDICES = (0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44)
NUM_JOINTS_UBHGCN = 14
NUM_FRAMES_UBHGCN = 16
NUM_CHANNELS = 3   # (x, y, score)


def aide_pose_to_skeleton(pose_list: list, halpe_to_ubhgcn: dict[int, int]) -> np.ndarray:
    """Convert one AIDE clip's per-frame pose_list to a UbH-GCN tensor.

    Returns:
        np.ndarray of shape (C=3, T=16, V=14, M=1) float32. Frames not
        present in the clip are zero-filled. Joints whose Halpe index
        exceeds the clip's keypoint count are zero-filled.
    """
    out = np.zeros((NUM_CHANNELS, NUM_FRAMES_UBHGCN, NUM_JOINTS_UBHGCN, 1),
                   dtype=np.float32)
    for t, frame_idx in enumerate(FRAME_INDICES):
        if frame_idx >= len(pose_list):
            continue
        frame_pose = pose_list[frame_idx]
        # Defensive: pose_list[frame_idx] might be None or missing 'result'.
        if not frame_pose or "result" not in frame_pose:
            continue
        if not frame_pose["result"]:
            continue
        det = frame_pose["result"][0]   # take the first detection
        kpts_flat = det.get("keypoints", [])
        # Halpe-136 = 136 × 3 = 408 floats.
        if len(kpts_flat) < 408:
            # Fewer keypoints than expected — could be a different format.
            continue
        kpts = np.array(kpts_flat, dtype=np.float32).reshape(-1, 3)
        # kpts shape: (136, 3) where rows are (x, y, score).
        for v_idx, halpe_idx in halpe_to_ubhgcn.items():
            if halpe_idx >= kpts.shape[0]:
                continue
            out[:, t, v_idx, 0] = kpts[halpe_idx]
    return out


def convert_split(
    csv_path: Path,
    annotation_root: Path,
    halpe_to_ubhgcn: dict[int, int],
    progress_label: str = "split",
) -> tuple[np.ndarray, list[int]]:
    """Read the CSV row of (frames_path, annotation_path) for one
    train/val/test split and produce the stacked .npy + label list.
    """
    # AIDE CSVs start with a "0,1" header row (column names = ["0", "1"]).
    # Skip the header and rename columns to our internal schema.
    rows = pd.read_csv(csv_path)
    rows.columns = ["frames_path", "ann_path"]
    n = len(rows)
    data = np.zeros((n, NUM_CHANNELS, NUM_FRAMES_UBHGCN, NUM_JOINTS_UBHGCN, 1),
                    dtype=np.float32)
    labels: list[int] = []
    skipped: list[str] = []
    for i, (_, row) in enumerate(rows.iterrows()):
        ann_rel = row["ann_path"].strip()
        ann_file = annotation_root / Path(ann_rel).name
        if not ann_file.exists():
            # The CSV stores paths like "AIDE_Dataset/annotation/0001.json"; try absolute.
            ann_file = annotation_root.parent / ann_rel
        if not ann_file.exists():
            skipped.append(ann_rel)
            labels.append(-1)
            continue
        try:
            ann = json.loads(ann_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            skipped.append(f"{ann_rel} ({exc})")
            labels.append(-1)
            continue
        pose_list = ann.get("pose_list", [])
        if not pose_list:
            skipped.append(f"{ann_rel} (empty pose_list)")
            labels.append(-1)
            continue
        emo_str = ann.get("emotion_label", "").capitalize()
        if emo_str not in AIDE_EMOTION_LABELS:
            skipped.append(f"{ann_rel} (unknown emotion: {emo_str!r})")
            labels.append(-1)
            continue
        labels.append(AIDE_EMOTION_LABELS.index(emo_str))
        data[i] = aide_pose_to_skeleton(pose_list, halpe_to_ubhgcn)
        if (i + 1) % 200 == 0:
            click.echo(f"  [{progress_label}] {i+1}/{n} converted")
    if skipped:
        click.echo(f"  [{progress_label}] skipped {len(skipped)} (first 5: {skipped[:5]})")
    return data, labels


@click.command()
@click.option("--aide-root", default="data/aide/AIDE_Dataset",
              show_default=True,
              help="Path to the extracted AIDE_Dataset/ directory.")
@click.option("--annotation-root", default=None,
              help="Override path to annotation/ subdir; defaults to "
                   "{aide-root}/annotation.")
@click.option("--csv-root", default="data/aide/AIDE_Dataset",
              show_default=True,
              help="Path to directory containing training.csv, validation.csv, "
                   "testing.csv (typically AIDE_Dataset/).")
@click.option("--output-root", default="data/aide/processed",
              show_default=True,
              help="Where to write train/val/test .npy + .pkl files.")
@click.option("--verify-only", is_flag=True,
              help="Print the inferred mapping + a single sample's pose layout, "
                   "then exit without converting (useful when AIDE extraction "
                   "just finished and we want to confirm the format).")
def main(aide_root: str, annotation_root: str | None, csv_root: str,
         output_root: str, verify_only: bool) -> None:
    aide_root_p = Path(aide_root)
    ann_root_p = Path(annotation_root) if annotation_root else aide_root_p / "annotation"
    csv_root_p = Path(csv_root)
    out_root_p = Path(output_root)

    if not aide_root_p.is_dir():
        raise click.ClickException(f"aide-root not found: {aide_root_p}")
    if not ann_root_p.is_dir():
        raise click.ClickException(f"annotation-root not found: {ann_root_p}")

    if verify_only:
        # Find any annotation JSON and print its Halpe keypoint count.
        candidates = sorted(ann_root_p.glob("*.json"))[:1]
        if not candidates:
            raise click.ClickException(f"No annotation JSONs in {ann_root_p}")
        sample_file = candidates[0]
        click.echo(f"Inspecting {sample_file}")
        ann = json.loads(sample_file.read_text(encoding="utf-8"))
        pose_list = ann.get("pose_list", [])
        click.echo(f"  pose_list length: {len(pose_list)}")
        click.echo(f"  emotion_label: {ann.get('emotion_label')!r}")
        if pose_list:
            first = pose_list[0]
            if "result" in first and first["result"]:
                det = first["result"][0]
                kpts_flat = det.get("keypoints", [])
                n_kpts = len(kpts_flat) // 3
                click.echo(f"  frame 0: {n_kpts} keypoints ({len(kpts_flat)} floats)")
                if n_kpts >= 136:
                    click.echo("  -> matches Halpe-136 expectation ✓")
                else:
                    click.echo(f"  -> WARNING: expected 136 keypoints, got {n_kpts}")
                click.echo(f"  bbox: {det.get('bbox')}")
                click.echo(f"  face_bbox: {det.get('face_bbox')}")
        click.echo("\nInferred Halpe → UbH-GCN-14 mapping:")
        for ub, hp in sorted(UBH_GCN_TO_HALPE.items()):
            click.echo(f"  UbH-GCN {ub+1:2d} ← Halpe-26 body[{hp:2d}]")
        return

    out_root_p.mkdir(parents=True, exist_ok=True)
    for split_name, csv_name, npy_name, pkl_name in (
        ("train", "training.csv", "train_data.npy", "train_label.pkl"),
        ("eval", "validation.csv", "eval_data.npy", "eval_label.pkl"),
        ("test", "testing.csv", "test_data.npy", "test_label.pkl"),
    ):
        csv_path = csv_root_p / csv_name
        if not csv_path.exists():
            click.echo(f"[skip] {csv_name} not found at {csv_path}")
            continue
        click.echo(f"Converting {split_name} from {csv_path}")
        data, labels = convert_split(csv_path, ann_root_p, UBH_GCN_TO_HALPE,
                                     progress_label=split_name)
        np.save(out_root_p / npy_name, data)
        with (out_root_p / pkl_name).open("wb") as f:
            pickle.dump(labels, f)
        click.echo(f"  saved {npy_name} (shape {data.shape}) + {pkl_name} "
                   f"(n_labels={len(labels)})")


if __name__ == "__main__":
    main()
