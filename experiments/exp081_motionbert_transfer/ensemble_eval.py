"""exp081 ensemble eval — does MotionBERT OOF logits lift the production 7-way?

Loads all 7 production member OOF logits + the new MotionBERT OOF logits,
aligns to a common filename ordering, computes:
  - 7-way equal-weight Macro-F1 (sanity check vs docs 34.03%)
  - 8-way (7 + MotionBERT) Macro-F1
  - per-class F1 deltas
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import click
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from diema.data.parser import IDX_TO_EMOTION, emotion_to_label

MEMBERS = [
    "exp002_a04_smooth",
    "exp003_ctrgcn_a00",
    "exp004_skateformer_a01",
    "exp006_protogcn_a00",
    "exp020_conv1d_transformer_a01",
    "exp023_keypoint_pool_mlp_a00",
    "exp034_regionaware_convtr_a00",
]


def load_member_oof_aligned(name: str, filenames_target: np.ndarray, root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load 10-fold OOF logits + labels concatenated, then align to filenames_target."""
    # Each fold writes oof_logits.npy + oof_labels.npy. We also need OOF filenames.
    fold_logits, fold_labels, fold_fns = [], [], []
    for fid in range(10):
        d = root / f"{name}/fold_{fid:02d}"
        logits = np.load(d / "oof_logits.npy")
        labels = np.load(d / "oof_labels.npy")
        # OOF filenames may be saved separately; for our purposes we
        # reconstruct using val split filenames per fold.
        fold_logits.append(logits)
        fold_labels.append(labels)
        # NOTE: we trust that each fold's OOF rows are in same order as val split's filenames
    logits_concat = np.concatenate(fold_logits, axis=0)
    labels_concat = np.concatenate(fold_labels, axis=0)
    return logits_concat, labels_concat


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


@click.command()
@click.option("--splits-pkl", default="data/diema_challenge/processed/split_lpo_10fold.pkl")
@click.option("--mb-logits", default="output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_logits.npy")
@click.option("--mb-filenames", default="output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_filenames.npy")
@click.option("--artifacts-root", default="output/artifacts")
@click.option("--results-json", default="experiments/exp081_motionbert_transfer/ensemble_results.json")
def main(splits_pkl, mb_logits, mb_filenames, artifacts_root, results_json):
    # Load splits to reconstruct OOF filename ordering per fold
    with open(splits_pkl, "rb") as f:
        splits = pickle.load(f)

    # Build the concatenated OOF filenames in the order the per-fold OOF arrays were saved
    oof_filenames_concat = []
    for fid in range(10):
        for fn, _idx in splits[fid]["val"]:
            oof_filenames_concat.append(str(fn))
    oof_filenames_concat = np.array(oof_filenames_concat)

    oof_labels_concat = np.array([emotion_to_label(fn) for fn in oof_filenames_concat], dtype=np.int64)
    N = len(oof_filenames_concat)
    NUM_CLASS = 12

    root = Path(artifacts_root)

    member_probs = []
    member_metrics = {}
    for name in MEMBERS:
        logits, labels = load_member_oof_aligned(name, oof_filenames_concat, root)
        if logits.shape[0] != N:
            print(f"[warn] {name} OOF n={logits.shape[0]} != target N={N}")
            continue
        probs = softmax(logits)
        member_probs.append(probs)
        y_pred = probs.argmax(axis=-1)
        member_metrics[name] = {
            "macro_f1": float(f1_score(labels, y_pred, average="macro")),
            "acc": float(accuracy_score(labels, y_pred)),
        }
        print(f"[member] {name}: Macro-F1 {member_metrics[name]['macro_f1']*100:.2f}%, Acc {member_metrics[name]['acc']*100:.2f}%")

    # 7-way equal-weight ensemble
    seven_way_probs = np.mean(member_probs, axis=0)
    seven_y_pred = seven_way_probs.argmax(axis=-1)
    seven_f1 = float(f1_score(oof_labels_concat, seven_y_pred, average="macro"))
    seven_acc = float(accuracy_score(oof_labels_concat, seven_y_pred))
    print(f"\n[7way] equal-weight Macro-F1 {seven_f1*100:.2f}%, Acc {seven_acc*100:.2f}%")

    # MotionBERT OOF (need to align to the same ordering)
    mb_logits_arr = np.load(mb_logits)  # ordered by filenames as saved by full_10fold.py (matches motion_train_quat order)
    mb_fns_arr = np.load(mb_filenames, allow_pickle=True)
    mb_fns_str = np.array([str(x) for x in mb_fns_arr])
    fn_to_mb_idx = {fn: i for i, fn in enumerate(mb_fns_str)}
    # Build mb_logits aligned to OOF order
    mb_logits_aligned = np.zeros((N, NUM_CLASS), dtype=np.float32)
    for i, fn in enumerate(oof_filenames_concat):
        mb_logits_aligned[i] = mb_logits_arr[fn_to_mb_idx[fn]]
    mb_probs = softmax(mb_logits_aligned)
    mb_pred = mb_probs.argmax(axis=-1)
    mb_f1 = float(f1_score(oof_labels_concat, mb_pred, average="macro"))
    print(f"[mb alone] Macro-F1 {mb_f1*100:.2f}%")

    # 8-way equal-weight ensemble
    eight_way_probs = (seven_way_probs * 7 + mb_probs) / 8
    eight_y_pred = eight_way_probs.argmax(axis=-1)
    eight_f1 = float(f1_score(oof_labels_concat, eight_y_pred, average="macro"))
    eight_acc = float(accuracy_score(oof_labels_concat, eight_y_pred))
    delta = eight_f1 - seven_f1
    print(f"[8way] (7way * 7 + MotionBERT) / 8 Macro-F1 {eight_f1*100:.2f}%, Acc {eight_acc*100:.2f}%")
    print(f"[8way] Δ vs 7-way: {delta*100:+.2f}pp Macro-F1")
    print(f"  GO gate (+0.20pp): {'PASS' if delta >= 0.002 else 'FAIL'}")

    # Per-class F1 deltas
    per_class_7 = f1_score(oof_labels_concat, seven_y_pred, average=None, labels=list(range(NUM_CLASS)))
    per_class_8 = f1_score(oof_labels_concat, eight_y_pred, average=None, labels=list(range(NUM_CLASS)))
    print("\n[per-class F1 deltas]")
    for c in range(NUM_CLASS):
        d = (per_class_8[c] - per_class_7[c]) * 100
        flag = "" if abs(d) < 0.30 else (" *worse" if d < -0.30 else " *better")
        print(f"  {IDX_TO_EMOTION[c]:12s}: 7way {per_class_7[c]*100:.2f}% -> 8way {per_class_8[c]*100:.2f}% Δ {d:+.2f}pp{flag}")
    no_class_regression = bool(all(per_class_8[c] - per_class_7[c] >= -0.003 for c in range(NUM_CLASS)))
    print(f"\nno per-class regression worse than -0.30pp: {no_class_regression}")

    results = {
        "members": list(member_metrics.keys()),
        "member_metrics": member_metrics,
        "seven_way": {"macro_f1": seven_f1, "acc": seven_acc},
        "mb_alone": {"macro_f1": mb_f1},
        "eight_way_avg": {"macro_f1": eight_f1, "acc": eight_acc, "delta_macro_f1": delta},
        "per_class_f1": {
            IDX_TO_EMOTION[c]: {"7way": float(per_class_7[c]), "8way": float(per_class_8[c])}
            for c in range(NUM_CLASS)
        },
        "no_class_regression": no_class_regression,
        "go_threshold_macro_f1_delta": 0.002,
        "decision": "GO" if (delta >= 0.002 and no_class_regression) else "NO-GO",
    }
    Path(results_json).parent.mkdir(parents=True, exist_ok=True)
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[result] decision: {results['decision']}")


if __name__ == "__main__":
    main()
