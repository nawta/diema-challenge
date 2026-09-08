"""exp072 ensemble eval — does C3D contact OOF lift 7-way to 8-way?"""

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


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


@click.command()
@click.option("--splits-pkl", default="data/diema_challenge/processed/split_lpo_10fold.pkl")
@click.option("--c3d-logits", default="output/predictions/c3d_contact_only/oof_logits.npy")
@click.option("--c3d-filenames", default="output/predictions/c3d_contact_only/oof_filenames.npy")
@click.option("--artifacts-root", default="output/artifacts")
@click.option("--results-json", default="experiments/exp072_c3d_contact_only/ensemble_results.json")
def main(splits_pkl, c3d_logits, c3d_filenames, artifacts_root, results_json):
    with open(splits_pkl, "rb") as f:
        splits = pickle.load(f)

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
    for name in MEMBERS:
        fold_logits = []
        for fid in range(10):
            d = root / f"{name}/fold_{fid:02d}"
            logits = np.load(d / "oof_logits.npy")
            fold_logits.append(logits)
        logits_concat = np.concatenate(fold_logits, axis=0)
        probs = softmax(logits_concat)
        member_probs.append(probs)

    seven_way_probs = np.mean(member_probs, axis=0)
    seven_y_pred = seven_way_probs.argmax(axis=-1)
    seven_f1 = float(f1_score(oof_labels_concat, seven_y_pred, average="macro"))
    seven_acc = float(accuracy_score(oof_labels_concat, seven_y_pred))
    print(f"[7way] Macro-F1 {seven_f1*100:.2f}%, Acc {seven_acc*100:.2f}%")

    c3d_arr = np.load(c3d_logits)
    c3d_fns = np.array([str(x) for x in np.load(c3d_filenames, allow_pickle=True)])
    fn_to_idx = {fn: i for i, fn in enumerate(c3d_fns)}
    c3d_aligned = np.zeros((N, NUM_CLASS), dtype=np.float32)
    for i, fn in enumerate(oof_filenames_concat):
        c3d_aligned[i] = c3d_arr[fn_to_idx[fn]]
    c3d_probs = softmax(c3d_aligned)
    c3d_pred = c3d_probs.argmax(axis=-1)
    c3d_f1 = float(f1_score(oof_labels_concat, c3d_pred, average="macro"))
    print(f"[c3d alone] Macro-F1 {c3d_f1*100:.2f}%")

    eight_way_probs = (seven_way_probs * 7 + c3d_probs) / 8
    eight_y_pred = eight_way_probs.argmax(axis=-1)
    eight_f1 = float(f1_score(oof_labels_concat, eight_y_pred, average="macro"))
    eight_acc = float(accuracy_score(oof_labels_concat, eight_y_pred))
    delta = eight_f1 - seven_f1
    print(f"[8way] (7way * 7 + C3D) / 8 Macro-F1 {eight_f1*100:.2f}%, Acc {eight_acc*100:.2f}%")
    print(f"[8way] Δ vs 7-way: {delta*100:+.2f}pp")

    per_class_7 = f1_score(oof_labels_concat, seven_y_pred, average=None, labels=list(range(NUM_CLASS)))
    per_class_8 = f1_score(oof_labels_concat, eight_y_pred, average=None, labels=list(range(NUM_CLASS)))
    print("\n[per-class F1 deltas]")
    regression = []
    for c in range(NUM_CLASS):
        d = (per_class_8[c] - per_class_7[c]) * 100
        flag = "" if abs(d) < 0.30 else (" *worse" if d < -0.30 else " *better")
        if d < -0.30:
            regression.append((IDX_TO_EMOTION[c], d))
        print(f"  {IDX_TO_EMOTION[c]:12s}: 7way {per_class_7[c]*100:.2f}% -> 8way {per_class_8[c]*100:.2f}% Δ {d:+.2f}pp{flag}")

    no_regression = len(regression) == 0
    print(f"\nno per-class regression > 0.30pp: {no_regression}")

    f1_gate_pass = delta >= 0.002
    print(f"F1 gate (Δ ≥ +0.20pp): {'PASS' if f1_gate_pass else 'FAIL'}")
    print(f"No-regression gate: {'PASS' if no_regression else 'FAIL'}")
    decision = "GO" if (f1_gate_pass and no_regression) else "NO-GO"
    print(f"\nDecision: {decision}")

    out = {
        "members": MEMBERS,
        "seven_way_macro_f1": seven_f1,
        "seven_way_acc": seven_acc,
        "c3d_alone_macro_f1": c3d_f1,
        "eight_way_macro_f1": eight_f1,
        "eight_way_acc": eight_acc,
        "delta_macro_f1": delta,
        "per_class": {IDX_TO_EMOTION[c]: {"7way": float(per_class_7[c]), "8way": float(per_class_8[c])} for c in range(NUM_CLASS)},
        "regressions": regression,
        "decision": decision,
    }
    Path(results_json).parent.mkdir(parents=True, exist_ok=True)
    with open(results_json, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
