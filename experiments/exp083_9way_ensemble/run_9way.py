"""exp083 — 9-way ensemble: 7-way production + MotionBERT + C3D.

Hypothesis (from session-summary multi-model consultation):
MotionBERT (exp081) and C3D contact (exp072) have ORTHOGONAL per-class
regressions:
  MotionBERT regresses: joy/surprise/gratitude/guilt/contempt (social-context)
  C3D regresses:        jealousy/fear/guilt/contempt (subtle-posture)
  Both gain:            anger, sadness, pride (body-energy)

Averaging them in a 9-way ensemble may cancel the regressions:
  jealousy: MB +2.04 + C3D -1.08 ≈ avg ~ +0.48
  joy:       MB -1.69 + C3D +0.35 ≈ avg ~ -0.67 (halved)

Cost: 1 min CPU (both OOF logits already saved).

Inputs:
  output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_logits.npy
  output/predictions/c3d_contact_only/oof_logits.npy
  output/artifacts/{7 production members}/fold_*/oof_logits.npy

Gates (pre-registered, matches exp081/072 protocol):
  - F1: 9-way Δ ≥ +0.20pp vs 7-way
  - No per-class regression > 0.30pp (strict, original)
  - Optional loosen: ≤ 2 classes regress > 0.30pp (relaxed)

Output:
  experiments/exp083_9way_ensemble/results.json
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


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


@click.command()
@click.option("--splits-pkl", default="data/diema_challenge/processed/split_lpo_10fold.pkl")
@click.option("--mb-logits", default="output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_logits.npy")
@click.option("--mb-filenames", default="output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_filenames.npy")
@click.option("--c3d-logits", default="output/predictions/c3d_contact_only/oof_logits.npy")
@click.option("--c3d-filenames", default="output/predictions/c3d_contact_only/oof_filenames.npy")
@click.option("--artifacts-root", default="output/artifacts")
@click.option("--results-json", default="experiments/exp083_9way_ensemble/results.json")
def main(splits_pkl, mb_logits, mb_filenames, c3d_logits, c3d_filenames, artifacts_root, results_json):
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
            fold_logits.append(np.load(d / "oof_logits.npy"))
        logits_concat = np.concatenate(fold_logits, axis=0)
        member_probs.append(softmax(logits_concat))
    seven_way_probs = np.mean(member_probs, axis=0)
    seven_y_pred = seven_way_probs.argmax(axis=-1)
    seven_f1 = float(f1_score(oof_labels_concat, seven_y_pred, average="macro"))
    seven_acc = float(accuracy_score(oof_labels_concat, seven_y_pred))
    print(f"[7way] Macro-F1 {seven_f1*100:.2f}%, Acc {seven_acc*100:.2f}%")

    def align_oof(logits_path: str, fns_path: str) -> np.ndarray:
        arr = np.load(logits_path)
        fns = np.array([str(x) for x in np.load(fns_path, allow_pickle=True)])
        fn_to_idx = {fn: i for i, fn in enumerate(fns)}
        aligned = np.zeros((N, NUM_CLASS), dtype=np.float32)
        for i, fn in enumerate(oof_filenames_concat):
            aligned[i] = arr[fn_to_idx[fn]]
        return softmax(aligned)

    mb_probs = align_oof(mb_logits, mb_filenames)
    c3d_probs = align_oof(c3d_logits, c3d_filenames)

    mb_pred = mb_probs.argmax(axis=-1)
    c3d_pred = c3d_probs.argmax(axis=-1)
    print(f"[mb alone]  Macro-F1 {f1_score(oof_labels_concat, mb_pred, average='macro')*100:.2f}%")
    print(f"[c3d alone] Macro-F1 {f1_score(oof_labels_concat, c3d_pred, average='macro')*100:.2f}%")

    eight_mb_probs = (seven_way_probs * 7 + mb_probs) / 8
    eight_c3d_probs = (seven_way_probs * 7 + c3d_probs) / 8
    nine_probs = (seven_way_probs * 7 + mb_probs + c3d_probs) / 9

    for name, probs in [
        ("8way+MB", eight_mb_probs),
        ("8way+C3D", eight_c3d_probs),
        ("9way (MB+C3D)", nine_probs),
    ]:
        y_pred = probs.argmax(axis=-1)
        f1 = float(f1_score(oof_labels_concat, y_pred, average="macro"))
        acc = float(accuracy_score(oof_labels_concat, y_pred))
        delta = f1 - seven_f1
        print(f"[{name}] Macro-F1 {f1*100:.2f}%, Acc {acc*100:.2f}%, Δ {delta*100:+.2f}pp")

    nine_y_pred = nine_probs.argmax(axis=-1)
    per_class_7 = f1_score(oof_labels_concat, seven_y_pred, average=None, labels=list(range(NUM_CLASS)))
    per_class_9 = f1_score(oof_labels_concat, nine_y_pred, average=None, labels=list(range(NUM_CLASS)))
    print("\n[9-way per-class F1 deltas]")
    regression_strict = []
    regression_loose = []
    for c in range(NUM_CLASS):
        delta_c = (per_class_9[c] - per_class_7[c]) * 100
        flag = ""
        if delta_c < -0.30:
            flag = " *worse"
            regression_strict.append((IDX_TO_EMOTION[c], delta_c))
        elif delta_c > 0.30:
            flag = " *better"
        if delta_c < -1.00:
            regression_loose.append((IDX_TO_EMOTION[c], delta_c))
        print(f"  {IDX_TO_EMOTION[c]:12s}: 7way {per_class_7[c]*100:.2f}% -> 9way {per_class_9[c]*100:.2f}% Δ {delta_c:+.2f}pp{flag}")

    nine_f1 = float(f1_score(oof_labels_concat, nine_y_pred, average="macro"))
    delta = nine_f1 - seven_f1
    f1_gate_pass = delta >= 0.002
    no_regression_strict = len(regression_strict) == 0
    no_regression_loose = len(regression_loose) == 0

    print(f"\n=== 9-way verdict ===")
    print(f"F1 gate (Δ ≥ +0.20pp): {'PASS' if f1_gate_pass else 'FAIL'} ({delta*100:+.2f}pp)")
    print(f"Strict no-regression (-0.30pp): {'PASS' if no_regression_strict else f'FAIL ({len(regression_strict)} classes)'}")
    print(f"Relaxed no-regression (-1.00pp): {'PASS' if no_regression_loose else f'FAIL ({len(regression_loose)} classes)'}")
    decision_strict = "GO" if (f1_gate_pass and no_regression_strict) else "NO-GO"
    decision_loose = "GO" if (f1_gate_pass and no_regression_loose) else "NO-GO"
    print(f"Decision (strict gate): {decision_strict}")
    print(f"Decision (loose gate, ≤-1.00pp regression OK): {decision_loose}")

    out = {
        "members": MEMBERS + ["motionbert_lite", "c3d_contact"],
        "seven_way_macro_f1": seven_f1,
        "eight_way_mb_macro_f1": float(f1_score(oof_labels_concat, eight_mb_probs.argmax(-1), average="macro")),
        "eight_way_c3d_macro_f1": float(f1_score(oof_labels_concat, eight_c3d_probs.argmax(-1), average="macro")),
        "nine_way_macro_f1": nine_f1,
        "nine_way_delta": delta,
        "per_class_7way": {IDX_TO_EMOTION[c]: float(per_class_7[c]) for c in range(NUM_CLASS)},
        "per_class_9way": {IDX_TO_EMOTION[c]: float(per_class_9[c]) for c in range(NUM_CLASS)},
        "regressions_strict_0.30pp": regression_strict,
        "regressions_loose_1.00pp": regression_loose,
        "f1_gate_pass": f1_gate_pass,
        "no_regression_strict_pass": no_regression_strict,
        "no_regression_loose_pass": no_regression_loose,
        "decision_strict": decision_strict,
        "decision_loose": decision_loose,
    }
    Path(results_json).parent.mkdir(parents=True, exist_ok=True)
    with open(results_json, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
