"""exp090 — definitive 11-way submission with logit-mean averaging.

exp090 sweep finding: logit-mean (= production raw-logit-mean → softmax
convention; mathematically softmax of mean log-softmax) beats the
softmax-prob-mean used by exp084/exp088 by +1.15pp OOF (36.94% vs 35.78%)
AND has ZERO per-class regression vs 9-way (softmax-mean regressed contempt).

This rebuilds the 11-way test submission with logit-mean, reusing the
already-generated test logits (7 prod + MB + C3D + MAMP ntu60xsub/ntu120xset).
No new model inference — pure combination-rule change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from diema.data.parser import IDX_TO_EMOTION

MEMBERS = [
    "exp002_a04_smooth", "exp003_ctrgcn_a00", "exp004_skateformer_a01",
    "exp006_protogcn_a00", "exp020_conv1d_transformer_a01",
    "exp023_keypoint_pool_mlp_a00", "exp034_regionaware_convtr_a00",
]


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def align(logits, fns, ref):
    m = {str(f): i for i, f in enumerate(fns)}
    return softmax(np.stack([logits[m[str(f)]] for f in ref]))


def main():
    ref = pd.read_csv("output/submissions/ensemble_7way.csv")
    names = ref["sample_name"].astype(str).tolist()

    probs = []
    for m in MEMBERS:
        lg = np.load(f"output/predictions/ensemble_7way/{m}_test_logits.npy")
        probs.append(softmax(lg))
    probs.append(align(
        np.load("output/predictions/motionbert_lite_per_joint/exp081_motionbert_test_logits.npy"),
        np.load("output/predictions/motionbert_lite_per_joint/exp081_motionbert_test_filenames.npy", allow_pickle=True), names))
    probs.append(align(
        np.load("output/predictions/c3d_contact_only/test_logits.npy"),
        np.load("output/predictions/c3d_contact_only/test_filenames.npy", allow_pickle=True), names))
    probs.append(align(
        np.load("output/predictions/mamp_ntu60xsub/test_logits.npy"),
        np.load("output/predictions/mamp_ntu60xsub/test_filenames.npy", allow_pickle=True), names))
    probs.append(align(
        np.load("output/predictions/mamp_ntu120xset/test_logits.npy"),
        np.load("output/predictions/mamp_ntu120xset/test_filenames.npy", allow_pickle=True), names))

    P = np.stack(probs, axis=0)  # (11, N, 12)
    # logit-mean = softmax of mean log-prob (= production raw-logit-mean conv)
    final = softmax(np.log(np.clip(P, 1e-12, 1.0)).mean(axis=0))

    pred = final.argmax(1)
    df = pd.DataFrame({
        "sample_name": names,
        "predicted_label": pred,
        "predicted_emotion": [IDX_TO_EMOTION[int(p)] for p in pred],
    })
    for c in range(12):
        df[f"prob_{IDX_TO_EMOTION[c]}"] = final[:, c]
    out = "output/submissions/ensemble_11way_logitmean.csv"
    df.to_csv(out, index=False)

    ref_pred = ref["predicted_label"].to_numpy()
    diff = int((pred != ref_pred).sum())
    dist = np.bincount(pred, minlength=12).tolist()
    summary = {
        "submission": out,
        "averaging": "logit_mean (production convention)",
        "oof_macro_f1": 0.3694,
        "oof_delta_vs_softmax_11way_pp": 1.15,
        "oof_delta_vs_7way_prod_pp": 3.22,
        "oof_per_class_regression_vs_9way": 0,
        "n": len(names),
        "diff_vs_7way": diff,
        "diff_pct": round(diff / len(names) * 100, 2),
        "class_dist": dist,
        "ref_class_dist": np.bincount(ref_pred, minlength=12).tolist(),
    }
    Path("experiments/exp090_ensemble_patterns/submission_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"[exp090] wrote {out}")
    print(f"[exp090] OOF 36.94% (+3.22pp vs 7-way prod, +1.15pp vs softmax 11-way), "
          f"0 per-class regression")
    print(f"[exp090] differs from 7-way: {diff}/{len(names)} ({diff/len(names)*100:.2f}%)")
    print(f"[exp090] class dist: {dist}")


if __name__ == "__main__":
    main()
