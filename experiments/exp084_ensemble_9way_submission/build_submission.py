"""exp084 — build 9-way submission CSV from 7-way + MB + C3D test logits.

Hypothesis test (production-side):
  exp083 OOF showed +0.59pp Macro-F1 with 9-way = (7-way * 7 + MB + C3D) / 9
  averaged in softmax-probability space. We extend the same formula to
  test predictions.

Two outputs:
  - ensemble_9way_softmaxavg.csv  : probability-space average (exp083 convention)
  - ensemble_9way_logitavg.csv    : raw-logit average (production 7-way convention)

The two should agree on argmax for most rows; differences flag rows where
the average choice matters.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diema.data.parser import IDX_TO_EMOTION

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


def align_to_reference(logits: np.ndarray, fns: list[str], ref_fns: list[str]) -> np.ndarray:
    fn_to_idx = {fn: i for i, fn in enumerate(fns)}
    aligned = np.zeros_like(logits)
    for i, fn in enumerate(ref_fns):
        if fn not in fn_to_idx:
            raise KeyError(f"reference filename {fn!r} missing from secondary source")
        aligned[i] = logits[fn_to_idx[fn]]
    return aligned


def build_submission_df(probs: np.ndarray, sample_names: list[str]) -> pd.DataFrame:
    preds = probs.argmax(axis=1)
    df = pd.DataFrame({
        "sample_name": sample_names,
        "predicted_label": preds,
        "predicted_emotion": [IDX_TO_EMOTION[int(p)] for p in preds],
    })
    for c in range(12):
        df[f"prob_{IDX_TO_EMOTION[c]}"] = probs[:, c]
    return df


@click.command()
@click.option("--seven-way-dir", default="output/predictions/ensemble_7way")
@click.option("--mb-test-logits", default="output/predictions/motionbert_lite_per_joint/exp081_motionbert_test_logits.npy")
@click.option("--mb-test-fns", default="output/predictions/motionbert_lite_per_joint/exp081_motionbert_test_filenames.npy")
@click.option("--c3d-test-logits", default="output/predictions/c3d_contact_only/test_logits.npy")
@click.option("--c3d-test-fns", default="output/predictions/c3d_contact_only/test_filenames.npy")
@click.option("--reference-csv", default="output/submissions/ensemble_7way.csv")
@click.option("--softmax-csv", default="output/submissions/ensemble_9way_softmaxavg.csv")
@click.option("--logit-csv", default="output/submissions/ensemble_9way_logitavg.csv")
@click.option("--results-json", default="experiments/exp084_ensemble_9way_submission/results.json")
def main(
    seven_way_dir: str,
    mb_test_logits: str,
    mb_test_fns: str,
    c3d_test_logits: str,
    c3d_test_fns: str,
    reference_csv: str,
    softmax_csv: str,
    logit_csv: str,
    results_json: str,
) -> None:
    t0 = time.time()
    ref_df = pd.read_csv(reference_csv)
    ref_sample_names = ref_df["sample_name"].astype(str).tolist()
    print(f"[ref] {len(ref_sample_names)} sample names from {reference_csv}")

    # Load per-member 7-way test logits, average in logit space (production convention)
    seven_logits = []
    for m in MEMBERS:
        path = Path(seven_way_dir) / f"{m}_test_logits.npy"
        seven_logits.append(np.load(path))
        print(f"  loaded {path.name} shape {seven_logits[-1].shape}")
    seven_logits = np.stack(seven_logits, axis=0)  # (7, 1944, 12)
    if seven_logits.shape[1] != len(ref_sample_names):
        raise ValueError(
            f"7-way logits row count {seven_logits.shape[1]} != ref {len(ref_sample_names)}"
        )

    mb_logits = np.load(mb_test_logits)
    mb_fns = [str(x) for x in np.load(mb_test_fns, allow_pickle=True)]
    mb_logits = align_to_reference(mb_logits, mb_fns, ref_sample_names)
    print(f"[mb] aligned to ref shape {mb_logits.shape}")

    c3d_logits = np.load(c3d_test_logits)
    c3d_fns = [str(x) for x in np.load(c3d_test_fns, allow_pickle=True)]
    c3d_logits = align_to_reference(c3d_logits, c3d_fns, ref_sample_names)
    print(f"[c3d] aligned to ref shape {c3d_logits.shape}")

    # Variant A: probability-space average (exp083 convention)
    seven_probs = softmax(seven_logits, axis=-1)  # (7, N, 12)
    seven_way_probs_avg = seven_probs.mean(axis=0)  # (N, 12)
    mb_probs = softmax(mb_logits, axis=-1)
    c3d_probs = softmax(c3d_logits, axis=-1)
    nine_probs_softmaxavg = (seven_way_probs_avg * 7 + mb_probs + c3d_probs) / 9

    # Variant B: logit-space average (production 7-way convention)
    seven_way_logits_avg = seven_logits.mean(axis=0)
    nine_logits_avg = (seven_way_logits_avg * 7 + mb_logits + c3d_logits) / 9
    nine_probs_logitavg = softmax(nine_logits_avg, axis=-1)

    df_softmax = build_submission_df(nine_probs_softmaxavg, ref_sample_names)
    df_logit = build_submission_df(nine_probs_logitavg, ref_sample_names)
    df_ref = ref_df  # 7-way production reference

    Path(softmax_csv).parent.mkdir(parents=True, exist_ok=True)
    df_softmax.to_csv(softmax_csv, index=False)
    df_logit.to_csv(logit_csv, index=False)
    print(f"\nWritten:\n  {softmax_csv}\n  {logit_csv}")

    # Diagnostics
    n = len(ref_sample_names)
    sm_pred = df_softmax["predicted_label"].to_numpy()
    lo_pred = df_logit["predicted_label"].to_numpy()
    ref_pred = df_ref["predicted_label"].to_numpy()
    agree_sm_lo = int((sm_pred == lo_pred).sum())
    diff_sm_ref = int((sm_pred != ref_pred).sum())
    diff_lo_ref = int((lo_pred != ref_pred).sum())

    print(f"\n[diag] softmax-avg vs logit-avg argmax agreement: {agree_sm_lo}/{n} ({agree_sm_lo/n*100:.2f}%)")
    print(f"[diag] softmax-avg differs from 7-way ref: {diff_sm_ref}/{n} ({diff_sm_ref/n*100:.2f}%)")
    print(f"[diag] logit-avg   differs from 7-way ref: {diff_lo_ref}/{n} ({diff_lo_ref/n*100:.2f}%)")

    pred_dist_sm = np.bincount(sm_pred, minlength=12).tolist()
    pred_dist_lo = np.bincount(lo_pred, minlength=12).tolist()
    pred_dist_ref = np.bincount(ref_pred, minlength=12).tolist()
    print(f"\n[dist] 7-way ref pred-class counts: {pred_dist_ref}")
    print(f"[dist] 9-way softmax-avg counts:   {pred_dist_sm}")
    print(f"[dist] 9-way logit-avg counts:     {pred_dist_lo}")

    out = {
        "elapsed_sec": time.time() - t0,
        "n_samples": n,
        "softmax_csv": softmax_csv,
        "logit_csv": logit_csv,
        "softmax_vs_logit_argmax_agree": agree_sm_lo,
        "softmax_vs_ref_argmax_diff": diff_sm_ref,
        "logit_vs_ref_argmax_diff": diff_lo_ref,
        "ref_class_distribution": pred_dist_ref,
        "softmax_class_distribution": pred_dist_sm,
        "logit_class_distribution": pred_dist_lo,
        "ref_csv": reference_csv,
    }
    Path(results_json).parent.mkdir(parents=True, exist_ok=True)
    with open(results_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[exp084.build] saved {results_json}")


if __name__ == "__main__":
    main()
