"""exp084 — format check the 9-way submission CSVs against the 7-way reference.

Validates:
  - row count == reference (1944)
  - column set == reference (header equality)
  - sample_name set + order == reference
  - prob_* row sums = 1.0 ± 1e-5
  - argmax(prob_*) == predicted_label
  - predicted_emotion matches IDX_TO_EMOTION[predicted_label]
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diema.data.parser import IDX_TO_EMOTION


def check_csv(path: str, reference_path: str) -> tuple[bool, list[str]]:
    issues: list[str] = []
    df = pd.read_csv(path)
    ref = pd.read_csv(reference_path)
    if len(df) != len(ref):
        issues.append(f"row count {len(df)} != ref {len(ref)}")
    if list(df.columns) != list(ref.columns):
        issues.append(f"columns mismatch: {list(df.columns)} vs {list(ref.columns)}")
    if (df["sample_name"].astype(str).tolist()
            != ref["sample_name"].astype(str).tolist()):
        issues.append("sample_name order/values mismatch")
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    row_sums = df[prob_cols].sum(axis=1).to_numpy()
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        bad = (np.abs(row_sums - 1.0) > 1e-5).sum()
        issues.append(f"prob_* row sums != 1.0 for {bad}/{len(df)} rows (max dev {np.abs(row_sums - 1.0).max():.2e})")
    probs = df[prob_cols].to_numpy()
    arg = probs.argmax(axis=1)
    pred = df["predicted_label"].to_numpy()
    if not (arg == pred).all():
        diff = (arg != pred).sum()
        issues.append(f"argmax(prob_*) != predicted_label for {diff}/{len(df)} rows")
    emo = df["predicted_emotion"].astype(str).tolist()
    expected = [IDX_TO_EMOTION[int(p)] for p in pred]
    if emo != expected:
        diff = sum(a != b for a, b in zip(emo, expected))
        issues.append(f"predicted_emotion mismatch for {diff}/{len(df)} rows")
    return len(issues) == 0, issues


@click.command()
@click.option("--csv", required=True, type=str)
@click.option("--reference", default="output/submissions/ensemble_7way.csv")
def main(csv: str, reference: str) -> None:
    ok, issues = check_csv(csv, reference)
    print(f"[check] {csv}")
    print(f"[check] reference: {reference}")
    if ok:
        print("[check] PASS — all format checks passed")
    else:
        print("[check] FAIL — issues:")
        for it in issues:
            print(f"  - {it}")
        sys.exit(1)


if __name__ == "__main__":
    main()
