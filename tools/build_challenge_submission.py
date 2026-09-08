#!/usr/bin/env python3
"""Build the official MMAC@ACII 2026 DIEM-A challenge test-set submission CSV.

Converts the locked production ensemble prediction file
(``output/submissions/ensemble_11way_logitmean.csv`` — 11-way logit-mean,
36.94% OOF Macro-F1, exp090, zero per-class regression) into the
challenge-required two-column format::

    anonymized_name,predicted_emotion

The ID set and row order are validated against the official test ID list
(``test0001`` .. ``test1944``); predicted emotions are checked against the 12
official lowercase class names. The script refuses to emit a file that would
fail the organizers' ``validate_submission.py`` checker.

Design refs:
  * docs/analysis/exp090_ensemble_patterns_verdict.md — logit-mean averaging rule
  * docs/analysis/exp088_final_submission.md           — 11 member set
  * docs/results_summary.md §"Best 成績"                — 36.94% locked best
"""
import argparse
import csv
import os
from collections import Counter

EMOTIONS = [
    "anger", "contempt", "disgust", "fear", "joy", "sadness",
    "surprise", "jealousy", "shame", "guilt", "gratitude", "pride",
]
EMOTION_SET = set(EMOTIONS)
N_TEST = 1944  # official test clips test0001 .. test1944


def _find(fieldnames, *names):
    """Return the actual header matching any of `names` (case/space-insensitive)."""
    wanted = {n.lower() for n in names}
    for h in fieldnames or []:
        if (h or "").strip().lower() in wanted:
            return h
    return None


def read_official_ids(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        col = _find(reader.fieldnames, "anonymized_name")
        if col is None:
            raise SystemExit(f"{path}: no 'anonymized_name' column")
        ids = [(r.get(col) or "").strip() for r in reader]
    return [i for i in ids if i]


def read_predictions(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        id_col = _find(reader.fieldnames, "anonymized_name", "sample_name")
        pred_col = _find(reader.fieldnames, "predicted_emotion")
        if id_col is None or pred_col is None:
            raise SystemExit(f"{path}: need an ID column and 'predicted_emotion'")
        preds = {}
        for r in reader:
            name = (r.get(id_col) or "").strip()
            preds[name] = (r.get(pred_col) or "").strip().lower()
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default="output/submissions/ensemble_11way_logitmean.csv",
                    help="locked production ensemble prediction CSV")
    ap.add_argument("--ids", default="data/diema_challenge/raw/test_data.csv",
                    help="official test_data.csv (provides ID set + order)")
    ap.add_argument("--out", default="output/submissions/MMAC_ACII2026_test_predictions.csv",
                    help="output submission CSV (2 columns)")
    args = ap.parse_args()

    if os.path.exists(args.ids):
        ids = read_official_ids(args.ids)
    else:
        ids = [f"test{i:04d}" for i in range(1, N_TEST + 1)]
        print(f"[warn] {args.ids} missing; using standard {N_TEST} IDs (test0001..test{N_TEST:04d})")

    preds = read_predictions(args.preds)

    # --- integrity gates (mirror the organizers' validator) ---
    id_set, pred_set = set(ids), set(preds)
    if not (len(ids) == len(id_set) == N_TEST):
        raise SystemExit(f"official IDs are not {N_TEST} unique values: "
                         f"{len(ids)} rows / {len(id_set)} unique")
    missing = sorted(id_set - pred_set)
    extra = sorted(pred_set - id_set)
    if missing:
        raise SystemExit(f"{len(missing)} test IDs have no prediction, e.g. {missing[:10]}")
    if extra:
        raise SystemExit(f"{len(extra)} predicted IDs are not in the test set, e.g. {extra[:10]}")
    bad = {n: e for n, e in preds.items() if e not in EMOTION_SET}
    if bad:
        raise SystemExit(f"{len(bad)} invalid emotion labels, e.g. {list(bad.items())[:10]}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["anonymized_name", "predicted_emotion"])
        for name in ids:  # official order: test0001 .. test1944
            w.writerow([name, preds[name]])

    dist = Counter(preds[n] for n in ids)
    print(f"[ok] wrote {args.out}: {len(ids)} rows (header + {len(ids)} predictions)")
    print("[dist] " + " ".join(f"{e}={dist[e]}" for e in EMOTIONS))


if __name__ == "__main__":
    main()
