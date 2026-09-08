"""error_correlation_probe.py — fold-0 diversity probe for the
joint_pos member vs the production 11-way ensemble.

The cross-country retarget's whole rationale is *error decorrelation*: a
member that is slightly weaker standalone can still help an ensemble if
it makes different mistakes. The Stage 3 / pre-flight smokes showed the
retarget does not help *standalone* F1 — this probe asks the only
remaining question cheaply, on fold 0, before committing to a 30 hr
10-fold run.

For each candidate joint_pos member (mix=0.4 retarget, mix=0.0
no-retarget) it reports, on the 864 fold-0 samples:
  - standalone fold-0 Macro-F1
  - 11-way vs 12-way fold-0 Macro-F1 (the direct "does it help" test)
  - max pairwise error cosine + Cohen's kappa vs the 11 members
    (framework §8 decision rule: <0.55 add / 0.55-0.70 conditional /
    >0.70 redundant) — with a high-error-regime caveat.

Reuses paper/reconcile_f1.py's OOF loader (11-member logit set).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "paper"))

from reconcile_f1 import load_oof, combine, PROD, EXTRA4, SPLITS_PKL  # noqa: E402

ART = REPO / "output" / "artifacts" / "exp099_phase1_smoke"
CANDIDATES = {
    "jointpos_honest_mix0.4_retarget": "jointpos_honest_fold_00_e65_upidx2",
    "jointpos_honest_mix0.0_noretarget": "jointpos_honest_mix00_fold_00_e65",
}


def _ref_filenames() -> list[str]:
    """Reconstruct the OOF reference filename order used by reconcile_f1."""
    with open(SPLITS_PKL, "rb") as f:
        splits = pickle.load(f)
    return [str(fn) for fid in range(10) for fn, _ in splits[fid]["val"]]


def _load_member_oof(run_dir: Path):
    """Load a smoke run's OOF: (logits (N,12), filenames list, labels (N,))."""
    logits = np.load(run_dir / "oof_logits.npy").astype(np.float64)
    fns = (run_dir / "oof_filenames.txt").read_text().splitlines()
    fns = [f for f in fns if f]
    labels = np.load(run_dir / "oof_labels.npy").astype(np.int64)
    assert len(fns) == logits.shape[0] == labels.shape[0], (
        f"OOF length mismatch in {run_dir}: "
        f"logits {logits.shape[0]}, fns {len(fns)}, labels {labels.shape[0]}")
    return logits, fns, labels


def _err_cosine(err_a: np.ndarray, err_b: np.ndarray) -> float:
    """Cosine similarity of two binary misclassification vectors."""
    na = np.linalg.norm(err_a)
    nb = np.linalg.norm(err_b)
    if na == 0 or nb == 0:
        return 0.0
    return float((err_a @ err_b) / (na * nb))


def main():
    print("[probe] loading 11-member production OOF ...")
    raw, y, fold_ids = load_oof()
    ref = _ref_filenames()
    assert len(ref) == len(y) == 7992, f"ref/y length {len(ref)}/{len(y)}"
    members = PROD + EXTRA4
    assert len(members) == 11, f"expected 11 members, got {len(members)}"

    fold0 = fold_ids == 0
    ref0 = [ref[i] for i in range(len(ref)) if fold0[i]]
    ref0_idx = {fn: gi for gi, fn in
                [(i, ref[i]) for i in range(len(ref)) if fold0[i]]}
    y0 = y[fold0]
    print(f"[probe] fold 0: {len(ref0)} samples")

    # 11-way predictions on the full OOF, then fold-0 slice
    pred11_full = combine(raw, members, "logit_mean")
    pred11_0 = pred11_full[fold0]
    f1_11 = f1_score(y0, pred11_0, average="macro")
    err11 = (pred11_0 != y0).astype(np.float64)
    # per-member fold-0 predictions + error vectors
    member_pred0 = {m: raw[m][fold0].argmax(-1) for m in members}
    member_err0 = {m: (member_pred0[m] != y0).astype(np.float64)
                   for m in members}

    print(f"\n[probe] 11-way fold-0 Macro-F1 = {f1_11*100:.2f}%  "
          f"(error rate {err11.mean()*100:.1f}%)")
    print("=" * 68)

    for cand_name, cand_dir in CANDIDATES.items():
        run_dir = ART / cand_dir
        if not run_dir.exists():
            print(f"\n[probe] SKIP {cand_name}: {run_dir} missing")
            continue
        logits, fns, labels = _load_member_oof(run_dir)

        # Align the candidate's fold-0 samples to the ref fold-0 order.
        # The candidate IS a fold-0 run, so its filenames should equal ref0.
        cand_idx = {fn: i for i, fn in enumerate(fns)}
        missing = [fn for fn in ref0 if fn not in cand_idx]
        extra = [fn for fn in fns if fn not in ref0_idx]
        if missing or extra:
            print(f"\n[probe] {cand_name}: filename set mismatch "
                  f"(missing {len(missing)}, extra {len(extra)}) — "
                  f"aligning on the intersection")
        aligned = [cand_idx[fn] for fn in ref0 if fn in cand_idx]
        keep = [i for i, fn in enumerate(ref0) if fn in cand_idx]
        cand_logits0 = logits[aligned]              # (n_keep, 12)
        y0_keep = y0[keep]
        # sanity: candidate's own labels must match the ref labels
        cand_labels_aligned = labels[aligned]
        assert np.array_equal(cand_labels_aligned, y0_keep), (
            f"{cand_name}: label mismatch after alignment")

        cand_pred = cand_logits0.argmax(-1)
        f1_cand = f1_score(y0_keep, cand_pred, average="macro")
        cand_err = (cand_pred != y0_keep).astype(np.float64)

        # 12-way = 11-way logits + candidate logits, logit-mean, on the
        # kept fold-0 samples.
        stack11 = np.stack([raw[m][fold0][keep] for m in members], axis=0)
        logit12 = (stack11.sum(axis=0) + cand_logits0) / 12.0
        pred12 = logit12.argmax(-1)
        f1_12 = f1_score(y0_keep, pred12, average="macro")
        f1_11_keep = f1_score(y0_keep, pred11_0[keep], average="macro")

        # Pairwise diversity vs each of the 11 members
        cosines, kappas = {}, {}
        for m in members:
            me = member_err0[m][keep]
            cosines[m] = _err_cosine(cand_err, me)
            # kappa on the PREDICTIONS (agreement beyond chance)
            kappas[m] = cohen_kappa_score(cand_pred, member_pred0[m][keep])
        max_cos_m = max(cosines, key=cosines.get)
        max_cos = cosines[max_cos_m]

        print(f"\n### {cand_name}")
        print(f"  standalone fold-0 F1   : {f1_cand*100:.2f}%  "
              f"(error rate {cand_err.mean()*100:.1f}%)")
        print(f"  11-way fold-0 F1       : {f1_11_keep*100:.2f}%")
        print(f"  12-way fold-0 F1 (+cand): {f1_12*100:.2f}%  "
              f"(delta {(f1_12-f1_11_keep)*100:+.2f} pp)")
        print(f"  max error-cosine vs 11 : {max_cos:.3f}  (vs {max_cos_m})")
        print(f"  mean error-cosine      : {np.mean(list(cosines.values())):.3f}")
        kv = np.array(list(kappas.values()))
        print(f"  pred-kappa vs 11       : "
              f"min {kv.min():.3f} / mean {kv.mean():.3f} / max {kv.max():.3f}")

    print("\n" + "=" * 68)
    print("[probe] NOTE: at ~76% error rate, two INDEPENDENT classifiers")
    print("  already have error-cosine ~0.76 by construction — the")
    print("  framework's <0.55 cosine threshold is not meaningful in this")
    print("  high-error regime. The 12-way-minus-11-way F1 delta and the")
    print("  prediction-kappa are the decision-relevant numbers.")


if __name__ == "__main__":
    main()
