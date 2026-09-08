"""probe_3fold.py — 3-fold ensemble-contribution probe.

The fold-0-only error_correlation_probe used the weakest jointpos member
(fold 0 = 22.4%, the unlucky fold). The multifold runs revealed folds 1-2
are ~3 pp stronger. This script extends the "does the member lift the
ensemble" test to folds 0/1/2 for both mix=0.0 and mix=0.4, giving a
3-fold mean 12-way-minus-11-way F1 delta — the decision-relevant number
for whether the 10-fold run is worth ~30 hr.

Reuses paper/reconcile_f1.py's 11-member OOF loader.
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
MEMBERS = PROD + EXTRA4

# Per-fold smoke run dirs for each candidate config.
RUNS = {
    "mix0.4_retarget": {
        0: "jointpos_honest_fold_00_e65_upidx2",
        1: "jointpos_honest_mix04_fold_01_e65",
        2: "jointpos_honest_mix04_fold_02_e65",
    },
    "mix0.0_noretarget": {
        0: "jointpos_honest_mix00_fold_00_e65",
        1: "jointpos_honest_mix00_fold_01_e65",
        2: "jointpos_honest_mix00_fold_02_e65",
    },
}


def _ref_filenames() -> list[str]:
    with open(SPLITS_PKL, "rb") as f:
        splits = pickle.load(f)
    return [str(fn) for fid in range(10) for fn, _ in splits[fid]["val"]]


def _load_run_oof(run_dir: Path):
    logits = np.load(run_dir / "oof_logits.npy").astype(np.float64)
    fns = [f for f in (run_dir / "oof_filenames.txt").read_text().splitlines() if f]
    labels = np.load(run_dir / "oof_labels.npy").astype(np.int64)
    assert len(fns) == logits.shape[0] == labels.shape[0]
    return logits, fns, labels


def main():
    print("[probe3] loading 11-member production OOF ...")
    raw, y, fold_ids = load_oof()
    ref = _ref_filenames()
    assert len(ref) == len(y) == 7992

    for cfg_name, fold_dirs in RUNS.items():
        print(f"\n{'='*66}\n### {cfg_name}\n{'='*66}")
        deltas, f11s, f12s, standalone, kappas = [], [], [], [], []
        for fold in (0, 1, 2):
            run_dir = ART / fold_dirs[fold]
            if not run_dir.exists():
                print(f"  fold {fold}: SKIP ({run_dir} missing)")
                continue
            logits, fns, labels = _load_run_oof(run_dir)
            mask = fold_ids == fold
            ref_f = [ref[i] for i in range(len(ref)) if mask[i]]
            cand_idx = {fn: i for i, fn in enumerate(fns)}
            keep = [i for i, fn in enumerate(ref_f) if fn in cand_idx]
            aligned = [cand_idx[ref_f[i]] for i in keep]
            global_keep = np.where(mask)[0][keep]

            cand_logits = logits[aligned]
            yk = y[global_keep]
            assert np.array_equal(labels[aligned], yk), "label mismatch"

            # 11-way and 12-way on the kept samples of this fold
            pred11 = combine(raw, MEMBERS, "logit_mean")[global_keep]
            stack11 = np.stack([raw[m][global_keep] for m in MEMBERS], axis=0)
            pred12 = ((stack11.sum(0) + cand_logits) / 12.0).argmax(-1)
            cand_pred = cand_logits.argmax(-1)

            f11 = f1_score(yk, pred11, average="macro")
            f12 = f1_score(yk, pred12, average="macro")
            fc = f1_score(yk, cand_pred, average="macro")
            kap = cohen_kappa_score(cand_pred, pred11)
            deltas.append(f12 - f11)
            f11s.append(f11); f12s.append(f12)
            standalone.append(fc); kappas.append(kap)
            print(f"  fold {fold}: standalone {fc*100:5.2f}%  "
                  f"11-way {f11*100:5.2f}%  12-way {f12*100:5.2f}%  "
                  f"delta {(f12-f11)*100:+5.2f}pp  kappa(vs 11-way) {kap:.3f}")

        if deltas:
            d = np.array(deltas)
            print(f"  ----")
            print(f"  3-fold mean: standalone {np.mean(standalone)*100:.2f}%  "
                  f"11-way {np.mean(f11s)*100:.2f}%  12-way {np.mean(f12s)*100:.2f}%")
            print(f"  3-fold mean 12-way delta: {d.mean()*100:+.2f}pp  "
                  f"(per-fold {[round(x*100,2) for x in d]})")
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan')
            print(f"  delta std {d.std(ddof=1)*100:.2f}pp, "
                  f"SE {se*100:.2f}pp → {'spans 0' if abs(d.mean())<2*se else 'clear'}")


if __name__ == "__main__":
    main()
