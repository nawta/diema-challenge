"""paper/tables/build_table1.py — Table 1 main results (P9).

Design doc: paper/data_inventory.md §Reconciliation (P0).

CANONICAL convention = logit-mean × 10-fold LPO per-fold mean ± SD (NOT
the 33.72 % softmax-mean internal baseline).  Every ensemble row is
recomputed from the SAME OOF artifacts via reconcile_f1 (identical to
Fig.2 / P3) so the table is internally consistent.  Row 1 (STGCN++
official) is the challenge-reported number (92-perf full LPO), marked
★: not re-evaluated with our script, so no sample bootstrap CI.

95 % CI = sample-level paired bootstrap, 1000 iter, seed 42 (OOF rows).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # paper/
sys.path.insert(0, str(Path(__file__).resolve().parent))          # _tbl

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

import reconcile_f1 as R  # noqa: E402
from _tbl import emit  # noqa: E402

CANON_7WAY_PERFOLD = 33.86          # data_inventory §Reconciliation
CANON_7WAY_POOLED = 34.03

raw, y, fid = R.load_oof()


def _stats(pred, yy, ff):
    pf = [f1_score(yy[ff == k], pred[ff == k], average="macro")
          for k in range(10)]
    pa = [accuracy_score(yy[ff == k], pred[ff == k]) for k in range(10)]
    rng = np.random.default_rng(42)
    N = len(yy)
    bs = np.empty(1000)
    for b in range(1000):
        idx = rng.integers(0, N, N)
        bs[b] = f1_score(yy[idx], pred[idx], average="macro")
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return (100 * np.mean(pf), 100 * np.std(pf, ddof=1),
            100 * np.mean(pa), 100 * np.std(pa, ddof=1),
            100 * lo, 100 * hi, 100 * f1_score(yy, pred, average="macro"))


def _exp001():
    d = R.ART / "exp001_benchmark_repro"
    P, Y, F = [], [], []
    for k in range(10):
        L = np.load(d / f"fold_{k:02d}/oof_logits.npy")
        yy = np.load(d / f"fold_{k:02d}/oof_labels.npy")
        P.append(L.argmax(-1))
        Y.append(yy)
        F.append(np.full(len(yy), k))
    return _stats(np.concatenate(P), np.concatenate(Y), np.concatenate(F))


def _nparams(member):
    ck = torch.load(R.ART / f"{member}/fold_00/best.ckpt",
                    map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck \
        else ck
    return sum(v.numel() for v in sd.values() if hasattr(v, "numel"))


sum7 = sum(_nparams(m) for m in R.PROD) / 1e6
p001 = _nparams("exp001_benchmark_repro") / 1e6
p034 = _nparams("exp034_regionaware_convtr_a00") / 1e6

S = {
    "best": _stats(R.combine(raw, ["exp034_regionaware_convtr_a00"],
                             "logit_mean"), y, fid),
    "e7": _stats(R.combine(raw, R.PROD, "logit_mean"), y, fid),
    "e11l": _stats(R.combine(raw, R.PROD + R.EXTRA4, "logit_mean"), y, fid),
}
# 2026-05-20 (post-E1.3, a review): the 9-way / 11-way-softmax
# intermediate computations were retained as commented "dead code" after the
# E1.3 row-drop; deleted entirely since they were never consumed by row()
# nor written to any sidecar — eliminates re-enable foot-gun (guide §5).
b001 = _exp001()

# P9 constraint: fail loudly if the canonical reconciliation is violated
e7_pf, _, _, _, _, _, e7_pool = S["e7"]
assert abs(e7_pf - CANON_7WAY_PERFOLD) < 0.1 and \
    abs(e7_pool - CANON_7WAY_POOLED) < 0.1, (
        f"CANONICAL MISMATCH: 7-way per-fold={e7_pf:.2f} "
        f"(expect {CANON_7WAY_PERFOLD}), pooled={e7_pool:.2f} "
        f"(expect {CANON_7WAY_POOLED}) — P0 reconciliation violated")


def row(sysname, pm, st, official=False):
    if official:                                  # STGCN++ official ★
        return [sysname, f"{pm:.2f}★",
                "25.21 ± 4.49★", "n/a★", "27.11 ± 3.67★"]
    f1m, f1s, am, asd, lo, hi, _ = st
    return [sysname, f"{pm:.2f}",
            f"{f1m:.2f} ± {f1s:.2f}", f"[{lo:.2f}, {hi:.2f}]",
            f"{am:.2f} ± {asd:.2f}"]

rows = [
    row("STGCN++ official baseline", 1.40, None, official=True),
    row("STGCN++ reproduced", p001, b001),
    row("Best single (Region-Aware)", p034, S["best"]),
    row("7-way ensemble", sum7, S["e7"]),
    row("11-way logit-mean (submitted)", sum7, S["e11l"]),
]
# 2026-05-20 (E1.3, review consensus + post-E1.3 cleanup):
# the 9-way (+MotionBERT +C3D) and 11-way softmax-mean intermediate rows are
# dropped from the main table. The +1.15 pp logit-vs-softmax ablation is
# reported in §4.2 prose only. The external-pretraining contribution is
# carried by Fig.4 orthogonality. The unused computations were deleted from
# S above (no sidecar consumer) to eliminate the re-enable foot-gun.

notes = [
    "★ Challenge-reported (92-performer full LPO, official 25.2 % ± 4.5 %, "
    "no bootstrap CI), NOT re-evaluated with our script; our reproduction "
    "(25.73 ± 4.03) is within SD of the official number, so the official "
    "baseline serves only as an external anchor and all improvement "
    "numbers are same-split comparisons.",
    "‡ The table reports trainable parameters; frozen external branches "
    "add fewer than 0.01 M trainable parameters each.",
]
# 2026-05-20 (E1.3): the previous third footnote stating the 33.72 % softmax-mean
# value and the internal reconciliation path was removed — guide §5
# (softmax-mean appears NOWHERE, not even a footnote) + §11 (no internal-doc
# provenance). The +1.15 pp logit-vs-softmax ablation lives in §4.2 prose only.

out = emit("table1_main_results",
           ["System", "Trainable params (M)‡", "Macro-F1 (mean ± SD)",
            "Macro-F1 95% CI", "Accuracy (mean ± SD)"],
           rows, colspec="l r c c c", bold_last=True, notes=notes,
           caption="Main results on DIEM-A (10-fold LPO, 74-performer train "
                   "split). Final row = submitted model. Macro-F1 / Accuracy "
                   "= 10-fold LPO per-fold mean ± SD (the official "
                   "convention); fusion = logit-mean (mean of raw logits); "
                   "95% CI = sample-level paired bootstrap, 1000 iter, seed "
                   "42, on pooled OOF. Pooled-OOF F1 is a descriptive "
                   "secondary: 7-way 34.03%, 11-way logit-mean 36.94%.")

print("Table 1 written:", out)
for r in rows:
    print(" |", " | ".join(str(c) for c in r))
print(f"\ncanonical check PASS: 7-way per-fold {e7_pf:.2f} / pooled "
      f"{e7_pool:.2f}  (Σ7 params = {sum7:.2f} M)")
