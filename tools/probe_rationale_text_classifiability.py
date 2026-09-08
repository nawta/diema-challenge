""" (Tier 0): linear probe on rationale text embeddings → 12-class.

See: / docs/cards/audit_sampling_protocol_3AD-7_v1.0.md
Related:
- diema/features/rationale_text_cache.py (cache loader)
- diema/data/splits.py (LPO 3-fold splits — same as exp052)

Diagnostic question: does the Qwen3-VL rationale text contain enough
emotion-discriminative signal for a linear classifier to predict the 12
emotion classes from the 6-part mpnet embeddings?

If yes (F1 high) → text has signal, current alignment failure is geometry-
classifier mismatch, Tier 1 ablations may help.
If no (F1 low) → text bottleneck, only path is VLM logit KD (Tier 2).

Usage::

    python tools/probe_rationale_text_classifiability.py \\
        --rationale-cache output/rationale_text_cache_mpnet.pt \\
        --num-folds 3 \\
        --aggregation concat
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Same canonical 12 emotion order as DIEM-A.
EMOTIONS = (
    "anger", "contempt", "disgust", "fear", "gratitude", "guilt",
    "jealousy", "joy", "pride", "sadness", "shame", "surprise",
)
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}


def emotion_from_stem(stem: str) -> int:
    """Stems look like ``JP_06_anger_1_H``. Return 12-class label or -1."""
    parts = stem.split("_")
    if len(parts) < 5:
        return -1
    return EMOTION_TO_IDX.get(parts[2], -1)


def aggregate_part_embeddings(
    embeddings: torch.Tensor, part_mask: torch.Tensor, mode: str,
) -> torch.Tensor:
    """Reduce (N, P, D) to (N, D_out).

    - "mean": placeholder-aware mean → (N, D)
    - "concat": raw concat over parts → (N, P*D), zero-fill placeholder cells
    """
    if mode == "mean":
        m = part_mask.to(torch.float32).unsqueeze(-1)        # (N, P, 1)
        denom = m.sum(dim=1).clamp_min(1e-9)                  # (N, 1)
        return (embeddings * m).sum(dim=1) / denom            # (N, D)
    if mode == "concat":
        m = part_mask.to(torch.float32).unsqueeze(-1)
        masked = embeddings * m  # zero-fill placeholders
        N, P, D = masked.shape
        return masked.reshape(N, P * D)
    raise ValueError(f"unknown mode {mode!r}; expected mean|concat")


@click.command()
@click.option(
    "--rationale-cache", default="output/rationale_text_cache_mpnet.pt",
    show_default=True,
)
@click.option(
    "--data-path", default="data/diema_challenge/processed/motion_train_quat.npz",
    show_default=True, help="Used only to derive LPO fold splits (same as exp052).",
)
@click.option("--num-folds", default=3, type=int, show_default=True)
@click.option(
    "--aggregation", type=click.Choice(["mean", "concat"]), default="concat",
    show_default=True,
    help="How to reduce 6 per-part embeddings: mean (D-dim) or concat (6*D-dim).",
)
@click.option("--C-grid", "c_grid", default="0.1,1.0,10.0", show_default=True)
@click.option(
    "--output-md", default="docs/analysis/rationale_linear_probe_3fold.md",
    show_default=True,
)
def main(
    rationale_cache: str, data_path: str, num_folds: int,
    aggregation: str, c_grid: str, output_md: str,
) -> None:
    import pybvh_ml
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.preprocessing import StandardScaler

    from diema.data.splits import generate_lpo_splits
    from diema.features.rationale_text_cache import RationaleTextCache

    cache = RationaleTextCache(rationale_cache)
    click.echo(
        f"Loaded cache: N={cache.num_rows}, P={cache.num_parts}, D={cache.embedding_dim}"
    )

    X = aggregate_part_embeddings(
        cache._embeddings, cache._part_mask, mode=aggregation,
    ).numpy().astype(np.float32)

    # Stems from cache
    stems = list(cache._stem_to_idx.keys())
    y = np.array([emotion_from_stem(s) for s in stems], dtype=np.int64)
    valid = y >= 0
    X = X[valid]
    y = y[valid]
    stems = [s for s, v in zip(stems, valid) if v]
    click.echo(f"Valid (stem, label) rows: {len(stems)}")

    # Build LPO fold splits — same logic as exp052 trainer.
    preprocessed = pybvh_ml.load_preprocessed(data_path)
    filenames = preprocessed["filenames"]
    splits = generate_lpo_splits(filenames, num_folds=num_folds)

    stem_to_row = {s: i for i, s in enumerate(stems)}

    cs = [float(x) for x in c_grid.split(",")]
    fold_results: list[dict] = []
    def _stem_of(entry) -> str:
        # Splits return (filename_str, idx) tuples or sometimes plain strings.
        if isinstance(entry, tuple):
            entry = entry[0]
        return Path(str(entry)).stem

    for fold_id in range(num_folds):
        split = splits[fold_id]
        train_stems = [_stem_of(e) for e in split["train"]]
        val_stems = [_stem_of(e) for e in split["val"]]
        train_idx = np.array([stem_to_row[s] for s in train_stems if s in stem_to_row])
        val_idx = np.array([stem_to_row[s] for s in val_stems if s in stem_to_row])

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_z = scaler.fit_transform(X_tr)
        X_va_z = scaler.transform(X_va)

        # Sweep C and pick best on val.
        per_c: list[dict] = []
        for c in cs:
            clf = LogisticRegression(
                C=c, max_iter=2000, multi_class="multinomial",
                solver="lbfgs", n_jobs=-1,
            )
            clf.fit(X_tr_z, y_tr)
            preds = clf.predict(X_va_z)
            f1 = f1_score(y_va, preds, average="macro")
            acc = accuracy_score(y_va, preds)
            per_c.append({"C": c, "macro_f1": f1, "acc": acc})

        best = max(per_c, key=lambda r: r["macro_f1"])
        fold_results.append({
            "fold": fold_id,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "best_C": best["C"],
            "macro_f1": best["macro_f1"],
            "acc": best["acc"],
            "per_C": per_c,
        })
        click.echo(
            f"  fold {fold_id}: n_train={len(train_idx)}, n_val={len(val_idx)}, "
            f"best C={best['C']}, F1={best['macro_f1']:.4f}, acc={best['acc']:.4f}"
        )

    f1s = [r["macro_f1"] for r in fold_results]
    accs = [r["acc"] for r in fold_results]
    f1_mean, f1_std = float(np.mean(f1s)), float(np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0)
    acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)

    # Verdict
    if f1_mean < 0.25:
        verdict = "TEXT BOTTLENECK confirmed (F1 < 25%) — root cause 2; pivot to Tier 2 VLM logit KD"
    elif f1_mean > 0.28:
        verdict = "TEXT HAS SIGNAL (F1 > 28%) — root cause 1 likely (geometry-classifier mismatch); Tier 1 ablations may help"
    else:
        verdict = f"GREY ZONE (F1 = {f1_mean:.4f}) — try Tier 1 + Tier 2 in parallel"

    click.echo("")
    click.echo(f"3-fold mean macro-F1: {f1_mean:.4f} ± {f1_std:.4f}")
    click.echo(f"3-fold mean acc:      {acc_mean:.4f} ± {acc_std:.4f}")
    click.echo(f"Verdict: {verdict}")

    out = Path(output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# Rationale text linear probe — (Tier 0)",
        "",
        f"- cache: `{rationale_cache}`",
        f"- aggregation: **{aggregation}** ({'D-dim mean' if aggregation == 'mean' else f'{cache.num_parts}*D concat'})",
        f"- num_folds: {num_folds}",
        f"- C grid: {cs}",
        f"- valid rows (stem ↔ emotion label): {len(stems)} of {cache.num_rows}",
        "",
        "## 3-fold result",
        "",
        f"- macro-F1: **{f1_mean*100:.2f}% ± {f1_std*100:.2f}%**",
        f"- acc:      **{acc_mean*100:.2f}% ± {acc_std*100:.2f}%**",
        "",
        f"### Verdict",
        "",
        f"**{verdict}**",
        "",
        "## Per-fold detail",
        "",
        "| fold | n_train | n_val | best C | macro-F1 | acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in fold_results:
        md.append(
            f"| {r['fold']} | {r['n_train']} | {r['n_val']} | {r['best_C']} | "
            f"{r['macro_f1']*100:.2f}% | {r['acc']*100:.2f}% |"
        )
    md.extend([
        "",
        "## Reference points",
        "",
        "- Random baseline (1/12 classes, balanced): F1 ≈ 8.3%",
        "- Backbone-only baseline (exp034 a00, motion only): 30.30% ± 3.23% (3-fold)",
        "- exp052 v4 (motion + part rationale alignment): 29.66% ± 2.43% (3-fold)",
        "",
        "If text-only F1 is comparable to motion-only F1, then the rationale carries",
        "discriminative signal — and the alignment failure is mechanistic, not data-",
        "limited. If text-only F1 is much lower than motion-only F1, the text is the",
        "bottleneck and Tier 2 (VLM logit KD) becomes mandatory.",
    ])
    out.write_text("\n".join(md), encoding="utf-8")
    click.echo(f"Saved: {out}")


if __name__ == "__main__":
    main()
