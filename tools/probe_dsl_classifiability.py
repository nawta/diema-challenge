""" / exp062 — DSL feature linear-probe (3-fold LPO).

See: / diema/features/qwen_dsl.py
Related:
- tools/extract_rule_dsl.py (DSL extractor; produces input cache)
- tools/probe_rationale_text_classifiability_v2.py (free-form probe baseline)

Reads every DSL JSON under ``output/dsl_cache_rule/<rat_key>.json`` (or
``--dsl-dir`` override), one-hot encodes the 36 categorical attributes,
joins with the ``train_data.csv`` emotion labels via filename stem, and runs
a 3-fold LPO LogReg probe identical in protocol to v1's free-form probe.

Output: a markdown report at ``docs/analysis/exp062_dsl_probe.md`` with
the 3-fold mean F1 and per-fold detail. Reference:
  - random (1/12): 8.33%
  - v1 free-form mpnet+concat: 10.94%
  - exp062 GATE target (DSL ≥ free-form + 2pt): **12.94%**

Usage::

    python tools/probe_dsl_classifiability.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diema.features.qwen_dsl import (  # noqa: E402
    DSL_COLUMNS, GLOBAL_ATTR_VALUES, PART_ATTR_VALUES, PART_NAMES,
    flatten_dsl,
)

EMOTIONS = (
    "anger", "contempt", "disgust", "fear", "gratitude", "guilt",
    "jealousy", "joy", "pride", "sadness", "shame", "surprise",
)
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}


def emotion_from_stem(stem: str) -> int:
    parts = stem.split("_")
    if len(parts) < 5:
        return -1
    return EMOTION_TO_IDX.get(parts[2], -1)


def load_render_index(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path.is_file():
        return mapping
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ck = rec.get("cache_key")
            stem = rec.get("stem")
            if isinstance(ck, str) and isinstance(stem, str):
                mapping[ck] = stem
    return mapping


def load_dsl_cache(dsl_dir: Path, stem_lookup: dict[str, str]) -> tuple[list[str], list[dict[str, str]]]:
    """Return (stems, flat-dsl-dicts) for every (resolvable) clip."""
    stems: list[str] = []
    rows: list[dict[str, str]] = []
    for p in sorted(dsl_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rk = data.get("render_cache_key")
        if not rk:
            continue
        stem = stem_lookup.get(rk)
        if not stem:
            continue
        flat = flatten_dsl(data.get("dsl", {}))
        # Skip if any expected column is missing (shouldn't happen for rule extractor)
        if any(c not in flat for c in DSL_COLUMNS):
            continue
        stems.append(stem)
        rows.append(flat)
    return stems, rows


def one_hot_encode(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[str]]:
    """Build (N, P) float32 array; columns are stable across calls.

    Total feature count = sum(|values|) for each DSL column. With the schema
    in qwen_dsl.py: global 6 attrs (3+3+3+3+3+3 = 18) + 6 parts × 5 attrs
    (9+7+3+3+3 = 25 each, total 150) = **168 one-hot features**.
    """
    cols: list[tuple[str, str]] = []
    for col in DSL_COLUMNS:
        section, attr = col.split("__", 1)
        valspace = GLOBAL_ATTR_VALUES[attr] if section == "global" else PART_ATTR_VALUES[attr]
        for v in valspace:
            cols.append((col, v))
    feat_names = [f"{col}=={v}" for col, v in cols]
    P = len(cols)
    N = len(rows)
    X = np.zeros((N, P), dtype=np.float32)
    col_idx = {(col, v): i for i, (col, v) in enumerate(cols)}
    for i, row in enumerate(rows):
        for col, val in row.items():
            key = (col, val)
            if key in col_idx:
                X[i, col_idx[key]] = 1.0
    return X, feat_names


def fit_logreg(X_tr, y_tr, X_va, y_va, c_grid):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(with_mean=False)  # one-hot is already 0/1; with_mean=False keeps sparsity
    X_tr_z = scaler.fit_transform(X_tr)
    X_va_z = scaler.transform(X_va)

    per_c = []
    for c in c_grid:
        clf = LogisticRegression(C=c, max_iter=2000, solver="lbfgs", n_jobs=-1)
        clf.fit(X_tr_z, y_tr)
        preds = clf.predict(X_va_z)
        per_c.append({
            "C": c,
            "macro_f1": f1_score(y_va, preds, average="macro"),
            "acc": accuracy_score(y_va, preds),
        })
    best = max(per_c, key=lambda r: r["macro_f1"])
    return best, per_c


@click.command()
@click.option("--dsl-dir", default="output/dsl_cache_rule", show_default=True)
@click.option("--render-index", default="output/rationale_cache/renders/render_index.jsonl",
              show_default=True)
@click.option("--data-path", default="data/diema_challenge/processed/motion_train_quat.npz",
              show_default=True)
@click.option("--num-folds", default=3, type=int, show_default=True)
@click.option("--c-grid", "c_grid", default="0.1,1.0,10.0", show_default=True)
@click.option("--output-md", default="docs/analysis/exp062_dsl_probe.md",
              show_default=True)
def main(
    dsl_dir: str, render_index: str, data_path: str, num_folds: int,
    c_grid: str, output_md: str,
) -> None:
    import pybvh_ml
    from diema.data.splits import generate_lpo_splits

    cs = [float(x) for x in c_grid.split(",")]
    dsl_path = Path(dsl_dir)
    stem_lookup = load_render_index(Path(render_index))
    stems, rows = load_dsl_cache(dsl_path, stem_lookup)
    click.echo(f"Loaded {len(rows)} DSL rows from {dsl_path}")

    y = np.array([emotion_from_stem(s) for s in stems], dtype=np.int64)
    valid_mask = y >= 0
    stems = [s for s, v in zip(stems, valid_mask) if v]
    rows = [r for r, v in zip(rows, valid_mask) if v]
    y = y[valid_mask]
    click.echo(f"Valid (stem, label): {len(stems)}")

    X, feat_names = one_hot_encode(rows)
    click.echo(f"Feature dim: {X.shape[1]} ({len(DSL_COLUMNS)} cols × avg ~{X.shape[1]/len(DSL_COLUMNS):.1f} values)")

    preprocessed = pybvh_ml.load_preprocessed(data_path)
    filenames = preprocessed["filenames"]
    splits = generate_lpo_splits(filenames, num_folds=num_folds)

    def _stem_of(entry):
        if isinstance(entry, tuple):
            entry = entry[0]
        return Path(str(entry)).stem

    stem_to_row = {s: i for i, s in enumerate(stems)}

    fold_results: list[dict] = []
    for fold_id in range(num_folds):
        split = splits[fold_id]
        train_stems = [_stem_of(e) for e in split["train"]]
        val_stems = [_stem_of(e) for e in split["val"]]
        train_idx = np.array([stem_to_row[s] for s in train_stems if s in stem_to_row])
        val_idx = np.array([stem_to_row[s] for s in val_stems if s in stem_to_row])
        X_tr = X[train_idx]; y_tr = y[train_idx]
        X_va = X[val_idx];   y_va = y[val_idx]
        best, per_c = fit_logreg(X_tr, y_tr, X_va, y_va, cs)
        fold_results.append({
            "fold": fold_id, "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
            "best_C": best["C"], "macro_f1": best["macro_f1"], "acc": best["acc"], "per_C": per_c,
        })
        click.echo(
            f"  fold {fold_id}: F1={best['macro_f1']:.4f} acc={best['acc']:.4f} (best C={best['C']})"
        )

    f1s = [r["macro_f1"] for r in fold_results]
    accs = [r["acc"] for r in fold_results]
    f1_mean = float(np.mean(f1s))
    f1_std = float(np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0)
    acc_mean = float(np.mean(accs))
    acc_std = float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)

    if f1_mean < 0.13:
        verdict = "STRONG NO-GO (F1 < 13%) — DSL representation hits the same ceiling as free-form text"
    elif f1_mean < 0.18:
        verdict = "WEAK signal (13% ≤ F1 < 18%) — DSL beats free-form by some pp; check if ≥ +2pt gate met"
    elif f1_mean < 0.25:
        verdict = "MODERATE signal (18% ≤ F1 < 25%) — DSL appears to extract latent class signal"
    else:
        verdict = "STRONG signal (F1 ≥ 25%) — re-evaluate direction; DSL is meaningfully informative"

    click.echo(f"\n=== {verdict} ===")
    click.echo(f"3-fold mean macro-F1: {f1_mean*100:.2f}% ± {f1_std*100:.2f}%")
    click.echo(f"3-fold mean acc:      {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")

    md = [
        "# exp062 DSL classifiability linear probe",
        "",
        f"- DSL dir: `{dsl_dir}`",
        f"- num_folds: {num_folds}",
        f"- valid rows: {len(stems)}",
        f"- feature dim (one-hot): {X.shape[1]}",
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
        "- Random (1/12): 8.33%",
        "- v1 free-form mpnet+concat: 10.94%",
        "- exp062 GATE (DSL ≥ free-form + 2pt): **≥ 12.94%**",
        "- v2 ranker probe (mini-run, exp061 verdict): 9.19%",
        "- Backbone-only motion model (exp034 a00): 30.30%",
    ])
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(md), encoding="utf-8")
    click.echo(f"Saved: {output_md}")


if __name__ == "__main__":
    main()
