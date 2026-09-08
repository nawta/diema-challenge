""" / exp065 — 8-attribute scenario probe (3-fold LPO).

See: / diema/features/scenario_attrs.py
Related:
- tools/probe_scenario_text_classifiability.py (raw scenario text upper bound)

Reads each clip's masked scenario text from train_data.csv, extracts the 8
categorical attributes via the rule-based extractor, one-hot encodes (24
dim), and runs a 3-fold LPO LogReg probe.

Reference points
----------------
- Random (1/12): 8.33%
- Raw scenario mpnet probe (`docs/analysis/exp065_scenario_probe.md`):
  53.09% — the upper bound for any scenario-derived feature
- This probe: tells us how much of the 53% survives the 8-attribute
  compression. If F1 ≥ ~30%, the auxiliary head has substantial gradient
  signal to give the motion backbone (gate: +0.20pt downstream OR qualitative).

Usage::

    python tools/probe_scenario_attrs_classifiability.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diema.features.scenario_attrs import (  # noqa: E402
    ATTR_ORDER, ATTR_VALUES, ONEHOT_COLUMNS, ONEHOT_FEATURE_NAMES,
    extract_attrs,
)
from tools.probe_scenario_text_classifiability import (  # noqa: E402
    EMOTION_TO_IDX, emotion_from_stem, load_clip_to_scenario,
)


def one_hot_encode(rows: list[dict[str, str]]) -> np.ndarray:
    P = len(ONEHOT_COLUMNS)
    N = len(rows)
    X = np.zeros((N, P), dtype=np.float32)
    col_idx = {col: i for i, col in enumerate(ONEHOT_COLUMNS)}
    for i, row in enumerate(rows):
        for attr, val in row.items():
            key = (attr, val)
            if key in col_idx:
                X[i, col_idx[key]] = 1.0
    return X


@click.command()
@click.option("--train-csv", default="data/diema_challenge/raw/train_data.csv",
              show_default=True)
@click.option("--data-path", default="data/diema_challenge/processed/motion_train_quat.npz",
              show_default=True)
@click.option("--num-folds", default=3, type=int, show_default=True)
@click.option("--mask-emotion/--no-mask-emotion", default=True, show_default=True)
@click.option("--forbidden", default="configs/leakage/forbidden_tokens.yaml",
              show_default=True)
@click.option("--c-grid", "c_grid", default="0.1,1.0,10.0", show_default=True)
@click.option("--output-md", default="docs/analysis/exp065_scenario_attrs_probe.md",
              show_default=True)
@click.option("--dump-attrs", default="output/scenario_attrs_rule.json",
              show_default=True,
              help="JSON dump of the {stem: attrs} dict for downstream aux-head training.")
def main(
    train_csv: str, data_path: str, num_folds: int,
    mask_emotion: bool, forbidden: str,
    c_grid: str, output_md: str, dump_attrs: str,
) -> None:
    import json
    import pybvh_ml
    from diema.data.splits import generate_lpo_splits
    from collections import Counter

    cs = [float(x) for x in c_grid.split(",")]
    stem_to_scenario = load_clip_to_scenario(
        Path(train_csv), mask_emotion=mask_emotion, forbidden_yaml=Path(forbidden),
    )
    click.echo(f"Loaded {len(stem_to_scenario)} clip→scenario mappings (mask_emotion={mask_emotion})")

    preprocessed = pybvh_ml.load_preprocessed(data_path)
    filenames = preprocessed["filenames"]
    splits = generate_lpo_splits(filenames, num_folds=num_folds)

    def _stem_of(entry):
        if isinstance(entry, tuple):
            entry = entry[0]
        return Path(str(entry)).stem

    all_stems_in_splits: list[str] = []
    seen = set()
    for s in splits:
        for entry in s["train"]:
            ss = _stem_of(entry)
            if ss not in seen and ss in stem_to_scenario:
                seen.add(ss); all_stems_in_splits.append(ss)
        for entry in s["val"]:
            ss = _stem_of(entry)
            if ss not in seen and ss in stem_to_scenario:
                seen.add(ss); all_stems_in_splits.append(ss)
    stems = all_stems_in_splits
    y = np.array([emotion_from_stem(s) for s in stems], dtype=np.int64)
    valid = y >= 0
    stems = [s for s, v in zip(stems, valid) if v]
    y = y[valid]
    click.echo(f"Valid (stem, label): {len(stems)}")

    rows: list[dict[str, str]] = []
    distrib: dict[str, Counter] = {a: Counter() for a in ATTR_ORDER}
    stem_to_attrs: dict[str, dict[str, str]] = {}
    for s in stems:
        scenario = stem_to_scenario[s]
        attrs = extract_attrs(scenario)
        rows.append(attrs)
        stem_to_attrs[s] = attrs
        for a, v in attrs.items():
            distrib[a][v] += 1

    if dump_attrs:
        Path(dump_attrs).parent.mkdir(parents=True, exist_ok=True)
        Path(dump_attrs).write_text(
            json.dumps(stem_to_attrs, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        click.echo(f"Dumped attrs JSON: {dump_attrs}")

    X = one_hot_encode(rows)
    click.echo(f"Feature shape: {X.shape}")
    click.echo("Per-attr distribution:")
    for a in ATTR_ORDER:
        c = distrib[a]
        n = sum(c.values())
        dist = ", ".join(f"{v}={c.get(v, 0)/n*100:.0f}%" for v in ATTR_VALUES[a])
        click.echo(f"  {a:20s}: {dist}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.preprocessing import StandardScaler

    stem_to_row = {s: i for i, s in enumerate(stems)}

    fold_results: list[dict] = []
    for fold_id in range(num_folds):
        split = splits[fold_id]
        train_stems = [_stem_of(e) for e in split["train"]]
        val_stems = [_stem_of(e) for e in split["val"]]
        train_idx = np.array([stem_to_row[s] for s in train_stems if s in stem_to_row])
        val_idx = np.array([stem_to_row[s] for s in val_stems if s in stem_to_row])

        scaler = StandardScaler(with_mean=False)
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        y_tr = y[train_idx]; y_va = y[val_idx]

        per_c = []
        for c in cs:
            clf = LogisticRegression(C=c, max_iter=2000, solver="lbfgs", n_jobs=-1)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_va)
            per_c.append({
                "C": c, "macro_f1": f1_score(y_va, preds, average="macro"),
                "acc": accuracy_score(y_va, preds),
            })
        best = max(per_c, key=lambda r: r["macro_f1"])
        fold_results.append({
            "fold": fold_id, "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
            "best_C": best["C"], "macro_f1": best["macro_f1"], "acc": best["acc"],
        })
        click.echo(
            f"  fold {fold_id}: F1={best['macro_f1']:.4f} acc={best['acc']:.4f}"
        )

    f1s = [r["macro_f1"] for r in fold_results]
    accs = [r["acc"] for r in fold_results]
    f1_mean = float(np.mean(f1s))
    f1_std = float(np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0)
    acc_mean = float(np.mean(accs))
    acc_std = float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)

    if f1_mean < 0.13:
        verdict = "STRONG NO-GO (F1 < 13%) — 8-attribute extraction lost the scenario signal"
    elif f1_mean < 0.25:
        verdict = "MODERATE — 8-attribute compression captures only weak class signal"
    elif f1_mean < 0.40:
        verdict = "STRONG signal — substantial portion of scenario-text 53% survived discretization"
    else:
        verdict = "VERY STRONG signal — 8-attribute representation preserves most of the scenario information"

    click.echo(f"\n=== {verdict} ===")
    click.echo(f"3-fold mean macro-F1: {f1_mean*100:.2f}% ± {f1_std*100:.2f}%")
    click.echo(f"3-fold mean acc:      {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")

    md = [
        "# exp065 — 8-attribute scenario probe (rule-based extraction, 3-fold LPO)",
        "",
        f"- mask_emotion: **{mask_emotion}**",
        f"- num_folds: {num_folds}",
        f"- valid rows: {len(stems)}",
        f"- feature dim (one-hot): {X.shape[1]} (= 8 attrs × 3 values)",
        "",
        "## Per-attribute distribution",
        "",
        "| attribute | distribution |",
        "|---|---|",
    ]
    for a in ATTR_ORDER:
        c = distrib[a]
        n = sum(c.values())
        dist = ", ".join(f"{v}: {c.get(v, 0)} ({c.get(v, 0)/n*100:.1f}%)" for v in ATTR_VALUES[a])
        md.append(f"| `{a}` | {dist} |")
    md.extend([
        "",
        "## 3-fold result",
        "",
        f"- macro-F1: **{f1_mean*100:.2f}% ± {f1_std*100:.2f}%**",
        f"- acc:      **{acc_mean*100:.2f}% ± {acc_std*100:.2f}%**",
        "",
        "### Verdict",
        "",
        f"**{verdict}**",
        "",
        "## Per-fold detail",
        "",
        "| fold | n_train | n_val | best C | macro-F1 | acc |",
        "|---|---:|---:|---:|---:|---:|",
    ])
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
        "- Rationale free-form mpnet probe: 10.94% (Tier 0 baseline)",
        "- v2 ranker mpnet probe: 9.19% (exp061 verdict)",
        "- Rule-based DSL probe: 10.91% (exp062 verdict)",
        "- **Raw scenario mpnet probe: 53.09%** (`docs/analysis/exp065_scenario_probe.md`)",
        "- Backbone-only motion model: 30.30%",
        "",
        "## Implications for exp065 auxiliary head",
        "",
        "- **F1 ≥ 30%**: 24-dim attributes preserve most scenario signal.",
        "  Aux head training likely gives motion backbone meaningful gradient.",
        "- **F1 ∈ [14%, 30%]**: partial signal preservation.",
        "  Aux head may help but no guarantee of +0.2pp F1 lift.",
        "- **F1 < 14%**: extraction lost the signal — stick with raw scenario",
        "  embedding (exp051 was -1.42pt; this is a known failure mode).",
    ])
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(md), encoding="utf-8")
    click.echo(f"Saved: {output_md}")


if __name__ == "__main__":
    main()
