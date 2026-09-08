""" / exp065 prep — scenario-text linear probe (3-fold LPO).

See: / docs/cards/scenario_data_provenance.md
Related:
- tools/build_scenario_cache.py (produces scenario_text_cache_masked.pt)
- tools/probe_rationale_text_classifiability_v2.py (rationale equivalent)

Quick upper-bound diagnostic for the **scenario** channel of
how much class signal does the scenario text alone carry, before we invest
in 8-attribute extraction or auxiliary-head training?

Workflow
--------
1. Map each clip stem → (its masked scenario text via train_data.csv).
2. Encode with `sentence-transformers/all-mpnet-base-v2` (matches the
   rationale baseline encoder exactly so the F1 comparison is honest).
3. 3-fold LPO LogReg.

Reference points
----------------
- Random (1/12): 8.33%
- v1 rationale free-form mpnet+concat: 10.94%
- exp065 GATE for auxiliary attribute head: F1 +0.20pt downstream OR qualitative
- This probe: tells us **what scenario alone could give to an aux head** —
  if scenario probe is 30%+, the aux head has real headroom; if <14%, the
  scenario channel is just as bottlenecked as rationale.

Usage::

    python tools/probe_scenario_text_classifiability.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import click
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def mask_emotion_tokens(text: str, emotion_aliases: list[str]) -> str:
    """Apply the same emotion-token masking as build_scenario_cache.py.

    Each alias is a whole-word case-insensitive replace with ``[MASK]``.
    """
    import re
    if not emotion_aliases:
        return text
    uniq = sorted({a.strip() for a in emotion_aliases if a.strip()}, key=len, reverse=True)
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(a) for a in uniq) + r")\b", re.IGNORECASE)
    return pattern.sub("[MASK]", text)


def load_clip_to_scenario(
    csv_path: Path, mask_emotion: bool, forbidden_yaml: Path,
) -> dict[str, str]:
    """Read train_data.csv → {filename_stem: masked_scenario_text}."""
    import yaml
    aliases: list[str] = []
    if mask_emotion:
        try:
            with forbidden_yaml.open("r", encoding="utf-8") as f:
                tokens = yaml.safe_load(f)
            aliases = list(tokens.get("emotion_aliases", []))
        except (FileNotFoundError, OSError):
            click.echo(f"[warn] {forbidden_yaml} not found; skipping mask", err=True)

    out: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = row.get("original_name", "").strip()
            scenario = row.get("scenario", "").strip()
            if not stem or not scenario:
                continue
            if mask_emotion:
                scenario = mask_emotion_tokens(scenario, aliases)
            out[stem] = scenario
    return out


@click.command()
@click.option("--train-csv", default="data/diema_challenge/raw/train_data.csv",
              show_default=True)
@click.option("--data-path", default="data/diema_challenge/processed/motion_train_quat.npz",
              show_default=True)
@click.option("--num-folds", default=3, type=int, show_default=True)
@click.option("--encoder-model", default="sentence-transformers/all-mpnet-base-v2",
              show_default=True)
@click.option("--mask-emotion/--no-mask-emotion", default=True, show_default=True,
              help="Apply the same forbidden-token mask used to build "
                   "scenario_text_cache_masked.pt. Disable only for diagnostic "
                   "comparison — masked is the canonical setting.")
@click.option("--forbidden", default="configs/leakage/forbidden_tokens.yaml",
              show_default=True)
@click.option("--c-grid", "c_grid", default="0.1,1.0,10.0", show_default=True)
@click.option("--device", default=None)
@click.option("--output-md", default="docs/analysis/exp065_scenario_probe.md",
              show_default=True)
def main(
    train_csv: str, data_path: str, num_folds: int,
    encoder_model: str, mask_emotion: bool, forbidden: str,
    c_grid: str, device: str | None, output_md: str,
) -> None:
    import pybvh_ml
    from diema.data.splits import generate_lpo_splits

    cs = [float(x) for x in c_grid.split(",")]
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    click.echo(f"setup: encoder={encoder_model}, mask_emotion={mask_emotion}, device={dev}")

    stem_to_scenario = load_clip_to_scenario(
        Path(train_csv), mask_emotion=mask_emotion, forbidden_yaml=Path(forbidden),
    )
    click.echo(f"Loaded {len(stem_to_scenario)} clip→scenario mappings")

    preprocessed = pybvh_ml.load_preprocessed(data_path)
    filenames = preprocessed["filenames"]
    splits = generate_lpo_splits(filenames, num_folds=num_folds)

    def _stem_of(entry):
        if isinstance(entry, tuple):
            entry = entry[0]
        return Path(str(entry)).stem

    # Build the universe of stems we'll probe over (intersection of CSV +
    # render-aware split stems). LPO splits guarantee both train and val are
    # disjoint by performer, so we can encode once.
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

    texts = [stem_to_scenario[s] for s in stems]

    # Encode with mpnet (matches v1 rationale baseline)
    from sentence_transformers import SentenceTransformer
    click.echo(f"Encoding {len(texts)} scenarios via {encoder_model} ...")
    model = SentenceTransformer(encoder_model, device=dev, trust_remote_code=True)
    X = model.encode(
        texts, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False,
    ).astype(np.float32)
    click.echo(f"Feature shape: {X.shape}")

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

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        y_tr = y[train_idx]; y_va = y[val_idx]

        per_c = []
        for c in cs:
            clf = LogisticRegression(C=c, max_iter=2000, solver="lbfgs", n_jobs=-1)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_va)
            per_c.append({
                "C": c,
                "macro_f1": f1_score(y_va, preds, average="macro"),
                "acc": accuracy_score(y_va, preds),
            })
        best = max(per_c, key=lambda r: r["macro_f1"])
        fold_results.append({
            "fold": fold_id, "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
            "best_C": best["C"], "macro_f1": best["macro_f1"], "acc": best["acc"],
            "per_C": per_c,
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
        verdict = "STRONG NO-GO (F1 < 13%) — scenario text alone has no class signal under the mask"
    elif f1_mean < 0.25:
        verdict = "MODERATE signal — scenario channel is informative but bounded"
    elif f1_mean < 0.45:
        verdict = "STRONG signal — scenario text carries substantial class information"
    else:
        verdict = "VERY STRONG signal — scenario text is highly class-discriminative"

    click.echo(f"\n=== {verdict} ===")
    click.echo(f"3-fold mean macro-F1: {f1_mean*100:.2f}% ± {f1_std*100:.2f}%")
    click.echo(f"3-fold mean acc:      {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")

    md = [
        "# exp065 prep — scenario-text linear probe (3-fold LPO)",
        "",
        f"- encoder: `{encoder_model}`",
        f"- mask_emotion: **{mask_emotion}**",
        f"- num_folds: {num_folds}",
        f"- valid rows: {len(stems)}",
        f"- feature dim: {X.shape[1]}",
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
        "- v1 rationale mpnet+concat: 10.94%",
        "- v2 ranker mpnet (mini-run, 500 clips): 9.19%",
        "- Rule-based DSL probe (exp062): 10.91%",
        "- Backbone-only motion model (exp034 a00): 30.30%",
        "",
        "## Implications for exp065",
        "",
        "- **F1 < 14%**: scenario channel is bottlenecked similarly to rationale.",
        "  The 8-attribute auxiliary head is unlikely to give +0.2pp F1 lift.",
        " exp065 STOP recommended.",
        "- **F1 ∈ [14%, 25%]**: scenario carries some signal. The auxiliary head",
        "  could provide moderate explainability gains; F1 lift uncertain.",
        "- **F1 ≥ 25%**: scenario channel is strong. Auxiliary head training is",
        "  worth pursuing.",
    ])
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(md), encoding="utf-8")
    click.echo(f"Saved: {output_md}")


if __name__ == "__main__":
    main()
