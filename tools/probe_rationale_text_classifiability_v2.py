""" redux: extended linear probe for the rationale-text bottleneck verdict.

See: / docs/analysis/rationale_linear_probe_3fold.md (v1, mpnet+concat → 10.94%)
Related:
- tools/probe_rationale_text_classifiability.py (v1, kept for reproducibility)
- diema/features/rationale_text_cache.py
- output/rationale_cache/rationales/*.json (raw text source)

Extensions over v1 (driven by a code review,
2026-04-27):

1. ``--encoder-model`` upgrade: mpnet-base (768d, MTEB EmotionClf F1=35) →
   bge-large-en-v1.5 (1024d, F1=46) → stella_en_1.5B (1024d, F1=80) →
   NV-Embed-v2 (4096d, F1=90). Computed fresh per call (no cache reuse).

2. ``--aggregation``: ``concat`` (4608d flatten, v1 default) →
   ``masked_mean`` (D-dim, placeholder excluded from the mean) →
   ``attention`` (1-head softmax over parts).

3. ``--prepend-part-name`` flag: re-encodes each part text with
   ``"head: ..."`` etc. so the encoder sees the part name as context.

4. ``--probe``: ``logreg`` (default), ``mlp`` (256-hidden, dropout=0.3),
   ``tfidf`` (skip encoder entirely; whitespace-concat all parts → TF-IDF
   1-2 gram → LogReg). TF-IDF mode is the **token-level baseline**
   that confirms whether the verdict is encoder choice or token-level
   signal absence.

5. ``--permutation-trials``: run N label-shuffle trials to build a null
   F1 distribution; report the p-value of the observed F1.

This is meant to be a single-shot decisive diagnostic. Run 4 setups
(A baseline, B tfidf, C bge-large+masked_mean+prefix, D stella+masked_mean+prefix)
and read the F1 trajectory.

Usage::

    # Setup B: TF-IDF baseline
    python tools/probe_rationale_text_classifiability_v2.py \\
        --probe tfidf --output-md docs/analysis/rationale_probe_tfidf.md

    # Setup C: bge-large + masked_mean + part-prefix
    python tools/probe_rationale_text_classifiability_v2.py \\
        --encoder-model BAAI/bge-large-en-v1.5 \\
        --aggregation masked_mean --prepend-part-name \\
        --output-md docs/analysis/rationale_probe_bge_large.md

    # Setup D: stella_en_1.5B + masked_mean + part-prefix
    python tools/probe_rationale_text_classifiability_v2.py \\
        --encoder-model NovaSearch/stella_en_1.5B_v5 \\
        --aggregation masked_mean --prepend-part-name \\
        --output-md docs/analysis/rationale_probe_stella.md
"""

from __future__ import annotations

import json
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

PART_NAMES = (
    "head", "torso", "left_arm", "right_arm", "left_leg", "right_leg",
)

PART_DISPLAY = {
    "head": "head",
    "torso": "torso",
    "left_arm": "left arm",
    "right_arm": "right arm",
    "left_leg": "left leg",
    "right_leg": "right leg",
}


def emotion_from_stem(stem: str) -> int:
    parts = stem.split("_")
    if len(parts) < 5:
        return -1
    return EMOTION_TO_IDX.get(parts[2], -1)


def is_placeholder(text: str) -> bool:
    import re
    return bool(re.match(
        r"^\s*(no\s+distinctive\s+motion|none|n/?a|)\s*\.?\s*$",
        str(text), re.IGNORECASE,
    ))


def load_rationale_texts(
    rationales_dir: Path, prompt_version: str | None = "1.0",
) -> list[dict]:
    """Re-load raw rationale JSONs (we need text, not pre-encoded vectors).

    ``prompt_version`` filters by the rationale's recorded version. Pass
    ``None`` to accept any version (v2 cache uses ``"2.0"``).
    """
    rows: list[dict] = []
    for p in sorted(rationales_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if prompt_version is not None and str(data.get("prompt_version", "")) != prompt_version:
            continue
        render_key = data.get("render_cache_key")
        rationale = data.get("rationale") or {}
        if not render_key or not rationale:
            continue
        rows.append({"render_key": render_key, "rationale": rationale})
    return rows


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
                ck = rec.get("cache_key")
                stem = rec.get("stem")
                if isinstance(ck, str) and isinstance(stem, str):
                    mapping[ck] = stem
            except json.JSONDecodeError:
                continue
    return mapping


def build_part_texts(
    rationales: list[dict],
    stem_lookup: dict[str, str],
    prepend_part_name: bool,
) -> tuple[list[str], list[list[str]], list[list[bool]]]:
    """Return (stems, per-clip per-part texts, per-clip per-part valid mask)."""
    stems_out: list[str] = []
    texts_out: list[list[str]] = []
    masks_out: list[list[bool]] = []
    for d in rationales:
        stem = stem_lookup.get(str(d["render_key"]))
        if not stem:
            continue
        rationale = d["rationale"]
        per_part_text: list[str] = []
        per_part_valid: list[bool] = []
        for p in PART_NAMES:
            v = rationale.get(p, "")
            v = str(v).strip().lower()
            valid = not is_placeholder(v)
            per_part_valid.append(valid)
            if not valid:
                per_part_text.append("")
                continue
            if prepend_part_name:
                v = f"{PART_DISPLAY[p]}: {v}"
            per_part_text.append(v)
        stems_out.append(stem)
        texts_out.append(per_part_text)
        masks_out.append(per_part_valid)
    return stems_out, texts_out, masks_out


def encode_texts(
    texts_flat: list[str], encoder_model: str, device: str,
) -> np.ndarray:
    """Encode all texts in one batch via sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    click.echo(f"  Loading encoder: {encoder_model} ...")
    model = SentenceTransformer(encoder_model, device=device, trust_remote_code=True)
    click.echo(f"  Encoding {len(texts_flat)} texts ...")
    vecs = model.encode(
        texts_flat, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False,
    )
    return vecs.astype(np.float32)


def aggregate_features(
    per_part_emb: np.ndarray,    # (N, P, D)
    per_part_mask: np.ndarray,    # (N, P) bool
    mode: str,
) -> np.ndarray:
    if mode == "concat":
        N, P, D = per_part_emb.shape
        m = per_part_mask[:, :, None].astype(np.float32)
        return (per_part_emb * m).reshape(N, P * D)
    if mode == "mean":
        m = per_part_mask[:, :, None].astype(np.float32)
        denom = np.clip(m.sum(axis=1), 1e-9, None)
        return (per_part_emb * m).sum(axis=1) / denom
    if mode == "masked_mean":
        # Identical to mean but explicit name; placeholder rows have mask all-False
        # → falls back to zero (placeholder rationales are skipped upstream anyway).
        m = per_part_mask[:, :, None].astype(np.float32)
        denom = np.clip(m.sum(axis=1), 1e-9, None)
        return (per_part_emb * m).sum(axis=1) / denom
    if mode == "attention":
        # Single-head softmax over parts using L2-normalized embeddings as queries.
        # No learning here — pure self-attention with key=query, mask out invalid.
        N, P, D = per_part_emb.shape
        x = per_part_emb / (np.linalg.norm(per_part_emb, axis=2, keepdims=True) + 1e-9)
        scores = (x * x.mean(axis=1, keepdims=True)).sum(axis=2)  # (N, P)
        scores = np.where(per_part_mask, scores, -1e9)
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        weights = weights / weights.sum(axis=1, keepdims=True)    # (N, P)
        return (per_part_emb * weights[..., None]).sum(axis=1)
    raise ValueError(f"unknown aggregation: {mode!r}")


def fit_probe(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    probe: str, c_grid: list[float],
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.preprocessing import StandardScaler

    if probe == "tfidf":
        # The caller passes raw whitespace-joined text strings as X_tr/X_va.
        # No scaling; sklearn's TfidfVectorizer + LogReg pipeline.
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
        Z_tr = vec.fit_transform(X_tr)
        Z_va = vec.transform(X_va)
        per_c: list[dict] = []
        for c in c_grid:
            clf = LogisticRegression(
                C=c, max_iter=2000, multi_class="multinomial",
                solver="lbfgs", n_jobs=-1,
            )
            clf.fit(Z_tr, y_tr)
            preds = clf.predict(Z_va)
            per_c.append({
                "C": c, "macro_f1": f1_score(y_va, preds, average="macro"),
                "acc": accuracy_score(y_va, preds),
            })
        best = max(per_c, key=lambda r: r["macro_f1"])
        return best, per_c

    scaler = StandardScaler()
    X_tr_z = scaler.fit_transform(X_tr)
    X_va_z = scaler.transform(X_va)

    if probe == "logreg":
        per_c = []
        for c in c_grid:
            clf = LogisticRegression(
                C=c, max_iter=2000, multi_class="multinomial",
                solver="lbfgs", n_jobs=-1,
            )
            clf.fit(X_tr_z, y_tr)
            preds = clf.predict(X_va_z)
            per_c.append({
                "C": c, "macro_f1": f1_score(y_va, preds, average="macro"),
                "acc": accuracy_score(y_va, preds),
            })
        best = max(per_c, key=lambda r: r["macro_f1"])
        return best, per_c

    if probe == "mlp":
        from sklearn.neural_network import MLPClassifier
        # Use C-grid as a stand-in for `alpha` (L2 weight decay).
        per_c = []
        for c in c_grid:
            alpha = 1.0 / max(c, 1e-9)  # roughly invert
            clf = MLPClassifier(
                hidden_layer_sizes=(256,), activation="relu",
                solver="adam", alpha=alpha, batch_size=128,
                learning_rate_init=1e-3, max_iter=200,
                early_stopping=True, validation_fraction=0.1,
                random_state=42,
            )
            clf.fit(X_tr_z, y_tr)
            preds = clf.predict(X_va_z)
            per_c.append({
                "C": c, "macro_f1": f1_score(y_va, preds, average="macro"),
                "acc": accuracy_score(y_va, preds),
            })
        best = max(per_c, key=lambda r: r["macro_f1"])
        return best, per_c
    raise ValueError(f"unknown probe: {probe!r}")


@click.command()
@click.option(
    "--rationales-dir", default="output/rationale_cache/rationales", show_default=True,
)
@click.option(
    "--render-index",
    default="output/rationale_cache/renders/render_index.jsonl", show_default=True,
)
@click.option(
    "--data-path", default="data/diema_challenge/processed/motion_train_quat.npz",
    show_default=True,
)
@click.option("--num-folds", default=3, type=int, show_default=True)
@click.option(
    "--encoder-model", default="sentence-transformers/all-mpnet-base-v2",
    show_default=True,
    help="HF model id; ignored if --probe=tfidf.",
)
@click.option(
    "--aggregation",
    type=click.Choice(["concat", "mean", "masked_mean", "attention"]),
    default="concat", show_default=True,
)
@click.option(
    "--prepend-part-name", is_flag=True,
    help='Prepend "head: ..." etc. before encoding so the encoder sees the part name.',
)
@click.option(
    "--probe", type=click.Choice(["logreg", "mlp", "tfidf"]),
    default="logreg", show_default=True,
)
@click.option("--C-grid", "c_grid", default="0.1,1.0,10.0", show_default=True)
@click.option(
    "--permutation-trials", default=0, type=int, show_default=True,
    help="Run N label-shuffle null trials to compute a p-value (0=skip).",
)
@click.option("--device", default=None, help="cuda / cpu, default auto.")
@click.option(
    "--prompt-version", default="1.0", show_default=True,
    help='Rationale prompt_version to load. "1.0" = v1 cache, "2.0" = v2 '
         'selected cache, "any" / empty string = accept all versions.',
)
@click.option(
    "--output-md", default="docs/analysis/rationale_probe_v2.md", show_default=True,
)
def main(
    rationales_dir: str, render_index: str, data_path: str, num_folds: int,
    encoder_model: str, aggregation: str, prepend_part_name: bool,
    probe: str, c_grid: str,
    permutation_trials: int, device: str | None,
    prompt_version: str,
    output_md: str,
) -> None:
    import pybvh_ml
    from diema.data.splits import generate_lpo_splits

    cs = [float(x) for x in c_grid.split(",")]
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    click.echo(f"setup: encoder={encoder_model}, aggregation={aggregation}, "
               f"prefix={prepend_part_name}, probe={probe}, device={dev}")

    pv = None if prompt_version.strip().lower() in ("", "any") else prompt_version
    rationales = load_rationale_texts(Path(rationales_dir), prompt_version=pv)
    stem_lookup = load_render_index(Path(render_index))
    stems, per_part_texts, per_part_masks = build_part_texts(
        rationales, stem_lookup, prepend_part_name,
    )
    click.echo(f"loaded {len(stems)} rationales")

    y = np.array([emotion_from_stem(s) for s in stems], dtype=np.int64)
    valid_y = y >= 0
    stems = [s for s, v in zip(stems, valid_y) if v]
    per_part_texts = [t for t, v in zip(per_part_texts, valid_y) if v]
    per_part_masks = [m for m, v in zip(per_part_masks, valid_y) if v]
    y = y[valid_y]
    click.echo(f"valid (stem, label): {len(stems)}")

    # Build feature matrix.
    if probe == "tfidf":
        # Whitespace-join 6 parts (placeholder texts are already empty strings).
        X = np.array(
            [" ".join(t for t in parts if t) for parts in per_part_texts],
            dtype=object,
        )
        feature_dim = "tfidf"
    else:
        # Encode every (clip, part) text. Empty placeholders pass through as ""
        # which most encoders will return a degenerate vector for, but the
        # downstream mask zeros them out so it doesn't matter.
        flat_texts: list[str] = []
        for parts in per_part_texts:
            flat_texts.extend(parts)
        flat_emb = encode_texts(flat_texts, encoder_model, device=dev)
        D = int(flat_emb.shape[1])
        per_part_emb = flat_emb.reshape(len(stems), len(PART_NAMES), D)
        per_part_mask = np.array(per_part_masks, dtype=bool)
        X = aggregate_features(per_part_emb, per_part_mask, mode=aggregation)
        feature_dim = X.shape[1]
        click.echo(f"feature dim: {feature_dim}")

    # LPO splits.
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

        X_tr = X[train_idx] if probe != "tfidf" else X[train_idx]
        X_va = X[val_idx] if probe != "tfidf" else X[val_idx]
        y_tr = y[train_idx]
        y_va = y[val_idx]
        best, per_c = fit_probe(X_tr, y_tr, X_va, y_va, probe=probe, c_grid=cs)
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
            f"  fold {fold_id}: F1={best['macro_f1']:.4f} acc={best['acc']:.4f} "
            f"(best C={best['C']})"
        )

    f1s = [r["macro_f1"] for r in fold_results]
    accs = [r["acc"] for r in fold_results]
    f1_mean, f1_std = float(np.mean(f1s)), float(np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0)
    acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)

    # Permutation test (cheap: re-fit only the probe on shuffled labels).
    p_value = None
    null_mean = None
    null_std = None
    if permutation_trials > 0:
        click.echo(f"\nRunning permutation test ({permutation_trials} trials) ...")
        rng = np.random.default_rng(2026)
        # Use only fold 0 for speed (TF-IDF re-fits encoder per call too).
        split0 = splits[0]
        train_idx = np.array([
            stem_to_row[_stem_of(e)]
            for e in split0["train"]
            if _stem_of(e) in stem_to_row
        ])
        val_idx = np.array([
            stem_to_row[_stem_of(e)]
            for e in split0["val"]
            if _stem_of(e) in stem_to_row
        ])
        null_f1s = []
        for trial in range(permutation_trials):
            y_shuffled = y.copy()
            rng.shuffle(y_shuffled)
            best_p, _ = fit_probe(
                X[train_idx], y_shuffled[train_idx],
                X[val_idx], y_shuffled[val_idx],
                probe=probe, c_grid=[1.0],   # single C for speed
            )
            null_f1s.append(best_p["macro_f1"])
            if (trial + 1) % max(1, permutation_trials // 10) == 0:
                click.echo(f"  trial {trial + 1}/{permutation_trials}: F1={best_p['macro_f1']:.4f}")
        null_arr = np.array(null_f1s)
        null_mean = float(null_arr.mean())
        null_std = float(null_arr.std(ddof=1))
        observed_fold0 = fold_results[0]["macro_f1"]
        p_value = float((null_arr >= observed_fold0).mean())
        click.echo(
            f"null F1: mean={null_mean:.4f}, std={null_std:.4f}, "
            f"observed (fold 0) = {observed_fold0:.4f}, p-value = {p_value:.4f}"
        )

    # Verdict
    if f1_mean < 0.13:
        verdict = "STRONG NO-GO (F1 < 13%) — text bottleneck confirmed at this setup"
    elif f1_mean < 0.18:
        verdict = "WEAK signal (13% ≤ F1 < 18%) — encoder/aggregation may help, but ceiling probably low"
    elif f1_mean < 0.25:
        verdict = "MODERATE signal (18% ≤ F1 < 25%) — text encodes class info; Tier 1 ablations may now help"
    else:
        verdict = "STRONG signal (F1 ≥ 25%) — original NO-GO verdict was wrong; revisit Tier 1/2"

    click.echo(f"\n=== {verdict} ===")
    click.echo(f"3-fold mean macro-F1: {f1_mean*100:.2f}% ± {f1_std*100:.2f}%")
    click.echo(f"3-fold mean acc:      {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")

    out = Path(output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# Rationale text linear probe v2 — redux",
        "",
        f"- encoder: `{encoder_model}`",
        f"- aggregation: **{aggregation}**",
        f"- prepend-part-name: **{prepend_part_name}**",
        f"- probe: **{probe}**",
        f"- num_folds: {num_folds}",
        f"- valid rows: {len(stems)}",
        f"- feature dim: {feature_dim}",
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
    ]
    if permutation_trials > 0:
        md.extend([
            "## Permutation test (fold 0)",
            "",
            f"- trials: {permutation_trials}",
            f"- null F1 distribution: {null_mean*100:.2f}% ± {null_std*100:.2f}%",
            f"- observed F1 (fold 0): {fold_results[0]['macro_f1']*100:.2f}%",
            f"- p-value: **{p_value:.4f}**",
            "",
        ])
    md.extend([
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
        "- v1 baseline (mpnet+concat): 10.94%",
        "- Backbone-only motion model (exp034 a00): 30.30% ± 3.23%",
    ])
    out.write_text("\n".join(md), encoding="utf-8")
    click.echo(f"Saved: {out}")


if __name__ == "__main__":
    main()
