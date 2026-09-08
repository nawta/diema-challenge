""" / exp066 — per-clip evidence-to-explanation report.

See: / docs/analysis/exp065_aux_head_verdict.md
Related:
- output/artifacts/exp034_regionaware_convtr_a00/fold_*/oof_*.{npy,txt}
- output/dsl_cache_rule/<rationale_cache_key>.json (exp062)
- output/scenario_attrs_rule.json (exp065)
- output/rationale_cache/rationales/<rationale_cache_key>.json (v1 free-form text)
- docs/analysis/part_masking/*_faithfulness.json (corpus body-part importance)

Aggregates the per-clip evidence we have for each validation clip into a
single markdown report. The intended audience is a human auditing
explainability quality (groundedness / faithfulness / coherence) and the
paper appendix figure pipeline.

Components per clip (current implementation):
  1. predicted emotion + confidence + ground truth (from OOF logits)
  2. body-part importance (corpus-level ranking, the same for
     every clip — explainability artifact rather than instance-specific)
  3. DSL summary (per-clip, exp062)
  4. scenario attributes (per-clip, exp065)
  5. v1 free-form rationale text (per-clip, kinematic-only)

Skipped for now (require additional GPU runs / model downloads):
  6. nearest-neighbor exemplars by rationale similarity (requires
     pre-computed text embeddings + retrieval)
  7. counterfactual motion edit narratives (requires re-running
     tools/counterfactual_motion_edit.py per clip)
  8. natural-language Qwen3 narrative synthesis (requires Qwen3 text
     model download and inference)

Components 6-8 can be layered in later if the manual audit on 1-5 shows
the artifact is paper-grade.

Usage::

    # Generate reports for the first 50 val clips of fold 0
    python tools/generate_explanation_report.py \\
        --exp-id exp034_regionaware_convtr_a00 --fold 0 --max-clips 50

    # Stratified sample: top-1 highest-confidence correct per emotion
    python tools/generate_explanation_report.py \\
        --exp-id exp034_regionaware_convtr_a00 --fold 0 \\
        --strategy top1_per_class
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EMOTIONS = (
    "anger", "contempt", "disgust", "fear", "gratitude", "guilt",
    "jealousy", "joy", "pride", "sadness", "shame", "surprise",
)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def load_oof(artifact_dir: Path) -> dict:
    logits = np.load(artifact_dir / "oof_logits.npy")
    labels = np.load(artifact_dir / "oof_labels.npy")
    filenames = (artifact_dir / "oof_filenames.txt").read_text().splitlines()
    return {"logits": logits, "labels": labels, "filenames": filenames}


def load_render_index(path: Path) -> dict[str, str]:
    """{cache_key: stem} from render_index.jsonl."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ck = rec.get("cache_key"); stem = rec.get("stem")
            if isinstance(ck, str) and isinstance(stem, str):
                out[ck] = stem
    return out


def build_stem_to_dsl(dsl_dir: Path, render_index: dict[str, str]) -> dict[str, dict]:
    """{stem: dsl_dict} via the per-cache-key DSL JSONs."""
    out: dict[str, dict] = {}
    for p in dsl_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rk = data.get("render_cache_key")
        stem = render_index.get(str(rk)) if rk else None
        if stem and "dsl" in data:
            out[stem] = data["dsl"]
    return out


def build_stem_to_rationale(rat_dir: Path, render_index: dict[str, str]) -> dict[str, dict]:
    """{stem: rationale_dict (v1 free-form text fields)}."""
    out: dict[str, dict] = {}
    for p in rat_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if str(data.get("prompt_version", "")) != "1.0":
            continue
        rk = data.get("render_cache_key")
        stem = render_index.get(str(rk)) if rk else None
        rat = data.get("rationale") or {}
        if stem and isinstance(rat, dict):
            out[stem] = rat
    return out


def render_report(
    stem: str,
    label_idx: int,
    logits: np.ndarray,
    dsl: dict | None,
    attrs: dict | None,
    rationale: dict | None,
    body_part_order: list[str] | None,
) -> str:
    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))
    pred_emo = EMOTIONS[pred_idx]
    true_emo = EMOTIONS[label_idx]
    is_correct = pred_idx == label_idx
    pred_conf = float(probs[pred_idx])
    top3 = sorted(enumerate(probs), key=lambda kv: -kv[1])[:3]

    lines: list[str] = []
    lines.append(f"# Explainability report — `{stem}`")
    lines.append("")

    # 1. Predicted emotion + confidence
    lines.append("## 1. Prediction")
    lines.append("")
    lines.append(f"- **predicted**: `{pred_emo}` (confidence {pred_conf*100:.1f}%)")
    lines.append(f"- **ground truth**: `{true_emo}`")
    lines.append(f"- **correct**: {'✅' if is_correct else '❌'}")
    lines.append("")
    lines.append("Top-3 class probabilities:")
    lines.append("")
    lines.append("| rank | class | prob |")
    lines.append("|---|---|---:|")
    for r, (i, p) in enumerate(top3, 1):
        lines.append(f"| {r} | `{EMOTIONS[i]}` | {p*100:.2f}% |")
    lines.append("")

    # 2. Body-part importance (corpus-level)
    if body_part_order:
        lines.append("## 2. Body-part importance (corpus-level)")
        lines.append("")
        lines.append("Order from most to least important (mean across val folds):")
        lines.append("")
        lines.append("1. " + "\n2. ".join(f"`{p}`" for p in body_part_order))
        lines.append("")
        lines.append("_Note: this ranking is corpus-averaged, not per-clip. Per-clip "
                     "importance requires re-running joint-masking per stem (deferred)._")
        lines.append("")

    # 3. DSL summary
    if dsl:
        lines.append("## 3. Body-motion DSL (exp062, rule-based)")
        lines.append("")
        glob = dsl.get("global", {})
        if glob:
            lines.append("### Global")
            lines.append("")
            for a in ("openness", "verticality", "locomotion", "energy", "tempo", "asymmetry"):
                v = glob.get(a, "—")
                lines.append(f"- `{a}`: **{v}**")
            lines.append("")
        lines.append("### Per body part")
        lines.append("")
        lines.append("| part | action | direction | amplitude | tempo | confidence |")
        lines.append("|---|---|---|---|---|---:|")
        for p in ("head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"):
            row = dsl.get(p, {})
            lines.append(
                f"| `{p}` | {row.get('action', '—')} | {row.get('direction', '—')} | "
                f"{row.get('amplitude', '—')} | {row.get('tempo', '—')} | {row.get('confidence', '—')} |"
            )
        lines.append("")

    # 4. Scenario attributes
    if attrs:
        lines.append("## 4. Scenario attributes (exp065, rule-based 8-dim)")
        lines.append("")
        for a in ("valence", "arousal", "dominance", "sociality",
                  "agency", "approach_avoidance", "energy", "openness"):
            v = attrs.get(a, "—")
            lines.append(f"- `{a}`: **{v}**")
        lines.append("")

    # 5. v1 free-form rationale text
    if rationale:
        lines.append("## 5. Free-form rationale (Qwen3-VL kinematic description, v1)")
        lines.append("")
        glob = rationale.get("global_motion", "")
        if glob:
            lines.append(f"**Global motion**: {glob}")
            lines.append("")
        lines.append("**Per body part**:")
        lines.append("")
        for p in ("head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"):
            v = rationale.get(p, "")
            if v:
                lines.append(f"- `{p}`: {v}")
        intervals = rationale.get("salient_intervals") or []
        if intervals:
            lines.append("")
            lines.append("**Salient intervals**:")
            for it in intervals[:3]:
                if isinstance(it, dict):
                    s = it.get("start_sec"); e = it.get("end_sec"); n = it.get("note")
                    lines.append(f"- {s:.1f}-{e:.1f}s: {n}" if isinstance(s, (int, float)) and isinstance(e, (int, float)) else f"- {it}")
        unc = rationale.get("uncertainty_notes")
        if unc:
            lines.append("")
            lines.append(f"**Uncertainty**: {unc}")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("_Components 6-8 (nearest-neighbor exemplars, counterfactual edits, "
                 "Qwen3 narrative synthesis) deferred — see "
                 "`tools/generate_explanation_report.py` docstring._")
    return "\n".join(lines)


@click.command()
@click.option("--exp-id", default="exp034_regionaware_convtr_a00", show_default=True,
              help="Experiment ID (used to locate output/artifacts/<exp_id>/fold_<NN>/).")
@click.option("--fold", default=0, type=int, show_default=True)
@click.option("--max-clips", default=50, type=int, show_default=True,
              help="Number of reports to generate.")
@click.option("--strategy",
              type=click.Choice(["first", "top1_per_class", "stratified"]),
              default="stratified", show_default=True,
              help="Selection strategy for the --max-clips sample. "
                   "'first' = first N. "
                   "'top1_per_class' = highest-confidence correct per class. "
                   "'stratified' = balanced sample across (correct/incorrect, class).")
@click.option("--render-index",
              default="output/rationale_cache/renders/render_index.jsonl",
              show_default=True)
@click.option("--dsl-dir", default="output/dsl_cache_rule", show_default=True)
@click.option("--attrs-json", default="output/scenario_attrs_rule.json", show_default=True)
@click.option("--rationale-dir", default="output/rationale_cache/rationales", show_default=True)
@click.option("--part-masking-json",
              default="docs/analysis/part_masking/exp034_regionaware_convtr_a00_fold00_faithfulness.json",
              show_default=True)
@click.option("--output-dir", default="output/explainability_reports", show_default=True)
@click.option("--index-md",
              default="docs/analysis/exp066_explainability_index.md",
              show_default=True)
def main(
    exp_id: str, fold: int, max_clips: int, strategy: str,
    render_index: str, dsl_dir: str, attrs_json: str, rationale_dir: str,
    part_masking_json: str, output_dir: str, index_md: str,
) -> None:
    artifact_dir = Path("output/artifacts") / exp_id / f"fold_{fold:02d}"
    if not artifact_dir.is_dir():
        raise click.ClickException(f"artifact_dir not found: {artifact_dir}")

    oof = load_oof(artifact_dir)
    logits = oof["logits"]; labels = oof["labels"]; stems = oof["filenames"]

    rendered_index = load_render_index(Path(render_index))
    stem_to_dsl = build_stem_to_dsl(Path(dsl_dir), rendered_index)
    stem_to_rat = build_stem_to_rationale(Path(rationale_dir), rendered_index)
    if Path(attrs_json).is_file():
        stem_to_attrs = json.loads(Path(attrs_json).read_text(encoding="utf-8"))
    else:
        stem_to_attrs = {}

    body_part_order = None
    pm_path = Path(part_masking_json)
    if pm_path.is_file():
        try:
            pm = json.loads(pm_path.read_text())
            body_part_order = pm.get("order_important")
        except (json.JSONDecodeError, OSError):
            click.echo(f"[warn] could not parse {pm_path}", err=True)

    click.echo(
        f"OOF: {len(stems)} clips. Lookups: dsl={len(stem_to_dsl)}, "
        f"rat={len(stem_to_rat)}, attrs={len(stem_to_attrs)}"
    )

    probs = softmax(logits, axis=1)
    preds = probs.argmax(axis=1)
    correct = preds == labels

    if strategy == "first":
        sel = list(range(min(max_clips, len(stems))))
    elif strategy == "top1_per_class":
        sel = []
        for c in range(len(EMOTIONS)):
            mask = (labels == c) & correct
            if not mask.any():
                continue
            confs = probs[mask, c]
            best = int(np.where(mask)[0][confs.argmax()])
            sel.append(best)
        sel = sel[:max_clips]
    else:
        # stratified: roughly equal correct + incorrect per class
        sel = []
        per_class_quota = max(1, max_clips // (len(EMOTIONS) * 2))
        for c in range(len(EMOTIONS)):
            for is_corr in (True, False):
                mask = (labels == c) & (correct == is_corr)
                idxs = np.where(mask)[0]
                if len(idxs) == 0:
                    continue
                # Sort by confidence and take top per_class_quota
                confs = probs[idxs, preds[idxs]]
                ord_ = idxs[np.argsort(-confs)]
                sel.extend(ord_[:per_class_quota].tolist())
        sel = sel[:max_clips]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[tuple[str, str, int, int, float, bool]] = []
    n_with_full_evidence = 0
    for idx in sel:
        stem = stems[idx]
        rep = render_report(
            stem=stem, label_idx=int(labels[idx]),
            logits=logits[idx],
            dsl=stem_to_dsl.get(stem),
            attrs=stem_to_attrs.get(stem),
            rationale=stem_to_rat.get(stem),
            body_part_order=body_part_order,
        )
        out_path = out_dir / f"{stem}.md"
        out_path.write_text(rep, encoding="utf-8")
        has_dsl = stem in stem_to_dsl
        has_attrs = stem in stem_to_attrs
        has_rat = stem in stem_to_rat
        if has_dsl and has_attrs and has_rat:
            n_with_full_evidence += 1
        pred_emo = EMOTIONS[int(preds[idx])]
        true_emo = EMOTIONS[int(labels[idx])]
        rendered.append(
            (stem, true_emo, int(labels[idx]), int(preds[idx]),
             float(probs[idx][preds[idx]]), bool(correct[idx])),
        )

    click.echo(
        f"Rendered {len(rendered)} reports to {out_dir} "
        f"({n_with_full_evidence} with all 4 evidence channels: "
        f"dsl + attrs + rationale + part-masking)"
    )

    # Index file
    md = ["# exp066 explainability report index",
          "",
          f"- exp_id: `{exp_id}`",
          f"- fold: {fold}",
          f"- strategy: `{strategy}`",
          f"- rendered: **{len(rendered)}** reports",
          f"- with all 4 evidence channels: **{n_with_full_evidence}**",
          "",
          "## Reports",
          "",
          "| stem | true | predicted | confidence | correct |",
          "|---|---|---|---:|---|"]
    # Compute relative path from the index_md dir to output_dir for clickable
    # markdown links. Falls back to absolute path if they are on different
    # roots (rare on this project).
    try:
        rel_out = Path("../..") / Path(output_dir)
    except Exception:
        rel_out = Path(output_dir).resolve()
    for stem, true_emo, lbl, pred, conf, corr in rendered:
        md.append(
            f"| [`{stem}`]({rel_out}/{stem}.md) "
            f"| {true_emo} "
            f"| {EMOTIONS[pred]} "
            f"| {conf*100:.1f}% "
            f"| {'✅' if corr else '❌'} |"
        )
    Path(index_md).parent.mkdir(parents=True, exist_ok=True)
    Path(index_md).write_text("\n".join(md), encoding="utf-8")
    click.echo(f"Saved index: {index_md}")


if __name__ == "__main__":
    main()
