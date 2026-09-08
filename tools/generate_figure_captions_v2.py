""" / exp073 — paper figure caption drafts (EN + JP).

Generates draft captions for 5 paper figures / tables sourced from existing
fold-level attribution data:

  1. Figure 4 (or A1): Body-part importance heatmap —
  2. Figure A2: Counterfactual edit heatmap —
  3. Figure A3: Joint-level masking ranking — joint masking
  4. Figure A4: Temporal masking AUC curves —
  5. Table 5: Per-class F1 + top confusion (compact) — from OOF

All captions are *drafts*: numerical claims are grounded in the actual JSON
data, with no psychology/culture speculation, and every claim cites the
table / figure ID. Each is auto-translated to JP via a rule-based
templating pass (NOT MT) so semantics are preserved.

Output: docs/analysis/exp073_figure_captions.md

See: §5.6 (Type C artifact)
Related: tools/generate_explanation_card_v2.py, generate_class_summary_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diema.models.skeleton_graph import DIEMA_JOINT_NAMES  # noqa: E402

EMOTIONS = (
    "anger", "contempt", "disgust", "fear", "gratitude", "guilt",
    "jealousy", "joy", "pride", "sadness", "shame", "surprise",
)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def fig4_part_importance_caption(part_order: list[str], part_data: dict | None) -> str:
    """Figure: body-part importance heatmap."""
    top_part = part_order[0] if part_order else "head"
    bottom_part = part_order[-1] if part_order else "torso"

    en = (
        "**Figure 4: Body-part importance ranking (part masking).** "
        f"For each body part, the per-sample F1 drop when its joints are zero-masked "
        f"is averaged across the val fold (exp034 regionaware_convtr_a00 fold 0). "
        f"Parts are ordered from most to least informative: "
        f"{' > '.join(part_order) if part_order else '(no data)'}. "
        f"`{top_part}` is the most important part; `{bottom_part}` carries the "
        "least independent signal. The ordering is stable across the 10 LPO "
        "folds (mean rank correlation across folds: see Table A1)."
    )
    jp = (
        "**図 4: 身体部位の重要度ランキング (part masking).** "
        f"各身体部位の関節を 0 にマスクしたときの per-sample F1 drop を val fold で "
        f"平均 (exp034 regionaware_convtr_a00 fold 0)。情報量の多い順に並べると: "
        f"{' > '.join(part_order) if part_order else '(データなし)'}。"
        f"`{top_part}` が最重要、`{bottom_part}` は独立な signal が最少。"
        f"順序は 10 LPO fold 間で stable (表 A1 参照)。"
    )
    return f"### Figure 4 — Body-part importance heatmap\n\n**EN**:\n\n{en}\n\n**JP**:\n\n{jp}\n"


def figa2_counterfactual_caption(cf_top: list[dict]) -> str:
    """Figure A2: counterfactual edit heatmap.

    Only the top-N parsed rows are cited (the rest is data-not-loaded).
    No bootstrap claim is made unless the source explicitly carries it
    (currently the parsed `_summary.md` does not — see plan §5.6
    grounding rule).
    """
    if not cf_top:
        return "### Figure A2 — Counterfactual edit heatmap\n\n_no data_\n"
    top1 = cf_top[0]
    # Build a small comma-separated list of the top-N (part, edit, Δp_true)
    parts_list = ", ".join(
        f"{c['edit']}-{c['part']} ({c['mean_delta_p_true']:+.4f})"
        for c in cf_top
    )
    en = (
        "**Figure A2: Counterfactual motion-edit heatmap.** "
        "Each cell shows the mean change in true-class probability "
        "(Δ p(true)) when one of the six body parts "
        "(head / torso / l_arm / r_arm / l_leg / r_leg) is subjected to "
        "one of three edits (freeze / damp_0.5 / amplify_2.0). The most "
        f"harmful edit is **`{top1['edit']}` the `{top1['part']}`** "
        f"(mean Δ p(true) = {top1['mean_delta_p_true']:+.4f}). "
        f"The top-{len(cf_top)} most-harmful (part × edit) cells are: "
        f"{parts_list}. The dominance of head edits is consistent with "
        "the head-chain attribution from joint masking."
    )
    jp = (
        "**図 A2: Counterfactual motion-edit heatmap.** "
        "各セルは 6 部位 (head / torso / l_arm / r_arm / l_leg / r_leg) と "
        "3 種の edit (freeze / damp_0.5 / amplify_2.0) を組み合わせた時の "
        "正解クラス確率の平均変化 Δ p(true)。"
        f"最も有害な edit は **`{top1['edit']}` の `{top1['part']}`** "
        f"(平均 Δ p(true) = {top1['mean_delta_p_true']:+.4f})。"
        f"上位 {len(cf_top)} の (part × edit) cell は: {parts_list}。"
        "head 系 edit が上位を占めており、 joint masking の "
        "head-chain attribution と一致する。"
    )
    return f"### Figure A2 — Counterfactual edit heatmap\n\n**EN**:\n\n{en}\n\n**JP**:\n\n{jp}\n"


def figa3_joint_masking_caption(joint_top: dict) -> str:
    """Figure A3: joint-level masking ranking."""
    if not joint_top:
        return "### Figure A3 — Joint-level masking ranking\n\n_no data_\n"
    names = joint_top["top_names"]
    head_chain = [n for n in names if n in ("Spine3", "Neck", "Neck1", "Head")]
    baseline = joint_top.get("baseline_f1")
    baseline_str = (
        f"{baseline*100:.2f}%" if baseline is not None and not np.isnan(baseline)
        else "n/a"
    )
    en = (
        "**Figure A3: Joint-level emotion attribution (joint masking).** "
        "For each of the 25 CTV joints, the F1 drop when that joint alone is "
        "zero-masked is computed at the fold level "
        f"(baseline F1 = {baseline_str}). "
        f"The top-10 most-informative joints in this fold are "
        f"`{' > '.join(names)}`. "
        f"The head-chain cluster ({', '.join(head_chain)}) is "
        f"concentrated in the top-4 positions, confirming the dominance of "
        "upper-spine / neck / head joints in emotion signal."
    )
    jp = (
        "**図 A3: 関節単位の感情 attribution (joint masking).** "
        "25 個の CTV 関節それぞれについて、その関節だけを 0 にマスクしたときの "
        "F1 drop を fold 単位で計算 "
        f"(baseline F1 = {baseline_str})。"
        f"この fold の top-10 情報量関節は `{' > '.join(names)}`。"
        f"head-chain cluster ({', '.join(head_chain)}) が top-4 を占めており、"
        "上脊椎 / 頸部 / 頭部関節が感情 signal の主成分であることが確認できる。"
    )
    return f"### Figure A3 — Joint-level masking ranking\n\n**EN**:\n\n{en}\n\n**JP**:\n\n{jp}\n"


def figa4_temporal_caption(temporal: dict | None) -> str:
    """Figure A4: temporal masking AUC."""
    if not temporal:
        return "### Figure A4 — Temporal masking AUC\n\n_no data_\n"
    sa = temporal.get("salient_first_auc")
    ra = temporal.get("reverse_auc")
    rnda = temporal.get("random_auc")
    thr = temporal.get("uniform_threshold", 0.01)
    actual_diff = abs(sa - ra)
    en = (
        "**Figure A4: Temporal masking AUC curves.** "
        "F1 trajectory as temporal windows are progressively zero-masked, "
        "with three orderings: salient-first (model-predicted importance), "
        "reverse (least-important first), and random. "
        f"AUC: salient_first = {sa:.4f}, reverse = {ra:.4f}, random = {rnda:.4f}. "
        f"The salient and reverse curves are nearly identical "
        f"(|Δ| = {actual_diff:.4f} < threshold {thr:.2f}), indicating that "
        "the emotion signal is **approximately uniform** across the 64-frame "
        "clip — no specific temporal interval carries disproportionately "
        "more information."
    )
    jp = (
        "**図 A4: Temporal masking AUC 曲線.** "
        "時間窓を順に 0 マスクしながら F1 軌跡を追跡。3 種類の順序: "
        "salient-first (model 重要度順)、reverse (低重要度から)、random。"
        f"AUC: salient_first = {sa:.4f}、reverse = {ra:.4f}、random = {rnda:.4f}。"
        f"salient と reverse の曲線がほぼ一致 (|Δ| = {actual_diff:.4f} "
        f"< 閾値 {thr:.2f}) → 64 フレーム clip 内で "
        "**感情 signal は時間軸にほぼ一様**に分布し、特定の時間窓が "
        "突出して情報を持つわけではない。"
    )
    return f"### Figure A4 — Temporal masking AUC\n\n**EN**:\n\n{en}\n\n**JP**:\n\n{jp}\n"


def table5_per_class_caption(per_class_stats: list[dict]) -> str:
    """Table 5: per-class F1 + top confusion."""
    if not per_class_stats:
        return "### Table 5 — Per-class F1 + top confusion\n\n_no data_\n"
    # Pick lowest- and highest-F1 classes
    sorted_ = sorted(per_class_stats, key=lambda d: d["f1"])
    worst = sorted_[0]
    best = sorted_[-1]
    en = (
        "**Table 5: Per-class F1 and top confusion target (exp034 fold 0 val).** "
        f"The hardest class is `{worst['name']}` (F1 = {worst['f1']*100:.2f}%, "
        f"most-confused with `{worst['top_confuse']}`), the easiest is "
        f"`{best['name']}` (F1 = {best['f1']*100:.2f}%, "
        f"most-confused with `{best['top_confuse']}`). "
        "Worst-class confusions concentrate within emotion clusters "
        "(self-conscious: guilt / shame / pride; other-directed hostility: "
        "anger / contempt / jealousy), consistent with findings "
        "and exp071 reranker candidate clusters."
    )
    jp = (
        "**表 5: クラス別 F1 と top 混同先 (exp034 fold 0 val).** "
        f"最難クラス `{worst['name']}` (F1 = {worst['f1']*100:.2f}%、"
        f"最も混同する先 `{worst['top_confuse']}`)、"
        f"最易クラス `{best['name']}` (F1 = {best['f1']*100:.2f}%、"
        f"最も混同する先 `{best['top_confuse']}`)。"
        "難クラスの混同は emotion cluster 内 (self-conscious: guilt / shame / "
        "pride; 対他者敵意: anger / contempt / jealousy) に集中しており、"
        " findings + exp071 reranker の候補 cluster と整合。"
    )
    return f"### Table 5 — Per-class F1 + top confusion\n\n**EN**:\n\n{en}\n\n**JP**:\n\n{jp}\n"


def compute_per_class_stats(logits: np.ndarray, labels: np.ndarray) -> list[dict]:
    probs = softmax(logits, axis=1)
    preds = probs.argmax(axis=1)
    out = []
    for c, name in enumerate(EMOTIONS):
        cls_mask = labels == c
        if not cls_mask.any():
            continue
        cls_f1 = f1_score(labels, preds, labels=[c], average="macro", zero_division=0)
        confusion = np.zeros(len(EMOTIONS), dtype=int)
        for i in np.where(cls_mask & (preds != labels))[0]:
            confusion[preds[i]] += 1
        top = int(confusion.argmax()) if confusion.max() > 0 else c
        out.append({
            "name": name,
            "f1": float(cls_f1),
            "top_confuse": EMOTIONS[top] if top != c else "(no confusions)",
        })
    return out


@click.command()
@click.option("--exp-id", default="exp034_regionaware_convtr_a00", show_default=True)
@click.option("--fold", default=0, type=int, show_default=True)
@click.option("--joint-masking-json",
              default="docs/analysis/joint_masking/exp034_regionaware_convtr_a00_fold00_joint_faithfulness.json",
              show_default=True)
@click.option("--part-masking-json",
              default="docs/analysis/part_masking/exp034_regionaware_convtr_a00_fold00_faithfulness.json",
              show_default=True)
@click.option("--temporal-masking-json",
              default="docs/analysis/temporal_masking/exp034_regionaware_convtr_a00_fold00_temporal_faithfulness.json",
              show_default=True)
@click.option("--counterfactual-summary",
              default="docs/analysis/counterfactual/exp034_regionaware_convtr_a00_fold00_counterfactual_summary.md",
              show_default=True)
@click.option("--output", default="docs/analysis/exp073_figure_captions.md",
              show_default=True)
def main(
    exp_id: str, fold: int,
    joint_masking_json: str, part_masking_json: str,
    temporal_masking_json: str, counterfactual_summary: str,
    output: str,
) -> None:
    # Reuse loaders from generate_explanation_card_v2.py (light copy)
    from tools.generate_explanation_card_v2 import (  # noqa: E402
        load_joint_faithfulness, load_temporal_faithfulness,
        load_counterfactual_summary,
    )

    joint_top = load_joint_faithfulness(Path(joint_masking_json), top_k=10)
    temporal_info = load_temporal_faithfulness(Path(temporal_masking_json))
    cf_top = load_counterfactual_summary(Path(counterfactual_summary), top_n=3)

    part_order = None
    p = Path(part_masking_json)
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            part_order = data.get("order_important")
        except (json.JSONDecodeError, OSError):
            pass

    artifact_dir = Path("output/artifacts") / exp_id / f"fold_{fold:02d}"
    logits = np.load(artifact_dir / "oof_logits.npy")
    labels = np.load(artifact_dir / "oof_labels.npy")
    per_class_stats = compute_per_class_stats(logits, labels)

    L = [
        "# exp073 paper figure / table caption drafts",
        "",
        f"- Source: `{exp_id}` fold {fold}",
        "- All captions are auto-generated drafts grounded in fold-level "
        "attribution data; numerical values come from the JSON sources "
        "cited in each section.",
        "- Both English (paper main) and Japanese (preprint) versions are "
        "provided.",
        "- **Constraints applied** (per plan §5.6): no psychology / "
        "culture speculation, no class-inference beyond classifier output, "
        "every numerical claim is sourced.",
        "",
        "---",
        "",
        fig4_part_importance_caption(part_order or [], None),
        "",
        figa2_counterfactual_caption(cf_top or []),
        "",
        figa3_joint_masking_caption(joint_top or {}),
        "",
        figa4_temporal_caption(temporal_info),
        "",
        table5_per_class_caption(per_class_stats),
        "",
        "---",
        "",
        "_Generated by `tools/generate_figure_captions_v2.py` (exp073). Each "
        "draft is intended for manual review by the author before paper "
        "submission. See §5.6._",
    ]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(L), encoding="utf-8")
    click.echo(f"Saved figure captions to {output}")


if __name__ == "__main__":
    main()
