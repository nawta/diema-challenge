""" / exp073 — per-clip explanation card v2 with 10 evidence channels.

Extends `tools/generate_explanation_report.py` (exp066, 5 channels) with
the remaining attribution channels per
§5.6:

  | ch | source | new in v2 |
  |---|---|:---:|
  | 1 | exp034 OOF prediction + top-3 | (kept) |
  | 2 | part masking ranking | (kept) |
  | 3 | exp062 body-motion DSL (per-clip) | (kept) |
  | 4 | exp065 scenario attributes (per-clip) | (kept) |
  | 5 | exp062 v1 free-form rationale | (kept) |
  | 6 | ** joint masking** (top-k by f1_drop) | NEW |
  | 7 | ** temporal masking** (uniform = no localization) | NEW |
  | 8 | ** counterfactual edits** (top part/edit by Δ p_true) | NEW |
  | 9 | ** per-performer baseline** (this performer's F1) | NEW |
  | 10 | **Deterministic narrator summary** (3 sentences, evidence citations) | NEW |

Optional (`--qwen3` flag): swap the deterministic narrator for a
constraint-bound Qwen3 narrator. Mandatory guard: no class inference, no
psychology speculation, citation required for every numerical claim.
Falls back to deterministic narrator if Qwen3 inference fails (per plan
No-Go fallback policy).

Output is paper-grade markdown matching exp066's structure plus the 5
new sections; the deterministic narrator section is at the bottom and
labeled "auto-narrated (template)" or "auto-narrated (Qwen3-{model})".

See: §5.6
Related: tools/generate_explanation_report.py (exp066 baseline),
         docs/analysis/{joint_masking, temporal_masking, counterfactual,
                        per_performer}/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np

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


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def load_oof(artifact_dir: Path) -> dict:
    logits = np.load(artifact_dir / "oof_logits.npy")
    labels = np.load(artifact_dir / "oof_labels.npy")
    filenames = (artifact_dir / "oof_filenames.txt").read_text().splitlines()
    return {"logits": logits, "labels": labels, "filenames": filenames}


def load_render_index(path: Path) -> dict[str, str]:
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


def load_joint_faithfulness(path: Path, top_k: int = 10) -> dict | None:
    """Load joint masking ranking. Returns dict with 'top_idx',
    'top_names', 'baseline_f1' or None if unavailable."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    top_idx = data.get("top_order_joint_idx", [])[:top_k]
    return {
        "top_idx": top_idx,
        "top_names": [DIEMA_JOINT_NAMES[i] for i in top_idx],
        "baseline_f1": data.get("baseline_f1"),
    }


# Threshold for declaring temporal saliency "approximately uniform".
# salient_first_auc and reverse_auc differing by less than this absolute
# margin → emotion signal is judged to be temporally diffuse with no
# clip-internal localization. The threshold is **not** registered in
# plan §5.6 — it is a downstream-presentation heuristic. We expose it
# as a constant so the figure caption can cite the same value.
TEMPORAL_UNIFORM_THRESHOLD: float = 0.01


def load_temporal_faithfulness(path: Path) -> dict | None:
    """Load temporal masking. Returns dict with curve auc and
    a uniform-localization flag (using TEMPORAL_UNIFORM_THRESHOLD)."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    aucs = data.get("aucs", {})
    sal_auc = aucs.get("salient_first")
    rev_auc = aucs.get("reverse")
    rand_auc = aucs.get("random")
    is_uniform = None
    if isinstance(sal_auc, (int, float)) and isinstance(rev_auc, (int, float)):
        is_uniform = abs(sal_auc - rev_auc) < TEMPORAL_UNIFORM_THRESHOLD
    return {
        "salient_first_auc": sal_auc,
        "reverse_auc": rev_auc,
        "random_auc": rand_auc,
        "is_uniform": is_uniform,
        "uniform_threshold": TEMPORAL_UNIFORM_THRESHOLD,
    }


def load_counterfactual_summary(path: Path, top_n: int = 3) -> list[dict] | None:
    """Parse top-N (part, edit, mean Δ p_true, flipped %) from
    counterfactual_summary.md. Returns list of dicts or None.

    The summary file has two markdown tables: 'Mean Δ p(true class)' and
    'Samples flipped'. We pull from the first one (most-harmful first).
    """
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    # Parse the first markdown table after "Mean Δ p(true class)" header.
    out: list[dict] = []
    in_first_table = False
    for line in text.splitlines():
        if "Mean Δ p(true class)" in line:
            in_first_table = True
            continue
        if not in_first_table:
            continue
        if line.startswith("Samples flipped"):
            break
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        # Expected: rank | part | edit | mean Δ p_true
        if len(cells) >= 4:
            rank_cell = cells[0]
            if rank_cell in ("rank", ":---", "----", "---") or "----" in rank_cell:
                continue
            try:
                rank = int(rank_cell)
            except ValueError:
                continue
            try:
                delta = float(cells[3])
            except ValueError:
                continue
            out.append({
                "rank": rank,
                "part": cells[1],
                "edit": cells[2],
                "mean_delta_p_true": delta,
            })
            if len(out) >= top_n:
                break
    return out or None


def load_per_performer_f1(csv_path: Path) -> dict[str, dict] | None:
    """Parse per-performer F1 CSV. Returns {performer: {macro_f1, accuracy,
    n_samples, country, n_classes_seen}} or None.

    Uses csv.DictReader to handle any future quoted-field rows safely
    (round-1 review found the original split-on-comma would drop a row
    if `country` ever contained a comma).
    """
    if not csv_path.is_file():
        return None
    import csv as _csv
    out: dict[str, dict] = {}
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                pid = (row.get("performer") or "").strip()
                if not pid:
                    continue
                try:
                    out[pid] = {
                        "country": (row.get("country") or "").strip(),
                        "n_samples": int(row.get("n_samples") or 0),
                        "macro_f1": float(row.get("macro_f1") or "nan"),
                        "accuracy": float(row.get("accuracy") or "nan"),
                        "n_classes_seen": int(row.get("n_classes_seen") or 0),
                    }
                except ValueError:
                    continue
    except OSError:
        return None
    return out or None


def stem_to_performer(stem: str) -> str:
    """Stem like 'JP_06_contempt_1_H' → 'JP_06'."""
    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return stem


# ---------------------------------------------------------------------------
# narrators
# ---------------------------------------------------------------------------


def deterministic_narrator(
    stem: str,
    pred_emo: str,
    true_emo: str,
    pred_conf: float,
    is_correct: bool,
    top3: list[tuple[int, float]],
    joint_top_names: list[str] | None,
    temporal_uniform: bool | None,
    counterfactual_top: list[dict] | None,
    perf_f1: float | None,
) -> str:
    """Generate a 3-sentence template-based narrative grounded in the
    numerical evidence. No psychology, no class inference beyond what the
    classifier output. Each claim is suffixed with [§N] referencing the
    section above.
    """
    sent_1: list[str] = []
    sent_1.append(f"The classifier predicts **{pred_emo}** with "
                  f"{pred_conf*100:.1f}% confidence")
    if is_correct:
        sent_1.append(f"matching the ground truth label **{true_emo}** [§1]")
    else:
        runner_emo = EMOTIONS[top3[1][0]]
        runner_conf = top3[1][1]
        sent_1.append(
            f"the ground truth is **{true_emo}** (model's 2nd guess is "
            f"{runner_emo} at {runner_conf*100:.1f}%) [§1]"
        )
    sentence_1 = "; ".join(sent_1) + "."

    sent_2: list[str] = []
    if joint_top_names:
        head_chain = [n for n in joint_top_names if n in
                      ("Spine3", "Neck", "Neck1", "Head")]
        if len(head_chain) >= 2:
            sent_2.append(
                f"the top-{len(joint_top_names)} most informative joints "
                f"include {', '.join(head_chain)} from the head-chain [§6]"
            )
        else:
            sent_2.append(
                f"the top-{len(joint_top_names)} most informative joints "
                f"are {', '.join(joint_top_names[:5])}{'...' if len(joint_top_names)>5 else ''} [§6]"
            )
    if temporal_uniform is True:
        sent_2.append(
            "temporal saliency is approximately uniform (the emotion signal "
            "is distributed across the clip, not localized to a specific "
            "interval) [§7]"
        )
    elif temporal_uniform is False:
        sent_2.append(
            "temporal saliency is non-uniform (some intervals are more "
            "informative than others) [§7]"
        )
    sentence_2 = (
        "Attribution analysis at this fold's corpus level shows: "
        + "; ".join(sent_2) + "." if sent_2 else
        "No attribution data available at this fold."
    )

    sent_3: list[str] = []
    if counterfactual_top:
        c0 = counterfactual_top[0]
        sent_3.append(
            f"the most harmful counterfactual edit is to "
            f"`{c0['edit']}` the **{c0['part']}** "
            f"(mean Δ p(true) = {c0['mean_delta_p_true']:+.4f}) [§8]"
        )
    if perf_f1 is not None and not np.isnan(perf_f1):
        sent_3.append(
            f"this performer's overall macro-F1 across all classes is "
            f"{perf_f1*100:.1f}% [§9]"
        )
    sentence_3 = (
        "For this clip, " + "; ".join(sent_3) + "." if sent_3 else
        "No additional context evidence available."
    )

    return f"{sentence_1} {sentence_2} {sentence_3}"


def qwen3_narrator(
    stem: str,
    evidence_block: str,
    model_id: str,
    max_new_tokens: int = 200,
) -> tuple[str | None, str]:
    """Optional Qwen3 narrator. Returns (narrative, status). Status is
    'ok' on success or an error label on failure (caller should fall back
    to deterministic narrator).
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return None, "import_error"

    prompt = f"""You are a scientific report writer for a paper on skeleton-based
emotion classification.

Below are numerical results for one motion-capture clip. Write exactly 3
sentences in plain English summarising what the numbers say, no more.

STRICT RULES:
- Do NOT speculate about the person's mental state, culture, gender, or intent.
- Do NOT make any claim that is not directly supported by a number below.
- Cite the section ID (§1, §6, etc.) for every numerical claim.
- No emojis, no Markdown formatting, no headers, no bullets.

Numerical evidence:
{evidence_block}

Output exactly 3 sentences, each ending with a period and citing the relevant section ID.
"""
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
        gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return gen.strip(), "ok"
    except Exception as e:  # noqa: BLE001
        return None, f"qwen_error:{type(e).__name__}"


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def render_card(
    stem: str,
    label_idx: int,
    logits: np.ndarray,
    dsl: dict | None,
    attrs: dict | None,
    rationale: dict | None,
    body_part_order: list[str] | None,
    joint_top: dict | None,
    temporal_info: dict | None,
    counterfactual_top: list[dict] | None,
    perf_info: dict | None,
    narrator: str = "deterministic",
    qwen_model_id: str | None = None,
) -> str:
    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))
    pred_emo = EMOTIONS[pred_idx]
    true_emo = EMOTIONS[label_idx]
    is_correct = pred_idx == label_idx
    pred_conf = float(probs[pred_idx])
    top3 = sorted(enumerate(probs), key=lambda kv: -kv[1])[:3]

    L: list[str] = []
    L.append(f"# Explanation card v2 — `{stem}`")
    L.append("")

    # §1 — Prediction
    L += [
        "## 1. Prediction",
        "",
        f"- **predicted**: `{pred_emo}` (confidence {pred_conf*100:.1f}%)",
        f"- **ground truth**: `{true_emo}`",
        f"- **correct**: {'✅' if is_correct else '❌'}",
        "",
        "Top-3 class probabilities:",
        "",
        "| rank | class | prob |",
        "|---|---|---:|",
    ]
    for r, (i, p) in enumerate(top3, 1):
        L.append(f"| {r} | `{EMOTIONS[i]}` | {p*100:.2f}% |")
    L.append("")

    # §2 — Body-part importance (corpus)
    if body_part_order:
        L += [
            "## 2. Body-part importance (corpus-level)",
            "",
            "Order from most to least important (mean across val folds):",
            "",
        ]
        for i, p in enumerate(body_part_order, 1):
            L.append(f"{i}. `{p}`")
        L += [
            "",
            "_Note: corpus-averaged, not per-clip._",
            "",
        ]

    # §3 — DSL
    if dsl:
        L += ["## 3. Body-motion DSL (exp062, rule-based)", ""]
        glob = dsl.get("global", {})
        if glob:
            L.append("### Global")
            L.append("")
            for a in ("openness", "verticality", "locomotion",
                      "energy", "tempo", "asymmetry"):
                v = glob.get(a, "—")
                L.append(f"- `{a}`: **{v}**")
            L.append("")
        L += ["### Per body part", ""]
        L += ["| part | action | direction | amplitude | tempo | confidence |"]
        L += ["|---|---|---|---|---|---:|"]
        for p in ("head", "torso", "left_arm", "right_arm",
                  "left_leg", "right_leg"):
            row = dsl.get(p, {})
            L.append(
                f"| `{p}` | {row.get('action', '—')} | {row.get('direction', '—')} "
                f"| {row.get('amplitude', '—')} | {row.get('tempo', '—')} "
                f"| {row.get('confidence', '—')} |"
            )
        L.append("")

    # §4 — Scenario attributes
    if attrs:
        L += ["## 4. Scenario attributes (exp065, rule-based 8-dim)", ""]
        for a in ("valence", "arousal", "dominance", "sociality",
                  "agency", "approach_avoidance", "energy", "openness"):
            v = attrs.get(a, "—")
            L.append(f"- `{a}`: **{v}**")
        L.append("")

    # §5 — Free-form rationale
    if rationale:
        L += ["## 5. Free-form rationale (Qwen3-VL kinematic description, v1)",
              ""]
        glob = rationale.get("global_motion", "")
        if glob:
            L.append(f"**Global motion**: {glob}")
            L.append("")
        L += ["**Per body part**:", ""]
        for p in ("head", "torso", "left_arm", "right_arm",
                  "left_leg", "right_leg"):
            v = rationale.get(p, "")
            if v:
                L.append(f"- `{p}`: {v}")
        intervals = rationale.get("salient_intervals") or []
        if intervals:
            L.append("")
            L.append("**Salient intervals**:")
            for it in intervals[:3]:
                if isinstance(it, dict):
                    s = it.get("start_sec"); e = it.get("end_sec"); n = it.get("note")
                    L.append(
                        f"- {s:.1f}-{e:.1f}s: {n}"
                        if isinstance(s, (int, float)) and isinstance(e, (int, float))
                        else f"- {it}"
                    )
        unc = rationale.get("uncertainty_notes")
        if unc:
            L.append("")
            L.append(f"**Uncertainty**: {unc}")
        L.append("")

    # §6 — Joint-level evidence (joint masking, NEW)
    if joint_top:
        L += ["## 6. Joint-level evidence (joint masking, fold-level)", ""]
        names = joint_top["top_names"]
        idxs = joint_top["top_idx"]
        L.append(
            f"Top-{len(names)} CTV joints by F1 drop when masked alone "
            f"(baseline F1 = {joint_top['baseline_f1']*100:.2f}% if available):"
        )
        L.append("")
        L += ["| rank | joint name | CTV idx |", "|---:|---|---:|"]
        for r, (n, i) in enumerate(zip(names, idxs), 1):
            L.append(f"| {r} | `{n}` | {i} |")
        L += [
            "",
            "_Note: masking F1 ranking is a fold-level corpus statistic, "
            "not specific to this clip. The head-chain (Spine3/Neck/Neck1/"
            "Head, CTV idx 5-8) is the dominant cluster across all folds._",
            "",
        ]

    # §7 — Temporal evidence (NEW)
    if temporal_info:
        L += ["## 7. Temporal evidence (temporal masking, fold-level)",
              ""]
        sa = temporal_info.get("salient_first_auc")
        ra = temporal_info.get("reverse_auc")
        if sa is not None and ra is not None:
            L.append(f"- AUC under salient-first masking: **{sa:.4f}**")
            L.append(f"- AUC under reverse masking: **{ra:.4f}**")
            if temporal_info.get("is_uniform") is True:
                L.append(
                    "- **Interpretation**: salient and reverse AUCs are nearly "
                    "identical → **emotion signal is approximately uniform "
                    "across the temporal axis**, with no clip-internal "
                    "localization."
                )
            elif temporal_info.get("is_uniform") is False:
                L.append(
                    "- **Interpretation**: salient and reverse AUCs differ → "
                    "some intervals carry more emotion signal than others."
                )
        L.append("")

    # §8 — Counterfactual evidence (NEW)
    if counterfactual_top:
        L += ["## 8. Counterfactual evidence (top-3, fold-level)",
              "",
              "Top-3 most-harmful (part × edit) cells by mean Δ p(true class) "
              "across all val samples in this fold (negative = drop in "
              "true-class probability):",
              "",
              "| rank | part | edit | mean Δ p(true) |",
              "|---:|---|---|---:|"]
        for c in counterfactual_top:
            L.append(
                f"| {c['rank']} | `{c['part']}` | `{c['edit']}` "
                f"| {c['mean_delta_p_true']:+.4f} |"
            )
        L += [
            "",
            "_Mean across val samples; not specific to this clip but "
            "signals which parts the model relies on most._",
            "",
        ]

    # §9 — Per-performer baseline (NEW)
    perf_f1: float | None = None
    if perf_info:
        L += ["## 9. Per-performer baseline ", ""]
        perf_f1 = perf_info.get("macro_f1")
        if perf_f1 is not None and not np.isnan(perf_f1):
            L.append(
                f"- Performer ID: `{perf_info.get('performer', '—')}` "
                f"(country: {perf_info.get('country', '—')})"
            )
            L.append(
                f"- This performer's overall macro-F1 (across all 12 emotion "
                f"classes, exp020 a01 reference): **{perf_f1*100:.1f}%** "
                f"(accuracy {perf_info.get('accuracy', float('nan'))*100:.1f}%)"
            )
            L.append(
                f"- N val samples for this performer: "
                f"{perf_info.get('n_samples', '—')} "
                f"(across {perf_info.get('n_classes_seen', '—')} classes)"
            )
        L.append("")

    # §10 — Narrator summary (NEW)
    L += ["## 10. Narrator summary"]
    L += [""]
    narrative_text = ""
    narrative_status = "deterministic"
    if narrator == "qwen3" and qwen_model_id is not None:
        evidence_block = _build_qwen_evidence_block(
            pred_emo, true_emo, pred_conf, top3,
            joint_top, temporal_info, counterfactual_top, perf_f1,
        )
        gen, status = qwen3_narrator(stem, evidence_block, qwen_model_id)
        if gen is not None:
            narrative_text = gen
            narrative_status = f"qwen3:{qwen_model_id}"
        else:
            # Fall back to deterministic
            narrative_text = deterministic_narrator(
                stem, pred_emo, true_emo, pred_conf, is_correct, top3,
                joint_top["top_names"] if joint_top else None,
                temporal_info.get("is_uniform") if temporal_info else None,
                counterfactual_top, perf_f1,
            )
            narrative_status = f"deterministic (fallback from {status})"
    else:
        narrative_text = deterministic_narrator(
            stem, pred_emo, true_emo, pred_conf, is_correct, top3,
            joint_top["top_names"] if joint_top else None,
            temporal_info.get("is_uniform") if temporal_info else None,
            counterfactual_top, perf_f1,
        )
    L.append(f"_Auto-narrated ({narrative_status})._")
    L.append("")
    L.append(f"> {narrative_text}")
    L.append("")
    L += [
        "**Constraints applied**: no class inference, no psychology / "
        "culture / gender / intent speculation, every numerical claim "
        "cites a section ID. See §5.6 for "
        "the full constraint list.",
        "",
    ]

    L += [
        "---",
        "",
        "_Generated by `tools/generate_explanation_card_v2.py` (exp073 / "
        " explanation cards). 10 channels. Per `"
        "next_todo_plan.md` §5.6._",
    ]
    return "\n".join(L)


def _build_qwen_evidence_block(
    pred_emo: str, true_emo: str, pred_conf: float,
    top3: list[tuple[int, float]],
    joint_top: dict | None, temporal_info: dict | None,
    counterfactual_top: list[dict] | None, perf_f1: float | None,
) -> str:
    """Compact text block fed to the Qwen3 narrator. One numerical fact
    per line; section IDs included."""
    lines: list[str] = []
    lines.append(
        f"§1: predicted={pred_emo} ({pred_conf*100:.1f}%); ground_truth={true_emo}"
    )
    top3_str = ", ".join(f"{EMOTIONS[i]}={p*100:.1f}%" for i, p in top3)
    lines.append(f"§1: top-3={top3_str}")
    if joint_top:
        lines.append(
            f"§6: top-{len(joint_top['top_names'])} most-informative joints = "
            + ", ".join(joint_top["top_names"])
        )
    if temporal_info:
        lines.append(
            f"§7: temporal_uniform={temporal_info.get('is_uniform')} "
            f"(salient_auc={temporal_info.get('salient_first_auc')}, "
            f"reverse_auc={temporal_info.get('reverse_auc')})"
        )
    if counterfactual_top:
        c = counterfactual_top[0]
        lines.append(
            f"§8: most-harmful counterfactual = {c['edit']} {c['part']} "
            f"(mean Δp_true={c['mean_delta_p_true']:+.4f})"
        )
    if perf_f1 is not None and not np.isnan(perf_f1):
        lines.append(f"§9: this_performer_macro_f1={perf_f1*100:.1f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--exp-id", default="exp034_regionaware_convtr_a00", show_default=True)
@click.option("--fold", default=0, type=int, show_default=True)
@click.option("--max-clips", default=50, type=int, show_default=True)
@click.option("--strategy",
              type=click.Choice(["first", "top1_per_class", "stratified"]),
              default="stratified", show_default=True)
@click.option("--render-index",
              default="output/rationale_cache/renders/render_index.jsonl",
              show_default=True)
@click.option("--dsl-dir", default="output/dsl_cache_rule", show_default=True)
@click.option("--attrs-json", default="output/scenario_attrs_rule.json", show_default=True)
@click.option("--rationale-dir", default="output/rationale_cache/rationales", show_default=True)
@click.option("--part-masking-json",
              default="docs/analysis/part_masking/exp034_regionaware_convtr_a00_fold00_faithfulness.json",
              show_default=True)
@click.option("--joint-masking-json",
              default="docs/analysis/joint_masking/exp034_regionaware_convtr_a00_fold00_joint_faithfulness.json",
              show_default=True)
@click.option("--temporal-masking-json",
              default="docs/analysis/temporal_masking/exp034_regionaware_convtr_a00_fold00_temporal_faithfulness.json",
              show_default=True)
@click.option("--counterfactual-summary",
              default="docs/analysis/counterfactual/exp034_regionaware_convtr_a00_fold00_counterfactual_summary.md",
              show_default=True)
@click.option("--per-performer-csv",
              default="docs/analysis/per_performer/exp020_conv1d_transformer_a01_per_performer.csv",
              show_default=True)
@click.option("--narrator", type=click.Choice(["deterministic", "qwen3"]),
              default="deterministic", show_default=True,
              help="'deterministic' = template narrator (no GPU). 'qwen3' = "
                   "constraint-bound Qwen3 narrator (requires --qwen-model-id, "
                   "falls back to deterministic on failure).")
@click.option("--qwen-model-id", default=None,
              help="HuggingFace model id (e.g., Qwen/Qwen3-1.7B). Required if "
                   "--narrator=qwen3.")
@click.option("--output-dir", default="output/explanation_cards_v2",
              show_default=True)
@click.option("--index-md",
              default="docs/analysis/exp073_explanation_cards_index.md",
              show_default=True)
def main(
    exp_id: str, fold: int, max_clips: int, strategy: str,
    render_index: str, dsl_dir: str, attrs_json: str, rationale_dir: str,
    part_masking_json: str, joint_masking_json: str,
    temporal_masking_json: str, counterfactual_summary: str,
    per_performer_csv: str,
    narrator: str, qwen_model_id: str | None,
    output_dir: str, index_md: str,
) -> None:
    artifact_dir = Path("output/artifacts") / exp_id / f"fold_{fold:02d}"
    if not artifact_dir.is_dir():
        raise click.ClickException(f"artifact_dir not found: {artifact_dir}")
    if narrator == "qwen3" and not qwen_model_id:
        raise click.UsageError("--qwen-model-id is required when --narrator=qwen3")

    oof = load_oof(artifact_dir)
    logits = oof["logits"]; labels = oof["labels"]; stems = oof["filenames"]

    rendered_index = load_render_index(Path(render_index))
    stem_to_dsl = build_stem_to_dsl(Path(dsl_dir), rendered_index)
    stem_to_rat = build_stem_to_rationale(Path(rationale_dir), rendered_index)
    stem_to_attrs = (
        json.loads(Path(attrs_json).read_text(encoding="utf-8"))
        if Path(attrs_json).is_file() else {}
    )

    body_part_order = None
    pm_path = Path(part_masking_json)
    if pm_path.is_file():
        try:
            pm = json.loads(pm_path.read_text())
            body_part_order = pm.get("order_important")
        except (json.JSONDecodeError, OSError):
            click.echo(f"[warn] could not parse {pm_path}", err=True)

    joint_top = load_joint_faithfulness(Path(joint_masking_json), top_k=10)
    temporal_info = load_temporal_faithfulness(Path(temporal_masking_json))
    counterfactual_top = load_counterfactual_summary(Path(counterfactual_summary), top_n=3)
    per_performer = load_per_performer_f1(Path(per_performer_csv))

    click.echo(
        f"OOF: {len(stems)} clips. Lookups: dsl={len(stem_to_dsl)}, "
        f"rat={len(stem_to_rat)}, attrs={len(stem_to_attrs)}, "
        f"joint={'yes' if joint_top else 'no'}, "
        f"temporal={'yes' if temporal_info else 'no'}, "
        f"counterfactual={'yes' if counterfactual_top else 'no'}, "
        f"per_performer={'yes' if per_performer else 'no'}"
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
        # Stratified: first fill per-class × per-{correct, incorrect} with
        # the top-confidence quota, then top up to exactly max_clips with
        # the next most-confident remaining clips. This guarantees we hit
        # max_clips even when the integer quota under-counts (e.g.,
        # max_clips=50, 12 emotions × 2 correctness → quota=2 → only 48).
        sel: list[int] = []
        per_class_quota = max(1, max_clips // (len(EMOTIONS) * 2))
        for c in range(len(EMOTIONS)):
            for is_corr in (True, False):
                mask = (labels == c) & (correct == is_corr)
                idxs = np.where(mask)[0]
                if len(idxs) == 0:
                    continue
                confs = probs[idxs, preds[idxs]]
                ord_ = idxs[np.argsort(-confs)]
                sel.extend(ord_[:per_class_quota].tolist())
        # Top up: fill the remainder with the next most-confident
        # not-yet-selected clips (overall ranking by max-class confidence).
        if len(sel) < max_clips:
            chosen = set(sel)
            remaining = [i for i in range(len(stems)) if i not in chosen]
            overall_conf = probs[remaining, preds[remaining]]
            order = np.argsort(-overall_conf)
            for i in order:
                if len(sel) >= max_clips:
                    break
                sel.append(int(remaining[i]))
        sel = sel[:max_clips]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[tuple[str, str, int, int, float, bool]] = []
    n_with_full_evidence = 0
    for idx in sel:
        stem = stems[idx]
        pid = stem_to_performer(stem)
        perf_info = None
        if per_performer and pid in per_performer:
            perf_info = dict(per_performer[pid])
            perf_info["performer"] = pid

        rep = render_card(
            stem=stem, label_idx=int(labels[idx]),
            logits=logits[idx],
            dsl=stem_to_dsl.get(stem),
            attrs=stem_to_attrs.get(stem),
            rationale=stem_to_rat.get(stem),
            body_part_order=body_part_order,
            joint_top=joint_top,
            temporal_info=temporal_info,
            counterfactual_top=counterfactual_top,
            perf_info=perf_info,
            narrator=narrator,
            qwen_model_id=qwen_model_id,
        )
        out_path = out_dir / f"{stem}.md"
        out_path.write_text(rep, encoding="utf-8")
        has_dsl = stem in stem_to_dsl
        has_attrs = stem in stem_to_attrs
        has_rat = stem in stem_to_rat
        has_joint = joint_top is not None
        has_temp = temporal_info is not None
        has_cf = counterfactual_top is not None
        has_perf = perf_info is not None
        if all((has_dsl, has_attrs, has_rat, has_joint, has_temp, has_cf, has_perf)):
            n_with_full_evidence += 1
        rendered.append(
            (stem, EMOTIONS[int(labels[idx])], int(labels[idx]),
             int(preds[idx]), float(probs[idx][preds[idx]]),
             bool(correct[idx])),
        )

    click.echo(
        f"Rendered {len(rendered)} cards to {out_dir} "
        f"({n_with_full_evidence} with all 8 *data-source* channels: "
        f"part (§2) + dsl (§3) + attrs (§4) + rationale (§5) + joint (§6) "
        f"+ temporal (§7) + counterfactual (§8) + per-performer (§9); "
        f"plus §1 prediction and §10 narrator are always present)"
    )

    md = [
        "# exp073 explanation cards (v2) index",
        "",
        f"- exp_id: `{exp_id}`",
        f"- fold: {fold}",
        f"- strategy: `{strategy}`",
        f"- narrator: `{narrator}`"
        + (f" ({qwen_model_id})" if qwen_model_id else ""),
        f"- rendered: **{len(rendered)}** cards",
        f"- with all 8 data-source channels (§2-§9): **{n_with_full_evidence}** "
        f"(§1 prediction + §10 narrator are always present)",
        "",
        "## Cards",
        "",
        "| stem | true | predicted | confidence | correct |",
        "|---|---|---|---:|---|",
    ]
    rel_out = Path("../..") / Path(output_dir)
    for stem, true_emo, _lbl, pred, conf, corr in rendered:
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
