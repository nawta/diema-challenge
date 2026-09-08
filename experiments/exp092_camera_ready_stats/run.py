#!/usr/bin/env python3
"""Deterministic camera-ready statistics for the IEEE ACII MMAC paper.

Inputs (read-only, relative to nested repo root):
- experiments/exp091_ensemble_lma_faithfulness/r4_results.json
- experiments/exp091_ensemble_lma_faithfulness/results_single_model.json
- experiments/exp091_ensemble_lma_faithfulness/results_11way.json
- experiments/exp091_ensemble_lma_faithfulness/r3_results.json
- output/artifacts/exp002_a04_smooth/fold_00/oof_filenames.txt
- output/artifacts/exp002_a04_smooth/fold_01/oof_filenames.txt
- output/artifacts/exp002_a04_smooth/fold_02/oof_filenames.txt
- output/artifacts/exp002_a04_smooth/fold_03/oof_filenames.txt
- output/artifacts/exp002_a04_smooth/fold_04/oof_filenames.txt
- output/artifacts/exp002_a04_smooth/fold_05/oof_filenames.txt
- output/artifacts/exp002_a04_smooth/fold_06/oof_filenames.txt
- output/artifacts/exp002_a04_smooth/fold_07/oof_filenames.txt
- output/artifacts/exp002_a04_smooth/fold_08/oof_filenames.txt
- output/artifacts/exp002_a04_smooth/fold_09/oof_filenames.txt
- output/predictions/c3d_contact_only/test_filenames.npy
- experiments/exp091_ensemble_lma_faithfulness/eval_ensemble_lma.py

Outputs:
- experiments/exp092_camera_ready_stats/summary.md
- experiments/exp092_camera_ready_stats/summary.json
- experiments/exp092_camera_ready_stats
- ../tables/table_member_loo.tex (outer repo root)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
EXP_DIR = SCRIPT_PATH.parent
NESTED_ROOT = SCRIPT_PATH.parents[2]
OUTER_ROOT = SCRIPT_PATH.parents[3]

INPUT_R4 = NESTED_ROOT / "experiments/exp091_ensemble_lma_faithfulness/r4_results.json"
INPUT_SINGLE = NESTED_ROOT / "experiments/exp091_ensemble_lma_faithfulness/results_single_model.json"
INPUT_ENSEMBLE = NESTED_ROOT / "experiments/exp091_ensemble_lma_faithfulness/results_11way.json"
INPUT_R3 = NESTED_ROOT / "experiments/exp091_ensemble_lma_faithfulness/r3_results.json"
INPUT_OOF_DIR = NESTED_ROOT / "output/artifacts/exp002_a04_smooth"
INPUT_TEST_FILENAMES = NESTED_ROOT / "output/predictions/c3d_contact_only/test_filenames.npy"
INPUT_EVAL = NESTED_ROOT / "experiments/exp091_ensemble_lma_faithfulness/eval_ensemble_lma.py"

OUTPUT_SUMMARY_MD = EXP_DIR / "summary.md"
OUTPUT_SUMMARY_JSON = EXP_DIR / "summary.json"
OUTPUT_NOTES = EXP_DIR / "camera_ready_notes.md"
OUTPUT_MEMBER_TABLE = OUTER_ROOT / "tables/table_member_loo.tex"
OUTPUT_SUP_MEMBER_STRATA_TABLE = OUTER_ROOT / "tables/sup_table_member_loo_strata.tex"

MEMBER_INFO = {
    "exp002_a04_smooth": ("STGCN++", "Graph-conv."),
    "exp003_ctrgcn_a00": ("CTR-GCN", "Graph-conv."),
    "exp004_skateformer_a01": ("SkateFormer", "Attention"),
    "exp006_protogcn_a00": ("ProtoGCN", "Graph-conv."),
    "exp020_conv1d_transformer_a01": ("Conv1D+Transformer", "Hybrid/MLP"),
    "exp023_keypoint_pool_mlp_a00": ("Keypoint-Pool-MLP", "Hybrid/MLP"),
    "exp034_regionaware_convtr_a00": ("Region-Aware ConvTr", "Graph-conv."),
    "motionbert": ("MotionBERT-Lite", "External (frozen)"),
    "c3d": ("C3D-marker-stats", "External (frozen)"),
    "mamp60": ("MAMP-NTU60", "External (frozen)"),
    "mamp120": ("MAMP-NTU120", "External (frozen)"),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def record_assert(assertions: dict[str, bool], name: str, condition: bool) -> None:
    assertions[name] = bool(condition)


def sign_test(values_by_emotion: dict[str, float]) -> dict[str, Any]:
    values = [float(v) for v in values_by_emotion.values()]
    n_positive = sum(v > 0.0 for v in values)
    n_negative = sum(v < 0.0 for v in values)
    n_zero = sum(v == 0.0 for v in values)
    n_used = n_positive + n_negative
    k = n_positive
    denominator = 2**n_used
    p_ge = sum(math.comb(n_used, i) for i in range(k, n_used + 1)) / denominator
    p_le = sum(math.comb(n_used, i) for i in range(0, k + 1)) / denominator
    return {
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": n_zero,
        "n_used": n_used,
        "k": k,
        "median_rho": float(np.median(values)),
        "one_sided_greater_p": float(p_ge),
        "two_sided_p": float(min(1.0, 2.0 * min(p_ge, p_le))),
        "values_by_emotion": values_by_emotion,
    }


def member_rows(r4: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for member in r4["members"]:
        display, family = MEMBER_INFO[member]
        deltas = {
            stratum: float(value)
            for stratum, value in r4["results"]["lomo"][member]["delta_vs_11way_pp"].items()
        }
        delta = deltas["all"]
        rows.append(
            {
                "internal_id": member,
                "display_name": display,
                "family": family,
                "delta_all": delta,
                "delta_all_rounded": f"{delta:+.2f}",
                "delta_vs_11way_pp": deltas,
            }
        )
    return sorted(rows, key=lambda row: row["delta_all"])


def read_oof_audit() -> dict[str, Any]:
    performers: set[str] = set()
    clip_counts: dict[str, int] = {}
    fold_line_counts: dict[str, int] = {}
    total_lines = 0
    for fold in range(10):
        fold_name = f"fold_{fold:02d}"
        path = INPUT_OOF_DIR / fold_name / "oof_filenames.txt"
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        fold_line_counts[fold_name] = len(lines)
        total_lines += len(lines)
        for filename in lines:
            parts = filename.split("_")
            performer = "_".join(parts[:2])
            performers.add(performer)
            clip_counts[performer] = clip_counts.get(performer, 0) + 1

    jp_01_05_absent = {f"JP_{i:02d}": f"JP_{i:02d}" not in performers for i in range(1, 6)}
    jp_06_12_present = [f"JP_{i:02d}" for i in range(6, 13) if f"JP_{i:02d}" in performers]
    jp_06_12_clip_count = sum(clip_counts.get(f"JP_{i:02d}", 0) for i in range(6, 13))

    test_filenames = np.load(INPUT_TEST_FILENAMES, allow_pickle=True)
    test_entries = [str(x) for x in test_filenames.tolist()]
    has_performer_prefix = any(("JP_" in entry or "TW_" in entry) for entry in test_entries)

    return {
        "total_oof_line_count": total_lines,
        "fold_line_counts": fold_line_counts,
        "total_unique_performers": len(performers),
        "n_jp": sum(p.startswith("JP_") for p in performers),
        "n_tw": sum(p.startswith("TW_") for p in performers),
        "jp_01_05_absent": jp_01_05_absent,
        "jp_06_12_present": jp_06_12_present,
        "jp_06_12_clip_count": jp_06_12_clip_count,
        "test_filename_count": len(test_entries),
        "test_filenames_contain_performer_prefix": has_performer_prefix,
        "test_filename_examples": test_entries[:5],
        "finding": (
            "The released test filenames are anonymized as testNNNN, so performer identity is "
            "not recoverable from the test list; therefore JP_01-05 presence in the test split "
            "cannot be confirmed from filenames. Absence from training is established from the "
            "OOF side only."
        ),
    }


def find_line(lines: list[str], needle: str) -> int:
    for index, line in enumerate(lines, start=1):
        if needle in line:
            return index
    raise ValueError(f"Could not find line containing: {needle}")


def find_line_after(lines: list[str], needle: str, start_line: int) -> int:
    for index, line in enumerate(lines[start_line - 1 :], start=start_line):
        if needle in line:
            return index
    raise ValueError(f"Could not find line containing after line {start_line}: {needle}")


def code_at(lines: list[str], line_no: int) -> str:
    return lines[line_no - 1].strip()


def permutation_description(single: dict[str, Any], ensemble: dict[str, Any]) -> dict[str, Any]:
    lines = INPUT_EVAL.read_text(encoding="utf-8").splitlines()
    rel = INPUT_EVAL.relative_to(NESTED_ROOT)
    line_def_spearman = find_line(lines, "def per_emotion_spearman")
    line_loop_emotion = find_line_after(lines, "for c in range(12):", line_def_spearman)
    line_def_perm = find_line(lines, "def permutation_null")
    line_perm = find_line(lines, "perm = rng.permutation(4)")
    line_shuffle = find_line(lines, "shuffled[c] = deep_reg[c, perm]")
    line_mean = find_line(lines, "nulls[k] = float(np.mean(per_emotion_spearman(lma_reg, shuffled)))")

    description = (
        f"{rel}:{line_def_spearman} `{code_at(lines, line_def_spearman)}` defines per-emotion "
        f"Spearman over the 12 emotion rows, and {rel}:{line_loop_emotion} "
        f"`{code_at(lines, line_loop_emotion)}` iterates one emotion at a time. "
        f"In permutation_null ({rel}:{line_def_perm} `{code_at(lines, line_def_perm)}`), "
        f"{rel}:{line_perm} `{code_at(lines, line_perm)}` permutes the four region columns "
        f"independently within each emotion, {rel}:{line_shuffle} `{code_at(lines, line_shuffle)}` "
        f"applies that permutation to deep_reg, and {rel}:{line_mean} `{code_at(lines, line_mean)}` "
        "recomputes the mean per-emotion Spearman for each null draw. With n_perm=1000 and "
        "seed=42, the stored p_value is the one-sided smoothed tail (1 + count(null >= observed)) "
        "/ (n_perm + 1)."
    )
    return {
        "description": description,
        "source_lines": {
            "per_emotion_spearman_def": f"{rel}:{line_def_spearman}",
            "emotion_loop": f"{rel}:{line_loop_emotion}",
            "permutation_null_def": f"{rel}:{line_def_perm}",
            "region_permutation": f"{rel}:{line_perm}",
            "apply_permutation": f"{rel}:{line_shuffle}",
            "null_mean": f"{rel}:{line_mean}",
        },
        "single_model_stored_p_value": float(single["permutation_null"]["p_value"]),
        "ensemble_stored_p_value": float(ensemble["permutation_null"]["p_value"]),
        "paper_phrase_verdict": (
            "Accurate: regions are shuffled within emotion, and 1/1001 = 0.000999000999... "
            "is below 0.001; this is the one-sided floor value under 1000 permutations with "
            "+1 smoothing."
        ),
    }


def latex_int(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def write_member_table(
    rows: list[dict[str, Any]], external_delta: float, r3: dict[str, Any], audit: dict[str, Any]
) -> None:
    full = r3["results"]["raw_logit_mean"]["full"]
    pooled_pct = float(full["pooled_f1"]) * 100.0
    per_fold_mean_pct = float(full["per_fold_mean"]) * 100.0
    per_fold_std_pct = float(full["per_fold_std"]) * 100.0
    oof_n = latex_int(int(audit["total_oof_line_count"]))
    caption = (
        "Leave-one-out ablation of the 11-member ensemble: change in pooled-OOF Macro-F1 "
        f"(pp, $n\\,{{=}}\\,{oof_n}$, 74-performer split) when one member is removed "
        "from the logit-mean fusion (last row: the whole frozen external block). "
        f"Pooled-OOF base ${pooled_pct:.2f}\\%$ (per-fold headline: "
        f"${per_fold_mean_pct:.2f}\\,{{\\pm}}\\,{per_fold_std_pct:.2f}\\%$). "
        "All deltas are negative."
    )

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\caption{{{caption}}}",
        "\\label{tab:member_loo}",
        "\\begin{tabular}{@{}l l r@{}}",
        "\\toprule",
        "Member & Family & $\\Delta$ Macro-F1 (pp) \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['display_name']} & {row['family']} & ${float(row['delta_all']):+.2f}$ \\\\"
        )
    lines.extend(
        [
            "\\midrule",
            f"All frozen externals (4) & External (frozen) & ${external_delta:+.2f}$ \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    OUTPUT_MEMBER_TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sup_member_strata_table(rows: list[dict[str, Any]], external_deltas: dict[str, float]) -> None:
    caption = (
        "Per-stratum leave-one-out deltas (pp, pooled OOF) for the 11-member logit-mean ensemble. "
        "`Excl.\\ earliest JP' scores after removing the seven earliest-captured Japanese training performers "
        "(JP\\_06--JP\\_12)."
    )
    strata = [
        ("all", "All"),
        ("JP_only", "JP"),
        ("TW_only", "TW"),
        ("OptiTrack_excluded", "Excl.\\ earliest JP"),
    ]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{{caption}}}",
        "\\label{tab:sup_member_loo_strata}",
        "\\begin{tabular}{@{}l l r r r r@{}}",
        "\\toprule",
        "Member & Family & " + " & ".join(label for _, label in strata) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        deltas = row["delta_vs_11way_pp"]
        values = " & ".join(f"${deltas[key]:+.2f}$" for key, _ in strata)
        lines.append(f"{row['display_name']} & {row['family']} & {values} \\\\")
    external_values = " & ".join(f"${external_deltas[key]:+.2f}$" for key, _ in strata)
    lines.extend(
        [
            "\\midrule",
            f"All frozen externals (4) & External (frozen) & {external_values} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )
    OUTPUT_SUP_MEMBER_STRATA_TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(data: dict[str, Any]) -> None:
    rows = data["member_table_rows"]
    single = data["sign_tests"]["single"]
    ensemble = data["sign_tests"]["ensemble"]
    audit = data["performer_audit"]
    perm = data["permutation_null"]
    assertions = data["assertions"]

    lines = [
        "# Camera-ready statistics summary",
        "",
        "## Sign tests",
        "",
        (
            f"- Single model: n_positive={single['n_positive']}, n_negative={single['n_negative']}, "
            f"n_zero={single['n_zero']}, median rho={single['median_rho']:.2f}, "
            f"one-sided p={single['one_sided_greater_p']:.8f}, two-sided p={single['two_sided_p']:.8f}."
        ),
        (
            f"- Eleven-member ensemble: n_positive={ensemble['n_positive']}, "
            f"n_negative={ensemble['n_negative']}, n_zero={ensemble['n_zero']}, "
            f"median rho={ensemble['median_rho']:.2f}, "
            f"one-sided p={ensemble['one_sided_greater_p']:.8f}, "
            f"two-sided p={ensemble['two_sided_p']:.8f}."
        ),
        "",
        "## Member leave-one-out table rows",
        "",
        "| Member | Family | Delta Macro-F1 (pp) |",
        "| --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['display_name']} | {row['family']} | {row['delta_all_rounded']} |")
    lines.extend(
        [
            (
                f"| All frozen externals (4) | Frozen external pretraining | "
                f"{data['external_block_delta_rounded']} |"
            ),
            "",
            "## Performer-ID audit",
            "",
            (
                f"- OOF total line count: {audit['total_oof_line_count']}; unique performers: "
                f"{audit['total_unique_performers']} ({audit['n_jp']} JP, {audit['n_tw']} TW)."
            ),
            (
                "- JP_01..JP_05 absent from OOF/training filenames: "
                + ", ".join(f"{k}={v}" for k, v in audit["jp_01_05_absent"].items())
                + "."
            ),
            (
                f"- JP_06..JP_12 present in OOF/training filenames: "
                f"{', '.join(audit['jp_06_12_present'])}; combined clip count "
                f"{audit['jp_06_12_clip_count']}."
            ),
            (
                f"- Test filename entries: {audit['test_filename_count']}; contains JP_/TW_ prefixes: "
                f"{audit['test_filenames_contain_performer_prefix']}."
            ),
            f"- Finding: {audit['finding']}",
            "",
            "## Permutation-null structure",
            "",
            perm["description"],
            "",
            (
                f"Stored permutation p-values: single={perm['single_model_stored_p_value']:.15f}; "
                f"ensemble={perm['ensemble_stored_p_value']:.15f}."
            ),
            perm["paper_phrase_verdict"],
            "",
            "## Cross-checks",
            "",
            (
                f"- All eleven all-stratum LOMO deltas are strictly negative: "
                f"{assertions['all_lomo_deltas_strictly_negative']}."
            ),
            (
                f"- Minimum rounded magnitude is 0.09 pp for CTR-GCN: "
                f"{assertions['min_magnitude_ctr_gcn_0_09']}."
            ),
            (
                f"- Maximum rounded magnitude is 0.84 pp for MAMP-NTU60: "
                f"{assertions['max_magnitude_mamp60_0_84']}."
            ),
            (
                f"- Rounded magnitude range is 0.09--0.84 pp: "
                f"{assertions['rounded_magnitude_range_0_09_0_84']}."
            ),
            "",
            "## Assertion verdicts",
            "",
        ]
    )
    for name in sorted(assertions):
        lines.append(f"- {name}: {'PASS' if assertions[name] else 'FAIL'}")
    OUTPUT_SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_session_notes(data: dict[str, Any]) -> None:
    rows = data["member_table_rows"]
    single = data["sign_tests"]["single"]
    ensemble = data["sign_tests"]["ensemble"]
    audit = data["performer_audit"]
    lines = [
        "# Session notes",
        "",
        "Purpose: deterministic offline camera-ready statistics for the accepted IEEE paper.",
        "",
        "Date: 2026-07-12",
        "",
        "## Input files",
        "",
        f"- {INPUT_R4}",
        f"- {INPUT_SINGLE}",
        f"- {INPUT_ENSEMBLE}",
        f"- {INPUT_R3}",
        f"- {INPUT_OOF_DIR}/fold_00..09/oof_filenames.txt",
        f"- {INPUT_TEST_FILENAMES}",
        f"- {INPUT_EVAL}",
        "",
        "## Output files",
        "",
        f"- {OUTPUT_SUMMARY_MD}",
        f"- {OUTPUT_SUMMARY_JSON}",
        f"- {OUTPUT_NOTES}",
        f"- {OUTPUT_MEMBER_TABLE}",
        f"- {OUTPUT_SUP_MEMBER_STRATA_TABLE}",
        "",
        "## Rerun",
        "",
        (
            "# from the repository root:\n"
            "uv run python experiments/exp092_camera_ready_stats/run.py"
        ),
        "",
        "## Key numbers",
        "",
        (
            f"- Single sign test: {single['n_positive']} positive, {single['n_negative']} negative, "
            f"{single['n_zero']} zero; median rho={single['median_rho']:.2f}; "
            f"one-sided p={single['one_sided_greater_p']:.8f}; "
            f"two-sided p={single['two_sided_p']:.8f}."
        ),
        (
            f"- Ensemble sign test: {ensemble['n_positive']} positive, {ensemble['n_negative']} negative, "
            f"{ensemble['n_zero']} zero; median rho={ensemble['median_rho']:.2f}; "
            f"one-sided p={ensemble['one_sided_greater_p']:.8f}; "
            f"two-sided p={ensemble['two_sided_p']:.8f}."
        ),
        "- Leave-one-out deltas:",
    ]
    for row in rows:
        lines.append(f"  - {row['display_name']}: {row['delta_all_rounded']} pp")
    lines.extend(
        [
            f"- External-block delta: {data['external_block_delta_rounded']} pp.",
            (
                f"- Performer audit: {audit['total_unique_performers']} unique performers "
                f"({audit['n_jp']} JP, {audit['n_tw']} TW), {audit['total_oof_line_count']} OOF lines."
            ),
            (
                f"- JP_01..JP_05 absent from OOF/training filenames; JP_06..JP_12 all present with "
                f"{audit['jp_06_12_clip_count']} combined clips."
            ),
            (
                f"- Test filenames are anonymized ({audit['test_filename_count']} entries) and contain "
                f"performer prefixes: {audit['test_filenames_contain_performer_prefix']}."
            ),
        ]
    )
    OUTPUT_NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    assertions: dict[str, bool] = {}

    r4 = load_json(INPUT_R4)
    single = load_json(INPUT_SINGLE)
    ensemble = load_json(INPUT_ENSEMBLE)
    r3 = load_json(INPUT_R3)

    record_assert(assertions, "member_ids_match_mapping_keys", set(r4["members"]) == set(MEMBER_INFO))

    single_sign = sign_test(single["per_emotion_spearman"])
    ensemble_sign = sign_test(ensemble["per_emotion_spearman_ensemble"])
    record_assert(assertions, "single_sign_n_positive", single_sign["n_positive"] == 10)
    record_assert(assertions, "single_sign_n_negative", single_sign["n_negative"] == 2)
    record_assert(assertions, "single_sign_n_zero", single_sign["n_zero"] == 0)
    record_assert(assertions, "single_sign_median", math.isclose(single_sign["median_rho"], 0.70, abs_tol=1e-12))
    record_assert(
        assertions,
        "single_sign_one_sided_p",
        math.isclose(single_sign["one_sided_greater_p"], 0.019287109375, abs_tol=1e-15),
    )
    record_assert(
        assertions,
        "single_sign_two_sided_p",
        math.isclose(single_sign["two_sided_p"], 0.03857421875, abs_tol=1e-15),
    )
    record_assert(assertions, "ensemble_sign_n_positive", ensemble_sign["n_positive"] == 11)
    record_assert(assertions, "ensemble_sign_n_negative", ensemble_sign["n_negative"] == 0)
    record_assert(assertions, "ensemble_sign_n_zero", ensemble_sign["n_zero"] == 1)
    record_assert(assertions, "ensemble_sign_n_used", ensemble_sign["n_used"] == 11)
    record_assert(assertions, "ensemble_sign_k", ensemble_sign["k"] == 11)
    record_assert(
        assertions, "ensemble_sign_median", math.isclose(ensemble_sign["median_rho"], 0.50, abs_tol=1e-12)
    )
    record_assert(
        assertions,
        "ensemble_sign_one_sided_p",
        math.isclose(ensemble_sign["one_sided_greater_p"], 0.00048828125, abs_tol=1e-15),
    )
    record_assert(
        assertions,
        "ensemble_sign_two_sided_p",
        math.isclose(ensemble_sign["two_sided_p"], 0.0009765625, abs_tol=1e-15),
    )

    rows = member_rows(r4)
    external_deltas = {
        stratum: float(value)
        for stratum, value in r4["results"]["group_ablations"]["minus_all_externals"]["delta_vs_11way_pp"].items()
    }
    external_delta = external_deltas["all"]
    record_assert(assertions, "external_block_delta_raw", abs(external_delta - (-2.9072)) < 0.001)
    record_assert(assertions, "external_block_delta_rounds_to_minus_2_91", f"{external_delta:+.2f}" == "-2.91")
    record_assert(assertions, "all_lomo_deltas_strictly_negative", all(row["delta_all"] < 0.0 for row in rows))
    min_mag = min(rows, key=lambda row: abs(row["delta_all"]))
    max_mag = max(rows, key=lambda row: abs(row["delta_all"]))
    record_assert(
        assertions,
        "min_magnitude_ctr_gcn_0_09",
        min_mag["display_name"] == "CTR-GCN" and f"{abs(min_mag['delta_all']):.2f}" == "0.09",
    )
    record_assert(
        assertions,
        "max_magnitude_mamp60_0_84",
        max_mag["display_name"] == "MAMP-NTU60" and f"{abs(max_mag['delta_all']):.2f}" == "0.84",
    )
    rounded_magnitudes = sorted(float(f"{abs(row['delta_all']):.2f}") for row in rows)
    record_assert(
        assertions,
        "rounded_magnitude_range_0_09_0_84",
        rounded_magnitudes[0] == 0.09 and rounded_magnitudes[-1] == 0.84,
    )

    audit = read_oof_audit()
    record_assert(assertions, "audit_unique_performers_74", audit["total_unique_performers"] == 74)
    record_assert(assertions, "audit_n_jp_40", audit["n_jp"] == 40)
    record_assert(assertions, "audit_n_tw_34", audit["n_tw"] == 34)
    record_assert(assertions, "audit_jp_01_05_all_absent", all(audit["jp_01_05_absent"].values()))
    record_assert(
        assertions,
        "audit_jp_06_12_all_present",
        audit["jp_06_12_present"] == [f"JP_{i:02d}" for i in range(6, 13)],
    )
    record_assert(assertions, "audit_jp_06_12_clip_count_756", audit["jp_06_12_clip_count"] == 756)
    record_assert(assertions, "audit_total_oof_line_count_7992", audit["total_oof_line_count"] == 7992)
    record_assert(assertions, "audit_test_filename_count_1944", audit["test_filename_count"] == 1944)
    record_assert(
        assertions,
        "audit_test_filenames_no_performer_prefix",
        audit["test_filenames_contain_performer_prefix"] is False,
    )

    perm = permutation_description(single, ensemble)
    expected_perm_p = 1.0 / 1001.0
    record_assert(
        assertions,
        "single_permutation_p_value_1_over_1001",
        math.isclose(perm["single_model_stored_p_value"], expected_perm_p, abs_tol=1e-18),
    )
    record_assert(
        assertions,
        "ensemble_permutation_p_value_1_over_1001",
        math.isclose(perm["ensemble_stored_p_value"], expected_perm_p, abs_tol=1e-18),
    )

    data = {
        "script_path": str(SCRIPT_PATH),
        "nested_repo_root": str(NESTED_ROOT),
        "outer_repo_root": str(OUTER_ROOT),
        "input_files": [
            str(INPUT_R4),
            str(INPUT_SINGLE),
            str(INPUT_ENSEMBLE),
            str(INPUT_R3),
            str(INPUT_OOF_DIR / "fold_00..09/oof_filenames.txt"),
            str(INPUT_TEST_FILENAMES),
            str(INPUT_EVAL),
        ],
        "output_files": [
            str(OUTPUT_SUMMARY_MD),
            str(OUTPUT_SUMMARY_JSON),
            str(OUTPUT_NOTES),
            str(OUTPUT_MEMBER_TABLE),
            str(OUTPUT_SUP_MEMBER_STRATA_TABLE),
        ],
        "sign_tests": {
            "single": single_sign,
            "ensemble": ensemble_sign,
        },
        "member_table_rows": [
            {
                "display_name": row["display_name"],
                "family": row["family"],
                "delta_all": row["delta_all"],
                "delta_all_rounded": row["delta_all_rounded"],
                "delta_vs_11way_pp": row["delta_vs_11way_pp"],
            }
            for row in rows
        ],
        "external_block_delta": external_delta,
        "external_block_delta_rounded": f"{external_delta:+.2f}",
        "external_block_delta_vs_11way_pp": external_deltas,
        "performer_audit": audit,
        "permutation_null": perm,
        "assertions": assertions,
    }

    write_member_table(rows, external_delta, r3, audit)
    write_sup_member_strata_table(rows, external_deltas)
    OUTPUT_SUMMARY_JSON.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_md(data)
    write_session_notes(data)

    failed = [name for name, passed in assertions.items() if not passed]
    print("Camera-ready stats assertion summary")
    for name in sorted(assertions):
        print(f"{'PASS' if assertions[name] else 'FAIL'} {name}")
    print("")
    print(f"single sign test: +{single_sign['n_positive']} -{single_sign['n_negative']} zero={single_sign['n_zero']}, "
          f"median={single_sign['median_rho']:.2f}, one-sided p={single_sign['one_sided_greater_p']:.8f}, "
          f"two-sided p={single_sign['two_sided_p']:.8f}")
    print(f"ensemble sign test: +{ensemble_sign['n_positive']} -{ensemble_sign['n_negative']} zero={ensemble_sign['n_zero']}, "
          f"median={ensemble_sign['median_rho']:.2f}, one-sided p={ensemble_sign['one_sided_greater_p']:.8f}, "
          f"two-sided p={ensemble_sign['two_sided_p']:.8f}")
    print(f"external block delta: {external_delta:+.2f} pp")
    print(f"OOF audit: performers={audit['total_unique_performers']} JP={audit['n_jp']} TW={audit['n_tw']} "
          f"OOF lines={audit['total_oof_line_count']} test filenames={audit['test_filename_count']}")
    print(f"outputs: {OUTPUT_SUMMARY_MD}, {OUTPUT_SUMMARY_JSON}, {OUTPUT_NOTES}, {OUTPUT_MEMBER_TABLE}")
    if failed:
        print(f"FAIL: {len(failed)} assertions failed")
        return 1
    print(f"PASS: all {len(assertions)} assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
