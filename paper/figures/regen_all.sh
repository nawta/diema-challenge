#!/usr/bin/env bash
# paper/figures/regen_all.sh — deterministic regeneration of every ACII
# 2026 paper figure + table (P12).
#
# Order: P1 style (imported, no output) -> P2..P8 figures -> P9..P11 tables.
# Determinism: every script uses seed_all(42) / fixed text and save_fig()
# strips the PDF CreationDate, so running this twice yields byte-identical
# PDFs (verified by P12). Run from anywhere; cwd is forced to repo root.
#
# Usage:  conda run -n acii2026 bash paper/figures/regen_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"

echo "[regen] repo root: $ROOT"
echo "[regen] python: $($PY -V 2>&1)"

# P0 reconciliation data (Table 1/Fig.2 depend on it; deterministic)
"$PY" paper/reconcile_f1.py

# P2..P8 figures (sorted, fixed order)
for f in fig01_system_overview fig02_lift_path fig03_confusion_jptw \
         fig04_member_corr fig05_saliency fig06_lma_counterfactual \
         fig07_cards ; do
  echo "[regen] figure: $f"
  "$PY" "paper/figures/${f}.py"
done

# P9..P11 tables
for t in build_table1 build_table2_negatives build_table3_explainability ; do
  echo "[regen] table: $t"
  "$PY" "paper/tables/${t}.py"
done

echo "[regen] DONE — $(ls paper/figures/*.pdf | wc -l) figure PDFs, "\
"$(ls paper/tables/*.tex | wc -l) table .tex"

# E1.3 (post-review): gate regen on the test suite so a future row-count /
# colspec / forbidden-token regression cannot land silently.
echo "[regen] running TDD gate"
"$PY" -m pytest tests/test_paper_tables.py tests/test_paper_style.py \
    tests/test_qwen_dataset_anon.py -q
