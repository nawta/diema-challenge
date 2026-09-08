#!/bin/bash
# exp061 smoke review — ranker + diff + leakage check on whatever's been
# generated so far. Safe to run on partial output (will skip groups missing
# variants).
#
# After tools/generate_rationale_candidates_qwen.py --max-clips 30 has
# completed, run this to get:
#   - docs/analysis/exp061_smoke_ranker.md (variant winner distribution,
#     score distribution, rejection counts)
#   - docs/analysis/exp061_smoke_diff.md (per-clip Jaccard / length spread)
#   - docs/analysis/exp061_smoke_leakage.md (must report 0 blocking hits)
#
# Decision gate (smoke):
#   - >= 90% ok rate per variant (target)
#   - Jaccard mean < 0.85 (target — 3 candidates differ enough to rank)
#   - 0 leakage blocks (target — paper-grade requirement)
#
# Pass → consider mini-run (500 clips × 3, ~6h) for noisy F1 estimate.
# Fail → debug or stop.

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acii2026
cd "$(git rev-parse --show-toplevel)"

CANDS="${CANDS:-output/rationale_cache_v2/rationales}"
SELECTED="${SELECTED:-output/rationale_cache_v2/selected_smoke}"
RANKER_REPORT="${RANKER_REPORT:-docs/analysis/exp061_smoke_ranker.md}"
DIFF_REPORT="${DIFF_REPORT:-docs/analysis/exp061_smoke_diff.md}"
LEAK_REPORT="${LEAK_REPORT:-docs/analysis/exp061_smoke_leakage.md}"
LEAK_CSV="${LEAK_CSV:-docs/analysis/exp061_smoke_leakage.csv}"

mkdir -p "$SELECTED" docs/analysis

echo "=== exp061 smoke review $(date -Iseconds) ==="
echo "  candidates dir: $CANDS"
n=$(ls "$CANDS" 2>/dev/null | wc -l)
echo "  candidate files: $n"

if [ "$n" -lt 3 ]; then
    echo "ABORT: <3 candidate files; smoke not yet started or failed early."
    exit 2
fi

echo ""
echo "--- 1/3 ranker ---"
python tools/rank_rationale_candidates.py \
    --candidates-dir "$CANDS" \
    --output-dir "$SELECTED" \
    --report-md "$RANKER_REPORT"

echo ""
echo "--- 2/3 diff (variant diversity) ---"
python tools/diff_rationale_candidates.py \
    --candidates-dir "$CANDS" \
    --report-md "$DIFF_REPORT" \
    --max-clips 5

echo ""
echo "--- 3/3 leakage scan ---"
# v1 leakage scanner returns nonzero exit when hits found; tolerate that
# here so the script always emits a summary instead of dying mid-step.
python tools/check_rationale_leakage.py \
    --rationales "$CANDS" \
    --output-md "$LEAK_REPORT" \
    --output-csv "$LEAK_CSV" || true

echo ""
echo "=== smoke review complete $(date -Iseconds) ==="
echo "  $RANKER_REPORT"
echo "  $DIFF_REPORT"
echo "  $LEAK_REPORT"
echo ""
echo "Decision: PASS if all 3 of"
echo "  (a) ranker report shows >=90% ok candidate rate per variant"
echo "  (b) diff report grand-mean Jaccard < 0.85"
echo "  (c) leakage report says 'BLOCKING hits: 0'"
echo "→ consider 500-clip mini-run (5-6h) for noisy F1 estimate."
