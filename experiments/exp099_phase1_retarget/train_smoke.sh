#!/usr/bin/env bash
# train_smoke.sh — Stage 3 of exp099
#
# 7 SkateFormer fold-runs across 3 modes:
#   - rotation_6d (3 folds): production-replica baseline
#   - jointpos_honest (3 folds): primary candidate; honest per-clip FK
#   - jointpos_flat (1 fold = fold 00): side-by-side for FK-rule choice
#
# Each run is ~50 min × 7 ≈ 6 hr serial. wandb project:
# acii2026-retarget-phase1-smoke.
#
# Usage:
#   bash experiments/exp099_phase1_retarget/train_smoke.sh \
#     | tee experiments/exp099_phase1_retarget/stage3_smoke.log

set -euo pipefail
cd "$(dirname "$0")/../.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acii2026

PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONPATH

SCRIPT="experiments/exp099_phase1_retarget/run_smoke.py"

run () {
    local mode=$1
    local fold=$2
    echo "================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START mode=$mode fold=$fold"
    echo "================================================================"
    python "$SCRIPT" mode="$mode" fold="$fold" dry_run=False
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] END mode=$mode fold=$fold"
}

# Configs (matches user decision A "Both, side-by-side" + B "Matched rotation_6d")
run rotation_6d 0
run rotation_6d 1
run rotation_6d 2
run jointpos_honest 0
run jointpos_honest 1
run jointpos_honest 2
run jointpos_flat 0

echo "================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL 7 SMOKE RUNS DONE"
echo "================================================================"
