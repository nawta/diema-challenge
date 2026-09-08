#!/usr/bin/env bash
# rerun_smoke_upidx2.sh — corrected Stage 3 smoke.
#
# Re-runs the matched-budget comparison after the up-axis fix:
#   rotation_6d fold 0 + jointpos_honest fold 0, both 65 epochs, up_idx=2.
# Artifacts are renamed with an _e65_upidx2 suffix so they don't collide
# with the archived up_idx=1 runs.
set -uo pipefail
cd "$(dirname "$0")/../.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acii2026

SCRIPT=experiments/exp099_phase1_retarget/run_smoke.py
ART=output/artifacts/exp099_phase1_smoke

run () {
    local mode=$1
    echo "================================================================"
    echo "[$(date '+%H:%M:%S')] START $mode fold 0 — 65 epochs, up_idx=2"
    echo "================================================================"
    rm -rf "$ART/${mode}_fold_00"
    python "$SCRIPT" mode="$mode" fold=0 max_epochs=65 \
        early_stopping_patience=20 dry_run=False
    if [ -d "$ART/${mode}_fold_00" ]; then
        mv "$ART/${mode}_fold_00" "$ART/${mode}_fold_00_e65_upidx2"
        echo "[$(date '+%H:%M:%S')] $mode done → ${mode}_fold_00_e65_upidx2"
    else
        echo "[$(date '+%H:%M:%S')] WARNING: $mode produced no artifacts"
    fi
}

run rotation_6d
run jointpos_honest

echo "================================================================"
echo "[$(date '+%H:%M:%S')] CORRECTED SMOKE DONE (up_idx=2, 65 epochs)"
echo "================================================================"
