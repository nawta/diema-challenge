#!/usr/bin/env bash
# run_preflight.sh — proper pre-flight.
#
# Two fold-0 runs (65 epochs, up_idx=2, corrected corpus) to answer the
# two questions the Stage 3 smoke left open:
#
#   A. jointpos_honest mix=0.0  — no-retarget baseline. Compare to the
#      Stage 3 jointpos_honest mix=0.4 result (0.2237): does retarget help?
#   B. jointpos_flat   mix=0.4  — flat-FK with retarget. Compare to the
#      Stage 3 jointpos_honest mix=0.4 (0.2237): honest vs flat FK?
#
# Artifacts get a mix-ratio suffix so they don't collide.
set -uo pipefail
cd "$(dirname "$0")/../.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acii2026

SCRIPT=experiments/exp099_phase1_retarget/run_smoke.py
ART=output/artifacts/exp099_phase1_smoke

run () {
    local mode=$1 mix=$2 tag=$3
    echo "================================================================"
    echo "[$(date '+%H:%M:%S')] START $mode mix=$mix — 65 epochs, up_idx=2"
    echo "================================================================"
    rm -rf "$ART/${mode}_fold_00"
    python "$SCRIPT" mode="$mode" fold=0 mix_ratio="$mix" max_epochs=65 \
        early_stopping_patience=20 dry_run=False
    if [ -d "$ART/${mode}_fold_00" ]; then
        mv "$ART/${mode}_fold_00" "$ART/${tag}"
        echo "[$(date '+%H:%M:%S')] $mode mix=$mix done → ${tag}"
    else
        echo "[$(date '+%H:%M:%S')] WARNING: $mode mix=$mix produced no artifacts"
    fi
}

# A. no-retarget baseline (honest FK, original-only)
run jointpos_honest 0.0 "jointpos_honest_mix00_fold_00_e65"
# B. flat-FK with retarget
run jointpos_flat   0.4 "jointpos_flat_mix04_fold_00_e65"

echo "================================================================"
echo "[$(date '+%H:%M:%S')] PRE-FLIGHT DONE"
echo "================================================================"
