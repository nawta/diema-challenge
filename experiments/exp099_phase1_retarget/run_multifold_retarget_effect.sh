#!/usr/bin/env bash
# run_multifold_retarget_effect.sh — firm up the retarget effect.
#
# Folds 1 & 2 of jointpos_honest at mix=0.0 (no-retarget) and mix=0.4
# (retarget), 65 ep, up_idx=2. Combined with the fold-0 pre-flight
# (mix0.0=0.2310, mix0.4=0.2237) this gives a 3-fold mean for the
# retarget effect, instead of a fold-0-only conclusion.
set -uo pipefail
cd "$(dirname "$0")/../.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acii2026

SCRIPT=experiments/exp099_phase1_retarget/run_smoke.py
ART=output/artifacts/exp099_phase1_smoke

run () {
    local fold=$1 mix=$2 mixtag=$3
    local tag="jointpos_honest_${mixtag}_fold_$(printf %02d "$fold")_e65"
    echo "================================================================"
    echo "[$(date '+%H:%M:%S')] START jointpos_honest fold=$fold mix=$mix"
    echo "================================================================"
    rm -rf "$ART/jointpos_honest_fold_$(printf %02d "$fold")"
    python "$SCRIPT" mode=jointpos_honest fold="$fold" mix_ratio="$mix" \
        max_epochs=65 early_stopping_patience=20 dry_run=False
    if [ -d "$ART/jointpos_honest_fold_$(printf %02d "$fold")" ]; then
        mv "$ART/jointpos_honest_fold_$(printf %02d "$fold")" "$ART/$tag"
        echo "[$(date '+%H:%M:%S')] fold=$fold mix=$mix done → $tag"
    else
        echo "[$(date '+%H:%M:%S')] WARNING: fold=$fold mix=$mix no artifacts"
    fi
}

# mix=0.0 runs first (faster, original-only), then mix=0.4
run 1 0.0 mix00
run 2 0.0 mix00
run 1 0.4 mix04
run 2 0.4 mix04

echo "================================================================"
echo "[$(date '+%H:%M:%S')] MULTIFOLD RETARGET-EFFECT DONE"
echo "================================================================"
