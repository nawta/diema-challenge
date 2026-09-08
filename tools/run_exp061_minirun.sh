#!/bin/bash
# exp061 v2 candidate generation MINI RUN — 500 clips × 3 variants.
#
# Purpose: produce a power-limited F1 estimate before deciding on full corpus
# generation. Smoke (30 clips) validates the pipeline; mini-run (500 clips)
# gives a 3-fold LPO probe number that is noisy (±2-3 pp) but distinguishes
# 11% (= NO-GO, same as v1) from 14% (= GATE PASS) with reasonable confidence.
#
# Cost: 500 × 3 × ~14.7 s/clip ≈ **6.1 GPU-hours**
# Output:
#   - output/rationale_cache_v2/rationales/<rat_key>__{a,b,c}.json (extends smoke)
#   - logs/exp061_v2_mini.log
#
# Idempotent: already-generated candidates are skipped (the smoke's 30 clips
# are reused for free).
#
# After completion:
#   bash tools/run_exp061_gate_test.sh
#
# Usage:
#   nohup bash tools/run_exp061_minirun.sh > /dev/null 2>&1 &

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acii2026
cd "$(git rev-parse --show-toplevel)"

REQUIRED_FREE_MIB=70000
MAX_CLIPS=500
echo "=== exp061 v2 MINI RUN (${MAX_CLIPS} clips × 3 variants) $(date -Iseconds) ==="

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ -z "$FREE_MIB" ] || [ "$FREE_MIB" -lt "$REQUIRED_FREE_MIB" ]; then
  echo "FAIL: GPU free=${FREE_MIB:-?} MiB < ${REQUIRED_FREE_MIB} MiB required. Aborting."
  exit 1
fi
echo "GPU OK: ${FREE_MIB} MiB free"

python tools/generate_rationale_candidates_qwen.py \
    --split train \
    --variants a,b,c \
    --max-clips ${MAX_CLIPS} \
    --batch-size 16 --max-num-seqs 4 \
    --gpu-mem-util 0.85 \
    --temperature 0.5 --top-p 0.95 \
    --max-new-tokens 6144 \
    2>&1 | tee logs/exp061_v2_mini.log

echo "=== exp061 v2 MINI RUN COMPLETE $(date -Iseconds) ==="
echo "Next: bash tools/run_exp061_gate_test.sh (uses output/rationale_cache_v2/selected/)"
