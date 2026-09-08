#!/bin/bash
# exp061 v2 candidate generation FULL RUN launcher.
#
# **WARNING**: This script generates ~24,000 rationale candidates
# (7975 clips × 3 variants) and is expected to take **60-100 GPU-hours**
# on a single 97GB Blackwell. Do NOT launch unless:
#
# 1. tools/generate_rationale_candidates_qwen.py --max-clips 30 has produced
#    a clean smoke (≥ 90% ok rate, 0 leakage hits, variant Jaccard < 0.85).
# 2. The user has explicitly confirmed the time/GPU investment.
# 3. nvidia-smi reports ≥ 70 GiB free.
#
# Outputs:
#   - output/rationale_cache_v2/rationales/<rat_key>__{a,b,c}.json
#   - output/rationale_cache_v2/audit_log.jsonl
#   - logs/exp061_v2_full.log
#
# After completion:
#   bash tools/run_exp061_gate_test.sh
#
# Usage:
#   nohup bash tools/run_exp061_full_generation.sh > /dev/null 2>&1 &

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acii2026
cd "$(git rev-parse --show-toplevel)"

REQUIRED_FREE_MIB=70000
echo "=== exp061 v2 FULL RUN $(date -Iseconds) ==="

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ -z "$FREE_MIB" ] || [ "$FREE_MIB" -lt "$REQUIRED_FREE_MIB" ]; then
  echo "FAIL: GPU free=${FREE_MIB:-?} MiB < ${REQUIRED_FREE_MIB} MiB required. Aborting."
  exit 1
fi
echo "GPU OK: ${FREE_MIB} MiB free"

# Run all 3 variants. Idempotent — already-generated candidates are skipped.
python tools/generate_rationale_candidates_qwen.py \
    --split train \
    --variants a,b,c \
    --batch-size 16 --max-num-seqs 4 \
    --gpu-mem-util 0.85 \
    --temperature 0.5 --top-p 0.95 \
    --max-new-tokens 6144 \
    2>&1 | tee logs/exp061_v2_full.log

echo "=== exp061 v2 FULL RUN COMPLETE $(date -Iseconds) ==="
echo "Next: bash tools/run_exp061_gate_test.sh"
