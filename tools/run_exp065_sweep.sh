#!/bin/bash
# exp065 3-fold × 3 λ-values sweep launcher.
#
# Runs:
#   exp=a00 (λ=0.05) × fold ∈ {0, 1, 2}
#   exp=a01 (λ=0.10) × fold ∈ {0, 1, 2}
#   exp=a02 (λ=0.20) × fold ∈ {0, 1, 2}
# = 9 sequential runs, ~30 min each, ~5 GPU-h total.
#
# Each run trains the region_aware_conv1d_transformer backbone with 8
# scenario-attribute auxiliary heads, total params ≈ 1.04 M, batch=128,
# max_epochs=80 (early-stop on val_loss patience=15).
#
# Output dir per run: output/artifacts/exp065_qwen_scenario_attributes_<exp_tag>/fold_<NN>/
# Aggregate metrics: extract from each fold_NN/metrics.json after the sweep.
#
# Gate:
#   3-fold mean val_f1 vs exp034 a00 baseline (30.30%) >= +0.20pt
#   OR explainability narrative の質的改善
#
# Usage:
#   nohup bash tools/run_exp065_sweep.sh > /dev/null 2>&1 &

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acii2026
cd "$(git rev-parse --show-toplevel)"

REQUIRED_FREE_MIB=20000
echo "=== exp065 sweep $(date -Iseconds) ==="

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ -z "$FREE_MIB" ] || [ "$FREE_MIB" -lt "$REQUIRED_FREE_MIB" ]; then
  echo "FAIL: GPU free=${FREE_MIB:-?} MiB < ${REQUIRED_FREE_MIB} MiB. Aborting."
  exit 1
fi
echo "GPU OK: ${FREE_MIB} MiB free"

mkdir -p logs/training

# NB: in this codebase, Hydra's `exp=a01` group override doesn't apply (see
# experiments/exp022_conformer).
# Workaround: pass parameters directly via CLI override.

declare -A LAMBDA_FOR
LAMBDA_FOR[a00]=0.05
LAMBDA_FOR[a01]=0.10
LAMBDA_FOR[a02]=0.20

for EXP_TAG in a00 a01 a02; do
  LAMBDA="${LAMBDA_FOR[$EXP_TAG]}"
  for FOLD in 0 1 2; do
    LOGFILE="logs/training/exp065_${EXP_TAG}_fold${FOLD}.log"
    echo ""
    echo "=== exp_tag=${EXP_TAG} fold=${FOLD} lambda_attr=${LAMBDA} START $(date -Iseconds) ==="
    python -m experiments.exp065_qwen_scenario_attributes.run \
        fold=${FOLD} exp_tag=${EXP_TAG} lambda_attr=${LAMBDA} \
        2>&1 | tee "$LOGFILE"
    EC=$?
    echo "=== exp_tag=${EXP_TAG} fold=${FOLD} EXIT=$EC $(date -Iseconds) ==="
    if [ "$EC" -ne 0 ]; then
        echo "ABORT: nonzero exit, stopping sweep."
        exit "$EC"
    fi
  done
done

echo ""
echo "=== ALL exp065 RUNS COMPLETE $(date -Iseconds) ==="
echo "Aggregate via:"
echo "  python -c \"import json, statistics as st; "
echo "    print(json.dumps({"
echo "      tag: ["
echo "        max(m['val_f1'] for m in json.load(open(f'output/artifacts/exp065_qwen_scenario_attributes_{tag}/fold_{f:02d}/metrics.json'))['history'] if 'val_f1' in m)"
echo "        for f in range(3)"
echo "      ] for tag in ['a00','a01','a02']"
echo "    }, indent=2))"
echo "  \""
