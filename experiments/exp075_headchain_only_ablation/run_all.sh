#!/usr/bin/env bash
# exp075: head-chain-only ablation matrix.
#   4 variants (a00_top4 / a01_top10 / a02_top10_arms / a03_top10_legs)
#   x 3 folds (0,1,2) = 12 trainings.
# Each fold ≈ 6-7 min @ 40 epochs (435K params, dim=128, num_blocks=2).
# Total ≈ 75-85 min on a single GPU.
set -e
set -o pipefail

cd "$(dirname "$0")/../.."
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-acii2026-diema}

LOG_DIR=logs/training/exp075
mkdir -p "$LOG_DIR"

VARIANTS=(a00_top4 a01_top10 a02_top10_arms a03_top10_legs)
FOLDS=(0 1 2)

for v in "${VARIANTS[@]}"; do
  for f in "${FOLDS[@]}"; do
    LOG="${LOG_DIR}/${v}_fold${f}.log"
    echo "[$(date +%H:%M:%S)] START exp075 ${v} fold ${f} -> ${LOG}"
    conda run -n acii2026 --no-capture-output \
      python experiments/exp075_headchain_only_ablation/run.py \
        exp=${v} fold=${f} \
        > "${LOG}" 2>&1
    if grep -q "Done. Artifacts saved" "${LOG}"; then
      F1=$(grep -oE "test_f1[^0-9]*[0-9.]+" "${LOG}" | tail -1 | grep -oE "[0-9.]+$")
      echo "[$(date +%H:%M:%S)] OK  exp075 ${v} fold ${f}: test_f1=${F1}"
    else
      echo "[$(date +%H:%M:%S)] FAIL exp075 ${v} fold ${f} (see ${LOG})"
      tail -20 "${LOG}"
      exit 2
    fi
  done
done

echo "[$(date +%H:%M:%S)] All 12 trainings completed."
