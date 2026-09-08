#!/usr/bin/env bash
# exp080 scale_a00 full 10-fold runner. Sequential (single GPU).
set -e
set -o pipefail
cd "$(dirname "$0")/../../"
LOG_ROOT=logs
mkdir -p "$LOG_ROOT"
for fold in 00 01 02 03 04 05 06 07 08 09; do
  LOG="${LOG_ROOT}/exp080_scale_a00_fold${fold}.log"
  WORK="output/artifacts/exp034_regionaware_convtr_scale_a00/fold_${fold}"
  if [ -f "${WORK}/metrics.json" ] && grep -q '"test_f1"' "${WORK}/metrics.json" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] SKIP fold ${fold} (already done)"
    continue
  fi
  echo "[$(date +%H:%M:%S)] START fold ${fold} -> ${LOG}"
  conda run -n acii2026 --no-capture-output \
    python -m experiments.exp034_regionaware_convtr.run \
      fold=${fold} exp_tag=scale_a00 \
      dim=224 num_conv_blocks=3 num_cross_blocks=2 num_heads=4 mlp_ratio=2.0 \
      drop_rate=0.35 late_dropout=0.80 late_dropout_start_step=1000 \
      label_smoothing=0.30 lr=3.0e-3 weight_decay=1.0e-3 warmup_epochs=5 \
      > "${LOG}" 2>&1
  if [ -f "${WORK}/metrics.json" ] && grep -q '"test_f1"' "${WORK}/metrics.json"; then
    echo "[$(date +%H:%M:%S)] OK fold ${fold}"
  else
    echo "[$(date +%H:%M:%S)] FAIL fold ${fold} (see ${LOG})"
    tail -20 "${LOG}"
    exit 2
  fi
done
echo "[$(date +%H:%M:%S)] All folds done."
