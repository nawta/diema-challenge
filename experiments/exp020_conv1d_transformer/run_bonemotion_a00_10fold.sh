#!/usr/bin/env bash
# Phase ACC Track A (A2): exp020 Conv1D+Transformer on the bone_motion stream.
# Best production hyperparams (a01: label_smoothing=0.3, drop_rate=0.3) + stream_type=bone_motion in_channels=3.
# exp020 has no time_shift_lags, so bone_motion → 3 base channels (no multiplication).
# See Phase ACC and ~.claude/plans/replicated-gathering-raven.md
set -e
set -o pipefail
cd "$(dirname "$0")/../../"
LOG_ROOT=logs
mkdir -p "$LOG_ROOT"
for fold in 00 01 02 03 04 05 06 07 08 09; do
  LOG="${LOG_ROOT}/exp020_bonemotion_a00_fold${fold}.log"
  WORK="output/artifacts/exp020_conv1d_transformer_bonemotion_a00/fold_${fold}"
  if [ -f "${WORK}/metrics.json" ] && grep -q '"test_f1"' "${WORK}/metrics.json" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] SKIP fold ${fold} (already done)"
    continue
  fi
  echo "[$(date +%H:%M:%S)] START fold ${fold} -> ${LOG}"
  conda run -n acii2026 --no-capture-output \
    python -m experiments.exp020_conv1d_transformer.run \
      fold=${fold} exp_tag=bonemotion_a00 \
      stream_type=bone_motion in_channels=3 \
      label_smoothing=0.3 drop_rate=0.3 late_dropout=0.8 \
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
