#!/usr/bin/env bash
set -euo pipefail

STUDENT_BACKBONES=("mobilenetv3_small_100" "ghostnet_100" "repvgg_a0" "repvgg_a1" "mobilevit_s" "fasternet_t0")
DEFAULT_FEATURE_OUT_INDICES="[1, 2, 3, 4]"

DATASET="visa"
TRAIN_DATA_PATH="/path/to/visa"
DEVICE="cuda:0"

for STUDENT in "${STUDENT_BACKBONES[@]}"; do
  FEATURE_OUT_INDICES="${DEFAULT_FEATURE_OUT_INDICES}"

  if [[ "${STUDENT}" == "fasternet_t0" ]]; then
    FEATURE_OUT_INDICES="[0, 1, 2, 3]"
  fi

  python /home/runner/work/tinyad2/tinyad2/train_timm_student.py \
    --config /home/runner/work/tinyad2/tinyad2/configs/default.yaml \
    --student_backbone "${STUDENT}" \
    --feature_out_indices "${FEATURE_OUT_INDICES}" \
    --dataset "${DATASET}" \
    --train_data_path "${TRAIN_DATA_PATH}" \
    --save_path "/home/runner/work/tinyad2/tinyad2/experiments/timm_distill/${DATASET}/${STUDENT}" \
    --device "${DEVICE}"
done
