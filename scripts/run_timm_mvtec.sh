#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STUDENT_BACKBONES=("mobilenetv3_small_100" "ghostnet_100" "repvgg_a0" "repvgg_a1" "mobilevit_s" "fasternet_t0")
DEFAULT_FEATURE_OUT_INDICES="[1, 2, 3, 4]"

DATASET="mvtec"
# Replace with your local MVTec root path before running.
TRAIN_DATA_PATH="<REPLACE_WITH_YOUR_MVTEC_PATH>"
DEVICE="cuda:0"

if [[ "${TRAIN_DATA_PATH}" == *"REPLACE_WITH_YOUR_MVTEC_PATH"* ]]; then
  echo "Please set TRAIN_DATA_PATH to your real MVTec path in scripts/run_timm_mvtec.sh"
  exit 1
fi

for STUDENT in "${STUDENT_BACKBONES[@]}"; do
  FEATURE_OUT_INDICES="${DEFAULT_FEATURE_OUT_INDICES}"

  if [[ "${STUDENT}" == "fasternet_t0" ]]; then
    FEATURE_OUT_INDICES="[0, 1, 2, 3]"
  fi

  python "${ROOT_DIR}/train_timm_student.py" \
    --config "${ROOT_DIR}/configs/default.yaml" \
    --student_backbone "${STUDENT}" \
    --feature_out_indices "${FEATURE_OUT_INDICES}" \
    --dataset "${DATASET}" \
    --train_data_path "${TRAIN_DATA_PATH}" \
    --save_path "${ROOT_DIR}/experiments/timm_distill/${DATASET}/${STUDENT}" \
    --device "${DEVICE}"
done
