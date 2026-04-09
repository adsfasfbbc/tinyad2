#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/experiments/onnx_exports"
IMAGE_SIZE=256

declare -A CKPTS=(
  ["mobilenetv3_small_100"]="<REPLACE_WITH_MOBILENET_STUDENT_CHECKPOINT_PTH>"
  ["ghostnet_100"]="<REPLACE_WITH_GHOSTNET_STUDENT_CHECKPOINT_PTH>"
  ["repvgg_a0"]="<REPLACE_WITH_REPVGG_A0_STUDENT_CHECKPOINT_PTH>"
  ["repvgg_a1"]="<REPLACE_WITH_REPVGG_A1_STUDENT_CHECKPOINT_PTH>"
  ["mobilevit_s"]="<REPLACE_WITH_MOBILEVIT_S_STUDENT_CHECKPOINT_PTH>"
)

mkdir -p "${OUTPUT_DIR}"

for MODEL in "${!CKPTS[@]}"; do
  CKPT="${CKPTS[$MODEL]}"
  if [[ "${CKPT}" == *"REPLACE_WITH_"* ]]; then
    echo "Please replace checkpoint path for ${MODEL} in scripts/export_students_onnx.sh"
    exit 1
  fi

  python "${ROOT_DIR}/export_timm_student_onnx.py" \
    --checkpoint_path "${CKPT}" \
    --student_backbone "${MODEL}" \
    --feature_out_indices "1,2,3,4" \
    --teacher_channels "256,512,1024,2048" \
    --image_size "${IMAGE_SIZE}" \
    --output_path "${OUTPUT_DIR}/${MODEL}.onnx"
done

echo "ONNX export completed: ${OUTPUT_DIR}"
