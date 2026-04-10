#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIGS=(
  "${ROOT_DIR}/benchmark_ad/configs/tinyvit_distill.yaml"
  "${ROOT_DIR}/benchmark_ad/configs/mobilvit_distill.yaml"
  "${ROOT_DIR}/benchmark_ad/configs/fastvit_distill.yaml"
  "${ROOT_DIR}/benchmark_ad/configs/mobilenetv4_distill.yaml"
  "${ROOT_DIR}/benchmark_ad/configs/unireplknet_distill.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "======================================================"
  echo "Training with config: ${cfg}"
  echo "======================================================"
  python "${ROOT_DIR}/benchmark_ad/train.py" --config "${cfg}"
done

