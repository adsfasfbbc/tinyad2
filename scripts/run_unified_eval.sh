#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/configs/eval_benchmark.yaml"
OUTPUT_DIR="${ROOT_DIR}/experiments/eval_benchmark"

echo "==============================================="
echo "Unified Eval: MVTec + VisA"
echo "Config: ${CONFIG_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "==============================================="

python "${ROOT_DIR}/evaluate_benchmark.py" \
  --config "${CONFIG_PATH}" \
  --output_dir "${OUTPUT_DIR}"
