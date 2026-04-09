#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/configs/vit_mobilevit_distill.yaml"

echo "======================================================"
echo "ViT->MobileViT Heterogeneous Distillation Training"
echo "Config: ${CONFIG_PATH}"
echo "======================================================"

python "${ROOT_DIR}/train_vit_mobilevit_distill.py" \
  --config "${CONFIG_PATH}"
