#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/experiments/benchmark_ad_eval"
mkdir -p "${OUT_DIR}"

declare -A CONFIGS
CONFIGS["tinyvit_11m"]="${ROOT_DIR}/benchmark_ad/configs/tinyvit_distill.yaml"
CONFIGS["mobilevit_s"]="${ROOT_DIR}/benchmark_ad/configs/mobilevit_distill.yaml"
CONFIGS["fastvit_t8"]="${ROOT_DIR}/benchmark_ad/configs/fastvit_distill.yaml"
CONFIGS["mobilenetv4_hybrid"]="${ROOT_DIR}/benchmark_ad/configs/mobilenetv4_distill.yaml"
CONFIGS["unireplknet_s"]="${ROOT_DIR}/benchmark_ad/configs/unireplknet_distill.yaml"

declare -A CKPTS
CKPTS["tinyvit_11m"]="${ROOT_DIR}/experiments/benchmark_ad/tinyvit_11m/tinyvit_11m_final.pth"
CKPTS["mobilevit_s"]="${ROOT_DIR}/experiments/benchmark_ad/mobilevit_s/mobilevit_s_final.pth"
CKPTS["fastvit_t8"]="${ROOT_DIR}/experiments/benchmark_ad/fastvit_t8/fastvit_t8_final.pth"
CKPTS["mobilenetv4_hybrid"]="${ROOT_DIR}/experiments/benchmark_ad/mobilenetv4_hybrid/mobilenetv4_hybrid_final.pth"
CKPTS["unireplknet_s"]="${ROOT_DIR}/experiments/benchmark_ad/unireplknet_s/unireplknet_s_final.pth"

for model in "${!CKPTS[@]}"; do
  cfg="${CONFIGS[$model]}"
  ckpt="${CKPTS[$model]}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "skip ${model}, checkpoint not found: ${ckpt}"
    continue
  fi
  echo "======================================================"
  echo "Unified benchmark eval+latency: ${model}"
  echo "======================================================"
  python "${ROOT_DIR}/benchmark_ad/eval.py" \
    --config "${cfg}" \
    --checkpoint "${ckpt}" \
    --save_dir "${OUT_DIR}/${model}" \
    --no_heatmap \
    --run_latency \
    --student_name "${model}"
done
