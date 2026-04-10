#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/experiments/benchmark_ad_latency"
mkdir -p "${OUT_DIR}"

declare -A CKPTS
CKPTS["tinyvit_11m"]="${ROOT_DIR}/experiments/benchmark_ad/tinyvit_11m/tinyvit_11m_final.pth"
CKPTS["mobilevit_s"]="${ROOT_DIR}/experiments/benchmark_ad/mobilevit_s/mobilevit_s_final.pth"
CKPTS["fastvit_t8"]="${ROOT_DIR}/experiments/benchmark_ad/fastvit_t8/fastvit_t8_final.pth"
CKPTS["mobilenetv4_hybrid"]="${ROOT_DIR}/experiments/benchmark_ad/mobilenetv4_hybrid/mobilenetv4_hybrid_final.pth"
CKPTS["unireplknet_s"]="${ROOT_DIR}/experiments/benchmark_ad/unireplknet_s/unireplknet_s_final.pth"

for model in "${!CKPTS[@]}"; do
  ckpt="${CKPTS[$model]}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "skip ${model}, checkpoint not found: ${ckpt}"
    continue
  fi
  echo "======================================================"
  echo "Latency benchmark: ${model}"
  echo "======================================================"
  python "${ROOT_DIR}/benchmark_ad/test_latency.py" \
    --student_name "${model}" \
    --checkpoint "${ckpt}" \
    --save_json "${OUT_DIR}/${model}.json"
done

