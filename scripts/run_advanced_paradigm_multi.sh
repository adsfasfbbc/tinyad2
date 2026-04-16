#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_ENTRY="${ROOT_DIR}/benchmark_ad/train.py"
EVAL_ENTRY="${ROOT_DIR}/benchmark_ad/eval.py"

if [[ $# -gt 0 ]]; then
  CONFIGS=("$@")
else
  CONFIGS=(
    "${ROOT_DIR}/benchmark_ad/configs/advanced_paradigm_fastvit.yaml"
  )
fi

EVAL_ROOT="${ROOT_DIR}/experiments/benchmark_ad_eval/advanced_paradigm"
mkdir -p "${EVAL_ROOT}"

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "${cfg}" ]]; then
    echo "[Advanced] skip, config not found: ${cfg}"
    continue
  fi

  echo "======================================================"
  echo "[Advanced] training with config: ${cfg}"
  echo "======================================================"
  if ! python "${TRAIN_ENTRY}" --config "${cfg}"; then
    echo "[Advanced] training failed, continue next config: ${cfg}"
    continue
  fi

  read -r save_dir student_name < <(
    python - <<'PY' "${cfg}"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8')) or {}
runtime = cfg.get('runtime', {}) or {}
student = cfg.get('student', {}) or {}
save_dir = runtime.get('save_dir', './experiments/benchmark_ad')
student_name = student.get('name', '')
print(save_dir, student_name)
PY
  )

  ckpt="${save_dir}/${student_name}_final.pth"
  out_dir="${EVAL_ROOT}/${student_name}"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[Advanced] skip eval, checkpoint not found: ${ckpt}"
    continue
  fi

  echo "======================================================"
  echo "[Advanced] eval+latency for model: ${student_name}"
  echo "======================================================"
  if ! python "${EVAL_ENTRY}" \
    --config "${cfg}" \
    --checkpoint "${ckpt}" \
    --save_dir "${out_dir}" \
    --no_heatmap \
    --run_latency \
    --student_name "${student_name}"; then
    echo "[Advanced] eval failed, continue next config: ${cfg}"
    continue
  fi
done
