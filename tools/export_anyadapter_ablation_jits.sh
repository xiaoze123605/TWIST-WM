#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/legged_gym/logs/g1_twist_anyadapter_safe/twist_anyadapter_safe_train}"
BASE_ACTOR="${BASE_ACTOR:-${REPO_ROOT}/legged_gym/logs/g1_stu_rl/0529_twist_rlbcstu/traced/0529_twist_rlbcstu-36500-jit.pt}"
DEVICE="${DEVICE:-cpu}"
PRESET="${PRESET:-safe}"

CKPT="${1:-}"
if [[ -z "${CKPT}" ]]; then
  latest_name="$(find "${RUN_DIR}" -maxdepth 1 -name 'model_*.pt' -printf '%f\n' | sort -V | tail -n 1)"
  if [[ -z "${latest_name}" ]]; then
    echo "No model_*.pt found in ${RUN_DIR}" >&2
    exit 1
  fi
  CKPT="${RUN_DIR}/${latest_name}"
fi

ckpt_tag="$(basename "${CKPT}" .pt)"
ckpt_tag="${ckpt_tag#model_}"
out_dir="$(dirname "${CKPT}")/traced"
run_name="$(basename "$(dirname "${CKPT}")")"
mkdir -p "${out_dir}"

echo "[A] Base TWIST actor:"
echo "    ${BASE_ACTOR}"

echo "[B] Exporting AnyAdapter adapter_gain=0"
python "${REPO_ROOT}/legged_gym/scripts/export_twist_anyadapter_jit.py" \
  --ckpt "${CKPT}" \
  --preset "${PRESET}" \
  --base_actor_jit_path "${BASE_ACTOR}" \
  --adapter_gain 0.0 \
  --device "${DEVICE}" \
  --out "${out_dir}/${run_name}-${ckpt_tag}-gain0-jit.pt"

echo "[C] Exporting AnyAdapter adapter_gain=1"
python "${REPO_ROOT}/legged_gym/scripts/export_twist_anyadapter_jit.py" \
  --ckpt "${CKPT}" \
  --preset "${PRESET}" \
  --base_actor_jit_path "${BASE_ACTOR}" \
  --adapter_gain 1.0 \
  --device "${DEVICE}" \
  --out "${out_dir}/${run_name}-${ckpt_tag}-gain1-jit.pt"

echo "Done."
echo "[B] ${out_dir}/${run_name}-${ckpt_tag}-gain0-jit.pt"
echo "[C] ${out_dir}/${run_name}-${ckpt_tag}-gain1-jit.pt"
