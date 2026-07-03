#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ROOT="${ROOT:-${CARACTGEN_OUTPUT_ROOT:-${REPO}/outputs}/caractgen_wheel_anchor}"
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN="${ROOT}/tmux_${STAMP}"

mkdir -p "${RUN}/logs"
printf '%s\n' "${RUN}" > "${ROOT}/latest_run.txt"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${RUN}/run.log"
}

run_model() {
  local model="$1"
  local gpu="$2"
  local seed="$3"
  log "Starting ${model} seed=${seed} on GPU ${gpu}"
  (
    cd "${REPO}" || exit 2
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PY}" -u experiments/paper_train_wheel_anchor_predictor.py \
      --model "${model}" \
      --seed "${seed}" \
      --pc_size 2048 \
      --epochs 3000 \
      --batch_size 64 \
      --eval_every 25 \
      --patience 650 \
      --output_dir "${ROOT}"
  ) > "${RUN}/logs/${model}_seed${seed}.log" 2>&1
  local status=$?
  printf '%s\n' "${status}" > "${RUN}/${model}_seed${seed}.status"
  log "${model} seed=${seed} exited with ${status}"
  return "${status}"
}

status=0
run_model bbox_mlp 2 20260702 &
pid_bbox=$!
run_model pointnet_anchor 3 20260702 &
pid_pointnet=$!

wait "${pid_bbox}" || status=1
wait "${pid_pointnet}" || status=1

printf '%s\n' "${status}" > "${RUN}/train.status"
log "Wheel anchor predictor training finished with status ${status}"
