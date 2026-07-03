#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ROOT="${ROOT:-${REPO}/outputs/caractgen_clean_partlocal}"
SPLIT_PATH="${SPLIT_PATH:-${CARACTGEN_SPLIT_PATH:-${REPO}/data/caractgen_metadata/splits/object_sketch_dinov2_partlocal_seed123456798.json}}"
BASE_DATASET="${BASE_DATASET:-${CARACTGEN_DATA_ROOT:-${REPO}/data/datasets}}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${ROOT}/runs/${STAMP}"
POLL_SEC="${POLL_SEC:-1200}"

mkdir -p "${RUN}/logs"
ln -sfn "${RUN}" "${ROOT}/runs/latest"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${RUN}/run.log"
}

latest_metrics() {
  local root="$1"
  find "${root}/lightning_logs" -type f -name metrics.csv -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
}

best_ckpt_by_metric_name() {
  local root="$1"
  local metric="$2"
  "${PY}" - "$root" "$metric" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
metric = sys.argv[2]
best = None
for path in root.rglob("*.ckpt"):
    if path.name == "last.ckpt":
        continue
    match = re.search(rf"{re.escape(metric)}=([0-9.]+)", path.name)
    if not match:
        continue
    value = float(match.group(1).rstrip("."))
    item = (value, path.stat().st_mtime, path)
    if best is None or item < best:
        best = item
if best is not None:
    print(best[2])
else:
    candidates = sorted(root.rglob("last.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        print(candidates[0])
PY
}

monitor_metric() {
  local pid="$1"
  local root="$2"
  local metric="$3"
  local label="$4"
  local patience_polls="$5"
  local min_epochs_after_best="$6"
  local min_delta="$7"
  local best_seen=""
  local best_epoch_seen=""
  local stale_polls=0

  log "Monitoring ${label}: metric=${metric}, poll=${POLL_SEC}s, patience=${patience_polls} polls"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep "${POLL_SEC}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    local metrics
    metrics="$(latest_metrics "${root}")"
    if [ -z "${metrics}" ]; then
      log "${label}: metrics.csv not found yet"
      continue
    fi
    local report
    report="$("${PY}" - "${metrics}" "${metric}" "${best_seen:-nan}" "${min_delta}" <<'PY'
import math
import sys
import pandas as pd

path, metric, previous_best, min_delta = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
df = pd.read_csv(path)
if metric not in df.columns:
    print("NO_METRIC")
    raise SystemExit
rows = df[["epoch", "step", metric]].dropna()
if rows.empty:
    print("NO_ROWS")
    raise SystemExit
idx = rows[metric].idxmin()
best = rows.loc[idx]
latest = rows.tail(1).iloc[0]
prev = math.inf if previous_best == "nan" else float(previous_best)
improved = float(best[metric]) < prev - min_delta
print(
    f"OK\t{int(latest['epoch'])}\t{float(latest[metric]):.8f}\t"
    f"{int(best['epoch'])}\t{float(best[metric]):.8f}\t{int(improved)}"
)
PY
)"
    if [[ "${report}" != OK$'\t'* ]]; then
      log "${label}: ${report}"
      continue
    fi
    IFS=$'\t' read -r _ latest_epoch latest_value best_epoch best_value improved <<< "${report}"
    log "${label}: latest epoch=${latest_epoch} ${metric}=${latest_value}; best epoch=${best_epoch} ${metric}=${best_value}"
    if [ "${improved}" = "1" ] || [ -z "${best_seen}" ]; then
      best_seen="${best_value}"
      best_epoch_seen="${best_epoch}"
      stale_polls=0
    else
      stale_polls=$((stale_polls + 1))
    fi
    local epochs_after_best=$((latest_epoch - best_epoch_seen))
    if [ "${stale_polls}" -ge "${patience_polls}" ] && [ "${epochs_after_best}" -ge "${min_epochs_after_best}" ]; then
      log "${label}: no improvement for ${stale_polls} polls and ${epochs_after_best} epochs; stopping process group ${pid}"
      kill -TERM "-${pid}" 2>/dev/null || true
      sleep 30
      kill -KILL "-${pid}" 2>/dev/null || true
      return 124
    fi
  done
  return 0
}

run_monitored() {
  local label="$1"
  local root="$2"
  local metric="$3"
  local patience="$4"
  local min_epochs_after_best="$5"
  local min_delta="$6"
  shift 6
  log "START ${label}"
  setsid bash -lc "$*" > "${RUN}/logs/${label}.log" 2>&1 &
  local pid=$!
  printf '%s\n' "${pid}" > "${RUN}/${label}.pid"
  monitor_metric "${pid}" "${root}" "${metric}" "${label}" "${patience}" "${min_epochs_after_best}" "${min_delta}" &
  local monitor_pid=$!
  wait "${pid}"
  local status=$?
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  printf '%s\n' "${status}" > "${RUN}/${label}.status"
  log "END ${label}: status=${status}"
  return "${status}"
}

log "Clean PartLocal pipeline root: ${ROOT}"
log "Run dir: ${RUN}"
log "Python: ${PY}"

cd "${REPO}" || exit 2

log "Preparing clean configs"
"${PY}" experiments/paper_prepare_clean_partlocal.py \
  --output_root "${ROOT}" \
  --split_path "${SPLIT_PATH}" \
  --base_dataset "${BASE_DATASET}" \
  --vae_epochs "${VAE_EPOCHS:-160}" \
  --diff_epochs "${DIFF_EPOCHS:-5000}" \
  > "${RUN}/logs/prepare.log" 2>&1
prepare_status=$?
printf '%s\n' "${prepare_status}" > "${RUN}/prepare.status"
if [ "${prepare_status}" -ne 0 ]; then
  log "Prepare failed"
  exit "${prepare_status}"
fi

VAE_CONFIG="${ROOT}/configs/train_function_aware_vae_clean.yaml"
DIFF_CONFIG="${ROOT}/configs/train_partlocal_clean.yaml"
LATENT_DATASET="${ROOT}/datasets/2.1_clean_trainonly_vae_latent_sketch_dinov2"

if [ "${SKIP_VAE:-0}" != "1" ]; then
  run_monitored \
    "train_clean_function_aware_vae" \
    "${ROOT}/vae" \
    "val_loss" \
    "${VAE_PATIENCE_POLLS:-2}" \
    "${VAE_MIN_EPOCHS_AFTER_BEST:-30}" \
    "${VAE_MIN_DELTA:-0.00002}" \
    "cd '${REPO}' && export CUDA_VISIBLE_DEVICES='${VAE_GPUS:-0,1,2,3,4,5,6,7}' && '${PY}' -u experiments/train_function_aware_sdf.py -c '${VAE_CONFIG}' --devices '${VAE_DEVICES:-8}'"
  vae_status=$?
  if [ "${vae_status}" -ne 0 ]; then
    log "VAE training ended with status ${vae_status}; continuing only if a checkpoint exists"
  fi
fi

VAE_CKPT="$(best_ckpt_by_metric_name "${ROOT}/vae_checkpoints" "val_loss")"
if [ -z "${VAE_CKPT}" ]; then
  log "No clean function-aware VAE checkpoint found"
  exit 3
fi
printf '%s\n' "${VAE_CKPT}" > "${RUN}/clean_vae_ckpt.txt"
log "Using clean VAE checkpoint: ${VAE_CKPT}"

if [ "${SKIP_EXTRACT:-0}" != "1" ]; then
  log "Extracting all latents with train-only VAE and train-only stats"
  (
    cd "${REPO}" || exit 2
    export CUDA_VISIBLE_DEVICES="${EXTRACT_GPU:-0}"
    "${PY}" -u experiments/extract_function_aware_latents.py \
      --sdf_ckpt_path "${VAE_CKPT}" \
      --dataset_dir "${BASE_DATASET}" \
      --mesh_info_dir "${BASE_DATASET}/1_preprocessed_info" \
      --existing_text_latent_dir "${BASE_DATASET}/2.1_text_n_latentcode" \
      --image_embedding_dir "${BASE_DATASET}/6_encoded_drivaer_sketch_image_dinov2" \
      --output_path "${LATENT_DATASET}" \
      --sdf_subdir 2_gensdf_dataset_adaptive \
      --stats_split_path "${SPLIT_PATH}" \
      --stats_split train \
      --batch_size "${EXTRACT_BATCH_SIZE:-24}" \
      --num_workers "${EXTRACT_WORKERS:-8}" \
      --samples_per_mesh 24000 \
      --pc_size 8192 \
      --uniform_sample_ratio 0.25 \
      --reset_output
  ) > "${RUN}/logs/extract_clean_latents.log" 2>&1
  extract_status=$?
  printf '%s\n' "${extract_status}" > "${RUN}/extract_clean_latents.status"
  if [ "${extract_status}" -ne 0 ]; then
    log "Latent extraction failed"
    exit "${extract_status}"
  fi
fi

if [ "${SKIP_DIFF:-0}" != "1" ]; then
  run_monitored \
    "train_clean_partlocal_diffusion" \
    "${ROOT}/partlocal_diffusion" \
    "val_loss" \
    "${DIFF_PATIENCE_POLLS:-2}" \
    "${DIFF_MIN_EPOCHS_AFTER_BEST:-600}" \
    "${DIFF_MIN_DELTA:-0.002}" \
    "cd '${REPO}' && export CUDA_VISIBLE_DEVICES='${DIFF_GPUS:-0,1,2,3}' && export RUN_NAME='clean_partlocal_trainonlyvae_${STAMP}' && '${PY}' -u experiments/train_adaptive_object_multimodal_diffusion.py -c '${DIFF_CONFIG}' --devices '${DIFF_DEVICES:-4}'"
  diff_status=$?
  if [ "${diff_status}" -ne 0 ]; then
    log "Diffusion training ended with status ${diff_status}; continuing only if a checkpoint exists"
  fi
fi

DIFF_CKPT="$(best_ckpt_by_metric_name "${ROOT}/partlocal_diffusion/checkpoint" "val_loss")"
if [ -z "${DIFF_CKPT}" ]; then
  log "No clean PartLocal diffusion checkpoint found"
  exit 4
fi
printf '%s\n' "${DIFF_CKPT}" > "${RUN}/clean_partlocal_ckpt.txt"
log "Using clean PartLocal checkpoint: ${DIFF_CKPT}"

if [ "${SKIP_EVAL:-0}" != "1" ]; then
  EVAL_DIR="${ROOT}/eval_clean_partlocal"
  mkdir -p "${EVAL_DIR}"
  log "Evaluating clean PartLocal on held-out test split"
  run_eval_chunk() {
    local start="$1"
    local end="$2"
    local gpu="$3"
    (
      cd "${REPO}" || exit 2
      export CUDA_VISIBLE_DEVICES="${gpu}"
      "${PY}" -u experiments/paper_diffusion_sample_eval.py \
        --preset partlocal \
        --checkpoint "${DIFF_CKPT}" \
        --dataset_path "${LATENT_DATASET}" \
        --split_path "${SPLIT_PATH}" \
        --split test \
        --max_shapes 48 \
        --start "${start}" \
        --end "${end}" \
        --output_dir "${EVAL_DIR}" \
        --sdf_resolution 128 \
        --max_batch 32768 \
        --clip_denoised 1.0 \
        --overwrite
    ) > "${RUN}/logs/eval_clean_partlocal_${start}_${end}.log" 2>&1
    printf '%s\n' "$?" > "${RUN}/eval_clean_partlocal_${start}_${end}.status"
  }
  run_eval_chunk 0 12 "${EVAL_GPU0:-0}" &
  p0=$!
  run_eval_chunk 12 24 "${EVAL_GPU1:-1}" &
  p1=$!
  run_eval_chunk 24 36 "${EVAL_GPU2:-2}" &
  p2=$!
  run_eval_chunk 36 48 "${EVAL_GPU3:-3}" &
  p3=$!
  eval_status=0
  for p in "${p0}" "${p1}" "${p2}" "${p3}"; do
    wait "${p}" || eval_status=1
  done
  printf '%s\n' "${eval_status}" > "${RUN}/eval_clean_partlocal.status"
  if [ "${eval_status}" -ne 0 ]; then
    log "Clean PartLocal evaluation failed"
    exit "${eval_status}"
  fi
  "${PY}" experiments/paper_summarize_diffusion_samples.py --output_dir "${EVAL_DIR}" \
    > "${RUN}/logs/summarize_clean_partlocal.log" 2>&1
  summarize_status=$?
  printf '%s\n' "${summarize_status}" > "${RUN}/summarize_clean_partlocal.status"
  if [ "${summarize_status}" -ne 0 ]; then
    log "Clean PartLocal summary failed"
    exit "${summarize_status}"
  fi
fi

log "Clean PartLocal pipeline complete"
