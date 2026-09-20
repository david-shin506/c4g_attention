#!/usr/bin/env bash
# Wait for the relative time-embedding run (max_steps 20000) to exit, then evaluate its step-20000
# checkpoint on ADT / NVIDIA / TUM / iPhone with the relative / period-100 eval configs (*_vggt_v43),
# matching the motion run's motion20k_* evals (*_vggt_ref45k differ only in the two time-embedding keys).
# Launch detached:
#   TRAIN_PGID=<pgid> CKPT_DIR=<run dir>/checkpoints setsid nohup bash scripts/eval_after_train_relative.sh \
#     > run_logs/eval_after_train_relative20k.log 2>&1 &
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STOP_STEP="${STOP_STEP:-20000}"
TRAIN_PGID="${TRAIN_PGID:?set TRAIN_PGID}"
CKPT_DIR="${CKPT_DIR:?set CKPT_DIR}"
ENV_NAME="${ENV_NAME:-l40s_anysplat}"
TAG="${TAG:-relative$((STOP_STEP / 1000))k}"

log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for training pgid ${TRAIN_PGID} to exit"
while pgrep -g "${TRAIN_PGID}" > /dev/null; do sleep 60; done
ckpt=$(ls "${CKPT_DIR}"/*-step_${STOP_STEP}.ckpt 2>/dev/null | head -1)
if [ -z "${ckpt}" ]; then
  log "training exited without a step-${STOP_STEP} checkpoint (node taken?); nothing to evaluate"
  ls -la "${CKPT_DIR}" 2>/dev/null
  echo "EVAL_DONE status=no_ckpt"
  exit 1
fi
ckpt="$(realpath "${ckpt}")"
log "checkpoint: ${ckpt}"

run_eval() {  # gpu evaluation name ckpt
  local gpu=$1 ev=$2 name=$3 ck=$4
  local out="run_logs/eval_${name}_$(date +%Y%m%d_%H%M).log"
  log "GPU${gpu}: ${name} (${ev}) -> ${out}"
  CUDA_VISIBLE_DEVICES=${gpu} PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 \
    conda run --no-capture-output -n "${ENV_NAME}" python -m src.main \
      +evaluation="${ev}" mode=test checkpointing.load="${ck}" wandb.name="${name}" > "${out}" 2>&1
  log "GPU${gpu}: ${name} finished (exit $?)"
}

run_eval 0 adt_vggt_v43    "${TAG}_adt"    "${ckpt}" &
run_eval 1 nvidia_vggt_v43 "${TAG}_nv"     "${ckpt}" &
run_eval 2 tum_vggt_v43    "${TAG}_tum"    "${ckpt}" &
run_eval 3 iphone_vggt_v43 "${TAG}_iphone" "${ckpt}" &
wait

for f in $(ls -t run_logs/eval_${TAG}_*.log 2>/dev/null); do
  psnr=$(grep "psnr_ours_epoch" "$f" | tr -s ' │' ' ' | awk '{print $2}')
  lpips=$(grep "lpips_ours_epoch" "$f" | tr -s ' │' ' ' | awk '{print $2}')
  ssim=$(grep "ssim_ours_epoch" "$f" | tr -s ' │' ' ' | awk '{print $2}')
  echo "RESULT $(basename "$f") psnr=${psnr:-NA} lpips=${lpips:-NA} ssim=${ssim:-NA}"
done
echo "EVAL_DONE status=ok"
