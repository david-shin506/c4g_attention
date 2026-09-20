#!/usr/bin/env bash
# Stop the motion davisK run once the step-STOP_STEP checkpoint is fully written, then evaluate it on
# ADT / NVIDIA / TUM / iPhone (C4G test path, align_pose on), followed by the VGGT 45k reference on the
# benchmarks it has no C4G-path number for yet (NVIDIA / TUM / iPhone; ADT align is exp_ref45k_adt_align).
# Launch detached (kill needs to run outside the tool sandbox):
#   setsid nohup bash scripts/stop_at_step_and_eval_motion.sh > run_logs/stop_eval_motion20k.log 2>&1 &
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STOP_STEP="${STOP_STEP:-20000}"
TRAIN_LOG="${TRAIN_LOG:-run_logs/motion_davisK_all_45k_20260916_r4.log}"
TRAIN_PGID="${TRAIN_PGID:-$(cat run_logs/motion_davisK_all_45k_20260916_r4.pgid)}"
CKPT_DIR="${CKPT_DIR:-outputs/exp_motion_davisK_all_45k/2026-09-16_11-44-01/checkpoints}"
REF_CKPT="${REPO_ROOT}/outputs/exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_45k_resume_from40k_4gpu/2026-07-04_162000/checkpoints/epoch_2-step_45000.ckpt"
ENV_NAME="${ENV_NAME:-l40s_anysplat}"
TAG="motion$((STOP_STEP / 1000))k"

log() { echo "[$(date '+%F %T')] $*"; }

# 1) Wait for the checkpoint; the run printing STOP_STEP+10 means the save (step ckpt + last.ckpt) returned.
log "waiting for step ${STOP_STEP} (pgid ${TRAIN_PGID})"
while true; do
  ckpt=$(ls "${CKPT_DIR}"/*-step_${STOP_STEP}.ckpt 2>/dev/null | head -1)
  if [ -n "${ckpt}" ] && grep -q "train step $((STOP_STEP + 10));" "${TRAIN_LOG}"; then
    break
  fi
  if ! pgrep -g "${TRAIN_PGID}" > /dev/null; then
    log "training process group is gone before step ${STOP_STEP}; ckpt='${ckpt}'"
    [ -n "${ckpt}" ] || { log "no checkpoint, abort"; echo "STOP_EVAL_DONE status=no_ckpt"; exit 1; }
    break
  fi
  sleep 60
done
ckpt="$(realpath "${ckpt}")"
log "checkpoint ready: ${ckpt} ($(stat -c %s "${ckpt}") bytes)"

# 2) Stop training.
if pgrep -g "${TRAIN_PGID}" > /dev/null; then
  log "stopping training"
  kill -TERM -- "-${TRAIN_PGID}" 2>/dev/null
  sleep 30
  kill -9 -- "-${TRAIN_PGID}" 2>/dev/null
fi
for _ in $(seq 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${used}" -lt 2000 ] && break
  sleep 5
done
log "GPU memory after stop: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"

# 3) Evaluations, one GPU each.
run_eval() {  # gpu evaluation name ckpt [extra overrides...]
  local gpu=$1 ev=$2 name=$3 ck=$4; shift 4
  local out="run_logs/eval_${name}_$(date +%Y%m%d_%H%M).log"
  log "GPU${gpu}: ${name} (${ev}) -> ${out}"
  CUDA_VISIBLE_DEVICES=${gpu} PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 \
    conda run --no-capture-output -n "${ENV_NAME}" python -m src.main \
      +evaluation="${ev}" mode=test checkpointing.load="${ck}" wandb.name="${name}" "$@" > "${out}" 2>&1
  log "GPU${gpu}: ${name} finished (exit $?)"
}

run_eval 0 adt_vggt_ref45k    "${TAG}_adt"    "${ckpt}" &
run_eval 1 nvidia_vggt_ref45k "${TAG}_nv"     "${ckpt}" &
run_eval 2 tum_vggt_ref45k    "${TAG}_tum"    "${ckpt}" &
run_eval 3 iphone_vggt_ref45k "${TAG}_iphone" "${ckpt}" &
wait

run_eval 0 nvidia_vggt_ref45k "ref45k_nv_align"     "${REF_CKPT}" &
run_eval 1 tum_vggt_ref45k    "ref45k_tum_align"    "${REF_CKPT}" &
run_eval 2 iphone_vggt_ref45k "ref45k_iphone_align" "${REF_CKPT}" &
wait

# 4) Summary.
for f in $(ls -t run_logs/eval_${TAG}_*.log run_logs/eval_ref45k_*_align_*.log 2>/dev/null); do
  psnr=$(grep "psnr_ours_epoch" "$f" | tr -s ' │' ' ' | awk '{print $2}')
  lpips=$(grep "lpips_ours_epoch" "$f" | tr -s ' │' ' ' | awk '{print $2}')
  ssim=$(grep "ssim_ours_epoch" "$f" | tr -s ' │' ' ' | awk '{print $2}')
  echo "RESULT $(basename "$f") psnr=${psnr:-NA} lpips=${lpips:-NA} ssim=${ssim:-NA}"
done
echo "STOP_EVAL_DONE status=ok"
