#!/bin/bash
#SBATCH -p sharedp
#SBATCH --account=collaborator
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/intrinsic_test/C4G/run_logs/slurm_%x_%j.out
#
# Preemption-safe train-then-evaluate job for C4G runs on sharedp (PreemptMode=REQUEUE).
# Every (re)start resumes from the newest complete checkpoint of outputs/exp_<RUN_NAME>, trains up to
# STOP_STEP, then evaluates the step-STOP_STEP checkpoint on ADT / NVIDIA / TUM / iPhone. Finished
# pieces (training, evals whose log already has metrics) are skipped after a requeue.
#
# Usage (from the login node):
#   sbatch -J <job name> scripts/sbatch_train_eval.sh <training cfg> <run name> <stop step> \
#       <eval cfg suffix: ref45k|v43> <eval tag> [extra hydra overrides...]
#   EXTRA_EVALS=ref45k also evaluates the VGGT 45k reference on NVIDIA / TUM / iPhone afterwards.
set -uo pipefail

REPO=/music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/intrinsic_test/C4G
cd "${REPO}"
# Batch jobs have no /opt/conda (only the interactive srun shells did), so call the env's python
# directly; the env is self-contained (no activate.d hooks), which is what `conda run` amounted to.
ENV_NAME=${ENV_NAME:-l40s_anysplat}
ENV_PREFIX=/home/jaewoo.jung/.conda/envs/${ENV_NAME}
export PATH=${ENV_PREFIX}/bin:${PATH}
export CONDA_PREFIX=${ENV_PREFIX} CONDA_DEFAULT_ENV=${ENV_NAME}
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
PY=${ENV_PREFIX}/bin/python

TRAIN_CFG=$1 RUN_NAME=$2 STOP_STEP=$3 EVAL_SUFFIX=$4 EVAL_TAG=$5
shift 5
EXTRA_EVALS=${EXTRA_EVALS:-}
EXP_DIR=outputs/exp_${RUN_NAME}
REF_CKPT=${REPO}/outputs/exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_45k_resume_from40k_4gpu/2026-07-04_162000/checkpoints/epoch_2-step_45000.ckpt
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

log() { echo "[$(date '+%F %T')] [job ${SLURM_JOB_ID:-?} restart ${SLURM_RESTART_COUNT:-0} $(hostname)] $*"; }

# Newest checkpoint whose size is within 1 MB of the largest one (a preempted save leaves a short file).
newest_complete_ckpt() {
  local pattern=$1 max f
  max=$(stat -c %s ${EXP_DIR}/*/checkpoints/*.ckpt 2>/dev/null | sort -n | tail -1)
  [ -n "${max}" ] || return 0
  for f in $(ls -t ${pattern} 2>/dev/null); do
    if [ $(( max - $(stat -c %s "${f}") )) -lt 1048576 ]; then
      realpath "${f}"
      return 0
    fi
  done
}

log "start: cfg=${TRAIN_CFG} run=${RUN_NAME} stop=${STOP_STEP} eval=${EVAL_SUFFIX}/${EVAL_TAG} gpus=${GPUS[*]} extra=$*"
log "TMPDIR=${TMPDIR:-unset}"
n_gpu=$("${PY}" -c "import torch; print(torch.cuda.device_count())" 2>&1 | tail -1)
log "torch sees ${n_gpu} GPU(s)"
[ "${n_gpu}" = "${#GPUS[@]}" ] || { log "GPU check failed, stopping"; exit 1; }

# 1) Train (skipped once the stop-step checkpoint exists).
final_ckpt=$(newest_complete_ckpt "${EXP_DIR}/*/checkpoints/*-step_${STOP_STEP}.ckpt")
if [ -z "${final_ckpt}" ]; then
  resume=$(newest_complete_ckpt "${EXP_DIR}/*/checkpoints/*.ckpt")
  load=()
  [ -n "${resume}" ] && load=("checkpointing.load=${resume}")
  log "training from ${resume:-scratch}"
  "${PY}" -m src.main \
    +training="${TRAIN_CFG}" wandb.mode=online wandb.name="${RUN_NAME}" \
    trainer.max_steps="${STOP_STEP}" "${load[@]}" "$@"
  log "training exited with $?"
  final_ckpt=$(newest_complete_ckpt "${EXP_DIR}/*/checkpoints/*-step_${STOP_STEP}.ckpt")
  [ -n "${final_ckpt}" ] || { log "no step-${STOP_STEP} checkpoint after training, stopping"; exit 1; }
fi
log "final checkpoint: ${final_ckpt}"

# 2) Evaluations, one GPU each; skip ones that already produced metrics.
run_eval() {  # gpu_slot evaluation name ckpt
  local slot=$1 ev=$2 name=$3 ck=$4 out
  if grep -qs "psnr_ours_epoch" run_logs/eval_${name}_*.log; then
    log "skip ${name}: already evaluated"
    return 0
  fi
  out="run_logs/eval_${name}_$(date +%Y%m%d_%H%M).log"
  log "GPU ${GPUS[$slot]}: ${name} (${ev}) -> ${out}"
  CUDA_VISIBLE_DEVICES=${GPUS[$slot]} "${PY}" -m src.main \
    +evaluation="${ev}" mode=test checkpointing.load="${ck}" wandb.name="${name}" > "${out}" 2>&1
  log "${name} finished (exit $?)"
}

# Run the (evaluation, name, checkpoint) triples one per GPU, in waves of however many we were given.
run_evals() {
  local i=0
  while [ $# -gt 0 ]; do
    run_eval $((i % ${#GPUS[@]})) "$1" "$2" "$3" &
    shift 3
    i=$((i + 1))
    [ $((i % ${#GPUS[@]})) -eq 0 ] && wait
  done
  wait
}

run_evals \
  "adt_vggt_${EVAL_SUFFIX}"    "${EVAL_TAG}_adt"    "${final_ckpt}" \
  "nvidia_vggt_${EVAL_SUFFIX}" "${EVAL_TAG}_nv"     "${final_ckpt}" \
  "tum_vggt_${EVAL_SUFFIX}"    "${EVAL_TAG}_tum"    "${final_ckpt}" \
  "iphone_vggt_${EVAL_SUFFIX}" "${EVAL_TAG}_iphone" "${final_ckpt}"

if [ "${EXTRA_EVALS}" = "ref45k" ]; then
  run_evals \
    nvidia_vggt_ref45k "ref45k_nv_align"     "${REF_CKPT}" \
    tum_vggt_ref45k    "ref45k_tum_align"    "${REF_CKPT}" \
    iphone_vggt_ref45k "ref45k_iphone_align" "${REF_CKPT}"
fi

# 3) Summary.
for f in $(ls -t run_logs/eval_${EVAL_TAG}_*.log run_logs/eval_ref45k_*_align_*.log 2>/dev/null); do
  psnr=$(grep "psnr_ours_epoch" "$f" | tr -s ' │' ' ' | awk '{print $2}')
  lpips=$(grep "lpips_ours_epoch" "$f" | tr -s ' │' ' ' | awk '{print $2}')
  ssim=$(grep "ssim_ours_epoch" "$f" | tr -s ' │' ' ' | awk '{print $2}')
  echo "RESULT $(basename "$f") psnr=${psnr:-NA} lpips=${lpips:-NA} ssim=${ssim:-NA}"
done
log "done"
