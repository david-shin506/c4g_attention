#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${ENV_NAME:-l40s_anysplat}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_NAME="${RUN_NAME:-intrinsic_test_release20k_tokens1024_4gpu}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_FROM="${RESUME_FROM:-}"

cd "${REPO_ROOT}"

find_latest_checkpoint() {
  local output_root="outputs/exp_${RUN_NAME}"
  local latest=""
  local latest_step=-1
  local ckpt basename step

  shopt -s nullglob
  for ckpt in "${output_root}"/*/checkpoints/*.ckpt; do
    basename="$(basename "${ckpt}")"
    if [[ "${basename}" =~ step[_=]([0-9]+)\.ckpt$ ]]; then
      step="${BASH_REMATCH[1]}"
    else
      continue
    fi

    if (( step > latest_step )); then
      latest_step="${step}"
      latest="${ckpt}"
    fi
  done
  shopt -u nullglob

  if [[ -n "${latest}" ]]; then
    printf '%s
' "${latest}"
  fi
}

RESUME_ARGS=()
if [[ "${RESUME_FROM}" == "none" ]]; then
  echo "Resume disabled via RESUME_FROM=none."
elif [[ -n "${RESUME_FROM}" ]]; then
  echo "Resuming from explicit checkpoint: ${RESUME_FROM}"
  RESUME_ARGS+=("checkpointing.load=${RESUME_FROM}")
elif [[ "${AUTO_RESUME}" != "0" ]]; then
  LATEST_CKPT="$(find_latest_checkpoint)"
  if [[ -n "${LATEST_CKPT}" ]]; then
    echo "Auto-resuming from latest checkpoint: ${LATEST_CKPT}"
    RESUME_ARGS+=("checkpointing.load=${LATEST_CKPT}")
  else
    echo "No checkpoint found for outputs/exp_${RUN_NAME}; starting from pretrained weights."
  fi
fi

conda run --no-capture-output -n "${ENV_NAME}" \
  python -m src.main +training=c4g_tokens_1024_release20k \
  wandb.mode="${WANDB_MODE}" \
  wandb.name="${RUN_NAME}" \
  "${RESUME_ARGS[@]}" \
  "$@"
