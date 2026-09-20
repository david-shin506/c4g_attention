#!/usr/bin/env bash
# Time-embedding ablation of the motion davisK run: v4-3 relative / period 100 instead of
# target_1 / period 10000 (config/training/c4g_motion_relative_davisK_all.yaml, max_steps 20000).
# Launch on a free 4-GPU node:
#   PYTHONUNBUFFERED=1 setsid nohup bash scripts/train_motion_relative_davisK.sh wandb.mode=online \
#     > run_logs/motion_relative_p100_davisK_all_20k_$(date +%Y%m%d).log 2>&1 &
# Resume: append checkpointing.load=<abs path>/checkpoints/last.ckpt
# Eval (normalize_type is not stored in the ckpt, so use the relative configs):
#   python -m src.main +evaluation={adt,nvidia,tum,iphone}_vggt_v43 mode=test checkpointing.load=<ckpt> wandb.name=<name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${ENV_NAME:-l40s_anysplat}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

cd "${REPO_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
  python -m src.main +training=c4g_motion_relative_davisK_all "$@"
