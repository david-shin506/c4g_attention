#!/usr/bin/env bash
# Motion-supervised C4G (scene flow + static anchor) on the davisK 4-dataset mix.
# Launch (survives slurm tmp cleanup, see memory note c4g-run-crash-slurm-tmpdir):
#   mkdir -p /tmp/jaewoo.jung/c4g_train_tmp
#   TMPDIR=/tmp/jaewoo.jung/c4g_train_tmp TMP=$TMPDIR TEMP=$TMPDIR \
#     setsid nohup bash scripts/train_motion_davisK.sh wandb.name=<name> > run_logs/<name>.log 2>&1 &
# Resume: append checkpointing.load=<abs path .ckpt>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${ENV_NAME:-l40s_anysplat}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

cd "${REPO_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
  python -m src.main +training=c4g_motion_davisK_all "$@"
