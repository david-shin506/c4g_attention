#!/bin/bash
#SBATCH -p sharedp
#SBATCH --account=collaborator
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J davis_vae_feat
#SBATCH -o /music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/intrinsic_test/C4G/run_logs/sbatch_davis_vae_features_%j.out
#
# DAVIS twin of the Spring VAE feature dataset (scripts/README_spring_vae_features.md):
#   1) prepare_davis_vae_latents.py  clean Wan2.1 VAE latents of every frame the windows use
#   2) export_davis_vae_features.py  per-window predicted features + 480x480 RGB predictions
# Output: C4G_prediction_dataset/DAVIS_VAE_480x480_ctx8. Both steps resume after a requeue.
# Runs inside the ~/3d_gpt image through singularity_run.sh (bash -i), like the interactive sessions the
# Spring export ran in: the vae16 rasterizer is a JIT extension that needs the image's toolchain and the
# real ninja in ~/.local/bin.
set -euo pipefail

REPO=/music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/intrinsic_test/C4G
PY=/home/jaewoo.jung/.conda/envs/l40s_anysplat/bin/python
cd "${REPO}"
echo "[$(date '+%F %T')] job ${SLURM_JOB_ID} restart ${SLURM_RESTART_COUNT:-0} on $(hostname) gpus=${CUDA_VISIBLE_DEVICES:-}"

# A job preempted during a JIT load leaves a lock that blocks every later load.
find /home/jaewoo.jung/.cache/torch_extensions -maxdepth 3 -name lock -mmin +15 -print -delete 2>/dev/null || true

/music-3d-shared-disk/user/KAIST/MG/singularity_run.sh \
    bash -c "${PY} -u scripts/prepare_davis_vae_latents.py && ${PY} -u scripts/export_davis_vae_features.py"

echo "[$(date '+%F %T')] done"
