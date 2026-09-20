#!/bin/bash
#SBATCH -p sharedp
#SBATCH --account=collaborator
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH -J diag_gs_motion
#SBATCH -o /music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/intrinsic_test/C4G/run_logs/diag_gaussian_motion_%j.out
# One-GPU diagnostic: do the Gaussians move with / without the motion losses? (scripts/diag_gaussian_motion.py)
# Runs in the ~/3d_gpt image because cv2 needs libGL, which the bare nodes may lack.
set -uo pipefail
REPO=/music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/intrinsic_test/C4G
cd "${REPO}"
O=outputs
newest() { ls -t "$@" 2>/dev/null | head -1; }
MOTION=c4g_motion_davisK_all
REL=c4g_motion_relative_davisK_all
# Rolling checkpoints of the running jobs first: save_top_k=1 deletes them about an hour after they appear.
CKPTS=(
  "motion_latest=$(newest ${O}/exp_motion_davisK_all_45k/*/checkpoints/epoch_*.ckpt)=${MOTION}"
  "relative_latest=$(newest ${O}/exp_motion_relative_p100_davisK_all_20k/*/checkpoints/epoch_*.ckpt)=${REL}"
  "motion_5k=${O}/exp_motion_davisK_all_45k/2026-09-16_11-44-01/checkpoints/epoch_0-step_5000.ckpt=${MOTION}"
  "relative_1k_nomotion=${O}/exp_motion_relative_p100_davisK_all_20k/2026-09-17_03-11-07/checkpoints/epoch_0-step_1000.ckpt=${REL}"
  "ref45k_nomotion=${O}/exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_45k_resume_from40k_4gpu/2026-07-04_162000/checkpoints/epoch_2-step_45000.ckpt=${MOTION}"
  "v43_20k_nomotion=${O}/exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_v4-3_temb_20k_4gpu/2026-07-04_16-35-35/checkpoints/epoch_1-step_20000.ckpt=${REL}"
)
args=()
for c in "${CKPTS[@]}"; do echo "ckpt: $c"; args+=(--ckpt "$c"); done
/opt/singularity-4.1.1/bin/singularity exec --nv \
  --bind /music-3d-shared-disk/dataset:/music-3d-shared-disk/dataset/ \
  --bind /music-3d-shared-disk/user/KAIST:/music-3d-shared-disk/user/KAIST \
  /home/jaewoo.jung/3d_gpt /home/jaewoo.jung/.conda/envs/l40s_anysplat/bin/python scripts/diag_gaussian_motion.py \
  "${args[@]}" --per-dataset 12 --out "run_logs/diag_gaussian_motion_${SLURM_JOB_ID}.json"
