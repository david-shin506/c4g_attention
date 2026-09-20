#!/bin/bash
set -euo pipefail
PROJECT_ROOT="/music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/intrinsic_test/C4G"
PYTHON_BIN="${SPRING_VAE_PYTHON:-/home/jaewoo.jung/.conda/envs/l40s_anysplat/bin/python}"
DATA_ROOT="${SPRING_VAE_ROOT:-/music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/C4G_prediction_dataset/Spring_VAE_480x480_ctx8}"
RUN_ROOT="${SPRING_VAE_RUN:-$PROJECT_ROOT/outputs/spring_vae_feature_ctx8_step45000/pilot_1000}"
export MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/.pipeline.lock"
flock -n 9
write_stage() {
  "$PYTHON_BIN" -c 'import json,os,sys,time; from pathlib import Path; p=Path(sys.argv[1]); tmp=p.with_suffix(".tmp"); tmp.write_text(json.dumps({"state":sys.argv[2],"updated_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())},indent=2)+"\n"); os.replace(tmp,p)' "$RUN_ROOT/pipeline_status.json" "$1"
}
trap 'write_stage failed' ERR
write_stage preparing
"$PYTHON_BIN" -u scripts/prepare_spring_vae_latents.py --output "$DATA_ROOT"
write_stage training
"$PYTHON_BIN" -u scripts/train_spring_vae_features.py --data "$DATA_ROOT" --run "$RUN_ROOT" --steps "${SPRING_VAE_STEPS:-1000}"
if [ "${SPRING_VAE_EXPORT:-0}" != "1" ]; then
  write_stage pilot_complete
  exit 0
fi
write_stage exporting
"$PYTHON_BIN" -u scripts/export_spring_vae_features.py --output "$DATA_ROOT" --checkpoint "$RUN_ROOT/best.ckpt"
write_stage validating
"$PYTHON_BIN" -u scripts/validate_spring_vae_features.py "$DATA_ROOT"
write_stage complete
