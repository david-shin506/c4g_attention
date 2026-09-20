#!/bin/bash
set -euo pipefail
PROJECT_ROOT=/music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/intrinsic_test/C4G
PYTHON_BIN=/home/jaewoo.jung/.conda/envs/l40s_anysplat/bin/python
DATA_ROOT=/music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/C4G_prediction_dataset/Spring_VAE_480x480_ctx8
RUN_ROOT="$PROJECT_ROOT/outputs/spring_vae_feature_ctx8_step45000/pilot_1000"
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=9.0
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"
exec 9>"$RUN_ROOT/.export_pipeline.lock"
flock -n 9
printf '%s\n' "$$" > "$RUN_ROOT/export_supervisor.pid"
write_stage() {
  "$PYTHON_BIN" - "$RUN_ROOT/export_pipeline_status.json" "$1" "$$" "$DATA_ROOT" <<'PY'
import json,os,sys,time
from pathlib import Path
p=Path(sys.argv[1]); tmp=p.with_suffix('.tmp')
tmp.write_text(json.dumps({'state':sys.argv[2],'supervisor_pid':int(sys.argv[3]),'dataset_root':sys.argv[4],'checkpoint_step':1000,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2)+'\n')
os.replace(tmp,p)
PY
}
trap 'write_stage failed' ERR
trap 'write_stage interrupted; exit 143' TERM INT
write_stage exporting
"$PYTHON_BIN" -u scripts/export_spring_vae_features.py --output "$DATA_ROOT" --checkpoint "$RUN_ROOT/checkpoint_step_1000.ckpt"
write_stage validating
"$PYTHON_BIN" -u scripts/validate_spring_vae_features.py "$DATA_ROOT"
write_stage complete
