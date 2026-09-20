from pathlib import Path
from collections import Counter,defaultdict
import sys,json,hashlib
import numpy as np
import torch
from omegaconf import OmegaConf
root=Path.cwd();sys.path.insert(0,str(root/'scripts'))
from camxtime_pose_normalization import normalize_camxtime_poses,POLICY
torch.set_num_threads(1)
out=root.parents[1]/'C4G_prediction_dataset/CamxTime_480x480'
raw=OmegaConf.to_container(OmegaConf.load(root/'outputs/exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_45k_resume_from40k_4gpu/2026-07-04_162000/.hydra/config.yaml'),resolve=True)['dataset']['spring']
plan=json.loads((out/'plan.json').read_text())['samples']; groups=defaultdict(list)
for row in plan:groups[row['scene'],row['trajectory']].append(row)
counts=Counter();rows=[];errors=[];existing_checks=0
for (scene,rig),windows in groups.items():
 try:
  data=json.loads((root/'datasets/CamxTime'/scene/rig/f'{rig}-camera.json').read_text())
  keys=sorted(data['trajectory'],key=int)
  src=np.array([data['trajectory'][k]['c2w'] for k in keys])
  assert len(src)==120
  for row in windows:
   s=row['start']; ts=list(range(s,s+33)); ctx=ts[::2]
   poses,scale,info=normalize_camxtime_poses(src,ctx,ts,raw['baseline_min'],raw['baseline_max'])
   counts[info['mode']]+=1;rows.append({'sample_id':row['sample_id'],**info})
   saved=out/'samples'/row['sample_id']/'cameras.npz'
   if saved.exists():
    with np.load(saved) as cams:
     assert np.array_equal(cams['normalized_c2w_before_alignment'],poses.numpy()),row['sample_id']
    existing_checks+=1
 except Exception as e:errors.append({'scene':scene,'trajectory':rig,'error':str(e)})
result={'policy':POLICY,'planned_windows':len(plan),'checked_windows':len(rows),'counts':dict(counts),
        'existing_completed_pose_arrays_exactly_equal':existing_checks,'errors':errors,'rows':rows}
p=root/'outputs/dataset_generation_ctx17/camxtime_baseline_audit.json';p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
assert len(rows)==len(plan) and not errors
# Rotation-only input must retain its rotation; a zero baseline is not an identity-pose instruction.
a=np.repeat(np.eye(4)[None],33,axis=0);a[:,:3,3]=[10,20,30]
for i,angle in enumerate(np.linspace(0,.8,33)):
 c,s=np.cos(angle),np.sin(angle);a[i,:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
p,scale,info=normalize_camxtime_poses(a,range(0,33,2),range(33),raw['baseline_min'],raw['baseline_max'])
assert scale==1 and not torch.allclose(p[-1,:3,:3],torch.eye(3)) and torch.allclose(p[0],torch.eye(4),atol=1e-5)
assert torch.allclose(p[:,:3,3],torch.zeros(33,3),atol=1e-5)
print('Stationary-center rotation-preservation check passed')
