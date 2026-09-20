from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from PIL import Image
root=Path('/music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/intrinsic_test/C4G/outputs/camxtime_preflight')
pilot=root/'pilot'
report=json.loads((pilot/'report.json').read_text())
assert report['state']=='complete'
counts=defaultdict(int); groups=defaultdict(list); paths={}
plan=json.loads((root/'target_camera_plan.json').read_text())
lookup={(r['scene'],r['trajectory'],r['start']):r['target_camera_index'] for r in plan['windows']}
for path in sorted((pilot/'samples').glob('*/sample.json')):
 m=json.loads(path.read_text()); s=m['start']; ts=list(range(s,s+33)); ctx=ts[::2]; c=m['target_camera_index']
 assert m['complete'] and m['context_time_indices']==m['context_camera_indices']==ctx
 assert lookup[m['scene'],m['trajectory'],s]==c
 for group in ['ctx','target']:
  items=m['groups'][group]
  assert len(items)==33 and [i['time_index'] for i in items]==ts
  assert [i['camera_index'] for i in items]==(ts if group=='ctx' else [c]*33)
  for item in items:
   assert item['source_pose_key']==m['source_json_keys_by_camera_index'][item['camera_index']]
   arrays=[]
   for key in ['prediction_path','gt_path']:
    p=pilot/item[key]
    with Image.open(p) as im:
     assert im.mode=='RGB' and im.size==(480,480)
     arrays.append(np.asarray(im,dtype=np.float64))
    paths[str(p)]=(480,480)
   mse=np.mean((arrays[0]-arrays[1])**2)
   value=float(10*np.log10(255**2/mse))
   assert abs(value-item['psnr_db'])<1e-10
   subgroup=('ctx_input' if item['time_index'] in ctx else 'ctx_intermediate') if group=='ctx' else 'target'
   groups[subgroup].append(value)
   counts[group]+=1
 for p in m['encoder_input_paths']:
  with Image.open(pilot/p) as im:
   assert im.size==(224,224) and im.mode=='RGB'; im.load()
  paths[str(pilot/p)]=(224,224)
 with np.load(pilot/m['camera_path']) as a:
  assert all(np.isfinite(v).all() for v in a.values())
  assert np.allclose(a['normalized_c2w'][s],np.eye(4),atol=1e-5)
  assert np.array_equal(a['context_time_indices'],ctx)
  assert np.array_equal(a['target_camera_indices'],[c]*33)
  assert np.allclose(a['render_K'],[[.8767,0,.5],[0,.8767,.5],[0,0,1]])
 # At the one overlapping ctx/target pair both branches must have identical predictions.
 cp=pilot/m['groups']['ctx'][c-s]['prediction_path']
 tp=pilot/m['groups']['target'][c-s]['prediction_path']
 with Image.open(cp) as x,Image.open(tp) as y: assert np.array_equal(np.array(x),np.array(y))
 counts['samples']+=1
assert counts['samples']==9 and counts['ctx']==counts['target']==297
for k,v in groups.items(): assert abs(np.mean(v)-report['psnr'][k]['mean_frame_psnr_db'])<1e-10
validation={'passed':True,'samples':9,'prediction_frames':594,'unique_pngs_fully_decoded':len(paths),
 'checks':['counts and pair indices','all referenced PNG RGB shape and decode','PSNR recomputed from saved PNGs','finite camera arrays','first-context identity','fixed DAVIS K','target selection matches full seed plan','overlapping predictions exactly equal']}
(pilot/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
base=json.loads((root/'storage_estimate.json').read_text())
parts=base['estimates']['mean_proxy']['parts_bytes'].copy()
parts['prediction']=base['counts']['prediction_pngs_without_overlap_deduplication']*report['prediction_mean_bytes']
total=sum(parts.values())
new={'basis':'9-window / 3-scene CamxTime prediction mean; 288-frame / 8-scene CamxTime GT mean; exact global GT dedup counts',
     'pilot_prediction_mean_bytes':report['prediction_mean_bytes'],'parts_bytes':parts,'decimal_TB':total/1e12,'TiB':total/2**40,
     'with_25_percent_margin_TB':total*1.25/1e12,'planning_budget_TB':2.0,
     'caveat':'Small pilot; not full-dataset measurement or a guaranteed upper bound; camera mapping remains provisional.',
     'pilot_file_bytes':sum(p.stat().st_size for p in pilot.rglob('*') if p.is_file())}
(root/'storage_estimate_with_pilot.json').write_text(json.dumps(new,indent=2)+'\n')
print(json.dumps({'validation':validation,'updated_estimate':new,'psnr':report['psnr'],'pilot_elapsed_s':report['elapsed_seconds']},indent=2))
