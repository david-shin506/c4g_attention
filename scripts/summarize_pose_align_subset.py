#!/usr/bin/env python3
"""Independently validate saved comparison pixels and plot diagnostic figures."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/pose_align_spring_camxtime_subset'
report=json.loads((OUT/'report.json').read_text()); spec=json.loads((OUT/'run_spec.json').read_text())
assert report['state']=='complete' and len(report['rows'])==27
seen=set(); samples=defaultdict(list)
for row in report['rows']:
 key=(row['dataset'],row['sample_id'],row['time_index'],row['camera_index'],row['group'])
 assert key not in seen; seen.add(key)
 arrays={}
 for name in ['gt','before','after']:
  with Image.open(OUT/row[f'{name}_path']) as im:
   assert im.mode=='RGB' and im.size==(480,480)
   arrays[name]=np.asarray(im,dtype=np.float64)
 for name in ['before','after']:
  mse=np.mean((arrays[name]-arrays['gt'])**2)
  value=float(10*np.log10(255**2/mse))
  assert abs(value-row[f'{name}_psnr_db'])<1e-10
 source=json.loads(Path(row['source_manifest']).read_text())
 if row['dataset']=='spring':
  j=source['target_indices'].index(row['time_index'])
  path=ROOT.parents[1]/'C4G_prediction_dataset/Spring_480x480'/source['prediction_paths'][j]
 else:
  group='target' if row['group']=='fixed_camera_target' else 'ctx'
  item=next(i for i in source['groups'][group] if i['time_index']==row['time_index'] and i['camera_index']==row['camera_index'])
  path=ROOT/'outputs/camxtime_preflight/pilot'/item['prediction_path']
 with Image.open(path) as im: delta=np.abs(np.array(im).astype(float)-arrays['before'])
 assert delta.max()<=1 and delta.mean()<=.01
 row['baseline_vs_saved_max_uint8_error']=float(delta.max())
 row['baseline_vs_saved_mean_uint8_error']=float(delta.mean())
 with np.load((OUT/row['before_path']).parent/'poses.npz') as p:
  assert all(np.isfinite(a).all() for a in p.values())
  assert np.allclose(p['before'][...,3,:],[0,0,0,1])
  assert np.allclose(p['after'][...,3,:],[0,0,0,1],rtol=0,atol=1e-5)  # float32 repeated matrix inversions
 samples[row['dataset'],row['sample_id']].append(row)
for dataset,summ in report['summary'].items():
 for group,entry in summ.items():
  rows=[r for r in report['rows'] if r['dataset']==dataset and (group=='all' or r['group']==group)]
  assert len(rows)==entry['frames']
  for name in ['before','after','delta']:
   assert abs(np.mean([r[f'{name}_psnr_db'] for r in rows])-entry[f'{name}_psnr_db'])<1e-10
qual=OUT/'qualitative'; qual.mkdir(exist_ok=True)
for (dataset,sid),rows in samples.items():
 rows=sorted(rows,key=lambda x:(x['group']=='fixed_camera_target',x['time_index']))
 fig,axes=plt.subplots(len(rows),3,figsize=(11,3.55*len(rows)),layout='constrained')
 for i,row in enumerate(rows):
  for j,name in enumerate(['gt','before','after']):
   with Image.open(OUT/row[f'{name}_path']) as im: axes[i,j].imshow(np.array(im))
   label={'gt':'GT','before':'Before align','after':'After align'}[name]
   metric='' if name=='gt' else f" | {row[name+'_psnr_db']:.2f} dB"
   axes[i,j].set_title(f'{label}{metric}\n{row["group"]}, c={row["camera_index"]}, t={row["time_index"]}',fontsize=10)
   axes[i,j].axis('off')
 fig.suptitle(f'{dataset}: {sid}\nGT-assisted eval pose optimization; fixed K and Gaussians',fontsize=12)
 fig.savefig(qual/f'{sid}.jpg',dpi=100)
 plt.close(fig)
with (OUT/'per_frame.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(report['rows'][0])); writer.writeheader();writer.writerows(report['rows'])
validation={'passed':True,'views':27,'pngs_fully_decoded':81,'checks':['saved-pixel PSNR independently recomputed',
 'baseline agrees with existing export <= 1 uint8 level and <= 0.01 mean level',
 'summary counts and means','finite homogeneous poses','unique view identities'],
 'max_baseline_difference':max(r['baseline_vs_saved_max_uint8_error'] for r in report['rows'])}
(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
lines=['# Pose alignment subset comparison','',
 'Completed on existing 480x480 exports. Spring uses 12 context / 23 targets; CamxTime uses 17 context / 33 timestamps. '
 'Three windows per dataset, three timestamps per window. Spring: first, middle context, adjacent interpolated timestamp. '
 'CamxTime: the same relative categories, with both diagonal ctx and fixed-camera target rendered. Existing data were not changed.','',
 '| Dataset / group | Frames | Before PSNR | After PSNR | Change |','|---|---:|---:|---:|---:|']
for dataset,summary in report['summary'].items():
 for group,r in summary.items():
  lines.append(f'| {dataset} / {group} | {r["frames"]} | {r["before_psnr_db"]:.3f} | {r["after_psnr_db"]:.3f} | +{r["delta_psnr_db"]:.3f} |')
lines+=['','## Method','',
 'The body of ModelWrapper.test_step_align_single_timestamp was executed unchanged via AST extraction; only progress-bar display was suppressed. '
 'The exact body and hash are saved. Checkpoint loss settings are unchanged: because no dynamic mask is passed by the eval align method, '
 'LossDynamicMask becomes (1+3)*MSE, plus 0.05*VGG LPIPS. Adam rotation/translation lr=0.005; pose_align_steps=100 means up to 1000 iterations, '
 'with the original patience-based stopping (minimum 101 iterations). Final iterate is reported, not the best PSNR-selected iterate. '
 'All Gaussian tensors were produced without gradients and remain fixed; only extrinsics are optimized. K and input RGB are unchanged.','',
 'This is GT-assisted evaluation, not GT-free inference. CamxTime camera poses are optimized independently per timestamp, '
 'so this does not enforce one temporally fixed pose per physical camera. The provisional camera mapping remains unchanged. '
 'Small selected subsets do not establish full-dataset performance.','',
 'Baseline reinference was checked against the saved predictions. All saved comparison PNGs and their PSNR were independently validated. '
 'See validation.json, report.json, per_frame.csv, and qualitative/*.jpg.','']
(OUT/'README.md').write_text('\n'.join(lines))
print(json.dumps({'summary':report['summary'],'validation':validation,'elapsed_seconds':report['elapsed_seconds']},indent=2))
