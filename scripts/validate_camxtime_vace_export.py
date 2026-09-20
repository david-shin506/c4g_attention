#!/usr/bin/env python3
"""Audit CamxTime aligned export manifests, cameras and image shapes."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import json
import struct
import time
import numpy as np
from PIL import Image
from export_spring_vace import atomic_json

def verify_context_origin(cams, metadata):
 """Accept tiny float32 cancellation only when source reconstruction is exact."""
 origin=cams['context_extrinsics'][0]
 residual=float(np.max(np.abs(origin-np.eye(4))))
 if np.allclose(origin,np.eye(4),atol=1e-5): return residual,False
 assert residual<=1e-4, f'Excessive context-origin residual: {residual}'
 import torch
 poses=torch.tensor(cams['source_c2w']@np.diag([1,-1,-1,1]),dtype=torch.float32)
 poses[:,:3,3]/=metadata['baseline_scale']
 indices=metadata['context_camera_indices']; first=indices[0]
 expected=(torch.linalg.inv(poses[first:first+1])@poses).numpy()
 assert np.array_equal(cams['normalized_c2w_before_alignment'],expected), 'Source pose reconstruction differs'
 assert np.array_equal(cams['context_extrinsics'],expected[indices]), 'Context poses differ from source reconstruction'
 return residual,True

def main(args):
 root=args.root.resolve(); spec=json.loads((root/'dataset.json').read_text())
 plan=json.loads((root/'plan.json').read_text())['samples']; planned={r['sample_id']:r for r in plan}
 status=json.loads((root/'status.json').read_text())
 if not args.allow_partial:
  assert status.get('completed')==len(plan) and status.get('remaining',0)==0, 'Generation is incomplete'
 checked={}; checked_lock=Lock(); count=0; start=time.monotonic(); max_origin_residual=0.; source_rechecked=0
 def check_sample(p):
  m=json.loads(p.read_text()); row=planned[m['sample_id']]; s=row['start']; c=row['target_camera_index']; ts=list(range(s,s+33))
  assert m['complete'] and m['pose_alignment'] and m['context_time_indices']==m['context_camera_indices']==ts[::2]
  assert m['target_camera_index']==c and len(m['encoder_input_paths'])==17
  paths=[(p,(224,224)) for p in m['encoder_input_paths']]
  for group in ['ctx','target']:
   items=m['groups'][group]
   assert [i['time_index'] for i in items]==ts
   assert [i['camera_index'] for i in items]==(ts if group=='ctx' else [c]*33)
   for item in items:
    assert item['source_pose_key']==m['source_json_keys_by_camera_index'][item['camera_index']]
    assert item['is_actual_input_pair']==(item['camera_index']==item['time_index'] and item['time_index'] in ts[::2])
    paths.extend((item[key],(480,480)) for key in ['prediction_path','gt_path'])
    if args.decode:
     with Image.open(root/item['prediction_path']) as im: pred=np.array(im,dtype=np.float64)
     with Image.open(root/item['gt_path']) as im: gt=np.array(im,dtype=np.float64)
     mse=np.mean((pred-gt)**2); psnr=10*np.log10(255**2/mse) if mse else np.inf
     assert abs(psnr-float(item['psnr_db']))<1e-8 or (np.isinf(psnr) and np.isinf(float(item['psnr_db'])))
  for rel,shape in paths:
   with checked_lock:
    if rel in checked: assert checked[rel]==shape; continue
    checked[rel]=shape
   path=(root/rel).resolve(); assert path.is_relative_to(root)
   with path.open('rb') as f: header=f.read(24)
   assert header[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',header[16:24])==shape
   if args.decode:
    with Image.open(path) as im: assert im.mode=='RGB'; im.load()
  with np.load(root/m['camera_path']) as cams:
   for key in ['context_extrinsics','ctx_aligned_extrinsics','target_aligned_extrinsics','render_K','alignment_iterations']:
    assert np.isfinite(cams[key]).all()
   assert cams['context_extrinsics'].shape==(17,4,4)
   assert cams['ctx_aligned_extrinsics'].shape==cams['target_aligned_extrinsics'].shape==(33,4,4)
   residual,rechecked=verify_context_origin(cams,m)
   assert np.array_equal(cams['context_time_indices'],ts[::2]) and np.array_equal(cams['target_time_indices'],ts)
   assert np.array_equal(cams['target_camera_indices'],[c]*33)
   assert np.allclose(cams['render_K'],spec['intrinsics'])
   assert (cams['alignment_iterations']>=101).all() and (cams['alignment_iterations']<=1000).all()
  return residual,rechecked
 # Parallelize independent samples to hide shared-disk metadata latency.
 paths=sorted((root/'samples').glob('*/sample.json'))
 with ThreadPoolExecutor(max_workers=args.workers) as pool:
  for residual,rechecked in pool.map(check_sample,paths):
   count+=1; max_origin_residual=max(max_origin_residual,residual); source_rechecked+=int(rechecked)
   if count%100==0:
    atomic_json(root/'validation_status.json',{'state':'running','completed_samples':count,'planned_samples':len(plan)})
 if not args.allow_partial: assert count==len(plan)==spec['planned_samples']
 result={'passed':True,'full_export':count==len(plan),'samples':count,'planned_samples':len(plan),
         'prediction_pngs':count*66,'unique_pngs_checked':len(checked),'decoded_and_psnr_checked':args.decode,
         'max_context_origin_residual':max_origin_residual,'float32_origin_source_rechecked':source_rechecked,
         'elapsed_seconds':time.monotonic()-start}
 atomic_json(root/('validation_partial.json' if args.allow_partial else 'validation.json'),result)
 atomic_json(root/'validation_status.json',{'state':'complete','completed_samples':count,'planned_samples':len(plan),'passed':True})
 print(json.dumps(result),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--allow-partial',action='store_true');p.add_argument('--decode',action='store_true');p.add_argument('--workers',type=int,default=16)
 main(p.parse_args())
