#!/usr/bin/env python3
"""Export trained per-window Gaussian VAE features with shared clean frame latents."""
import argparse,fcntl,hashlib,json,os,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from export_spring_vace import save_png
import numpy as np
import torch
from spring_vae_common import *
from src.model.vae_feature_lifting import VAEFeatureLifter


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DATA_ROOT)
    p.add_argument('--checkpoint',type=Path,default=RUN_ROOT/'best.ckpt');p.add_argument('--limit',type=int)
    p.add_argument('--fixed-bounds',action='store_true')
    p.add_argument('--save-rgb-pred',action='store_true',help='Also render native 480x480 RGB from the same Gaussian and camera as each latent prediction')
    p.add_argument('--allow-smoke-checkpoint',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);root=args.output.resolve()
    lock=(root/'.export.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((root/'prepare_status.json').read_text())['state']=='complete'
    raw,ds,plan,_=load_dataset(fixed_bounds=args.fixed_bounds)
    export_spec={'schema_version':1,'render_bounds':{'mode':'fixed' if args.fixed_bounds else 'baseline_scaled','near':0.1,'far':100.0,'divided_by_baseline':not args.fixed_bounds},'save_rgb_pred':args.save_rgb_pred,
                 'rgb_prediction_shape':[480,480] if args.save_rgb_pred else None,
                 'rgb_prediction_source':'same C4G Gaussian and target camera; native RGB rasterization, not VAE decoding' if args.save_rgb_pred else None}
    spec_path=root/'export_spec.json'
    if spec_path.exists():assert json.loads(spec_path.read_text())==export_spec,'Export settings differ; use a fresh output directory'
    else:
        if args.save_rgb_pred and any((root/'samples').glob('*/sample.json')):
            raise ValueError('Existing export lacks RGB predictions; use a fresh output directory')
        atomic_json(spec_path,export_spec)
    encoder,rgb_decoder,_=build_encoder(raw,DEFAULT_RUN/'checkpoints/epoch_2-step_45000.ckpt')
    lifter=VAEFeatureLifter(encoder).cuda()
    weights=torch.load(str(args.checkpoint),map_location='cpu')
    if weights['config']['smoke'] and not args.allow_smoke_checkpoint:
        raise ValueError('Smoke checkpoints cannot produce the final dataset')
    lifter.decoder.load_feature_state_dict(weights['features'])
    checkpoint_hash=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    provenance={'checkpoint':str(args.checkpoint.resolve()),'sha256':checkpoint_hash,'step':weights['step'],
                'validation':weights['validation'],
                'training_render_bounds':weights['config'].get('render_bounds','legacy baseline-scaled'),'smoke_checkpoint':weights['config']['smoke'],
                'base_checkpoint':weights['config']['checkpoint']}
    if (root/'feature_checkpoint.json').exists():assert json.loads((root/'feature_checkpoint.json').read_text())==provenance
    else:atomic_json(root/'feature_checkpoint.json',provenance)
    del weights
    started=time.monotonic();completed=[];new=0
    with ThreadPoolExecutor(max_workers=4) as png_pool, torch.no_grad():
        for row in plan:
            sid=row['sample_id'];sample=root/'samples'/sid;marker=sample/'sample.json'
            if marker.exists():
                m=json.loads(marker.read_text())
                assert m['complete'] and m['feature_checkpoint_sha256']==checkpoint_hash
                for rel in m['prediction_paths']:
                    a=np.load(root/rel,allow_pickle=False);assert a.shape==(16,60,60) and a.dtype==np.float16 and np.isfinite(a).all()
                if args.save_rgb_pred:
                    assert m['rgb_prediction_indices']==m['prediction_indices']
                    assert len(m['rgb_prediction_paths'])==len(m['prediction_indices'])
                    for rel in m['rgb_prediction_paths']:
                        with Image.open(root/rel) as im:
                            assert im.mode=='RGB' and im.size==(480,480);im.load()
                completed.append({'sample_id':sid,'path':str(marker.relative_to(root))});continue
            if args.limit is not None and new>=args.limit:break
            batch=make_batch(ds,row,lifter.encoder)
            ctx=batch['context']['index'][0].tolist();targets=batch['target']['index'][0].tolist()
            inputs=load_latents(root,row['scene'],ctx)[None]
            gs,features=lifter(batch['context'],inputs,batch['target']['index'][0])
            paths=[];mse=[];rgb_paths=[];rgb_jobs=[]
            for i,t in enumerate(targets):
                pred=render_feature(gs[t],features[t],batch['target'],i)[0]
                if not torch.isfinite(pred).all():raise ValueError(f'Nonfinite feature: {sid}/{t}')
                stored=pred.half()
                if not torch.isfinite(stored).all():raise ValueError('Feature exceeds FP16 range')
                rel=f'samples/{sid}/prediction/{t:05d}.npy';atomic_npy(root/rel,stored);paths.append(rel)
                gt=load_latents(root,row['scene'],[t])[0]
                mse.append(float((stored.float()-gt).square().mean()))
                if args.save_rgb_pred:
                    view=batch['target']
                    rgb=rgb_decoder(gs[t],view['extrinsics'][:,i:i+1],view['intrinsics'][:,i:i+1],
                                    view['near'][:,i:i+1],view['far'][:,i:i+1],(480,480)).color[0,0]
                    if not torch.isfinite(rgb).all():raise ValueError(f'Nonfinite RGB: {sid}/{t}')
                    pixels=(rgb.clamp(0,1)*255).round().byte().permute(1,2,0).cpu().numpy()
                    rgb_rel=f'samples/{sid}/rgb_prediction/{t:05d}.png';rgb_paths.append(rgb_rel)
                    rgb_jobs.append(png_pool.submit(save_png,root/rgb_rel,pixels))
            for job in rgb_jobs:job.result()
            camera_rel=f'samples/{sid}/cameras.npz';camera_path=root/camera_rel
            cam={f'{view}_{key}':batch[view][key][0].cpu().numpy() for view in ['context','target']
                 for key in ['index','extrinsics','intrinsics','near','far']}
            cam['source_w2c']=np.array([ds.scenes[row['scene']][i]['extrinsics'] for i in targets])
            tmp=camera_path.with_suffix('.npz.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,**cam)
            tmp.replace(camera_path)
            m={'schema_version':1,'complete':True,'sample_id':sid,'scene':row['scene'],'dataset':'spring',
               'context_camera':'left','target_camera':'left','context_indices':ctx,'target_indices':targets,
               'prediction_indices':targets,'index_base':0,'temporal_compression':False,'padding':False,
               'context_latent_paths':[latent_rel(row['scene'],i) for i in ctx],
               'target_latent_paths':[latent_rel(row['scene'],i) for i in targets],
               'prediction_paths':paths,'camera_path':camera_rel,'feature_checkpoint_sha256':checkpoint_hash,
               'feature_checkpoint_step':provenance['step'],'base_checkpoint_step':45000,
               'context_source_paths':[str(source_path(ds,row['scene'],i)) for i in ctx],
               'target_source_paths':[str(source_path(ds,row['scene'],i)) for i in targets],
               'render_bounds':export_spec['render_bounds'],
               'latent_shape':[16,60,60],'latent_dtype':'float16','mean_latent_mse':sum(mse)/len(mse),
               'feature_training_split':'validation_scene' if row['scene'] in {'0014','0038'} else 'training_scene'}
            if args.save_rgb_pred:
                m.update(rgb_prediction_paths=rgb_paths,rgb_prediction_indices=targets,
                         rgb_prediction_shape=[480,480],rgb_prediction_source=export_spec['rgb_prediction_source'])
            atomic_json(marker,m);new+=1;completed.append({'sample_id':sid,'path':str(marker.relative_to(root))})
            if new<=3 or len(completed)%10==0:
                elapsed=time.monotonic()-started
                progress={'state':'exporting','pid':os.getpid(),'completed':len(completed),'total':len(plan),
                          'seconds_per_new_sample':elapsed/new,'estimated_remaining_s':elapsed/new*(len(plan)-len(completed))}
                atomic_json(root/'status.json',progress);print(json.dumps(progress),flush=True)
    atomic_json(root/'index.json',{'schema_version':1,'samples':completed})
    atomic_json(root/'status.json',{'state':'complete' if len(completed)==len(plan) else 'partial',
                                  'completed':len(completed),'total':len(plan),'elapsed_s':time.monotonic()-started})

if __name__=='__main__':main()
