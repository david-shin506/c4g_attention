#!/usr/bin/env python3
"""Cache shared independent-frame Wan VAE latents for Spring 8-context windows."""
import argparse,fcntl,hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from spring_vae_common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DATA_ROOT);p.add_argument('--batch-size',type=int,default=4);args=p.parse_args()
    torch.set_num_threads(4)
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw,ds,plan,rejected=load_dataset()
    keys=sorted({(r['scene'],i) for r in plan for i in range(r['start'],r['end']+1)})
    spec={'schema_version':1,'dataset':'spring','source_root':str(ds.data_root),
          'base_checkpoint':str(DEFAULT_RUN/'checkpoints/epoch_2-step_45000.ckpt'),
          'vae_checkpoint':str(VAE_WEIGHTS),'vae_sha256':hashlib.sha256(VAE_WEIGHTS.read_bytes()).hexdigest(),
          'vae_source':str(VAE_SOURCE),'vae_source_sha256':hashlib.sha256(VAE_SOURCE.read_bytes()).hexdigest(),
          'context_count':8,'target_count':15,'context_gap':2,'start_stride':2,'planned_samples':len(plan),
          'index_base':0,'image_shape':[480,480],'encoder_image_shape':[224,224],
          'latent_shape':[16,60,60],'latent_dtype':'float16','vae_compute_dtype':'bfloat16',
          'vae_encoding':'independent frames; cache reset between encode calls; batch contains different independent images',
          'latent_normalization':'WanVideoVAE.single_encode applies (mu-mean)/std; never per-vector L2 normalized',
          'crop_box_after_1920x1080_resize':list(CROP_BOX),'crop_resize':'PIL LANCZOS to 480x480',
          'feature_input_downsample':'bilinear 60x60 ->16x16, align_corners=False',
          'feature_render_resolution':[60,60],'feature_render_low_pass':0.3,
          'temporal_compression':False,'padding':False,'unique_clean_frames':len(keys),
          'force_davis_intrinsics':True,'pose_normalization':'baseline=1; first context camera relative frame',
          'predictions':'per-window; clean GT/reference shared per original frame'}
    if (root/'dataset.json').exists():
        assert json.loads((root/'dataset.json').read_text())==spec,'Existing dataset specification differs'
    else:atomic_json(root/'dataset.json',spec)
    atomic_json(root/'plan.json',{'samples':plan,'rejected':rejected})
    atomic_json(root/'clean_index.json',{'frames':[{'scene':s,'index':i,'source_path':str(source_path(ds,s,i)),'path':latent_rel(s,i)} for s,i in keys]})
    pending=[]
    for s,i in keys:
        path=root/latent_rel(s,i)
        if path.exists():
            a=np.load(path,allow_pickle=False);assert a.shape==(16,60,60) and a.dtype==np.float16 and np.isfinite(a).all()
        else:pending.append((s,i))
    vae=load_vae();started=time.monotonic();done=len(keys)-len(pending)
    with ThreadPoolExecutor(max_workers=4) as pool,torch.inference_mode():
        for start in range(0,len(pending),args.batch_size):
            items=pending[start:start+args.batch_size]
            images=list(pool.map(read_square,[source_path(ds,s,i) for s,i in items]))
            x=torch.stack(images).unsqueeze(2).to(device='cuda',dtype=torch.bfloat16)
            z=vae.single_encode(x,device='cuda')[:,:,0]
            assert z.shape==(len(items),16,60,60) and torch.isfinite(z).all()
            for (s,i),latent in zip(items,z):atomic_npy(root/latent_rel(s,i),latent.half())
            done+=len(items)
            if start==0 or done%100<args.batch_size or done==len(keys):
                elapsed=time.monotonic()-started
                progress={'state':'running','completed_frames':done,'total_frames':len(keys),'elapsed_s':elapsed,
                          'remaining_s':elapsed/(start+len(items))*(len(pending)-start-len(items))}
                atomic_json(root/'prepare_status.json',progress);print(json.dumps(progress),flush=True)
    # Confirm independent-frame semantics: batched encoding agrees with encoding one image alone.
    s,i=keys[0]
    with torch.inference_mode():
        solo=vae.single_encode(read_square(source_path(ds,s,i))[None,:,None].cuda().bfloat16(),'cuda')[0,:,0].float()
    cached=torch.from_numpy(np.load(root/latent_rel(s,i))).cuda().float()
    delta=float((solo-cached).abs().max());assert delta<0.08,delta
    atomic_json(root/'prepare_status.json',{'state':'complete','completed_frames':len(keys),'total_frames':len(keys),
                'elapsed_s':time.monotonic()-started,'independent_frame_batch_max_abs_diff':delta})
    print('Shared framewise VAE cache complete',flush=True)

if __name__=='__main__':main()
