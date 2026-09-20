#!/usr/bin/env python3
"""Nine-window CamxTime pilot. Provisional sorted pose mapping; never full export."""
from collections import defaultdict
import hashlib
from pathlib import Path
import time
import json
import cv2
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from export_spring_vace import ROOT, DEFAULT_RUN, build_encoder, save_png, atomic_json, move
from estimate_camxtime_storage import target_camera, SEED

SOURCE=ROOT/'datasets/CamxTime'
OUT=ROOT/'outputs/camxtime_preflight/pilot'

def read_frames(rig, requests):
    result={}
    for camera, times in requests.items():
        cap=cv2.VideoCapture(str(rig/f'cam{camera+1:03d}_full_motion.mp4'))
        try:
            for t in range(max(times)+1):
                ok, pixels=cap.read()
                if not ok: raise ValueError(f'Decode failure {rig}/{camera}/{t}')
                if t in times:
                    result[camera,t]=Image.fromarray(cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB))
        finally: cap.release()
    return result

def main():
    start_time=time.monotonic()
    torch.set_num_threads(8); cv2.setNumThreads(1); torch.manual_seed(SEED)
    raw=OmegaConf.to_container(OmegaConf.load(DEFAULT_RUN/'.hydra/config.yaml'),resolve=True)
    cfg=raw['dataset']['spring']; checkpoint=DEFAULT_RUN/'checkpoints/epoch_2-step_45000.ckpt'
    assert cfg['make_baseline_1'] and cfg['relative_pose'] and raw['force_davis_intrinsics']
    OUT.mkdir(parents=True,exist_ok=True)
    spec={'dataset':'camxtime','pilot_only':True,'split':None,'seed':SEED,'context_count':17,
          'context_gap':2,'ctx_render_count':33,'target_count':33,'start_stride':2,
          'output_image_shape':[480,480],'encoder_image_shape':[224,224],
          'checkpoint':str(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,
          'source_root':str(SOURCE),'index_base':0,'target_policy':'one uniform deterministic camera from dense 33 candidates; see target_camera_plan.json',
          'pose_mapping':'PROVISIONAL: numeric sorted JSON keys paired with cam001..120; C2W right-multiplied by diag(1,-1,-1,1)',
          'pose_normalization':'max distance from first context camera -> 1; first context camera relative; near/far divided by same scale',
          'intrinsics':'fixed normalized DAVIS K; source native K saved for audit only',
          'rgb_processing':'512x512 decoded RGB, full image LANCZOS resize to 224 input and 480 GT',
          'time_embedding':'unchanged checkpoint encoder defaults, including target_1 normalization',
          'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    atomic_json(OUT/'dataset.json',spec)
    atomic_json(OUT/'status.json',{'state':'loading_model','planned_samples':9,'completed_samples':0})
    encoder,decoder,step=build_encoder(raw,checkpoint)
    K=torch.tensor([[.8767,0,.5],[0,.8767,.5],[0,0,1]],dtype=torch.float32)
    rows=[]; all_scores=defaultdict(list)
    with torch.inference_mode():
        for scene in ['Scene001','Scene002','Scene003']:
            rig=SOURCE/scene/'camera-trajectory-01'
            camera_file=rig/f'{rig.name}-camera.json'
            data=json.loads(camera_file.read_text()); keys=sorted(data['trajectory'],key=int)
            raw_poses=np.array([data['trajectory'][key]['c2w'] for key in keys])
            for start in [0,44,86]:
                t0=time.monotonic(); times=list(range(start,start+33)); contexts=times[::2]
                camera=target_camera(scene,rig.name,start)
                sid=f'{scene}_{rig.name}_{start:03d}_{start+32:03d}_target_c{camera:03d}'
                dest=OUT/'samples'/sid
                if (dest/'sample.json').exists(): raise RuntimeError(f'Pilot already exists: {dest}')
                dense=[(t,t) for t in times]; targets=[(camera,t) for t in times]
                requests=defaultdict(set)
                for c,t in dense+targets: requests[c].add(t)
                images=read_frames(rig,requests)
                input_pixels=[np.array(images[i,i].resize((224,224),Image.Resampling.LANCZOS)) for i in contexts]
                poses=torch.tensor(raw_poses@np.diag([1,-1,-1,1]),dtype=torch.float32)
                scale=(poses[contexts[1:],:3,3]-poses[contexts[0],:3,3]).norm(dim=1).max()
                if not cfg['baseline_min']<=scale<=cfg['baseline_max']:
                    raise ValueError(f'Baseline out of range {sid}: {scale}')
                poses[:,:3,3]/=scale
                poses=torch.linalg.inv(poses[start:start+1])@poses
                context={'image':torch.from_numpy(np.stack(input_pixels)).permute(0,3,1,2).float()/255,
                         'extrinsics':poses[contexts],'intrinsics':K.expand(17,-1,-1).clone(),
                         'index':torch.tensor(contexts),'camera':torch.tensor(contexts),
                         'near':torch.full((17,),float(cfg.get('near',.1)))/scale,
                         'far':torch.full((17,),float(cfg.get('far',100)))/scale}
                batch=encoder.get_data_shim()(move(default_collate([{'context':context}])))
                torch.cuda.synchronize(); gpu0=time.monotonic()
                gs=encoder(batch['context'],step,target_timestamps=torch.tensor(times,device='cuda'),visualization_dump=None)
                gpu_views={}
                for group,pairs in [('ctx',dense),('target',targets)]:
                    pix=[]
                    for c,t in pairs:
                        rgb=decoder(gs[t],poses[c:c+1][None].cuda(),K[None,None].cuda(),
                            torch.tensor([[float(cfg.get('near',.1))/scale]],device='cuda'),
                            torch.tensor([[float(cfg.get('far',100))/scale]],device='cuda'),(480,480)).color[0,0]
                        if not torch.isfinite(rgb).all(): raise ValueError(f'Nonfinite RGB {sid}')
                        pix.append((rgb.clamp(0,1)*255).round().byte().permute(1,2,0))
                    gpu_views[group]=torch.stack(pix).cpu().numpy()
                torch.cuda.synchronize(); gpu_s=time.monotonic()-gpu0
                metadata={'complete':False,'pilot_only':True,'sample_id':sid,'scene':scene,'trajectory':rig.name,
                          'start':start,'seed':SEED,'target_camera_index':camera,'context_time_indices':contexts,
                          'context_camera_indices':contexts,'source_camera_json':str(camera_file),
                          'source_camera_json_sha256':hashlib.sha256(camera_file.read_bytes()).hexdigest(),
                          'source_json_keys_by_camera_index':keys,'baseline_scale':float(scale),'checkpoint_step':step}
                groups={}
                for group,pairs in [('ctx',dense),('target',targets)]:
                    items=[]
                    for j,(c,t) in enumerate(pairs):
                        gt=np.array(images[c,t].resize((480,480),Image.Resampling.LANCZOS))
                        pred=gpu_views[group][j]
                        pred_path=dest/group/f'c{c:03d}_t{t:03d}.png'
                        gt_path=OUT/'clean'/scene/rig.name/f'c{c:03d}_t{t:03d}.png'
                        save_png(pred_path,pred)
                        if not gt_path.exists(): save_png(gt_path,gt)
                        mse=np.mean((pred.astype(np.float64)-gt.astype(np.float64))**2)
                        psnr=float(10*np.log10(255**2/mse)) if mse else float('inf')
                        subgroup=('ctx_input' if t in contexts else 'ctx_intermediate') if group=='ctx' else 'target'
                        all_scores[subgroup].append(psnr)
                        items.append({'camera_index':c,'time_index':t,'source_video':str(rig/f'cam{c+1:03d}_full_motion.mp4'),
                                      'source_pose_key':keys[c],'prediction_path':str(pred_path.relative_to(OUT)),
                                      'gt_path':str(gt_path.relative_to(OUT)),'is_actual_input_pair':c==t and t in contexts,'psnr_db':psnr})
                    groups[group]=items
                input_paths=[]
                for t,pixels in zip(contexts,input_pixels):
                    p=OUT/'encoder_input'/scene/rig.name/f'c{t:03d}_t{t:03d}.png'
                    if not p.exists(): save_png(p,pixels)
                    input_paths.append(str(p.relative_to(OUT)))
                np.savez_compressed(dest/'cameras.npz',source_c2w=raw_poses,source_K=np.array(data['intrinsics']['K']),
                    normalized_c2w=poses.numpy(),render_K=K.numpy(),context_camera_indices=contexts,
                    context_time_indices=contexts,ctx_camera_indices=times,ctx_time_indices=times,
                    target_camera_indices=[camera]*33,target_time_indices=times,
                    near=float(cfg.get('near',.1))/float(scale),far=float(cfg.get('far',100))/float(scale))
                metadata.update(groups=groups,encoder_input_paths=input_paths,camera_path=str((dest/'cameras.npz').relative_to(OUT)),complete=True)
                atomic_json(dest/'sample.json',metadata)
                rows.append({'sample_id':sid,'seconds':time.monotonic()-t0,'gpu_seconds':gpu_s,
                             'ctx_mean_psnr_db':float(np.mean([r['psnr_db'] for r in groups['ctx']])),
                             'target_mean_psnr_db':float(np.mean([r['psnr_db'] for r in groups['target']]))})
                atomic_json(OUT/'status.json',{'state':'running','planned_samples':9,'completed_samples':len(rows),'last':rows[-1]})
                print(json.dumps(rows[-1]),flush=True)
                del gs,batch,gpu_views,rgb,images
    predictions=[p.stat().st_size for p in (OUT/'samples').glob('*/*/*.png')]
    report={'state':'complete','pilot_only':True,'samples':rows,'prediction_png_count':len(predictions),
            'prediction_mean_bytes':float(np.mean(predictions)),
            'psnr':{group:{'count':len(v),'mean_frame_psnr_db':float(np.mean(v))} for group,v in all_scores.items()},
            'elapsed_seconds':time.monotonic()-start_time,
            'caveats':['provisional camera mapping','three scenes only, not representative full-dataset quality','fixed DAVIS K as requested']}
    assert len(rows)==9 and len(predictions)==594
    atomic_json(OUT/'report.json',report)
    atomic_json(OUT/'status.json',{'state':'complete','planned_samples':9,'completed_samples':9})
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
