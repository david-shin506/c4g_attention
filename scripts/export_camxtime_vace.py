#!/usr/bin/env python3
"""Resumable CamxTime 17-context / 33-time RGB export with original eval pose align."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from PIL import Image
from export_spring_vace import ROOT, DEFAULT_RUN, atomic_json, save_png, build_encoder, move
from pilot_camxtime_vace import read_frames
import eval_pose_align_dataset_subset as alignment
from src.loss.loss_dynamicmask import LossDynamicMask, LossDynamicMaskCfg, LossDynamicMaskCfgWrapper
from src.loss.loss_lpips import LossLpips, LossLpipsCfg, LossLpipsCfgWrapper
from src.misc.benchmarker import Benchmarker
from camxtime_pose_normalization import normalize_camxtime_poses, POLICY
from camxtime_sharding import partition_plan

SOURCE=ROOT/'datasets/CamxTime'
DEFAULT_OUT=ROOT.parents[1]/'C4G_prediction_dataset/CamxTime_480x480'
SEED=111123
# Keep this export on its original decoder while shared training code evolves.
PINNED_DECODER=ROOT/'scripts/camxtime_compat/decoder_splatting_cuda.py'
PINNED_DECODER_SHA256='ac300cd5a49e0356698a3204703771be2fb9f69a874219f8966d7ff748ad9c70'

def build_export_encoder(raw, checkpoint):
    if hashlib.sha256(PINNED_DECODER.read_bytes()).hexdigest()!=PINNED_DECODER_SHA256:
        raise ValueError('Pinned CamxTime decoder content changed')
    encoder, shared_decoder, step=build_encoder(raw,checkpoint)
    module_spec=importlib.util.spec_from_file_location('src.model.decoder.camxtime_export_pinned',PINNED_DECODER)
    module=importlib.util.module_from_spec(module_spec)
    import sys
    sys.modules[module_spec.name]=module
    module_spec.loader.exec_module(module)
    decoder=module.DecoderSplattingCUDA(shared_decoder.cfg).cuda().eval()
    return encoder,decoder,step

def build_plan():
    original=json.loads((ROOT/'outputs/camxtime_preflight/target_camera_plan.json').read_text())
    assert original['seed']==SEED
    groups=defaultdict(list)
    for row in original['windows']: groups[row['scene'],row['trajectory']].append(row)
    plan=[]
    for (scene,rig),rows in sorted(groups.items()):
        selected=sorted(rows,key=lambda r:hashlib.sha256(f"window-subset:{SEED}:{scene}:{rig}:{r['start']}".encode()).digest())[:3]
        for row in sorted(selected,key=lambda r:r['start']):
            s=row['start']; c=row['target_camera_index']
            plan.append({**row,'sample_id':f'{scene}_{rig}_{s:03d}_{s+32:03d}_target_c{c:03d}'})
    assert len(groups)==1664 and len(plan)==4992
    return plan

def save_npz(path,**arrays):
    with path.with_suffix('.tmp').open('wb') as f: np.savez_compressed(f,**arrays)
    os.replace(path.with_suffix('.tmp'),path)

def render_pixels(rgb):
    if not torch.isfinite(rgb).all(): raise ValueError('Nonfinite rendered RGB')
    return (rgb.detach().clamp(0,1)*255).round().byte().permute(0,2,3,1).cpu().numpy()

def main(args):
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.export.lock').open('a')
    fcntl.flock(lock,(fcntl.LOCK_SH if args.num_shards>1 else fcntl.LOCK_EX)|fcntl.LOCK_NB)
    state_dir=out if args.num_shards==1 else out/'workers'/f'shards_{args.num_shards}'/f'worker_{args.shard_id}'
    state_dir.mkdir(parents=True,exist_ok=True)
    if args.num_shards>1:
        worker_lock=(state_dir/'.worker.lock').open('a'); fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(8); torch.manual_seed(SEED)
    import cv2
    cv2.setNumThreads(1)
    raw=OmegaConf.to_container(OmegaConf.load(DEFAULT_RUN/'.hydra/config.yaml'),resolve=True)
    cfg=raw['dataset']['spring']; plan=build_plan()
    alignment.OUT=state_dir
    align,method_hash=alignment.original_align_method()
    checkpoint=DEFAULT_RUN/'checkpoints/epoch_2-step_45000.ckpt'
    spec={'schema_version':1,'dataset':'camxtime','split':None,'index_base':0,'source_root':str(SOURCE),
        'checkpoint':str(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,'checkpoint_mtime_ns':checkpoint.stat().st_mtime_ns,
        'config_sha256':hashlib.sha256((DEFAULT_RUN/'.hydra/config.yaml').read_bytes()).hexdigest(),
        'seed':SEED,'context_count':17,'context_gap':2,'ctx_render_count':33,'target_count':33,'start_stride':2,
        'windows_per_scene_trajectory':3,'planned_samples':len(plan),'output_image_shape':[480,480],'encoder_image_shape':[224,224],
        'window_selection':'three smallest SHA256(window-subset:111123:scene:trajectory:start) among starts 0,2,...,86',
        'target_selection':'saved deterministic uniform one of start..start+32, from preflight target_camera_plan.json',
        'pose_mapping':'Numeric sorted JSON keys paired with cam001..cam120; c2w @ diag(1,-1,-1,1). Provisional source provenance; geometrically tested on subset.',
        'pose_alignment':{'enabled':True,'method':'ModelWrapper.test_step_align_single_timestamp','sha256':method_hash,
                          'per_timestamp':True,'shared_across_time':False,'max_iterations':raw['test']['pose_align_steps']*10,
                          'rotation_lr':raw['test']['rot_opt_lr'],'translation_lr':raw['test']['trans_opt_lr'],
                          'objective':'4*MSE + 0.05*VGG_LPIPS, GT-assisted','intrinsics_optimized':False},
        'intrinsics':[[.8767,0,.5],[0,.8767,.5],[0,0,1]],'rgb_processing':'Full 512x512 RGB -> LANCZOS 224 input and 480 GT',
        'padding':'No padding: target already 4k+1=33','time_embedding':'Existing encoder target_1 normalization unchanged',
        'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256((PINNED_DECODER if p==ROOT/'src/model/decoder/decoder_splatting_cuda.py' else p).read_bytes()).hexdigest() for p in [ROOT/'src/model/encoder/encoder_vggt.py',ROOT/'src/misc/cam_utils.py',ROOT/'src/model/decoder/decoder_splatting_cuda.py']}}
    with (out/'.metadata.lock').open('a') as metadata_lock:
        fcntl.flock(metadata_lock,fcntl.LOCK_EX)
        if (out/'dataset.json').exists():
            assert json.loads((out/'dataset.json').read_text())==spec,'Existing export specification differs'
        else: atomic_json(out/'dataset.json',spec)
        atomic_json(out/'decoder_implementation.json', {'path':str(PINNED_DECODER),'sha256':PINNED_DECODER_SHA256,'reason':'Preserve original export decoder independently of shared training changes.'})
        atomic_json(out/'pose_normalization_policy.json', {**POLICY, 'implementation_sha256': hashlib.sha256((ROOT/'scripts/camxtime_pose_normalization.py').read_bytes()).hexdigest()})
        atomic_json(out/'plan.json',{'samples':plan}); atomic_json(out/'source_config.json',raw)
    if args.num_shards>1:
        plan=partition_plan(plan,args.num_shards)[args.shard_id]
    done=set()
    for row in plan:
        p=out/'samples'/row['sample_id']/'sample.json'
        if p.exists():
            m=json.loads(p.read_text()); assert m['complete'] and m['sample_id']==row['sample_id']
            assert m['context_time_indices']==list(range(row['start'],row['start']+33,2))
            assert m['target_camera_index']==row['target_camera_index']
            done.add(row['sample_id'])
    start=time.monotonic(); timings=[]
    def status(state,**extra):
        value={'state':state,'pid':os.getpid(),'total':len(plan),'completed':len(done),'remaining':len(plan)-len(done),
               'elapsed_seconds':time.monotonic()-start,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra}
        value.update(shard_id=args.shard_id,num_shards=args.num_shards)
        atomic_json(state_dir/'status.json',value)
        return value
    status('loading_model')
    encoder,decoder,step=build_export_encoder(raw,checkpoint); encoder.requires_grad_(False)
    losses=[LossDynamicMask(LossDynamicMaskCfgWrapper(LossDynamicMaskCfg(**raw['loss']['dynamicmask']))).cuda(),
            LossLpips(LossLpipsCfgWrapper(LossLpipsCfg(**raw['loss']['lpips']))).cuda().eval()]
    observed=alignment.ObservedDecoder(decoder)
    host=SimpleNamespace(encoder=encoder,decoder=observed,losses=losses,global_step=step,device=torch.device('cuda'),
                         test_cfg=SimpleNamespace(**raw['test']),benchmarker=Benchmarker())
    K=torch.tensor(spec['intrinsics'],dtype=torch.float32)
    processed=0
    try:
      with ThreadPoolExecutor(max_workers=8) as pool:
       for row in plan:
        if row['sample_id'] in done: continue
        if args.limit is not None and processed>=args.limit: break
        t0=time.monotonic(); scene,trajectory,sid,s,c=row['scene'],row['trajectory'],row['sample_id'],row['start'],row['target_camera_index']
        dest=out/'samples'/sid; dest.mkdir(parents=True,exist_ok=True)
        rig=SOURCE/scene/trajectory; camera_file=rig/f'{trajectory}-camera.json'
        data=json.loads(camera_file.read_text()); keys=sorted(data['trajectory'],key=int)
        assert len(keys)==120 and all((rig/f'cam{i:03d}_full_motion.mp4').is_file() for i in range(1,121))
        times=list(range(s,s+33)); contexts=times[::2]
        requests=defaultdict(set)
        for t in times: requests[t].add(t); requests[c].add(t)
        images=read_frames(rig,requests)
        gt={pair:np.array(im.resize((480,480),Image.Resampling.LANCZOS)) for pair,im in images.items()}
        input_pixels=[np.array(images[t,t].resize((224,224),Image.Resampling.LANCZOS)) for t in contexts]
        clean_paths={pair:f'clean/{scene}/{trajectory}/c{pair[0]:03d}_t{pair[1]:03d}.png' for pair in gt}
        input_paths=[f'encoder_input/{scene}/{trajectory}/c{t:03d}_t{t:03d}.png' for t in contexts]
        jobs=[pool.submit(save_png,out/clean_paths[pair],image) for pair,image in gt.items() if not (out/clean_paths[pair]).exists()]
        jobs.extend(pool.submit(save_png,out/p,pix) for p,pix in zip(input_paths,input_pixels) if not (out/p).exists())
        for job in jobs: job.result()
        raw_poses=np.array([data['trajectory'][key]['c2w'] for key in keys])
        poses,scale,normalization=normalize_camxtime_poses(raw_poses,contexts,times,cfg['baseline_min'],cfg['baseline_max'])
        near=float(cfg.get('near',.1))/float(scale); far=float(cfg.get('far',100))/float(scale)
        context={'image':torch.from_numpy(np.stack(input_pixels)).permute(0,3,1,2).float()/255,
                 'index':torch.tensor(contexts),'extrinsics':poses[contexts], 'intrinsics':K.repeat(17,1,1),
                 'near':torch.full((17,),near),'far':torch.full((17,),far)}
        with torch.no_grad():
            context=encoder.get_data_shim()(move(default_collate([{'context':context}])))['context']
            gaussians=encoder(context,step,target_timestamps=torch.tensor(times,device='cuda'),visualization_dump=None)
        groups={'ctx':[],'target':[]}; aligned=[]; iterations=[]
        for t in times:
            frame_path=dest/'frames'/f't{t:03d}.json'
            if frame_path.exists():
                frame=json.loads(frame_path.read_text())
                assert frame['complete'] and frame['time_index']==t
                for entry in frame['items']: assert (out/entry['prediction_path']).is_file()
            else:
                cams=[t,c]
                target={'extrinsics':poses[cams][None].cuda(),'intrinsics':K.repeat(2,1,1)[None].cuda(),
                        'near':torch.full((1,2),near,device='cuda'),'far':torch.full((1,2),far,device='cuda'),
                        'image':torch.from_numpy(np.stack([gt[(i,t)] for i in cams])).permute(0,3,1,2)[None].float().cuda()/255}
                observed.iterations=0
                output=align(host,{'context':context,'target':target},gaussians[t],torch.ones(2,dtype=torch.bool,device='cuda'),480,480)
                pix=render_pixels(output.color[0]); items=[]
                for j,group in enumerate(['ctx','target']):
                    cam=cams[j]; rel=f'samples/{sid}/{group}/c{cam:03d}_t{t:03d}.png'
                    save_png(out/rel,pix[j])
                    mse=float(np.mean((pix[j].astype(np.float64)-gt[cam,t].astype(np.float64))**2))
                    items.append({'group':group,'camera_index':cam,'time_index':t,'source_pose_key':keys[cam],
                        'source_video':str(rig/f'cam{cam+1:03d}_full_motion.mp4'),'prediction_path':rel,'gt_path':clean_paths[cam,t],
                        'is_actual_input_pair':cam==t and t in contexts,'psnr_db':float(10*np.log10(255**2/mse)) if mse else 'Infinity'})
                frame={'complete':True,'time_index':t,'items':items,'aligned_c2w':observed.final_poses[0].tolist(),'iterations':observed.iterations}
                atomic_json(frame_path,frame)
                del output,target
            for entry in frame['items']: groups[entry['group']].append(entry)
            aligned.append(frame['aligned_c2w']); iterations.append(frame['iterations'])
            status('running',active_sample=sid,completed_timestamps=t-s+1,seconds_per_sample=float(np.mean(timings[-20:])) if timings else None)
        aligned=np.array(aligned,dtype=np.float32)
        if not np.isfinite(aligned).all(): raise ValueError(f'Nonfinite aligned poses: {sid}')
        save_npz(dest/'cameras.npz',source_c2w=raw_poses,source_K=np.array(data['intrinsics']['K']),source_pose_keys=np.array(keys),
                 normalized_c2w_before_alignment=poses.numpy(),context_extrinsics=poses[contexts].numpy(),
                 ctx_aligned_extrinsics=aligned[:,0],target_aligned_extrinsics=aligned[:,1],render_K=K.numpy(),
                 context_camera_indices=contexts,context_time_indices=contexts,ctx_camera_indices=times,ctx_time_indices=times,
                 target_camera_indices=[c]*33,target_time_indices=times,near=near,far=far,alignment_iterations=iterations)
        metadata={'schema_version':1,'dataset':'camxtime','complete':True,'sample_id':sid,'scene':scene,'trajectory':trajectory,
            'start':s,'seed':SEED,'index_base':0,'context_time_indices':contexts,'context_camera_indices':contexts,
            'target_camera_index':c,'groups':groups,'encoder_input_paths':input_paths,
            'source_camera_json':str(camera_file),'source_camera_json_sha256':hashlib.sha256(camera_file.read_bytes()).hexdigest(),
            'source_json_keys_by_camera_index':keys,'baseline_scale':float(scale),'pose_normalization':normalization,'checkpoint_step':step,'pose_alignment':True,
            'camera_path':f'samples/{sid}/cameras.npz','export_spec':'dataset.json'}
        atomic_json(dest/'sample.json',metadata); done.add(sid); processed+=1; timings.append(time.monotonic()-t0)
        progress={'sample_id':sid,'seconds':timings[-1],'mean_alignment_iterations':float(np.mean(iterations)),
                  'ctx_psnr_db':float(np.mean([r['psnr_db'] for r in groups['ctx']])),'target_psnr_db':float(np.mean([r['psnr_db'] for r in groups['target']]))}
        with (state_dir/'progress.jsonl').open('a') as f: f.write(json.dumps(progress)+'\n')
        print(json.dumps(status('running',**progress,estimated_remaining_hours=(len(plan)-len(done))*np.mean(timings[-20:])/3600)),flush=True)
        del gaussians,context,images,gt,jobs
      atomic_json(state_dir/'index.json',{'schema_version':1,'complete':len(done)==len(plan),
                    'samples':[{'sample_id':r['sample_id'],'path':f"samples/{r['sample_id']}/sample.json"} for r in plan if r['sample_id'] in done]})
      status('complete' if len(done)==len(plan) else 'paused_after_limit')
    except BaseException as e:
      status('failed',error=f'{type(e).__name__}: {e}'); raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=DEFAULT_OUT); p.add_argument('--limit',type=int)
    p.add_argument('--num-shards',type=int,default=1); p.add_argument('--shard-id',type=int,default=0)
    args=p.parse_args()
    if args.num_shards<1 or not 0<=args.shard_id<args.num_shards: p.error('invalid shard configuration')
    if args.limit is not None and args.limit<1: p.error('limit must be positive')
    main(args)
