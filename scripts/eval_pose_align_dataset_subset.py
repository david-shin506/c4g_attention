#!/usr/bin/env python3
"""Compare original eval pose alignment on saved Spring/CamxTime samples.
The original method is AST-extracted unchanged, avoiding unrelated training imports.
"""
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import ast
import hashlib
import json
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from export_spring_vace import ROOT, DEFAULT_RUN, build_encoder, move, save_png, atomic_json, reuse_example
from src.misc.cam_utils import update_pose
from src.misc.benchmarker import Benchmarker
from src.loss.loss_dynamicmask import LossDynamicMask, LossDynamicMaskCfg, LossDynamicMaskCfgWrapper
from src.loss.loss_lpips import LossLpips, LossLpipsCfg, LossLpipsCfgWrapper

OUT=ROOT/'outputs/pose_align_spring_camxtime_subset'
SPRING=ROOT.parents[1]/'C4G_prediction_dataset/Spring_480x480'
CAM=ROOT/'outputs/camxtime_preflight/pilot'

def load_rgb(path):
    with Image.open(path) as im:
        assert im.mode=='RGB'
        return torch.from_numpy(np.array(im)).permute(2,0,1).float()/255

def pixels(rgb):
    return (rgb.detach().clamp(0,1)*255).round().byte().permute(0,2,3,1).cpu().numpy()

def psnr(a,b):
    mse=np.mean((a.astype(np.float64)-b.astype(np.float64))**2)
    return float(10*np.log10(255**2/mse)) if mse else float('inf')

class ObservedDecoder(nn.Module):
    def __init__(self,decoder):
        super().__init__(); self.base=decoder; self.iterations=0; self.final_poses=None
    def forward(self,*args,**kwargs):
        if kwargs.get('cam_rot_delta') is not None: self.iterations+=1
        else: self.final_poses=args[1].detach().cpu().numpy().copy()
        return self.base.forward(*args,**kwargs)

def original_align_method():
    path=ROOT/'src/model/model_wrapper.py'; source=path.read_text()
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelWrapper')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='test_step_align_single_timestamp')
    module=ast.Module(body=[method],type_ignores=[])
    scope={'torch':torch,'nn':nn,'rearrange':rearrange,'update_pose':update_pose,
           'tqdm':lambda sequence,**kwargs:sequence}
    exec(compile(module,str(path),'exec'),scope)
    method_text=ast.get_source_segment(source,method)
    (OUT/'original_align_method.py.txt').write_text(method_text+'\n')
    return scope[method.name],hashlib.sha256(method_text.encode()).hexdigest()

def select():
    manifests=defaultdict(list)
    for p in sorted((SPRING/'samples').glob('*/sample.json')):
        m=json.loads(p.read_text()); manifests[m['scene']].append((p,m))
    scenes=sorted(manifests); selected=[]
    for i in [0,(len(scenes)-1)//2,len(scenes)-1]:
        rows=sorted(manifests[scenes[i]],key=lambda x:x[1]['target_indices'][0])
        p,m=rows[len(rows)//2]; ts=m['target_indices']; s=ts[0]
        selected.append(('spring',p,[s,s+10,s+11]))
    for scene in ['Scene001','Scene002','Scene003']:
        p=next((CAM/'samples').glob(f'{scene}_*_044_076_*/sample.json'))
        selected.append(('camxtime',p,[44,60,61]))
    return selected

def example_for(dataset,path):
    m=json.loads(path.read_text())
    if dataset=='spring':
        ts=m['target_indices']
        ex=reuse_example(SPRING,{'sample_id':m['sample_id'],'start':ts[0],'end':ts[-1]})
        views={t:[{'group':'ctx_input' if t in m['context_indices'] else 'target_intermediate',
                   'camera':0,'gt_path':str(SPRING/m['target_image_paths'][j]),
                   'before_saved_path':str(SPRING/m['prediction_paths'][j]),
                   'extrinsics':ex['target']['extrinsics'][j], 'intrinsics':ex['target']['intrinsics'][j],
                   'near':ex['target']['near'][j],'far':ex['target']['far'][j]}]
               for j,t in enumerate(ts)}
        return m,ex['context'],ts,views
    with np.load(CAM/m['camera_path']) as c:
        poses=torch.from_numpy(c['normalized_c2w'].copy()); K=torch.from_numpy(c['render_K'].copy())
        near=float(c['near']); far=float(c['far'])
    contexts=m['context_time_indices']
    context={'image':torch.stack([load_rgb(CAM/p) for p in m['encoder_input_paths']]),
             'index':torch.tensor(contexts),'extrinsics':poses[contexts],
             'intrinsics':K.expand(len(contexts),-1,-1).clone(),
             'near':torch.full((len(contexts),),near),'far':torch.full((len(contexts),),far)}
    views=defaultdict(list)
    for group in ['ctx','target']:
        for item in m['groups'][group]:
            c,t=item['camera_index'],item['time_index']
            label=('ctx_input' if t in contexts else 'ctx_intermediate') if group=='ctx' else 'fixed_camera_target'
            views[t].append({'group':label,'camera':c,'gt_path':str(CAM/item['gt_path']),
                            'before_saved_path':str(CAM/item['prediction_path']),
                            'extrinsics':poses[c],'intrinsics':K,'near':torch.tensor(near),'far':torch.tensor(far)})
    return m,context,sorted(views),views

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8); torch.manual_seed(111123)
    align,method_hash=original_align_method()
    raw=OmegaConf.to_container(OmegaConf.load(DEFAULT_RUN/'.hydra/config.yaml'),resolve=True)
    selection=select()
    specification={'source_checkpoint':str(DEFAULT_RUN/'checkpoints/epoch_2-step_45000.ckpt'),
        'method':'ModelWrapper.test_step_align_single_timestamp (unchanged AST-extracted body; progress bar suppressed)',
        'method_sha256':method_hash,'pose_align_steps':raw['test']['pose_align_steps'],
        'max_optimizer_iterations':raw['test']['pose_align_steps']*10,
        'rot_lr':raw['test']['rot_opt_lr'],'trans_lr':raw['test']['trans_opt_lr'],
        'loss':raw['loss'],'effective_unmasked_objective':'4 * full RGB MSE + 0.05 * VGG LPIPS',
        'early_stop':'original: abs loss delta < 1e-5 for 10 consecutive iterations, i >= 100',
        'render_shape':[480,480],'scope':'3 windows per dataset; 3 timestamps per window; Spring 9 views, CamxTime 18 views',
        'spring_context_count':12,'spring_target_count':23,'camxtime_context_count':17,'camxtime_target_count':33,
        'selection':[{'dataset':d,'manifest':str(p),'timestamps':ts} for d,p,ts in selection],
        'caveats':['GT-assisted eval optimization, not GT-free inference.',
                   'CamxTime target pose optimized independently at each time; no temporal shared-pose constraint.',
                   'K and Gaussian geometry fixed; camera JSON mapping remains the existing provisional mapping.',
                   'Existing exported datasets unchanged; no large regeneration.']}
    spec_path=OUT/'run_spec.json'
    if spec_path.exists(): assert json.loads(spec_path.read_text())==specification
    else: atomic_json(spec_path,specification)
    atomic_json(OUT/'status.json',{'state':'loading_model','completed_views':0,'planned_views':27})
    encoder,decoder,step=build_encoder(raw,DEFAULT_RUN/'checkpoints/epoch_2-step_45000.ckpt')
    encoder.requires_grad_(False)
    losses=[LossDynamicMask(LossDynamicMaskCfgWrapper(LossDynamicMaskCfg(**raw['loss']['dynamicmask']))).cuda(),
            LossLpips(LossLpipsCfgWrapper(LossLpipsCfg(**raw['loss']['lpips']))).cuda().eval()]
    observed=ObservedDecoder(decoder)
    host=SimpleNamespace(encoder=encoder,decoder=observed,losses=losses,global_step=step,device=torch.device('cuda'),
                         test_cfg=SimpleNamespace(**raw['test']),benchmarker=Benchmarker())
    rows=[]; t0=time.monotonic()
    for dataset,path,chosen in selection:
        m,context,all_times,views=example_for(dataset,path)
        with torch.no_grad():
            context=encoder.get_data_shim()(move(default_collate([{'context':context}])))['context']
            gaussian=encoder(context,step,target_timestamps=torch.tensor(all_times,device='cuda'),visualization_dump=None)
        for t in chosen:
            dest=OUT/'samples'/dataset/m['sample_id']/f't{t:04d}'
            saved=dest/'result.json'
            if saved.exists():
                rows.extend(json.loads(saved.read_text())['views']); continue
            dest.mkdir(parents=True,exist_ok=True)
            v=views[t]
            target={k:torch.stack([item[k] for item in v])[None].cuda() for k in ['extrinsics','intrinsics','near','far']}
            target['image']=torch.stack([load_rgb(item['gt_path']) for item in v])[None].cuda()
            target['index']=torch.full((1,len(v)),t,dtype=torch.long,device='cuda')
            target['camera']=torch.tensor([[item['camera'] for item in v]],device='cuda')
            batch={'context':context,'target':target}
            with torch.no_grad():
                before=decoder(gaussian[t],target['extrinsics'],target['intrinsics'],target['near'],target['far'],(480,480)).color[0]
                before_px=pixels(before); gt_px=pixels(target['image'][0])
            for j,item in enumerate(v):
                with Image.open(item['before_saved_path']) as im: old=np.array(im)
                diff=np.abs(old.astype(int)-before_px[j].astype(int))
                if diff.max()>1 or diff.mean()>.01:
                    raise ValueError(f'Reinference baseline mismatch {dataset}/{m["sample_id"]}/{t}: {diff.max()}, {diff.mean()}')
            observed.iterations=0; tt=time.monotonic()
            output=align(host,batch,gaussian[t],torch.ones(len(v),dtype=torch.bool,device='cuda'),480,480)
            torch.cuda.synchronize(); seconds=time.monotonic()-tt
            after_px=pixels(output.color[0])
            np.savez_compressed(dest/'poses.npz',before=target['extrinsics'].cpu().numpy(),after=observed.final_poses,
                                intrinsics=target['intrinsics'].cpu().numpy(),camera_indices=[x['camera'] for x in v],time_index=t)
            current=[]
            for j,item in enumerate(v):
                tag=f'{item["group"]}_c{item["camera"]:03d}'
                for name,array in [('before',before_px[j]),('after',after_px[j]),('gt',gt_px[j])]:
                    save_png(dest/f'{tag}_{name}.png',array)
                row={'dataset':dataset,'sample_id':m['sample_id'],'scene':m['scene'],'time_index':t,'camera_index':item['camera'],
                     'group':item['group'],'before_psnr_db':psnr(before_px[j],gt_px[j]),'after_psnr_db':psnr(after_px[j],gt_px[j]),
                     'iterations':observed.iterations,'align_seconds_for_timestamp':seconds,
                     'before_path':str((dest/f'{tag}_before.png').relative_to(OUT)),
                     'after_path':str((dest/f'{tag}_after.png').relative_to(OUT)),
                     'gt_path':str((dest/f'{tag}_gt.png').relative_to(OUT)),'source_manifest':str(path)}
                row['delta_psnr_db']=row['after_psnr_db']-row['before_psnr_db']; current.append(row)
            atomic_json(saved,{'complete':True,'views':current}); rows.extend(current)
            atomic_json(OUT/'status.json',{'state':'running','completed_views':len(rows),'planned_views':27,
                        'elapsed_seconds':time.monotonic()-t0,'last':current})
            print(json.dumps(current),flush=True)
            del output,batch,target
        del gaussian,context
    summary={}
    for dataset in ['spring','camxtime']:
        subset=[r for r in rows if r['dataset']==dataset]
        summary[dataset]={}
        for group in ['all']+sorted({r['group'] for r in subset}):
            selected=subset if group=='all' else [r for r in subset if r['group']==group]
            summary[dataset][group]={'frames':len(selected),'before_psnr_db':float(np.mean([r['before_psnr_db'] for r in selected])),
                'after_psnr_db':float(np.mean([r['after_psnr_db'] for r in selected])),
                'delta_psnr_db':float(np.mean([r['delta_psnr_db'] for r in selected]))}
    assert len(rows)==27
    atomic_json(OUT/'report.json',{'state':'complete','summary':summary,'rows':rows,'elapsed_seconds':time.monotonic()-t0})
    atomic_json(OUT/'status.json',{'state':'complete','completed_views':27,'planned_views':27})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
