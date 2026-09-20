#!/usr/bin/env python3
"""Fixed-K TTO diagnosis. No mutations of production exports or parameters."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch import nn
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from export_spring_vace import ROOT,DEFAULT_RUN,build_encoder,move,save_png,atomic_json
import eval_pose_align_dataset_subset as alignmod
from eval_pose_align_dataset_subset import load_rgb,pixels,psnr
from src.misc.benchmarker import Benchmarker
from src.loss.loss_dynamicmask import LossDynamicMask,LossDynamicMaskCfg,LossDynamicMaskCfgWrapper
from src.loss.loss_lpips import LossLpips,LossLpipsCfg,LossLpipsCfgWrapper

OUT=ROOT/'outputs/camxtime_tto_diagnostic'
DATA=ROOT.parents[1]/'C4G_prediction_dataset/CamxTime_480x480'
SID='Scene003_camera-trajectory-01_022_054_target_c027'

class TraceDecoder(nn.Module):
 def __init__(self,base):super().__init__();self.base=base;self.trace=[];self.best=None
 def forward(self,*args,**kw):
  out=self.base(*args,**kw);self.last=out;self.pose=args[1].detach().clone();return out
class TraceLoss(nn.Module):
 def __init__(self,base,observer,last=False):super().__init__();self.base=base;self.observer=observer;self.last=last
 def forward(self,*args,**kw):
  loss=self.base(*args,**kw);o=self.observer
  if not self.last:o.partial=float(loss.detach())
  else:
   objective=o.partial+float(loss.detach());gt=kw['target_image'];mse=((o.last.color.detach()-gt)**2).mean(dim=(2,3,4))[0]
   row={'iteration':len(o.trace),'objective':objective,'psnr_db':(-10*torch.log10(mse)).tolist()};o.trace.append(row)
   if o.best is None or objective<o.best['objective']:
    o.best={**row,'rgb':pixels(o.last.color[0]),'pose':o.pose.cpu().numpy()}
  return loss

def main():
 OUT.mkdir(exist_ok=True,parents=True);torch.set_num_threads(4);torch.manual_seed(111123)
 m=json.loads((DATA/'samples'/SID/'sample.json').read_text())
 with np.load(DATA/m['camera_path']) as c:cam={k:c[k].copy() for k in c.files}
 times=cam['target_time_indices'].tolist();ctx=cam['context_time_indices'];K=torch.from_numpy(cam['render_K']);near=float(cam['near']);far=float(cam['far'])
 context={'image':torch.stack([load_rgb(DATA/p) for p in m['encoder_input_paths']]),'index':torch.from_numpy(ctx),'extrinsics':torch.from_numpy(cam['context_extrinsics']),'intrinsics':K.repeat(17,1,1),'near':torch.full((17,),near),'far':torch.full((17,),far)}
 raw=OmegaConf.to_container(OmegaConf.load(DEFAULT_RUN/'.hydra/config.yaml'),resolve=True)
 encoder,decoder,step=build_encoder(raw,DEFAULT_RUN/'checkpoints/epoch_2-step_45000.ckpt');encoder.requires_grad_(False)
 with torch.no_grad():
  context=encoder.get_data_shim()(move(default_collate([{'context':context}])))['context'];gs=encoder(context,step,target_timestamps=torch.tensor(times,device='cuda'),visualization_dump=None)
 torch.save({t:{k:v.detach().cpu() if v is not None else None for k,v in vars(g).items()} for t,g in gs.items()},OUT/'gaussians.pt')
 args=[K.repeat(2,1,1)[None].cuda(),torch.full((1,2),near,device='cuda'),torch.full((1,2),far,device='cuda'),(480,480)]
 rows=[]
 for i,t in enumerate(times):
  gt=pixels(torch.stack([load_rgb(DATA/m['groups'][g][i]['gt_path']) for g in ['ctx','target']]))
  poses0=torch.from_numpy(cam['normalized_c2w_before_alignment'][[t,27]])[None].cuda()
  poses1=torch.from_numpy(np.stack([cam['ctx_aligned_extrinsics'][i],cam['target_aligned_extrinsics'][i]]))[None].cuda()
  with torch.no_grad():
   pre=pixels(decoder(gs[t],poses0,*args).color[0]);post=pixels(decoder(gs[t],poses1,*args).color[0])
  for j,group in enumerate(['ctx','target']):
   saved=np.array(Image.open(DATA/m['groups'][group][i]['prediction_path']));err=np.abs(saved.astype(int)-post[j].astype(int))
   assert err.max()<=1 and err.mean()<.01,(t,group,err.max(),err.mean())
   rel=np.linalg.inv(poses0[0,j].cpu().numpy())@poses1[0,j].cpu().numpy();rot=float(np.degrees(np.arccos(np.clip((np.trace(rel[:3,:3])-1)/2,-1,1))))
   row={'time':t,'group':group,'before_psnr_db':psnr(pre[j],gt[j]),'after_psnr_db':psnr(post[j],gt[j]),'translation_delta':float(np.linalg.norm(rel[:3,3])),'rotation_delta_deg':rot,'saved_max_u8_error':int(err.max())};row['delta_psnr_db']=row['after_psnr_db']-row['before_psnr_db'];rows.append(row)
   for name,px in [('gt',gt[j]),('before',pre[j]),('after',post[j])]:save_png(OUT/'frames'/f'{group}_t{t:03d}_{name}.png',px)
 atomic_json(OUT/'before_after.json',{'sample_id':SID,'rows':rows});print('BEFORE_AFTER',json.dumps(rows),flush=True)
 # Replay the exact original TTO, observing the objective before each update.
 alignmod.OUT=OUT;align,method_hash=alignmod.original_align_method();observed=TraceDecoder(decoder)
 bases=[LossDynamicMask(LossDynamicMaskCfgWrapper(LossDynamicMaskCfg(**raw['loss']['dynamicmask']))).cuda(),LossLpips(LossLpipsCfgWrapper(LossLpipsCfg(**raw['loss']['lpips']))).cuda().eval()]
 losses=[TraceLoss(bases[0],observed),TraceLoss(bases[1],observed,True)]
 host=SimpleNamespace(encoder=encoder,decoder=observed,losses=losses,global_step=step,device=torch.device('cuda'),test_cfg=SimpleNamespace(**raw['test']),benchmarker=Benchmarker())
 traces=[]
 for i in [0,16,32]:
  t=times[i];observed.trace=[];observed.best=None
  target={'extrinsics':torch.from_numpy(cam['normalized_c2w_before_alignment'][[t,27]])[None].cuda(),'intrinsics':args[0],'near':args[1],'far':args[2],'image':torch.stack([load_rgb(DATA/m['groups'][g][i]['gt_path']) for g in ['ctx','target']])[None].cuda()}
  batch={'context':context,'target':target}
  out=align(host,batch,gs[t],torch.ones(2,dtype=torch.bool,device='cuda'),480,480)
  with torch.no_grad():
   final_obj=sum(float(l.forward(out,batch,gs[t],step,target_image=target['image'])) for l in bases)
  finalpx=pixels(out.color[0]);gt=pixels(target['image'][0]);best=observed.best
  result={'time':t,'iterations':len(observed.trace),'initial_objective':observed.trace[0]['objective'],'best_objective':best['objective'],'best_iteration':best['iteration'],'final_objective':final_obj,'best_psnr_db':[psnr(best['rgb'][j],gt[j]) for j in range(2)],'final_psnr_db':[psnr(finalpx[j],gt[j]) for j in range(2)],'trace':observed.trace}
  for j,group in enumerate(['ctx','target']):
   save_png(OUT/'frames'/f'{group}_t{t:03d}_best.png',best['rgb'][j]);save_png(OUT/'frames'/f'{group}_t{t:03d}_replay_final.png',finalpx[j])
  np.savez_compressed(OUT/f't{t:03d}_trace_poses.npz',best=best['pose'],final=observed.pose.cpu().numpy())
  traces.append(result);atomic_json(OUT/'trace.json',{'method_hash':method_hash,'traces':traces});print('TRACE',json.dumps({k:v for k,v in result.items() if k!='trace'}),flush=True)
 summary={g:{'frames':33,'before_psnr_db':float(np.mean([r['before_psnr_db'] for r in rows if r['group']==g])),'after_psnr_db':float(np.mean([r['after_psnr_db'] for r in rows if r['group']==g])),'worsened_frames':sum(r['delta_psnr_db']<0 for r in rows if r['group']==g)} for g in ['ctx','target']}
 atomic_json(OUT/'summary.json',{'state':'complete','sample_id':SID,'summary':summary});print('SUMMARY',json.dumps(summary),flush=True)
 for group in ['ctx','target']:
  canvas=Image.new('RGB',(4*320,3*345));draw=ImageDraw.Draw(canvas)
  for i,t in enumerate([22,38,54]):
   for j,name in enumerate(['gt','before','after','best']):
    canvas.paste(Image.open(OUT/'frames'/f'{group}_t{t:03d}_{name}.png').resize((320,320)),(j*320,i*345+25));draw.text((j*320,i*345),f'{group} t={t} {name}')
  canvas.save(OUT/f'{group}_comparison.jpg')
if __name__=='__main__':main()
