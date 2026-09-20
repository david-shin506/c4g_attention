#!/usr/bin/env python3
"""Isolated saved-sample reconstruction and render-only clip sweep; production unchanged."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from export_spring_vace import ROOT, DEFAULT_RUN, build_encoder, move, reuse_example, atomic_json, save_png
from eval_pose_align_dataset_subset import load_rgb,pixels,psnr
import src.model.decoder.cuda_splatting as raster

OUT=ROOT/'outputs/render_clip_diagnostic'
BASE=ROOT.parents[1]/'C4G_prediction_dataset'
SELECTION=[('spring','Spring_480x480','spring_0047_left_00000_00032_gap2'),('camxtime','CamxTime_480x480','Scene003_camera-trajectory-01_022_054_target_c027')]

def main():
 OUT.mkdir(exist_ok=True,parents=True); torch.set_num_threads(4); torch.manual_seed(111123)
 raw=OmegaConf.to_container(OmegaConf.load(DEFAULT_RUN/'.hydra/config.yaml'),resolve=True)
 encoder,decoder,step=build_encoder(raw,DEFAULT_RUN/'checkpoints/epoch_2-step_45000.ckpt'); encoder.requires_grad_(False)
 # Observe opacity/radii without altering raster inputs or returned tensors.
 original=raster.GaussianRasterizer; captured=[]
 class Observed(original):
  def forward(self,*a,**kw):
   out=super().forward(*a,**kw);captured.append((out[1].detach(),out[3].detach()));return out
 raster.GaussianRasterizer=Observed
 results=[]
 for dataset,dirname,sid in SELECTION:
  root=BASE/dirname;m=json.loads((root/'samples'/sid/'sample.json').read_text());dest=OUT/dataset;dest.mkdir(exist_ok=True)
  if dataset=='spring':
   times=m['target_indices'];ex=reuse_example(root,{'sample_id':sid,'start':times[0],'end':times[-1]});context=ex['context'];target=ex['target']
   gtpaths=m['target_image_paths'];predpaths=m['prediction_paths']
  else:
   with np.load(root/m['camera_path']) as c:
    times=c['target_time_indices'].tolist();ci=c['context_time_indices'].copy();K=torch.from_numpy(c['render_K'].copy());n=float(c['near']);f=float(c['far'])
    context={'image':torch.stack([load_rgb(root/p) for p in m['encoder_input_paths']]),'index':torch.from_numpy(ci),'extrinsics':torch.from_numpy(c['context_extrinsics'].copy()),'intrinsics':K.repeat(17,1,1),'near':torch.full((17,),n),'far':torch.full((17,),f)}
    target={'extrinsics':torch.from_numpy(c['target_aligned_extrinsics'].copy()),'intrinsics':K.repeat(33,1,1),'near':torch.full((33,),n),'far':torch.full((33,),f)}
   gtpaths=[x['gt_path'] for x in m['groups']['target']];predpaths=[x['prediction_path'] for x in m['groups']['target']]
  with torch.no_grad():
   context=encoder.get_data_shim()(move(default_collate([{'context':context}])))['context']
   gs=encoder(context,step,target_timestamps=torch.tensor(times,device='cuda'),visualization_dump=None)
   for idx in [0,1,8,16,24,32]:
    t=times[idx];v={k:x[idx:idx+1][None].cuda() for k,x in target.items() if k!='index'}
    gt=np.array(Image.open(root/gtpaths[idx]));saved=np.array(Image.open(root/predpaths[idx]));save_png(dest/f't{t:03d}_gt.png',gt)
    means=gs[t].means[0];w2c=v['extrinsics'][0,0].inverse();z=(means@w2c[:3,:3].T+w2c[:3,3])[:,2]
    row={'dataset':dataset,'sample_id':sid,'time':t,'near':v['near'].item(),'far':v['far'].item(),'gaussians':len(z),'z_quantiles':torch.quantile(z,torch.tensor([0.,.001,.01,.1,.5,.9,.99,1.],device='cuda')).tolist(),'z_behind_fraction':(z<=0).float().mean().item(),'z_below_effective_near_fraction':(z<=v['near'].item()*.2).float().mean().item(),'variants':{}}
    base=None
    for name,nscale,fscale in [('baseline',1.,1.),('near_div10',.1,1.),('near_div100',.01,1.),('far_mul100',1.,100.)]:
     captured.clear();output=decoder(gs[t],v['extrinsics'],v['intrinsics'],v['near']*nscale,v['far']*fscale,(480,480));px=pixels(output.color[0])[0]
     radii,alpha=captured[-1];alpha=alpha.cpu().numpy().squeeze()
     if base is None:
      base=px;diff=np.abs(px.astype(int)-saved.astype(int));row['saved_match_max']=int(diff.max());row['saved_match_mean']=float(diff.mean())
      if diff.max()>1 or diff.mean()>.01:raise ValueError(f'Baseline mismatch {row}')
      save_png(dest/f't{t:03d}_alpha.png',np.repeat(np.uint8(np.clip(alpha,0,1)*255)[...,None],3,axis=2))
     delta=np.abs(px.astype(int)-base.astype(int)); row['variants'][name]={'psnr_db':psnr(px,gt),'mean_abs_delta_u8':float(delta.mean()),'max_abs_delta_u8':int(delta.max()),'alpha_lt_01_fraction':float(np.mean(alpha<.1)),'alpha_lt_05_fraction':float(np.mean(alpha<.5)),'visible_gaussians':int((radii>0).sum()),'black_fraction':float(np.mean(px.max(axis=-1)<=3))}
     save_png(dest/f't{t:03d}_{name}.png',px)
    results.append(row);atomic_json(OUT/'report.json',{'state':'running','rows':results}); print(json.dumps(row),flush=True)
  del gs,context
 atomic_json(OUT/'report.json',{'state':'complete','rows':results})
 for dataset,_,_ in SELECTION:
  rows=[r for r in results if r['dataset']==dataset];canvas=Image.new('RGB',(5*240,len(rows)*265));d=ImageDraw.Draw(canvas)
  for i,r in enumerate(rows):
   for j,name in enumerate(['gt','baseline','near_div100','far_mul100','alpha']):
    im=Image.open(OUT/dataset/f't{r["time"]:03d}_{name}.png').resize((240,240));canvas.paste(im,(j*240,i*265+25));d.text((j*240,i*265),f't={r["time"]} {name}')
  canvas.save(OUT/f'{dataset}_comparison.jpg')
if __name__=='__main__': main()
