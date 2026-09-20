#!/usr/bin/env python3
"""Ablate near Gaussian layers in a separate diagnostic; not an export fix."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from omegaconf import OmegaConf
from export_spring_vace import ROOT,DEFAULT_RUN,save_png,atomic_json
from eval_pose_align_dataset_subset import pixels,psnr
from src.model.types import Gaussians
from src.model.decoder import get_decoder
from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
import src.model.decoder.cuda_splatting as raster
from diagnose_camxtime_tto import OUT,DATA,SID

def main():
 torch.set_num_threads(4);cfg=OmegaConf.to_container(OmegaConf.load(DEFAULT_RUN/'.hydra/config.yaml'),resolve=True)
 decoder=get_decoder(DecoderSplattingCUDACfg(**cfg['model']['decoder'])).cuda().eval()
 cache=torch.load(OUT/'gaussians.pt',map_location='cpu',weights_only=False)
 m=json.loads((DATA/'samples'/SID/'sample.json').read_text())
 with np.load(DATA/m['camera_path']) as c:cam={k:c[k].copy() for k in c.files}
 original=raster.GaussianRasterizer;captured=[]
 class Observed(original):
  def forward(self,*a,**kw):
   out=super().forward(*a,**kw);captured.append((out[1].detach(),out[3].detach(),out[4].detach()));return out
 raster.GaussianRasterizer=Observed
 result=[]
 with torch.no_grad():
  for t in [22,38,54]:
   g=Gaussians(**{k:v.cuda() if v is not None else None for k,v in cache[t].items()});i=t-22
   pose=torch.from_numpy(cam['target_aligned_extrinsics'][i])[None,None].cuda();K=torch.from_numpy(cam['render_K'])[None,None].cuda();near=torch.full((1,1),float(cam['near']),device='cuda');far=torch.full((1,1),float(cam['far']),device='cuda');args=[pose,K,near,far,(480,480)]
   w2c=pose[0,0].inverse();z=(g.means[0]@w2c[:3,:3].T+w2c[:3,3])[:,2];gt=np.array(Image.open(DATA/m['groups']['target'][i]['gt_path']))
   base=pixels(decoder(g,*args).color[0])[0];radii,alpha,touched=captured[-1]
   row={'time':t,'baseline_psnr':psnr(base,gt),'layers':[]}
   for threshold in [.5,1.,2.,3.,4.,5.,6.,8.]:
    mask=(z<threshold)&(z>0);fg=Gaussians(**vars(g));bg=Gaussians(**vars(g));fg.opacities=g.opacities*mask;bg.opacities=g.opacities*(~mask)
    layer=pixels(decoder(fg,*args).color[0])[0];fg_alpha=captured[-1][1].cpu().numpy().squeeze();remaining=pixels(decoder(bg,*args).color[0])[0]
    row['layers'].append({'z_lt':threshold,'gaussians':int(mask.sum()),'opacity_quantiles':torch.quantile(g.opacities[0,mask],torch.tensor([0.,.5,1.],device='cuda')).tolist() if mask.any() else [],'radius_quantiles':torch.quantile(radii[mask].float(),torch.tensor([0.,.5,1.],device='cuda')).tolist() if mask.any() else [],'pixel_fraction_alpha_gt_05':float(np.mean(fg_alpha>.5)),'removed_layer_psnr':psnr(remaining,gt)})
    for name,im in [('layer',layer),('removed',remaining)]:save_png(OUT/'layers'/f't{t:03d}_z{threshold}_{name}.png',im)
   result.append(row)
 atomic_json(OUT/'foreground_layers.json',{'scope':'Diagnostic only; deleting geometry can remove real content and is not a proposed production fix.','rows':result});print(json.dumps(result,indent=2))
 im=Image.new('RGB',(4*320,3*345));draw=ImageDraw.Draw(im)
 for i,t in enumerate([22,38,54]):
  for j,(title,p) in enumerate([('GT',OUT/'frames'/f'target_t{t:03d}_gt.png'),('current',OUT/'frames'/f'target_t{t:03d}_after.png'),('z<4 only',OUT/'layers'/f't{t:03d}_z4.0_layer.png'),('z<4 removed',OUT/'layers'/f't{t:03d}_z4.0_removed.png')]):
   im.paste(Image.open(p).resize((320,320)),(320*j,345*i+25));draw.text((j*320,i*345),f't={t} {title}')
 im.save(OUT/'foreground_comparison.jpg')
if __name__=='__main__':main()
