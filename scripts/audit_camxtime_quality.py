#!/usr/bin/env python3
"""Snapshot completed exports and produce reproducible visual audit sheets."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,random,csv
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parents[1]/'C4G_prediction_dataset/CamxTime_480x480'
OUT=ROOT/'outputs/camxtime_quality_audit'
def main():
 OUT.mkdir(exist_ok=True,parents=True);rows=[];manifests={}
 for p in sorted((DATA/'samples').glob('*/sample.json')):
  m=json.loads(p.read_text())
  if not m.get('complete'):continue
  r={k:m[k] for k in ['sample_id','scene','trajectory','start','target_camera_index','baseline_scale']}
  for g in ['ctx','target']:
   a=np.array([x['psnr_db'] for x in m['groups'][g]],dtype=float);r[g+'_mean_psnr']=float(a.mean());r[g+'_min_psnr']=float(a.min());r[g+'_mid_psnr']=float(a[16])
  rows.append(r);manifests[m['sample_id']]=m
 random_rows=random.Random(20260913).sample(rows,min(64,len(rows)))
 low_rows=sorted(rows,key=lambda x:x['target_mean_psnr'])[:16]
 snapshot={'snapshot_utc':datetime.now(timezone.utc).isoformat(),'completed_samples':len(rows),'scenes':len({x['scene'] for x in rows}),'rigs':len({(x['scene'],x['trajectory']) for x in rows}),'scene_counts':dict(Counter(x['scene'] for x in rows)),'trajectory_counts':dict(Counter(x['trajectory'] for x in rows)),'rows':rows,'visual_random_seed':20260913,'random_sample_ids':[x['sample_id'] for x in random_rows],'low_sample_ids':[x['sample_id'] for x in low_rows],'summary':{}}
 for g in ['ctx','target']:
  a=np.array([r[g+'_mean_psnr'] for r in rows]);snapshot['summary'][g]={'mean':float(a.mean()),'quantiles':dict(zip(['min','p10','p25','p50','p75','p90','max'],np.quantile(a,[0,.1,.25,.5,.75,.9,1]).tolist())),'sample_mean_below':{str(v):int((a<v).sum()) for v in [15,17,18,20,22,25]}}
 (OUT/'snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
 with (OUT/'samples.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for kind,selected in [('random',random_rows),('low',low_rows)]:
  for page in range((len(selected)+7)//8):
   canvas=Image.new('RGB',(1120,4*305),(25,25,25));draw=ImageDraw.Draw(canvas)
   for local,r in enumerate(selected[page*8:page*8+8]):
    n=page*8+local;m=manifests[r['sample_id']];entry=m['groups']['target'][16];x=(local%2)*560;y=(local//2)*305
    label=f'{kind} #{n:02d} {r["scene"]} tr{r["trajectory"][-2:]} s{r["start"]:03d} c{r["target_camera_index"]:03d} mean {r["target_mean_psnr"]:.1f}'
    draw.text((x,y),label,fill='white');draw.text((x,y+12),'GT                         PRED (middle frame)',fill='white')
    for j,k in enumerate(['gt_path','prediction_path']):canvas.paste(Image.open(DATA/entry[k]).resize((280,280)),(x+280*j,y+25))
   canvas.save(OUT/f'{kind}_{page:02d}.jpg',quality=88)
 print(json.dumps({k:v for k,v in snapshot.items() if k not in ['rows','random_sample_ids','low_sample_ids']},indent=2),flush=True)
if __name__=='__main__':main()
