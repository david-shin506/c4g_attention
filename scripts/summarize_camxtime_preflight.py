#!/usr/bin/env python3
"""Build qualitative comparison figures/videos and hypothetical subset estimates."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import subprocess
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/camxtime_preflight'
PILOT=OUT/'pilot'
QUAL=OUT/'qualitative'
QUAL.mkdir(parents=True,exist_ok=True)

full=json.loads((OUT/'storage_estimate_with_pilot.json').read_text())
base=json.loads((OUT/'storage_estimate.json').read_text())
plan=json.loads((OUT/'target_camera_plan.json').read_text())
rigs=defaultdict(list)
for r in plan['windows']: rigs[r['scene'],r['trajectory']].append(r)
meta_per_window=full['parts_bytes']['metadata_allowance']/base['counts']['windows']
results=[]
for n in [1,2,3,4,5,44]:
    gt_count=input_count=0
    for (scene,rig),rows in rigs.items():
        # Hypothetical deterministic random subset, nested as n increases.
        ranked=sorted(rows,key=lambda r:hashlib.sha256(f"window-subset:111123:{scene}:{rig}:{r['start']}".encode()).digest())
        clean=set(); inputs=set()
        for row in ranked[:n]:
            s=row['start']; c=row['target_camera_index']
            clean.update((t,t) for t in range(s,s+33))
            clean.update((c,t) for t in range(s,s+33))
            inputs.update((t,t) for t in range(s,s+33,2))
        gt_count+=len(clean); input_count+=len(inputs)
    windows=len(rigs)*n
    parts={'prediction':windows*66*full['pilot_prediction_mean_bytes'],
           'gt':gt_count*base['camxtime_gt_measurement']['mean_bytes'],
           'input':input_count*base['camxtime_input_measurement']['mean_bytes'],
           'metadata':windows*meta_per_window}
    total=sum(parts.values())
    results.append({'windows_per_scene_trajectory':n,'total_windows':windows,'prediction_count':windows*66,
                    'unique_gt_count':gt_count,'unique_input_count':input_count,'parts_bytes':parts,
                    'total_GB':total/1e9,'fraction_of_full_bytes':total/(full['decimal_TB']*1e12),
                    'full_divided_by_subset_bytes':full['decimal_TB']*1e12/total})
report={'hypothetical_only':True,'selection':'stable hash ranking of 44 starts, seed 111123; no generation plan changed',
        'basis':'measured pilot prediction mean, sampled CamxTime GT/input mean, exact subset dedup counts',
        'rows':results}
(OUT/'subset_storage_estimates.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)

for scene in ['Scene001','Scene002','Scene003']:
    matches=list((PILOT/'samples').glob(f'{scene}_*_044_076_*/sample.json'))
    assert len(matches)==1
    m=json.loads(matches[0].read_text()); groups=m['groups']
    fig,axes=plt.subplots(4,3,figsize=(12,15.5),layout='constrained',facecolor='#eeeeee')
    for col,j in enumerate([0,16,32]):
        for row,(group,key,label) in enumerate([('ctx','gt_path','Diagonal GT'),('ctx','prediction_path','Diagonal prediction'),
                                             ('target','gt_path','Fixed-camera GT'),('target','prediction_path','Fixed-camera prediction')]):
            item=groups[group][j]
            with Image.open(PILOT/item[key]) as im: axes[row,col].imshow(np.array(im))
            axes[row,col].set_title(f'{label}\nc={item["camera_index"]}, t={item["time_index"]}',fontsize=11)
            axes[row,col].axis('off')
    fig.suptitle(f'{scene} / trajectory-01 / start=44 / target camera={m["target_camera_index"]}\n'
                 '480x480 native render; fixed DAVIS K; provisional sorted pose mapping',fontsize=13)
    fig.savefig(QUAL/f'{scene}_start044_comparison.jpg',dpi=110)
    plt.close(fig)
    # A diagnostic 2x2 video, slowed to 8 fps for inspection, with fixed labels.
    import cv2
    width,height=960,1016
    video=QUAL/f'{scene}_start044_comparison.mp4'
    process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{width}x{height}','-r','8','-i','pipe:0','-an','-c:v','libx264','-crf','20','-preset','fast',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    try:
        for j in range(33):
            canvas=np.full((height,width,3),242,dtype=np.uint8)
            for row,group in enumerate(['ctx','target']):
                item=groups[group][j]
                for col,key in enumerate(['gt_path','prediction_path']):
                    y=row*508
                    with Image.open(PILOT/item[key]) as im: canvas[y+28:y+508,col*480:(col+1)*480]=np.array(im)
                    label=f'{group} {"GT" if col==0 else "prediction"}  c={item["camera_index"]} t={item["time_index"]}'
                    cv2.putText(canvas,label,(col*480+10,y+20),cv2.FONT_HERSHEY_SIMPLEX,.53,(0,0,0),1,cv2.LINE_AA)
            process.stdin.write(canvas.tobytes())
    finally: process.stdin.close()
    assert process.wait()==0
    print('qualitative',video,flush=True)
(QUAL/'README.md').write_text('Qualitative pilot comparisons: Scene001/002/003, trajectory-01, start 44.\n\n'
 'JPGs show t=44/60/76, with diagonal GT/prediction then fixed-camera GT/prediction. MP4s show all 33 timestamps at 8 fps for inspection (not original FPS). '\
 'Columns: GT then prediction. Rows: diagonal ctx then fixed target. Original indices are annotated. '\
 'All three scenes are shown without quality-based filtering. Camera mapping remains provisional.\n')
