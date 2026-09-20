#!/usr/bin/env python3
"""Read-only source probe: encode sampled RGB in memory and estimate export storage.
Prediction sizes are a Spring proxy until CamxTime calibration is verified.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import json
import shutil
import time
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'datasets/CamxTime'
SPRING = ROOT.parents[1] / 'C4G_prediction_dataset/Spring_480x480'
OUT = ROOT / 'outputs/camxtime_preflight'
SEED = 111123

def target_camera(scene, rig, start):
    # Rejection sampling avoids modulo bias; independent of enumeration/resume order.
    name = f'{SEED}:{scene}:{rig}:{start}'
    counter = 0
    while True:
        digest = hashlib.sha256(f'{name}:{counter}'.encode()).digest()
        for byte in digest:
            if byte < 231:
                return start + byte % 33
        counter += 1

def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    return {'count':len(a), 'mean_bytes':float(a.mean()),
            'p10_bytes':float(np.percentile(a,10)), 'p90_bytes':float(np.percentile(a,90)),
            'min_bytes':int(a.min()), 'max_bytes':int(a.max())}

def probe(path):
    cap = cv2.VideoCapture(str(path))
    rows = []
    try:
        for index in range(119):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode {path}:{index}')
            if index not in (0, 59, 118):
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            assert image.size == (512,512)
            row = {'path':str(path.relative_to(SOURCE)), 'time_index':index}
            for size, key in [(480,'gt_bytes'),(224,'encoder_bytes')]:
                buf = BytesIO()
                image.resize((size,size),Image.Resampling.LANCZOS).save(buf,format='PNG')
                row[key] = buf.tell()
            rows.append(row)
    finally:
        cap.release()
    return rows

def main():
    t0=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True)
    scenes=sorted(p for p in SOURCE.glob('Scene*') if p.is_dir())
    rigs=[p for scene in scenes for p in sorted(scene.glob('camera-trajectory-*')) if p.is_dir()]
    assert len(scenes)==416 and len(rigs)==1664
    starts=list(range(0,120-32,2))
    selected_scenes=[scenes[int(i)] for i in np.linspace(0,len(scenes)-1,8).round()]
    videos=[rig/f'cam{camera:03d}_full_motion.mp4'
            for scene in selected_scenes for rig in sorted(scene.glob('camera-trajectory-*'))
            for camera in (1,60,119)]
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        measured=[row for result in pool.map(probe,videos) for row in result]
    gt=summarize([r['gt_bytes'] for r in measured])
    inp=summarize([r['encoder_bytes'] for r in measured])
    print('CamxTime PNG measurement',json.dumps({'gt':gt,'input':inp}),flush=True)
    # Measure all existing Spring prediction sizes; metadata from sampled windows.
    spring_samples=sorted((SPRING/'samples').iterdir())
    sizes=[p.stat().st_size for sample in spring_samples for p in (sample/'prediction').glob('*.png')]
    pred=summarize(sizes)
    meta=np.mean([(p/'sample.json').stat().st_size+(p/'cameras.npz').stat().st_size for p in spring_samples[::17]])
    gt_count=0
    targets=[]
    for rig in rigs:
        pairs={(i,i) for i in range(119)}
        for start in starts:
            camera=target_camera(rig.parent.name,rig.name,start)
            pairs.update((camera,t) for t in range(start,start+33))
            targets.append({'scene':rig.parent.name,'trajectory':rig.name,'start':start,'target_camera_index':camera})
        gt_count+=len(pairs)
    windows=len(rigs)*len(starts)
    counts={'scenes':len(scenes),'trajectories':len(rigs),'windows_per_trajectory':len(starts),
            'windows':windows,'prediction_pngs_without_overlap_deduplication':windows*66,
            'prediction_pngs_with_ctx_target_overlap_deduplication':windows*65,
            'unique_gt_pngs':gt_count,'unique_encoder_input_pngs':len(rigs)*60,
            'sample_metadata_files':windows*2}
    estimates={}
    for label,pred_size,gt_size,input_size in [
        ('mean_proxy',pred['mean_bytes'],gt['mean_bytes'],inp['mean_bytes']),
        ('per_frame_p10_scenario',pred['p10_bytes'],gt['p10_bytes'],inp['p10_bytes']),
        ('per_frame_p90_scenario',pred['p90_bytes'],gt['p90_bytes'],inp['p90_bytes'])]:
        # x3 allowance for longer dual-camera lists + extra metadata/index files.
        parts={'prediction':windows*66*pred_size,'clean_gt':gt_count*gt_size,
               'encoder_input':len(rigs)*60*input_size,'metadata_allowance':windows*meta*3}
        total=sum(parts.values())
        estimates[label]={'parts_bytes':parts,'total_bytes':total,'decimal_TB':total/1e12,
                          'TiB':total/2**40,'with_25_percent_margin_TB':total*1.25/1e12}
    report={'state':'preliminary_capacity_estimate','source':str(SOURCE),'output_image_shape':[480,480],
        'seed':SEED,'target_policy':'sha256 rejection sampling uniform one of [start,start+32]; algorithm in script',
        'context_count':17,'context_gap':2,'ctx_render_count':33,'target_count':33,'start_stride':2,
        'split':None,'counts':counts,'camxtime_gt_measurement':gt,'camxtime_input_measurement':inp,
        'spring_prediction_proxy':pred,'estimates':estimates,'free_bytes_now':shutil.disk_usage(ROOT).free,
        'measurement_rows':measured,'elapsed_seconds':time.monotonic()-t0,
        'limitations':['CamxTime prediction not yet measured: camera mapping pending.',
            '120-frame assumption checked on one camera per rig, not all camera videos.',
            'No geometry-based window rejections applied; this is all candidate windows.',
            'PNG p10/p90 scenarios are sensitivity scenarios, not statistical confidence intervals.',
            'GT cached once per scene/trajectory/camera/time; input cached once per diagonal even index.',
            'Existing source archives/videos are excluded from incremental export size.',
            'Filesystem allocation and temporary caches are not measured; margin is planning only.']}
    (OUT/'storage_estimate.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'target_camera_plan.json').write_text(json.dumps({'seed':SEED,'windows':targets},indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['counts','spring_prediction_proxy','estimates','elapsed_seconds']},indent=2))

if __name__=='__main__': main()
