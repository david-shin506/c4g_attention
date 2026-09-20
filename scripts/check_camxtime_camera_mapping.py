#!/usr/bin/env python3
"""Empirical pose-order audit; geometric evidence, not provenance proof."""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'datasets/CamxTime'
OUT=ROOT/'outputs/camxtime_preflight'

def frame(rig,camera):
    cap=cv2.VideoCapture(str(rig/f'cam{camera+1:03d}_full_motion.mp4'))
    try: ok, image=cap.read()
    finally: cap.release()
    if not ok: raise ValueError(f'Cannot read {rig} cam{camera+1}')
    return image

def correspondences(a,b):
    sift=cv2.SIFT_create(nfeatures=3000)
    ka,da=sift.detectAndCompute(a,None)
    kb,db=sift.detectAndCompute(b,None)
    pairs=cv2.BFMatcher().knnMatch(da,db,k=2)
    good=[m for m,n in pairs if m.distance < .7*n.distance]
    p=np.float64([ka[m.queryIdx].pt for m in good])
    q=np.float64([kb[m.trainIdx].pt for m in good])
    if len(p)<20: raise ValueError(f'Only {len(p)} matches')
    _,mask=cv2.findFundamentalMat(p,q,cv2.FM_RANSAC,1.0,.999)
    keep=mask.ravel().astype(bool)
    return p[keep],q[keep],len(p)

def error(p,q,camera_a,camera_b,K):
    rel=np.linalg.inv(camera_b)@camera_a
    R=rel[:3,:3]; x,y,z=rel[:3,3]
    cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    ki=np.linalg.inv(K); F=ki.T@cross@R@ki
    p=np.c_[p,np.ones(len(p))]; q=np.c_[q,np.ones(len(q))]
    fp=p@F.T; fq=q@F
    numerator=np.abs(np.einsum('ni,ni->n',q,fp))
    denominator=np.sqrt(fp[:,0]**2+fp[:,1]**2+fq[:,0]**2+fq[:,1]**2)
    return float(np.median(numerator/np.maximum(denominator,1e-20)))

def main():
    cv2.setNumThreads(2)
    rows=[]
    for scene in ['Scene001','Scene002','Scene003']:
        for trajectory in [1,2]:
            rig=SOURCE/scene/f'camera-trajectory-{trajectory:02d}'
            data=json.loads((rig/f'{rig.name}-camera.json').read_text())
            keys=sorted(data['trajectory'],key=int)
            poses=np.array([data['trajectory'][k]['c2w'] for k in keys])
            K=np.array(data['intrinsics']['K'])
            images={i:frame(rig,i) for i in [0,8,16,32]}
            matches=[(a,b,*correspondences(images[a],images[b])) for a,b in [(0,8),(8,16),(16,32)]]
            hypotheses=[]
            for convention,axis in [('as_stored',np.eye(4)),('blender_to_opencv',np.diag([1,-1,-1,1]))]:
                cvposes=poses@axis
                for direction in [1,-1]:
                    for shift in range(120):
                        errors=[error(p,q,cvposes[(direction*a+shift)%120],cvposes[(direction*b+shift)%120],K)
                                for a,b,p,q,n in matches]
                        hypotheses.append({'convention':convention,'direction':direction,'cyclic_shift':shift,
                                           'median_pair_sampson_px':float(np.median(errors)),'pair_errors_px':errors})
            ranked=sorted(hypotheses,key=lambda x:x['median_pair_sampson_px'])
            sorted_h=[h for h in hypotheses if h['direction']==1 and h['cyclic_shift']==0]
            rows.append({'scene':scene,'trajectory':rig.name,'key_range':[keys[0],keys[-1]],
                'matched_pair_counts':[{'cameras':[a,b],'ratio_matches':n,'ransac_inliers':len(p)} for a,b,p,q,n in matches],
                'best_5':ranked[:5],'sorted_order_hypotheses':sorted_h})
            print(json.dumps(rows[-1]),flush=True)
    report={'method':'SIFT ratio 0.7; independent F-RANSAC inliers at t=0; median Sampson distance to JSON poses with native K',
            'scope':'3 scenes x 2 rigs x 3 camera pairs, first decoded time only',
            'caveat':'Empirical ranking against shifts/reversal only; not proof of source semantics or all 1664 rigs.',
            'rows':rows}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'camera_mapping_geometry.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
