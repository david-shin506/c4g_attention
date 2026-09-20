#!/usr/bin/env python3
"""Validate all framewise latent values, indices, paths and camera arrays."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from spring_vae_common import DATA_ROOT,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('root',nargs='?',type=Path,default=DATA_ROOT);p.add_argument('--allow-partial',action='store_true');args=p.parse_args()
    root=args.root.resolve();status=json.loads((root/'status.json').read_text())
    if not args.allow_partial:assert status['state']=='complete'
    index=json.loads((root/'index.json').read_text())['samples'];spec=json.loads((root/'dataset.json').read_text())
    export_spec=json.loads((root/'export_spec.json').read_text()) if (root/'export_spec.json').exists() else {}
    rgb_count=0;rgb_bytes=0
    checked={};mse=[];prediction_count=0;sample_ids=set()
    for entry in index:
        m=json.loads((root/entry['path']).read_text());assert m['complete'] and entry['sample_id']==m['sample_id']
        assert m['sample_id'] not in sample_ids;sample_ids.add(m['sample_id'])
        start=m['context_indices'][0]
        assert m['context_indices']==list(range(start,start+15,2))
        assert m['target_indices']==m['prediction_indices']==list(range(start,start+15))
        for field,indices in [('context_latent_paths',m['context_indices']),('target_latent_paths',m['target_indices']),('prediction_paths',m['prediction_indices'])]:
            assert len(m[field])==len(indices)
            for rel,i in zip(m[field],indices):
                path=(root/rel).resolve();assert path.is_relative_to(root) and path.stem==f'{i:05d}'
                if rel not in checked:
                    a=np.load(path,allow_pickle=False)
                    assert a.shape==(16,60,60) and a.dtype==np.float16 and np.isfinite(a).all(),path
                    checked[rel]=path.stat().st_size
        if export_spec.get('save_rgb_pred'):
            assert m['rgb_prediction_indices']==m['prediction_indices']
            assert len(m['rgb_prediction_paths'])==len(m['prediction_indices'])
            for rel,i in zip(m['rgb_prediction_paths'],m['prediction_indices']):
                path=(root/rel).resolve();assert path.is_relative_to(root) and path.stem==f'{i:05d}'
                with Image.open(path) as im:
                    assert im.mode=='RGB' and im.size==(480,480);im.load()
                rgb_count+=1;rgb_bytes+=path.stat().st_size
        with np.load(root/m['camera_path']) as camera:
            for view in ['context','target']:
                assert camera[f'{view}_index'].tolist()==m[f'{view}_indices']
                for key in ['extrinsics','intrinsics','near','far']:assert np.isfinite(camera[f'{view}_{key}']).all()
                if export_spec.get('render_bounds',{}).get('mode')=='fixed':
                    assert np.all(camera[f'{view}_near']==np.float32(.1))
                    assert np.all(camera[f'{view}_far']==np.float32(100.))
        prediction_count+=len(m['prediction_paths']);mse.append(m['mean_latent_mse'])
    assert len(index)==status['completed']
    if not args.allow_partial:assert len(index)==spec['planned_samples']
    report={'passed':True,'complete':status['state']=='complete','samples':len(index),'prediction_frames':prediction_count,
            'unique_latent_files':len(checked),'latent_file_bytes':sum(checked.values()),
            'rgb_prediction_frames':rgb_count,'rgb_prediction_bytes':rgb_bytes,
            'mean_export_latent_mse':sum(mse)/len(mse),'checks':'all latent shapes, FP16 types, finite values, index mappings, camera arrays and unique sample IDs'}
    atomic_json(root/('validation_partial.json' if args.allow_partial else 'validation.json'),report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
