#!/usr/bin/env python3
"""Audit every committed window, camera index and PNG header in a Spring export."""
import argparse
import json
from pathlib import Path
import struct
import time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args()
    root=args.root.resolve()
    spec=json.loads((root/'dataset.json').read_text())
    output_shape=tuple(reversed(spec['output_image_shape']))
    encoder_shape=tuple(reversed(spec['encoder_image_shape']))
    plan=json.loads((root/'plan.json').read_text())['samples']
    status=json.loads((root/'status.json').read_text())
    if not args.allow_partial:
        assert status['state']=='complete',status
    planned={r['sample_id']:r for r in plan}
    checked={}; completed=set(); prediction_files=0
    started=time.monotonic()
    for manifest in sorted((root/'samples').glob('*/sample.json')):
        m=json.loads(manifest.read_text()); sid=m['sample_id']
        assert m['complete'] is True and sid in planned and sid not in completed,manifest
        row=planned[sid]
        assert m['context_indices']==list(range(row['start'],row['end']+1,2)),manifest
        assert m['target_indices']==m['prediction_indices']==list(range(row['start'],row['end']+1)),manifest
        for view in ['context','target']:
            assert m[f'{view}_source_file_numbers']==[i+1 for i in m[f'{view}_indices']]
            for i,src in zip(m[f'{view}_indices'],m[f'{view}_source_paths']):
                assert Path(src).name==f'frame_left_{i+1:04d}.png',(manifest,src)
        with np.load(root/m['camera_path']) as cameras:
            for view in ['context','target']:
                assert cameras[f'{view}_index'].tolist()==m[f'{view}_indices'],manifest
                for key in ['extrinsics','intrinsics','near','far']:
                    assert len(cameras[f'{view}_{key}'])==len(m[f'{view}_indices'])
                    assert np.isfinite(cameras[f'{view}_{key}']).all(),(manifest,key)
                if spec.get('render_bounds',{}).get('mode')=='fixed':
                    assert np.all(cameras[f'{view}_near']==np.float32(.1)),manifest
                    assert np.all(cameras[f'{view}_far']==np.float32(100.)),manifest
        for field,n,shape in [('context_image_paths',spec['context_count'],output_shape),('target_image_paths',spec['target_count'],output_shape),
                              ('prediction_paths',spec['target_count'],output_shape),('encoder_input_image_paths',spec['context_count'],encoder_shape)]:
            assert len(m[field])==n,(manifest,field)
            for rel in m[field]:
                if rel in checked:
                    assert checked[rel]==shape
                    continue
                path=(root/rel).resolve()
                assert path.is_relative_to(root) and path.is_file(),path
                with path.open('rb') as f: header=f.read(24)
                assert header[:8]==b'\x89PNG\r\n\x1a\n',path
                assert struct.unpack('>II',header[16:24])==shape,path
                checked[rel]=shape
        completed.add(sid);prediction_files+=len(m['prediction_paths'])
        if len(completed)%100==0:
            progress={'state':'running','completed_samples':len(completed),
                      'planned_samples':len(planned),'unique_png_headers_checked':len(checked)}
            tmp=root/'validation_status.json.tmp'
            tmp.write_text(json.dumps(progress,indent=2)+'\n')
            tmp.replace(root/'validation_status.json')
            print(json.dumps(progress),flush=True)
    if not args.allow_partial:
        assert len(completed)==len(planned)==spec['planned_samples']
    result={'passed':True,'full_export':len(completed)==len(planned),'completed_samples':len(completed),
            'planned_samples':len(planned),'prediction_pngs':prediction_files,
            'unique_png_headers_checked':len(checked),
            'unique_clean_pngs':sum(p.startswith('clean/') for p in checked),
            'unique_encoder_input_pngs':sum(p.startswith('encoder_input/') for p in checked),
            'seconds':time.monotonic()-started,
            'output_image_shape':spec['output_image_shape'],
            'note':'Every manifest, camera index and PNG header checked; this audit does not fully decode PNG pixels.'}
    path=root/('validation_partial.json' if args.allow_partial else 'validation.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    (root/'validation_status.json').write_text(json.dumps({'state':'complete',**result},indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
