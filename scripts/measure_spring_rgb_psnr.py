#!/usr/bin/env python3
"""Measure saved Spring prediction PNGs against manifest-matched clean RGB PNGs."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from functools import lru_cache
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image


@lru_cache(maxsize=256)
def read_gt(path):
    with Image.open(path) as image:
        if image.mode != 'RGB':
            raise ValueError(f'Expected RGB: {path}')
        return np.array(image, dtype=np.uint8)


def evaluate_sample(root, entry, expected_shape):
    m = json.loads((root / entry['path']).read_text())
    assert m['complete'] and m['sample_id'] == entry['sample_id']
    assert m['prediction_indices'] == m['target_indices']
    assert len(m['prediction_paths']) == len(m['target_image_paths']) == len(m['target_indices'])
    context = set(m['context_indices'])
    rows = []
    for index, pred_path, gt_path in zip(m['prediction_indices'], m['prediction_paths'], m['target_image_paths']):
        gt = read_gt(str(root / gt_path))
        with Image.open(root / pred_path) as image:
            assert image.mode == 'RGB', pred_path
            pred = np.array(image, dtype=np.uint8)
        assert pred.shape == gt.shape == expected_shape, pred_path
        diff = np.subtract(pred, gt, dtype=np.int32)
        np.square(diff, out=diff)
        sse = int(diff.sum(dtype=np.int64))
        mse = sse / pred.size / (255.0 ** 2)
        psnr = -10.0 * math.log10(mse) if mse else math.inf
        group = 'context_timestamps' if index in context and m['context_camera'] == m['target_camera'] else 'interpolated_timestamps'
        rows.append(dict(sample_id=m['sample_id'], scene=m['scene'], frame_index=index,
                         group=group, psnr_db=psnr, mse_0_1=mse, squared_error_sum_uint8=sse,
                         channel_values=pred.size, prediction_path=pred_path, gt_path=gt_path))
    return rows


def summarize(rows):
    psnr = np.array([r['psnr_db'] for r in rows], dtype=np.float64)
    mse = sum(r['squared_error_sum_uint8'] for r in rows) / sum(r['channel_values'] for r in rows) / 255.0**2
    return dict(frames=len(rows), mean_frame_psnr_db=float(psnr.mean()),
                median_frame_psnr_db=float(np.median(psnr)),
                pooled_mse_psnr_db=-10 * math.log10(mse) if mse else 'Infinity',
                mean_mse_0_1=mse, min_psnr_db=float(psnr.min()), max_psnr_db=float(psnr.max()))


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--samples-per-scene', type=int, default=0,
                        help='Evaluate this many evenly spaced windows per scene; 0 evaluates all.')
    args = parser.parse_args()
    if args.samples_per_scene < 0:
        parser.error('samples-per-scene must be nonnegative')
    root = args.root.resolve()
    spec = json.loads((root / 'dataset.json').read_text())
    image_height, image_width = spec['output_image_shape']
    expected_shape = (image_height, image_width, 3)
    status = json.loads((root / 'status.json').read_text())
    assert status['state'] == 'complete'
    entries = json.loads((root / 'index.json').read_text())['samples']
    assert len(entries) == status['total'] == status['completed']
    assert len({e['sample_id'] for e in entries}) == len(entries)
    available_samples = len(entries)
    if args.samples_per_scene:
        by_scene_entries = defaultdict(list)
        for entry in entries:
            m = json.loads((root / entry['path']).read_text())
            by_scene_entries[m['scene']].append((m['context_indices'][0], entry))
        entries = []
        for scene, items in sorted(by_scene_entries.items()):
            items.sort(key=lambda item: item[0])
            positions = np.linspace(0, len(items)-1, min(args.samples_per_scene,len(items)),dtype=int)
            entries.extend(items[i][1] for i in positions)
    prefix = f'psnr_subset_{args.samples_per_scene}_per_scene' if args.samples_per_scene else 'psnr'

    started = time.monotonic()
    rows = []
    csv_path = root / f'{prefix}_per_frame.csv'
    csv_tmp = csv_path.with_suffix('.csv.tmp')
    with csv_tmp.open('w', newline='') as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        writer = None
        futures = [pool.submit(evaluate_sample, root, e, expected_shape) for e in entries]
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(result[0]))
                writer.writeheader()
            writer.writerows(result)
            rows.extend(result)
            if n == 1 or n % 100 == 0 or n == len(entries):
                elapsed = time.monotonic() - started
                progress = dict(state='running', completed_samples=n, total_samples=len(entries),
                                evaluated_frames=len(rows), elapsed_s=elapsed,
                                estimated_remaining_s=elapsed/n*(len(entries)-n))
                atomic_json(root / f'{prefix}_status.json', progress)
                print(json.dumps(progress), flush=True)
    csv_tmp.replace(csv_path)
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r['scene']].append(r)
    report = dict(state='complete', dataset_root=str(root), samples=len(entries), scenes=len(by_scene),
                  available_samples=available_samples, full_export_evaluated=len(entries)==available_samples,
                  selection='all windows' if not args.samples_per_scene else f'Up to {args.samples_per_scene} evenly spaced windows per scene, ordered by start index',
                  selected_sample_ids=[e['sample_id'] for e in entries] if args.samples_per_scene else None,
                  group_definitions={'context_timestamps':'prediction index is in context_indices (same camera)',
                                     'interpolated_timestamps':'target prediction index is not in context_indices'},
                  metric='RGB PSNR, data_range=255; mean of per-frame PSNR values',
                  evaluation=f'Saved {image_height}x{image_width} RGB PNG pairs, full image, no mask/resizing/alignment/padding; each window prediction evaluated separately even when GT repeats',
                  output_image_shape=[image_height, image_width],
                  all=summarize(rows),
                  groups={g:summarize([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})},
                  per_scene={scene:summarize(rs) for scene,rs in sorted(by_scene.items())},
                  elapsed_s=time.monotonic()-started,
                  completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  per_frame_csv=csv_path.name)
    atomic_json(root / f'{prefix}_report.json', report)
    atomic_json(root / f'{prefix}_status.json', dict(state='complete',completed_samples=len(entries),total_samples=len(entries),evaluated_frames=len(rows)))
    print(json.dumps({k:v for k,v in report.items() if k!='per_scene'}),flush=True)

if __name__ == '__main__':
    main()
