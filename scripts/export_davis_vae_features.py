#!/usr/bin/env python3
"""Export per-window Gaussian VAE features for DAVIS (DAVIS twin of export_spring_vae_features.py).

Uses the Spring-trained feature branch (fixed-bounds run) on frozen C4G step-45000 geometry, DA3 cameras,
fixed near/far bounds, and always saves the 480x480 RGB rendered from the same Gaussians and cameras.
Completed windows are committed by an atomic sample.json write; re-running resumes."""
import argparse, fcntl, hashlib, json, os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from davis_vae_common import *
from export_spring_vace import save_png
from src.model.vae_feature_lifting import VAEFeatureLifter


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=DATA_ROOT)
    p.add_argument('--checkpoint', type=Path, default=FEATURE_CKPT)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    torch.set_num_threads(4); root = args.output.resolve()
    lock = (root / '.export.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((root / 'prepare_status.json').read_text())['state'] == 'complete'
    scenes = load_scenes()
    plan = enumerate_windows(scenes)
    assert [r['sample_id'] for r in plan] == [r['sample_id'] for r in json.loads((root / 'plan.json').read_text())['samples']]
    export_spec = {'schema_version': 1, 'render_bounds': {'mode': 'fixed', 'near': NEAR, 'far': FAR, 'divided_by_baseline': False},
                   'save_rgb_pred': True, 'rgb_prediction_shape': [480, 480],
                   'rgb_prediction_source': 'same C4G Gaussian and target camera; native RGB rasterization, not VAE decoding',
                   'camera_source': 'DA3 monocular', 'static_camera_rel_threshold': STATIC_REL}
    spec_path = root / 'export_spec.json'
    if spec_path.exists():
        assert json.loads(spec_path.read_text()) == export_spec, 'Export settings differ; use a fresh output directory'
    else:
        atomic_json(spec_path, export_spec)
    raw = load_raw_config()
    encoder, rgb_decoder, _ = build_encoder(raw, BASE_CKPT)
    lifter = VAEFeatureLifter(encoder).cuda()
    weights = torch.load(str(args.checkpoint), map_location='cpu')
    if weights['config']['smoke']:
        raise ValueError('Smoke checkpoints cannot produce the final dataset')
    lifter.decoder.load_feature_state_dict(weights['features'])
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    provenance = {'checkpoint': str(args.checkpoint.resolve()), 'sha256': checkpoint_hash, 'step': weights['step'],
                  'validation': weights['validation'], 'trained_on': 'spring',
                  'training_render_bounds': weights['config'].get('render_bounds', 'legacy baseline-scaled'),
                  'smoke_checkpoint': weights['config']['smoke'], 'base_checkpoint': weights['config']['checkpoint']}
    if (root / 'feature_checkpoint.json').exists():
        assert json.loads((root / 'feature_checkpoint.json').read_text()) == provenance
    else:
        atomic_json(root / 'feature_checkpoint.json', provenance)
    del weights
    started = time.monotonic(); completed = []; new = 0
    with ThreadPoolExecutor(max_workers=4) as png_pool, torch.no_grad():
        for row in plan:
            sid = row['sample_id']; sample = root / 'samples' / sid; marker = sample / 'sample.json'
            if marker.exists():
                m = json.loads(marker.read_text())
                assert m['complete'] and m['feature_checkpoint_sha256'] == checkpoint_hash
                for rel in m['prediction_paths']:
                    a = np.load(root / rel, allow_pickle=False); assert a.shape == (16, 60, 60) and a.dtype == np.float16 and np.isfinite(a).all()
                for rel in m['rgb_prediction_paths']:
                    with Image.open(root / rel) as im:
                        assert im.mode == 'RGB' and im.size == (480, 480); im.load()
                completed.append({'sample_id': sid, 'path': str(marker.relative_to(root))}); continue
            if args.limit is not None and new >= args.limit:
                break
            batch = make_batch(scenes, row, lifter.encoder)
            ctx = batch['context']['index'][0].tolist(); targets = batch['target']['index'][0].tolist()
            inputs = load_latents(root, row['scene'], ctx)[None]
            gs, features = lifter(batch['context'], inputs, batch['target']['index'][0])
            paths, mse, rgb_paths, rgb_jobs = [], [], [], []
            view = batch['target']
            for i, t in enumerate(targets):
                pred = render_feature(gs[t], features[t], view, i)[0]
                if not torch.isfinite(pred).all():
                    raise ValueError(f'Nonfinite feature: {sid}/{t}')
                stored = pred.half()
                if not torch.isfinite(stored).all():
                    raise ValueError('Feature exceeds FP16 range')
                rel = f'samples/{sid}/prediction/{t:05d}.npy'; atomic_npy(root / rel, stored); paths.append(rel)
                gt = load_latents(root, row['scene'], [t])[0]
                mse.append(float((stored.float() - gt).square().mean()))
                rgb = rgb_decoder(gs[t], view['extrinsics'][:, i:i + 1], view['intrinsics'][:, i:i + 1],
                                  view['near'][:, i:i + 1], view['far'][:, i:i + 1], (480, 480)).color[0, 0]
                if not torch.isfinite(rgb).all():
                    raise ValueError(f'Nonfinite RGB: {sid}/{t}')
                pixels = (rgb.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
                rgb_rel = f'samples/{sid}/rgb_prediction/{t:05d}.png'; rgb_paths.append(rgb_rel)
                rgb_jobs.append(png_pool.submit(save_png, root / rgb_rel, pixels))
            for job in rgb_jobs:
                job.result()
            camera_rel = f'samples/{sid}/cameras.npz'; camera_path = root / camera_rel
            cam = {f'{v}_{key}': batch[v][key][0].cpu().numpy() for v in ['context', 'target']
                   for key in ['index', 'extrinsics', 'intrinsics', 'near', 'far']}
            cam['source_c2w_da3'] = np.stack([scenes[row['scene']]['frames'][i]['c2w'] for i in targets])
            tmp = camera_path.with_suffix('.npz.tmp')
            with tmp.open('wb') as f:
                np.savez_compressed(f, **cam)
            tmp.replace(camera_path)
            m = {'schema_version': 1, 'complete': True, 'sample_id': sid, 'scene': row['scene'], 'dataset': 'davis',
                 'context_indices': ctx, 'target_indices': targets, 'prediction_indices': targets, 'index_base': 0,
                 'temporal_compression': False, 'padding': False,
                 'context_latent_paths': [latent_rel(row['scene'], i) for i in ctx],
                 'target_latent_paths': [latent_rel(row['scene'], i) for i in targets],
                 'prediction_paths': paths, 'camera_path': camera_rel, 'camera_mode': row['camera_mode'],
                 'context_baseline_da3': row['baseline'], 'context_baseline_rel_depth': row['baseline_rel_depth'],
                 'feature_checkpoint_sha256': checkpoint_hash, 'feature_checkpoint_step': provenance['step'],
                 'base_checkpoint_step': 45000,
                 'context_source_paths': [str(source_path(scenes, row['scene'], i)) for i in ctx],
                 'target_source_paths': [str(source_path(scenes, row['scene'], i)) for i in targets],
                 'render_bounds': export_spec['render_bounds'], 'latent_shape': [16, 60, 60], 'latent_dtype': 'float16',
                 'mean_latent_mse': sum(mse) / len(mse), 'feature_training_split': 'unseen_dataset',
                 'rgb_prediction_paths': rgb_paths, 'rgb_prediction_indices': targets, 'rgb_prediction_shape': [480, 480],
                 'rgb_prediction_source': export_spec['rgb_prediction_source']}
            atomic_json(marker, m); new += 1; completed.append({'sample_id': sid, 'path': str(marker.relative_to(root))})
            if new <= 3 or len(completed) % 10 == 0:
                elapsed = time.monotonic() - started
                progress = {'state': 'exporting', 'pid': os.getpid(), 'completed': len(completed), 'total': len(plan),
                            'seconds_per_new_sample': elapsed / new,
                            'estimated_remaining_s': elapsed / new * (len(plan) - len(completed)),
                            'last_mean_latent_mse': m['mean_latent_mse']}
                atomic_json(root / 'status.json', progress); print(json.dumps(progress), flush=True)
    atomic_json(root / 'index.json', {'schema_version': 1, 'samples': completed})
    atomic_json(root / 'status.json', {'state': 'complete' if len(completed) == len(plan) else 'partial',
                                       'completed': len(completed), 'total': len(plan), 'elapsed_s': time.monotonic() - started})


if __name__ == '__main__':
    main()
