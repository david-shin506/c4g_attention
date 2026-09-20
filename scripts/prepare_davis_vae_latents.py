#!/usr/bin/env python3
"""Cache independent-frame Wan VAE latents for the DAVIS 8-context windows (DAVIS twin of
prepare_spring_vae_latents.py). Resumable: existing, valid latents are kept."""
import argparse, fcntl, hashlib, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from davis_vae_common import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=DATA_ROOT)
    p.add_argument('--batch-size', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(4)
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.prepare.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    scenes = load_scenes()
    plan = enumerate_windows(scenes)
    keys = sorted({(r['scene'], i) for r in plan for i in range(r['start'], r['end'] + 1)})
    spec = {'schema_version': 1, 'dataset': 'davis', 'source_root': str(DAVIS_JPG), 'camera_source': str(DAVIS_DA3),
            'camera_file': DA3_FILE, 'camera_model': 'DA3 monocular (cam_c2w); frame k = original frame k, k < min(#jpg, 81)',
            'base_checkpoint': str(BASE_CKPT),
            'vae_checkpoint': str(VAE_WEIGHTS), 'vae_sha256': hashlib.sha256(VAE_WEIGHTS.read_bytes()).hexdigest(),
            'vae_source': str(VAE_SOURCE), 'vae_source_sha256': hashlib.sha256(VAE_SOURCE.read_bytes()).hexdigest(),
            'context_count': CONTEXT, 'target_count': SPAN + 1, 'context_gap': GAP, 'start_stride': START_STRIDE,
            'planned_samples': len(plan), 'index_base': 0, 'image_shape': [480, 480], 'encoder_image_shape': [224, 224],
            'latent_shape': [16, 60, 60], 'latent_dtype': 'float16', 'vae_compute_dtype': 'bfloat16',
            'vae_encoding': 'independent frames; cache reset between encode calls; batch contains different independent images',
            'latent_normalization': 'WanVideoVAE.single_encode applies (mu-mean)/std; never per-vector L2 normalized',
            'crop_box_after_1920x1080_resize': list(CROP_BOX), 'crop_resize': 'PIL LANCZOS to 480x480',
            'feature_input_downsample': 'bilinear 60x60 ->16x16, align_corners=False',
            'feature_render_resolution': [60, 60], 'feature_render_low_pass': 0.3,
            'temporal_compression': False, 'padding': False, 'unique_clean_frames': len(keys),
            'force_davis_intrinsics': True,
            'pose_normalization': 'baseline=1 (max context baseline); first context camera relative frame',
            'static_camera_rule': f'context baseline < {STATIC_REL} x scene median DA3 depth -> identity poses',
            'render_bounds': {'mode': 'fixed', 'near': NEAR, 'far': FAR, 'divided_by_baseline': False},
            'predictions': 'per-window; clean GT/reference shared per original frame'}
    if (root / 'dataset.json').exists():
        assert json.loads((root / 'dataset.json').read_text()) == spec, 'Existing dataset specification differs'
    else:
        atomic_json(root / 'dataset.json', spec)
    atomic_json(root / 'plan.json', {'samples': plan, 'rejected': [],
                                     'scenes': {s: {'frames_used': len(v['frames']), 'source_frames': v['num_source_frames'],
                                                    'median_da3_depth': v['median_depth']} for s, v in scenes.items()}})
    atomic_json(root / 'clean_index.json', {'frames': [{'scene': s, 'index': i, 'source_path': str(source_path(scenes, s, i)),
                                                        'path': latent_rel(s, i)} for s, i in keys]})
    pending = []
    for s, i in keys:
        path = root / latent_rel(s, i)
        if path.exists():
            a = np.load(path, allow_pickle=False); assert a.shape == (16, 60, 60) and a.dtype == np.float16 and np.isfinite(a).all()
        else:
            pending.append((s, i))
    print(json.dumps({'windows': len(plan), 'static_windows': sum(r['camera_mode'] == 'static_identity' for r in plan),
                      'unique_frames': len(keys), 'pending_frames': len(pending)}), flush=True)
    vae = load_vae(); started = time.monotonic(); done = len(keys) - len(pending)
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for start in range(0, len(pending), args.batch_size):
            items = pending[start:start + args.batch_size]
            images = list(pool.map(read_square, [source_path(scenes, s, i) for s, i in items]))
            x = torch.stack(images).unsqueeze(2).to(device='cuda', dtype=torch.bfloat16)
            z = vae.single_encode(x, device='cuda')[:, :, 0]
            assert z.shape == (len(items), 16, 60, 60) and torch.isfinite(z).all()
            for (s, i), latent in zip(items, z):
                atomic_npy(root / latent_rel(s, i), latent.half())
            done += len(items)
            if start == 0 or done % 100 < args.batch_size or done == len(keys):
                elapsed = time.monotonic() - started
                progress = {'state': 'running', 'completed_frames': done, 'total_frames': len(keys), 'elapsed_s': elapsed,
                            'remaining_s': elapsed / (start + len(items)) * (len(pending) - start - len(items))}
                atomic_json(root / 'prepare_status.json', progress); print(json.dumps(progress), flush=True)
    # Independent-frame semantics: batched encoding must agree with encoding one image alone.
    s, i = keys[0]
    with torch.inference_mode():
        solo = vae.single_encode(read_square(source_path(scenes, s, i))[None, :, None].cuda().bfloat16(), 'cuda')[0, :, 0].float()
    cached = torch.from_numpy(np.load(root / latent_rel(s, i))).cuda().float()
    delta = float((solo - cached).abs().max()); assert delta < 0.08, delta
    atomic_json(root / 'prepare_status.json', {'state': 'complete', 'completed_frames': len(keys), 'total_frames': len(keys),
                                               'elapsed_s': time.monotonic() - started, 'independent_frame_batch_max_abs_diff': delta})
    print('DAVIS framewise VAE cache complete', flush=True)


if __name__ == '__main__':
    main()
