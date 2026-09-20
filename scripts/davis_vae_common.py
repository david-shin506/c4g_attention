"""DAVIS variant of the Spring 8+15 framewise VAE data contract (see spring_vae_common.py).

DAVIS ships no cameras. Poses come from the DA3 monocular reconstruction in
JB2/Dataset/DAVIS/processed/<scene>/da3_monocular_480x832.npz, which was run on 81-frame videos:
scenes longer than 81 frames were truncated to their first 81 frames and shorter ones padded by
repeating the last frame (checked against the 480p JPEGs). Only real frames are used here, so npz
frame k is original frame k for k < min(#jpgs, 81).

Everything else follows the Spring export: 1920x1080 JPEGs, the same centre crop for the 224x224
encoder input and the 480x480 VAE input, forced DAVIS intrinsics, context-baseline normalisation in the
first context camera's frame, and fixed near/far bounds. DA3's scale is arbitrary, so a window counts
as a static camera when its context baseline is below STATIC_REL of the scene's median DA3 depth; such
windows use identity poses (Spring's rule for fully static scenes) instead of dividing by a baseline
that is mostly DA3 noise.
"""
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from spring_vae_common import (ROOT, DEFAULT_RUN, VAE_SOURCE, VAE_WEIGHTS, CROP_BOX, atomic_json, atomic_npy,
                               build_encoder, load_vae, move, read_square, render_feature)
from export_spring_vace import cached_rgb
from src.misc.cam_utils import camera_normalization

DAVIS_JPG = Path('/music-3d-shared-disk/user/KAIST/JB2/Dataset/DAVIS/DAVIS/JPEGImages/1080p')
DAVIS_DA3 = Path('/music-3d-shared-disk/user/KAIST/JB2/Dataset/DAVIS/processed')
DATA_ROOT = Path('/music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/C4G_prediction_dataset/DAVIS_VAE_480x480_ctx8')
FEATURE_CKPT = ROOT / 'outputs/spring_vae_feature_fixed_bounds_20260913/selected_checkpoint.ckpt'
BASE_CKPT = DEFAULT_RUN / 'checkpoints/epoch_2-step_45000.ckpt'
DA3_FILE = 'da3_monocular_480x832.npz'
DA3_FRAMES = 81
CONTEXT, GAP, START_STRIDE = 8, 2, 2
SPAN = GAP * (CONTEXT - 1)  # 14: 8 context frames at gap 2, 15 target frames
STATIC_REL = 0.002
NEAR, FAR = 0.1, 100.0
K_DAVIS = torch.tensor([[.8767, 0, .5], [0, .8767, .5], [0, 0, 1]])


def load_raw_config():
    return OmegaConf.to_container(OmegaConf.load(DEFAULT_RUN / '.hydra/config.yaml'), resolve=True)


def load_scenes():
    scenes = {}
    for d in sorted(DAVIS_DA3.iterdir()):
        npz = d / DA3_FILE
        if not npz.is_file():
            continue
        jpgs = sorted((DAVIS_JPG / d.name).glob('*.jpg'))
        n = min(len(jpgs), DA3_FRAMES)
        with np.load(npz) as z:
            if list(z['frame_nums'][:n]) != list(range(n)):
                raise ValueError(f'Unexpected DA3 frame numbering: {npz}')
            c2w = z['cam_c2w'][:n].astype(np.float32)
            depth = z['depth'][:n]
        scenes[d.name] = {
            'frames': [{'rgb_file_path': str(jpgs[i]), 'c2w': c2w[i]} for i in range(n)],
            'median_depth': float(np.median(depth[depth > 0])),
            'num_source_frames': len(jpgs),
        }
    return scenes


def window_poses(scenes, row):
    """c2w for every frame of the scene, scale-normalised the way cached_example does for Spring."""
    frames = scenes[row['scene']]['frames']
    poses = torch.from_numpy(np.stack([f['c2w'] for f in frames])).float()
    ctx = torch.arange(row['start'], row['end'] + 1, GAP)
    baseline = (poses[ctx[1:], :3, 3] - poses[ctx[0], :3, 3]).norm(dim=1).max()
    return poses, ctx, baseline


def enumerate_windows(scenes):
    plan = []
    for scene, s in scenes.items():
        for start in range(0, len(s['frames']) - SPAN, START_STRIDE):
            row = {'scene': scene, 'start': start, 'end': start + SPAN,
                   'sample_id': f'davis_{scene}_{start:05d}_{start + SPAN:05d}_gap{GAP}'}
            _, _, baseline = window_poses(scenes, row)
            rel = float(baseline) / s['median_depth']
            row.update(baseline=float(baseline), baseline_rel_depth=rel,
                       camera_mode='static_identity' if rel < STATIC_REL else 'da3_baseline_normalized')
            plan.append(row)
    return plan


def make_example(scenes, row):
    poses, ctx, baseline = window_poses(scenes, row)
    tgt = torch.arange(row['start'], row['end'] + 1)
    if row['camera_mode'] == 'static_identity':
        poses = torch.eye(4).repeat(len(poses), 1, 1)
    else:
        poses[:, :3, 3] /= baseline
    poses = camera_normalization(poses[ctx[:1]], poses)
    frames = scenes[row['scene']]['frames']
    result = {name: {'extrinsics': poses[idx], 'intrinsics': K_DAVIS.repeat(len(idx), 1, 1), 'index': idx,
                     'near': torch.full((len(idx),), NEAR), 'far': torch.full((len(idx),), FAR)}
              for name, idx in [('context', ctx), ('target', tgt)]}
    # No encoder-input cache for DAVIS: cached_rgb crops the 1920x1080 source like the Spring loader.
    result['context']['image'] = torch.stack([cached_rgb(frames[i]['rgb_file_path'], '') for i in ctx.tolist()])
    return result


def make_batch(scenes, row, encoder):
    return encoder.get_data_shim()(move(default_collate([make_example(scenes, row)])))


def source_path(scenes, scene, index):
    return Path(scenes[scene]['frames'][index]['rgb_file_path'])


def latent_rel(scene, index):
    return f'clean_latents/{scene}/{index:05d}.npy'


def load_latents(root, scene, indices, device='cuda'):
    arrays = [np.load(root / latent_rel(scene, i), allow_pickle=False) for i in indices]
    if any(a.shape != (16, 60, 60) or a.dtype != np.float16 or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Invalid clean latent')
    return torch.from_numpy(np.stack(arrays)).to(device=device, dtype=torch.float32)
