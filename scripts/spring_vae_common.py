"""Shared, explicit data contract for the Spring 8+15 framewise VAE experiment."""
import importlib.util
import json
from pathlib import Path
from dataclasses import fields
import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from export_spring_vace import (ROOT, DEFAULT_RUN, FixedSampler, enumerate_windows,
    DatasetSpring, DatasetSpringCfg, cached_example, move, build_encoder, atomic_json)

DIFFSYNTH = Path('/music-3d-shared-disk/user/KAIST/MK/foundation_models/DiffSynth-Studio-ref_keyframes_fixed')
VAE_SOURCE = DIFFSYNTH / 'diffsynth/models/wan_video_vae.py'
VAE_WEIGHTS = DIFFSYNTH / 'models/Wan-AI/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth'
DATA_ROOT = Path('/music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/C4G_prediction_dataset/Spring_VAE_480x480_ctx8')
RGB_CACHE = DATA_ROOT.parent / 'Spring_480x832'
RUN_ROOT = ROOT / 'outputs/spring_vae_feature_ctx8_step45000'
CROP_BOX = (87 / 398 * 1920, 0, 311 / 398 * 1920, 1080)


def load_dataset(fixed_bounds=False):
    raw = OmegaConf.to_container(OmegaConf.load(DEFAULT_RUN / '.hydra/config.yaml'), resolve=True)
    sc = {k:v for k,v in raw['dataset']['spring'].items() if k in {f.name for f in fields(DatasetSpringCfg)}}
    sc.update(augment=False, force_davis_intrinsics=True)
    if fixed_bounds:sc['scale_bounds_with_baseline']=False
    ds = DatasetSpring(DatasetSpringCfg(**sc), 'train', FixedSampler(8))
    plan, rejected = enumerate_windows(ds, 2)
    return raw, ds, plan, rejected


def source_path(ds, scene, index):
    return Path(ds.scenes[scene][index]['rgb_file_path'])


def latent_rel(scene, index):
    return f'clean_latents/{scene}/left/{index:05d}.npy'


def atomic_npy(path, tensor):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else tensor
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.save(f, array, allow_pickle=False)
    tmp.replace(path)


def load_latents(root, scene, indices, device='cuda'):
    arrays = [np.load(root / latent_rel(scene, i), allow_pickle=False) for i in indices]
    if any(a.shape != (16,60,60) or a.dtype != np.float16 or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Invalid clean latent')
    return torch.from_numpy(np.stack(arrays)).to(device=device, dtype=torch.float32)


def make_batch(ds, row, encoder):
    example = cached_example(ds, row, RGB_CACHE)
    batch = encoder.get_data_shim()(move(default_collate([example])))
    return batch


def load_vae():
    spec = importlib.util.spec_from_file_location('spring_wan_vae', VAE_SOURCE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    vae = module.WanVideoVAE()
    state = torch.load(str(VAE_WEIGHTS), map_location='cpu', mmap=True)
    vae.load_state_dict(vae.state_dict_converter().from_civitai(state), strict=True)
    return vae.eval().requires_grad_(False).to(device='cuda', dtype=torch.bfloat16)


def read_square(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        if image.size != (1920,1080):
            image = image.resize((1920,1080), Image.Resampling.BILINEAR)
        image = image.crop(CROP_BOX).resize((480,480), Image.Resampling.LANCZOS)
        array = np.array(image)
    return torch.from_numpy(array).permute(2,0,1).float() / 127.5 - 1


def render_feature(gaussian, feature, view, i):
    from src.model.decoder.cuda_splatting_vae16 import render_cuda
    _, _, image = render_cuda(
        view['extrinsics'][:,i], view['intrinsics'][:,i], view['near'][:,i], view['far'][:,i],
        (60,60), torch.zeros((1,3),device=feature.device),
        gaussian.means.detach().float(), gaussian.covariances.detach().float(),
        gaussian.harmonics.detach().float(), gaussian.opacities.detach().float(),
        gaussian_features=feature.float().contiguous(), scale_invariant=True, low_pass_filter=0.3)
    return image
