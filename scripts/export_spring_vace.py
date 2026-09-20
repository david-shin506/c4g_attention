#!/usr/bin/env python3
"""Export Spring C4G predictions and explicit per-window metadata for VACE.

Completed windows are committed by an atomic sample.json write. Re-running the
same command resumes validated windows. Never launch diffusion training here.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from functools import lru_cache
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
from torch.utils.data._utils.collate import default_collate
from src.dataset.dataset_spring import DatasetSpring, DatasetSpringCfg
from src.model.encoder import get_encoder
from src.model.encoder.encoder_vggt import EncoderVGGTCfg, OpacityMappingCfg
from src.model.encoder.backbone.backbone_croco import BackboneCrocoCfg
from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from src.dataset.shims.crop_shim import rescale_and_crop
from src.misc.cam_utils import camera_normalization
from src.model.decoder import get_decoder
from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg

DEFAULT_RUN = ROOT / 'outputs/exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_45k_resume_from40k_4gpu/2026-07-04_162000'
DEFAULT_OUTPUT = Path('/music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/c4g/C4G_prediction_dataset/Spring_480x832')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def save_png(path, pixels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    if isinstance(pixels, Image.Image):
        pixels.save(tmp, format='PNG')
    else:
        Image.fromarray(pixels).save(tmp, format='PNG')
    os.replace(tmp, path)


def move(value):
    if torch.is_tensor(value):
        return value.cuda()
    if isinstance(value, dict):
        return {k: move(v) for k, v in value.items()}
    return value


class FixedSampler:
    def __init__(self, count=12):
        self.num_context_views = count
        self.target_count = 2 * count - 1

    cfg = SimpleNamespace(gap=2, initial_gap=2)
    start = 0

    def sample(self, scene, extrinsics, intrinsics):
        return (torch.arange(self.start, self.start + self.target_count, 2),
                torch.arange(self.start, self.start + self.target_count), torch.tensor([.5]))


def enumerate_windows(ds, stride):
    count = ds.view_sampler.target_count
    span = count - 1
    plan, rejected = [], []
    for scene in sorted(ds.scenes):
        # Same float32 positions and thresholds as DatasetSpring.getitem.
        ext = torch.tensor(np.linalg.inv(np.array([f['extrinsics'] for f in ds.scenes[scene]])), dtype=torch.float32)
        pos = ext[:, :3, 3]
        static = bool((pos - pos[:1]).norm(dim=1).max() < ds.cfg.baseline_min)
        for start in range(0, len(ext) - span, stride):
            ctx = pos[start:start + count:2]
            scale = (ctx[1:] - ctx[:1]).norm(dim=1).max()
            valid = (ds.cfg.baseline_min <= scale <= ds.cfg.baseline_max) or (scale < ds.cfg.baseline_min and static)
            row = {'scene': scene, 'start': start, 'end': start + span,
                   'sample_id': f'spring_{scene}_left_{start:05d}_{start+span:05d}_gap2'}
            if valid:
                plan.append(row)
            else:
                rejected.append({**row, 'reason': 'baseline_out_of_range'})
    return plan, rejected


def existing_complete(out, row):
    path = out / 'samples' / row['sample_id'] / 'sample.json'
    if not path.exists():
        return False
    m = json.loads(path.read_text())
    if m.get('complete') is not True or m['sample_id'] != row['sample_id']:
        raise ValueError(f'Invalid committed sample: {path}')
    if m['context_indices'] != list(range(row['start'], row['end'] + 1, 2)) or m['target_indices'] != list(range(row['start'], row['end'] + 1)):
        raise ValueError(f'Committed sample indices mismatch: {path}')
    paths = m['context_image_paths'] + m['encoder_input_image_paths'] + m['prediction_paths'] + m['target_image_paths'] + [m['camera_path']]
    if not all((out / p).is_file() and (out / p).stat().st_size > 0 for p in paths):
        raise ValueError(f'Committed sample has missing/empty assets: {path}')
    return True


def build_encoder(raw, checkpoint):
    ec = dict(raw['model']['encoder'])
    ec['backbone'] = BackboneCrocoCfg(**ec['backbone'])
    ec['gaussian_adapter'] = GaussianAdapterCfg(**ec['gaussian_adapter'])
    ec['opacity_mapping'] = OpacityMappingCfg(**ec['opacity_mapping'])
    ec['gradient_checkpoint'] = False
    encoder, _ = get_encoder(EncoderVGGTCfg(**ec))
    ckpt = torch.load(str(checkpoint), map_location='cpu', mmap=True)
    weights = {k[8:]: v for k, v in ckpt['state_dict'].items() if k.startswith('encoder.')}
    encoder.load_state_dict(weights, strict=True)
    step = int(ckpt['global_step'])
    encoder = encoder.cuda().eval()
    decoder = get_decoder(DecoderSplattingCUDACfg(**raw['model']['decoder'])).cuda().eval()
    del weights, ckpt
    gc.collect()
    return encoder, decoder, step


@lru_cache(maxsize=128)
def saved_input_tensor(path):
    with Image.open(path) as image:
        if image.mode != 'RGB' or image.size != (224, 224):
            raise ValueError(f'Unexpected saved encoder input: {path}')
        return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255


def reuse_example(source, row):
    m = json.loads((source / 'samples' / row['sample_id'] / 'sample.json').read_text())
    assert m['complete'] and m['sample_id'] == row['sample_id']
    assert m['context_indices'] == list(range(row['start'], row['end'] + 1, 2))
    assert m['prediction_indices'] == m['target_indices'] == list(range(row['start'], row['end'] + 1))
    with np.load(source / m['camera_path']) as cameras:
        example = {view: {key: torch.from_numpy(cameras[f'{view}_{key}'].copy())
                   for key in ['index', 'extrinsics', 'intrinsics', 'near', 'far']}
                   for view in ['context', 'target']}
    example['context']['image'] = torch.stack([
        saved_input_tensor(str(source / path)) for path in m['encoder_input_image_paths']])
    return example



@lru_cache(maxsize=256)
def cached_rgb(path, cached_path):
    if Path(cached_path).is_file():
        return saved_input_tensor(cached_path)
    with Image.open(path) as im:
        im = im.convert('RGB')
        if im.size != (1920, 1080):
            im = im.resize((1920, 1080), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255
    return rescale_and_crop(tensor, torch.eye(3), (224, 224))[0]


def cached_example(ds, row, cache):
    scene = row['scene']
    frames = ds.scenes[scene]
    poses = torch.tensor(np.linalg.inv(np.array([f['extrinsics'] for f in frames])), dtype=torch.float32)
    ctx = torch.arange(row['start'], row['end'] + 1, 2)
    tgt = torch.arange(row['start'], row['end'] + 1)
    scale = (poses[ctx[1:], :3, 3] - poses[ctx[0], :3, 3]).norm(dim=1).max()
    if scale < ds.cfg.baseline_min:
        assert (poses[:, :3, 3] - poses[:1, :3, 3]).norm(dim=1).max() < ds.cfg.baseline_min
        poses = torch.eye(4).repeat(len(poses), 1, 1)
        scale = torch.tensor(1.)
    else:
        assert scale <= ds.cfg.baseline_max
        poses[:, :3, 3] /= scale
    poses = camera_normalization(poses[ctx[:1]], poses)
    K = torch.tensor([[.8767, 0, .5], [0, .8767, .5], [0, 0, 1]])
    result = {name: {'extrinsics': poses[idx], 'intrinsics': K.repeat(len(idx), 1, 1),
                    'index': idx, 'near': ds.get_scaled_bound('near', len(idx), scale),
                    'far': ds.get_scaled_bound('far', len(idx), scale)}
              for name, idx in [('context', ctx), ('target', tgt)]}
    result['context']['image'] = torch.stack([
        cached_rgb(frames[i]['rgb_file_path'], str(cache / 'encoder_input' / scene / 'left' / f'{i:05d}.png'))
        for i in ctx.tolist()])
    return result


def export(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.export.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start_time = time.monotonic()
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(111123)
    np.random.seed(111123)
    raw = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    sc = {k: v for k, v in raw['dataset']['spring'].items() if k in {f.name for f in fields(DatasetSpringCfg)}}
    sc['augment'] = False
    if args.fixed_bounds:sc['scale_bounds_with_baseline'] = False
    sc['force_davis_intrinsics'] = bool(raw.get('force_davis_intrinsics', False) or sc['force_davis_intrinsics'])
    if sc['input_image_shape'] != [224, 224]:
        raise ValueError('This export expects the checkpoint\'s 224x224 encoder input')
    sampler = FixedSampler(args.context_count)
    ds = DatasetSpring(DatasetSpringCfg(**sc), 'train', sampler)
    plan, rejected = enumerate_windows(ds, args.stride)
    source = Path(ds.data_root)
    original_h, original_w = sc['original_image_shape']
    scaled_h, scaled_w = 224, round(original_w * 224 / original_h)
    crop_left = (scaled_w - 224) // 2
    crop_box = [crop_left / scaled_w * original_w, 0,
                (crop_left + 224) / scaled_w * original_w, original_h]
    checkpoint = args.checkpoint.resolve()
    spec = {'schema_version': 1, 'dataset': 'spring', 'split': 'train',
            'checkpoint': str(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size,
            'checkpoint_mtime_ns': checkpoint.stat().st_mtime_ns,
            'config_path': str(args.config.resolve()),
            'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
            'source_root': str(source), 'index_base': 0,
            'source_filename_rule': 'frame_left_{index+1:04d}.png',
            'context_count': args.context_count, 'context_gap': 2, 'target_count': sampler.target_count,
            'start_stride': args.stride, 'context_camera': 'left', 'target_camera': 'left',
            'encoder_image_shape': [224, 224], 'output_image_shape': [args.height, args.width],
            'force_davis_intrinsics': sc['force_davis_intrinsics'],
            'original_image_shape': [original_h, original_w],
            'encoder_rescale_shape': [scaled_h, scaled_w],
            'clean_crop_box_after_original_resize': crop_box,
            'clean_resize': f'PIL bilinear to original_image_shape; central crop; LANCZOS to {args.width}x{args.height}',
            'normalization': 'encoder image [0,1] -> (image-0.5)/0.5',
            'pose_normalization': {'make_baseline_1': sc['make_baseline_1'], 'relative_pose': sc['relative_pose']},
            'source_code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/dataset/dataset_spring.py', ROOT/'src/model/encoder/encoder_vggt.py', ROOT/'src/dataset/shims/crop_shim.py']},
            'planned_samples': len(plan), 'baseline_rejected_windows': len(rejected),
            'padding': f'Not saved. Loader pads to 4k+1 if needed; target count {sampler.target_count}.',
            'pose_alignment': False,
            'captions': 'Optional captions.json keyed by sample_id, spring_{scene}, or scene; empty prompt if absent.'}
    if args.fixed_bounds:
        spec['render_bounds'] = {'mode':'fixed','near':0.1,'far':100.0,'divided_by_baseline':False}
    if args.reuse_inputs_from is not None:
        args.reuse_inputs_from = args.reuse_inputs_from.resolve()
        old_spec = json.loads((args.reuse_inputs_from / 'dataset.json').read_text())
        old_status = json.loads((args.reuse_inputs_from / 'status.json').read_text())
        if old_status['state'] != 'complete':
            raise ValueError('Source export is incomplete')
        for key in spec:
            if key not in {'output_image_shape', 'clean_resize'} and old_spec.get(key) != spec[key]:
                raise ValueError(f'Source export setting differs: {key}')
        # Confirm saved PNGs reconstruct the actual pre-normalization input exactly.
        first = plan[0]
        sampler.start = first['start']
        scene_index = next(i for i, name in ds.scene_ids.items() if name == first['scene'])
        original = ds.getitem(scene_index, args.context_count, (16, 16))
        reused = reuse_example(args.reuse_inputs_from, first)
        for view in ['context', 'target']:
            for key in reused[view]:
                if not torch.equal(original[view][key], reused[view][key]):
                    raise ValueError(f'Saved input differs from raw dataset: {view}/{key}')
        atomic_json(out / 'input_reuse_validation.json', {
            'passed': True, 'source_export': str(args.reuse_inputs_from),
            'sample_id': first['sample_id'],
            'validation': 'Exact tensor equality with raw DatasetSpring for context RGB and context/target indices, extrinsics, intrinsics, near, far',
            'rendering': 'Fresh Gaussian inference and native-resolution rasterization; GT cropped from raw source RGB'})
        del original, reused
    if args.cached_rgb_from is not None:
        old_spec = json.loads((args.cached_rgb_from / 'dataset.json').read_text())
        for key in ['source_root', 'encoder_image_shape', 'original_image_shape', 'encoder_rescale_shape',
                    'clean_crop_box_after_original_resize', 'source_code_sha256']:
            old_value,new_value=old_spec[key],spec[key]
            if key=='source_code_sha256' and args.fixed_bounds:
                # Bounds changed in DatasetSpring; validate actual raw/cached tensors below.
                old_value={k:v for k,v in old_value.items() if k!='src/dataset/dataset_spring.py'}
                new_value={k:v for k,v in new_value.items() if k!='src/dataset/dataset_spring.py'}
            if old_value != new_value:
                raise ValueError(f'Cached RGB geometry/provenance differs: {key}')
        assert sc['make_baseline_1'] and sc['relative_pose'] and sc['force_davis_intrinsics']
        validations = []
        for pos in sorted(set([0, len(plan)//2, len(plan)-1])):
            row = plan[pos]
            sampler.start = row['start']
            scene_id = next(i for i, name in ds.scene_ids.items() if name == row['scene'])
            original = ds.getitem(scene_id, args.context_count, (16, 16))
            cached = cached_example(ds, row, args.cached_rgb_from)
            for view in ['context', 'target']:
                for key in cached[view]:
                    if not torch.equal(original[view][key], cached[view][key]):
                        raise ValueError(f'Cached construction differs from raw DatasetSpring: {row}/{view}/{key}')
            validations.append(row['sample_id'])
            del original, cached
        atomic_json(out / 'input_cache_validation.json', {'passed': True, 'samples': validations,
                    'check': 'Exact tensor equality of RGB, indices, poses, K, near/far against raw DatasetSpring with new context count',
                    'cached_rgb_source': str(args.cached_rgb_from)})
    if args.clean_cache_from is not None:
        old = json.loads((args.clean_cache_from / 'dataset.json').read_text())
        for key in ['source_root', 'output_image_shape', 'original_image_shape', 'clean_crop_box_after_original_resize', 'clean_resize']:
            if old[key] != spec[key]:
                raise ValueError(f'Clean cache mismatch: {key}')
    if (out / 'dataset.json').exists():
        if json.loads((out / 'dataset.json').read_text()) != spec:
            raise ValueError('Existing output has a different export specification')
    else:
        atomic_json(out / 'dataset.json', spec)
        atomic_json(out / 'source_config.json', raw)
    atomic_json(out / 'plan.json', {'samples': plan, 'rejected': rejected})
    done = {row['sample_id'] for row in plan if existing_complete(out, row)}
    pending = [row for row in plan if row['sample_id'] not in done]
    def status(state, **extra):
        value = {'state': state, 'pid': os.getpid(), 'total': len(plan), 'completed': len(done),
                 'remaining': len(plan)-len(done), 'elapsed_this_process_s': time.monotonic()-start_time,
                 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra}
        atomic_json(out / 'status.json', value)
        print(json.dumps(value), flush=True)
    status('loading_model')
    if not pending:
        status('complete')
        return
    encoder, decoder, checkpoint_step = build_encoder(raw, checkpoint)
    ids = {s: i for i, s in ds.scene_ids.items()}
    timings = []
    def save_clean(scene, index):
        path = out / 'clean' / scene / 'left' / f'{index:05d}.png'
        if path.exists():
            return
        if args.clean_cache_from is not None:
            old = args.clean_cache_from / 'clean' / scene / 'left' / f'{index:05d}.png'
            if old.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                os.link(old, path)
                return
        with Image.open(ds.scenes[scene][index]['rgb_file_path']) as im:
            im = im.convert('RGB').resize((original_w, original_h), Image.Resampling.BILINEAR)
            im = im.crop(crop_box).resize((args.width, args.height), Image.Resampling.LANCZOS)
            save_png(path, im)
    status('running', device=torch.cuda.get_device_name(), checkpoint_step=checkpoint_step)
    try:
        with ThreadPoolExecutor(max_workers=args.png_workers) as pool:
            for j, row in enumerate(pending):
                if args.limit is not None and j >= args.limit:
                    break
                t0 = time.monotonic()
                scene, start, sid = row['scene'], row['start'], row['sample_id']
                sampler.start = start
                # Call getitem directly: do not silently resample a different window on errors.
                example = (reuse_example(args.reuse_inputs_from, row) if args.reuse_inputs_from is not None
                           else cached_example(ds, row, args.cached_rgb_from) if args.cached_rgb_from is not None
                           else ds.getitem(ids[scene], args.context_count, (16, 16)))
                context_indices = example['context']['index'].tolist()
                target_indices = example['target']['index'].tolist()
                assert context_indices == list(range(start, row['end']+1, 2))
                assert target_indices == list(range(start, row['end']+1))
                paths = {
                    'context_image_paths': [f'clean/{scene}/left/{i:05d}.png' for i in context_indices],
                    'target_image_paths': [f'clean/{scene}/left/{i:05d}.png' for i in target_indices],
                    'encoder_input_image_paths': [f'encoder_input/{scene}/left/{i:05d}.png' for i in context_indices],
                    'prediction_paths': [f'samples/{sid}/prediction/{i:05d}.png' for i in target_indices],
                    'camera_path': f'samples/{sid}/cameras.npz'}
                inputs = (example['context']['image'] * 255).round().clamp(0,255).byte().permute(0,2,3,1).numpy()
                image_jobs = [pool.submit(save_clean, scene, i) for i in target_indices]
                for p, pixels in zip(paths['encoder_input_image_paths'], inputs):
                    if not (out / p).exists():
                        image_jobs.append(pool.submit(save_png, out / p, pixels.copy()))
                cameras = {f'{view}_{key}': example[view][key].numpy() for view in ['context','target'] for key in ['index','extrinsics','intrinsics','near','far']}
                cameras['source_w2c'] = np.array([ds.scenes[scene][i]['extrinsics'] for i in target_indices])
                cameras['source_normalized_intrinsics'] = np.array([ds.scenes[scene][i]['intrinsics'] for i in target_indices])
                batch = encoder.get_data_shim()(move(default_collate([example])))
                torch.cuda.synchronize()
                infer_start = time.monotonic()
                with torch.inference_mode():
                    requested_timestamps = batch['target']['index'][0]
                    gs = encoder(batch['context'], checkpoint_step, target_timestamps=requested_timestamps, visualization_dump=None)
                    rgb_frames = []
                    v = batch['target']
                    for i, timestamp in enumerate(requested_timestamps.tolist()):
                        rgb = decoder(gs[timestamp], v['extrinsics'][:,i:i+1], v['intrinsics'][:,i:i+1], v['near'][:,i:i+1], v['far'][:,i:i+1], (args.height,args.width)).color[0,0]
                        if not torch.isfinite(rgb).all():
                            raise ValueError(f'Nonfinite prediction {sid}/{timestamp}')
                        rgb_frames.append(rgb)
                    pixels = (torch.stack(rgb_frames).clamp(0,1)*255).round().byte().permute(0,2,3,1).cpu().numpy()
                infer_seconds = time.monotonic()-infer_start
                image_jobs.extend(pool.submit(save_png, out / p, pixels[i].copy()) for i, p in enumerate(paths['prediction_paths']))
                for future in image_jobs:
                    future.result()
                camera_path = out / paths['camera_path']
                camera_path.parent.mkdir(parents=True, exist_ok=True)
                with camera_path.with_suffix('.tmp').open('wb') as f:
                    np.savez_compressed(f, **cameras)
                os.replace(camera_path.with_suffix('.tmp'), camera_path)
                metadata = {'schema_version':1, 'complete':True, 'sample_id':sid, 'scene':scene,
                            'dataset':'spring', 'split':'train', 'index_base':0,
                            'context_camera':'left', 'target_camera':'left',
                            'context_indices':context_indices, 'target_indices':target_indices,
                            'prediction_indices':requested_timestamps.cpu().tolist(),
                            'context_source_paths':[ds.scenes[scene][i]['rgb_file_path'] for i in context_indices],
                            'target_source_paths':[ds.scenes[scene][i]['rgb_file_path'] for i in target_indices],
                            'context_source_file_numbers':[i+1 for i in context_indices],
                            'target_source_file_numbers':[i+1 for i in target_indices],
                            'checkpoint_step':checkpoint_step, 'export_spec':'dataset.json',
                            'caption_key':f'spring_{scene}', **paths}
                atomic_json(out / 'samples' / sid / 'sample.json', metadata)
                done.add(sid)
                timings.append(time.monotonic()-t0)
                with (out / 'progress.jsonl').open('a') as log:
                    log.write(json.dumps({'sample_id':sid,'seconds':timings[-1],'gpu_inference_render_s':infer_seconds,'completed':len(done)})+'\n')
                if j < 2 or len(done)%10==0 or len(done)==len(plan):
                    mean = sum(timings[-100:])/len(timings[-100:])
                    status('running', last_sample=sid, seconds_per_sample=mean,
                           estimated_remaining_hours=(len(plan)-len(done))*mean/3600)
                del example, batch, gs, rgb_frames, pixels, inputs, image_jobs, rgb, cameras
        manifest = [{'sample_id':row['sample_id'], 'path':f"samples/{row['sample_id']}/sample.json"} for row in plan if row['sample_id'] in done]
        atomic_json(out / 'index.json', {'schema_version':1, 'samples':manifest, 'complete':len(done)==len(plan)})
        status('complete' if len(done)==len(plan) else 'paused_after_limit')
    except BaseException as e:
        status('failed', error=f'{type(e).__name__}: {e}')
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_RUN/'checkpoints/epoch_2-step_45000.ckpt')
    p.add_argument('--config', type=Path, default=DEFAULT_RUN/'.hydra/config.yaml')
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--width', type=int, default=832)
    p.add_argument('--reuse-inputs-from', type=Path, default=None, help='Reuse verified saved 224x224 input RGB and cameras; render fresh predictions.')
    p.add_argument('--fixed-bounds', action='store_true', help='Use near=0.1 and far=100 in normalized coordinates without baseline division')
    p.add_argument('--context-count', type=int, default=12)
    p.add_argument('--cached-rgb-from', type=Path)
    p.add_argument('--clean-cache-from', type=Path)
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--limit', type=int, default=None, help='Commit at most this many new samples, then pause.')
    p.add_argument('--png-workers', type=int, default=8)
    p.add_argument('--cpu-threads', type=int, default=8)
    args = p.parse_args()
    if args.height < 1 or args.width < 1:
        p.error('height and width must be positive')
    if args.stride < 1 or args.png_workers < 1 or args.cpu_threads < 1 or (args.limit is not None and args.limit < 1):
        p.error('stride/workers/threads/limit must be positive')
    if args.context_count < 2 or (args.cached_rgb_from is not None and args.reuse_inputs_from is not None):
        p.error('context count must be >=2; cached RGB and full sample reuse are mutually exclusive')
    export(args)

if __name__ == '__main__':
    main()
