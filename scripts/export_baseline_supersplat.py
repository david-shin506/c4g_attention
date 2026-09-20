#!/usr/bin/env python3
"""Export official iPhone predictions from 4DGS baselines as 3DGS PLY sequences.

The baseline repositories are treated as read-only.  This script calls each
baseline's existing inference/renderer path and serializes the renderer-ready
Gaussians in the standard INRIA 3DGS PLY convention used by SuperSplat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


C0 = 0.28209479177387814
SCENES = ["apple", "block", "paper-windmill", "spin", "teddy"]
REQUIRED_PROPERTIES = {
    "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1",
    "rot_2", "rot_3",
}


def _cpu32(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _finite(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        count = int((~torch.isfinite(x)).sum())
        raise ValueError(f"{name} contains {count} NaN/Inf values")


def image_block_top1_mask(scores: torch.Tensor, block_size: int) -> torch.Tensor:
    """Keep the highest-scoring source Gaussian in every image-space block.

    Scores may have any leading dimensions, but its last two dimensions must
    be image height and width. Ties are deterministic (top-left wins). The
    returned mask has the same shape and never mixes source frames/views.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if scores.ndim < 2:
        raise ValueError(f"scores needs H,W dimensions, got {tuple(scores.shape)}")
    if block_size == 1:
        return torch.ones_like(scores, dtype=torch.bool)

    height, width = scores.shape[-2:]
    pad_h = (-height) % block_size
    pad_w = (-width) % block_size
    padded = F.pad(scores.float(), (0, pad_w, 0, pad_h), value=-torch.inf)
    blocks_h = padded.shape[-2] // block_size
    blocks_w = padded.shape[-1] // block_size
    prefix = list(padded.shape[:-2])
    prefix_dims = len(prefix)

    grouped = padded.reshape(
        *prefix, blocks_h, block_size, blocks_w, block_size
    ).permute(
        *range(prefix_dims), prefix_dims, prefix_dims + 2,
        prefix_dims + 1, prefix_dims + 3,
    )
    grouped = grouped.reshape(*prefix, blocks_h, blocks_w, block_size**2)
    winners = grouped.argmax(dim=-1, keepdim=True)
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(-1, winners, True)
    mask = mask.reshape(*prefix, blocks_h, blocks_w, block_size, block_size).permute(
        *range(prefix_dims), prefix_dims, prefix_dims + 2,
        prefix_dims + 1, prefix_dims + 3,
    )
    mask = mask.reshape(*prefix, padded.shape[-2], padded.shape[-1])
    return mask[..., :height, :width]


def contiguous_block_top1_mask(scores: torch.Tensor, group_size: int) -> torch.Tensor:
    """Keep one original row per contiguous native-order group."""
    scores = scores.float().reshape(-1)
    if group_size < 1:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if group_size == 1:
        return torch.ones_like(scores, dtype=torch.bool)
    pad = (-scores.numel()) % group_size
    padded = F.pad(scores, (0, pad), value=-torch.inf)
    winners = padded.reshape(-1, group_size).argmax(dim=-1)
    indices = torch.arange(len(winners), device=scores.device) * group_size + winners
    indices = indices[indices < scores.numel()]
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[indices] = True
    return mask


def decimation_metadata(block_size: int, method: str) -> dict[str, Any]:
    return {
        "enabled": block_size > 1,
        "block_size": block_size,
        "nominal_keep_fraction": 1.0 / (block_size**2),
        "method": method,
        "representative": "highest-opacity original Gaussian; attributes are not averaged",
        "temporal_policy": (
            "one source/native-row mask is selected once and reused for every target "
            "timestamp, preserving the selected Gaussian's motion and temporal identity"
        ),
        "lossy": block_size > 1,
    }


def compensate_block_decimation(
    opacities: torch.Tensor, scales: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate the opacity mass and footprint of an NxN source block."""
    if block_size <= 1:
        return opacities, scales
    coverage = block_size**2
    opacities = 1.0 - (1.0 - opacities.clamp(0, 1)).pow(coverage)
    scales = scales * float(block_size)
    return opacities, scales


def load_visibility_masks(paths: list[str], size: tuple[int, int]) -> torch.Tensor:
    from PIL import Image

    height, width = size
    masks = []
    for path in paths:
        image = Image.open(path).convert("L") if Path(path).exists() else Image.new("L", (width, height), 255)
        image = image.resize((width, height), Image.Resampling.NEAREST)
        masks.append(torch.from_numpy(np.asarray(image).copy()).float().div(255).gt(0.5))
    return torch.stack(masks).unsqueeze(1).float()


_LPIPS_VGG: Any = None


@torch.inference_mode()
def save_eval_artifacts(
    *,
    model_name: str,
    scene_dir: Path,
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    visibility: torch.Tensor | None,
    timestamp_indices: list[int],
) -> dict[str, Any]:
    """Save pred/GT-pred-diff panels and DyCheck-style evaluation summary."""
    from PIL import Image, ImageDraw
    from skimage.metrics import structural_similarity

    predictions = predictions.detach().float().cpu().clamp(0, 1)
    ground_truth = ground_truth.detach().float().cpu().clamp(0, 1)
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"Prediction/GT shape mismatch: {tuple(predictions.shape)} vs {tuple(ground_truth.shape)}"
        )
    if visibility is None:
        visibility = torch.ones(
            len(predictions), 1, *predictions.shape[-2:], dtype=torch.float32
        )
    visibility = visibility.detach().float().cpu()

    pred_eval = F.interpolate(predictions, size=(224, 224), mode="bilinear", align_corners=False)
    gt_eval = F.interpolate(ground_truth, size=(224, 224), mode="bilinear", align_corners=False)
    mask_eval = F.interpolate(visibility, size=(224, 224), mode="nearest").gt(0.5).float()

    diff_sq = (gt_eval - pred_eval).square().mean(dim=1, keepdim=True)
    mse = (diff_sq * mask_eval).sum(dim=(-3, -2, -1)) / mask_eval.sum(
        dim=(-3, -2, -1)
    ).clamp_min(1)
    psnr = -10 * mse.clamp_min(1e-12).log10()

    ssim_values = []
    for gt_frame, pred_frame, mask_frame in zip(gt_eval, pred_eval, mask_eval):
        ssim_map = structural_similarity(
            gt_frame.numpy(), pred_frame.numpy(), win_size=11,
            gaussian_weights=True, channel_axis=0, data_range=1.0, full=True,
        )[1]
        ssim_map = torch.from_numpy(ssim_map).float().mean(dim=0)
        mask_2d = mask_frame[0]
        ssim_values.append((ssim_map * mask_2d).sum() / mask_2d.sum().clamp_min(1))
    ssim = torch.stack(ssim_values)

    global _LPIPS_VGG
    if _LPIPS_VGG is None:
        from lpips import LPIPS
        _LPIPS_VGG = LPIPS(net="vgg").eval().cuda()
    lpips_chunks = []
    for start in range(0, len(gt_eval), 8):
        end = min(start + 8, len(gt_eval))
        chunk = _LPIPS_VGG(
            gt_eval[start:end].cuda() * 2 - 1,
            pred_eval[start:end].cuda() * 2 - 1,
        )
        lpips_chunks.append(chunk[:, 0, 0, 0].detach().cpu())
    lpips_values = torch.cat(lpips_chunks)

    pred_dir = scene_dir / "pred_images"
    compare_dir = scene_dir / "compare"
    pred_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)
    for index, (pred_frame, gt_frame) in enumerate(zip(predictions, ground_truth)):
        pred_array = (pred_frame.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        gt_array = (gt_frame.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        diff_array = (
            (pred_frame - gt_frame).abs().mul(4).clamp(0, 1).permute(1, 2, 0).numpy() * 255
        ).round().astype(np.uint8)
        Image.fromarray(pred_array).save(pred_dir / f"frame_{index:04d}.png")

        height, width = pred_array.shape[:2]
        panel = Image.new("RGB", (width * 3, height + 28), "white")
        panel.paste(Image.fromarray(gt_array), (0, 28))
        panel.paste(Image.fromarray(pred_array), (width, 28))
        panel.paste(Image.fromarray(diff_array), (width * 2, 28))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 7), "GT", fill="black")
        draw.text((width + 8, 7), f"PRED  PSNR {psnr[index]:.2f}", fill="black")
        draw.text((width * 2 + 8, 7), "ABS DIFF x4", fill="black")
        panel.save(compare_dir / f"frame_{index:04d}.png")

    metrics = {
        "model": model_name,
        "scene": scene_dir.name,
        "num_frames": len(predictions),
        "psnr": float(psnr.mean()),
        "ssim": float(ssim.mean()),
        "lpips": float(lpips_values.mean()),
        "metric_policy": "DyCheck visibility-masked PSNR/SSIM; unmasked VGG LPIPS at 224x224",
        "per_frame": [
            {
                "frame": index,
                "timestamp": int(timestamp_indices[index]),
                "psnr": float(psnr[index]),
                "ssim": float(ssim[index]),
                "lpips": float(lpips_values[index]),
            }
            for index in range(len(predictions))
        ],
    }
    lines = [
        f"model: {model_name}", f"scene: {scene_dir.name}",
        f"frames: {len(predictions)}", f"PSNR: {metrics['psnr']:.4f}",
        f"SSIM: {metrics['ssim']:.6f}", f"LPIPS: {metrics['lpips']:.6f}",
        f"policy: {metrics['metric_policy']}", "", "frame timestamp psnr ssim lpips",
    ]
    lines.extend(
        f"{row['frame']:04d} {row['timestamp']:04d} {row['psnr']:.4f} {row['ssim']:.6f} {row['lpips']:.6f}"
        for row in metrics["per_frame"]
    )
    metric_name = f"{model_name}_psnr_{metrics['psnr']:.2f}.txt"
    text = "\n".join(lines) + "\n"
    (scene_dir / metric_name).write_text(text, encoding="utf-8")
    (scene_dir.parent / metric_name).write_text(text, encoding="utf-8")
    (scene_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def write_standard_3dgs_ply(
    path: Path,
    *,
    means: torch.Tensor,
    harmonics: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    rotations_wxyz: torch.Tensor,
    harmonics_are_rgb_parameters: bool,
    comment: str,
) -> dict[str, Any]:
    """Write activated renderer attributes as inverse-activated INRIA 3DGS PLY."""
    means = _cpu32(means).reshape(-1, 3)
    harmonics = _cpu32(harmonics)
    if harmonics.ndim == 2:
        harmonics = harmonics.reshape(len(means), -1, 3)
    opacities = _cpu32(opacities).reshape(-1)
    scales = _cpu32(scales).reshape(-1, 3)
    rotations_wxyz = _cpu32(rotations_wxyz).reshape(-1, 4)

    n = len(means)
    if harmonics.ndim != 3 or harmonics.shape[0] != n or harmonics.shape[-1] != 3:
        raise ValueError(f"Invalid harmonics shape {tuple(harmonics.shape)} for N={n}")
    if len(opacities) != n or len(scales) != n or len(rotations_wxyz) != n:
        raise ValueError("Gaussian attribute row counts do not match")
    d_sh = harmonics.shape[1]
    degree = math.isqrt(d_sh) - 1
    if (degree + 1) ** 2 != d_sh:
        raise ValueError(f"SH coefficient count is not square: {d_sh}")
    for name, value in {
        "means": means, "harmonics": harmonics, "opacities": opacities,
        "scales": scales, "rotations_wxyz": rotations_wxyz,
    }.items():
        _finite(name, value)
    if n == 0:
        raise ValueError("Refusing to write an empty Gaussian scene")
    if not torch.all(scales > 0):
        raise ValueError("Renderer scales must be strictly positive")
    if not torch.all((opacities >= 0) & (opacities <= 1)):
        raise ValueError("Renderer opacities must be in [0, 1]")
    quat_norm = rotations_wxyz.norm(dim=-1)
    if not torch.all(quat_norm > 0):
        raise ValueError("Renderer rotations contain a zero quaternion")

    # 4DGT and MoVieS render RGB-like coefficients.  Their own PLY convention
    # maps these to SH with (rgb - 0.5) / C0.  NeoVerse already emits true SH.
    sh = (harmonics - 0.5) / C0 if harmonics_are_rgb_parameters else harmonics
    names = ["x", "y", "z", "nx", "ny", "nz"]
    parts = [means, torch.zeros_like(means)]
    names.extend(f"f_dc_{i}" for i in range(3))
    parts.append(sh[:, 0, :])
    if d_sh > 1:
        names.extend(f"f_rest_{i}" for i in range(3 * (d_sh - 1)))
        # INRIA convention is channel-major for non-DC coefficients.
        parts.append(sh[:, 1:, :].permute(0, 2, 1).reshape(n, -1))

    names.append("opacity")
    eps = torch.finfo(torch.float32).eps
    endpoint_count = int(((opacities == 0) | (opacities == 1)).sum())
    parts.append(torch.logit(opacities.clamp(eps, 1 - eps))[:, None])
    names.extend(f"scale_{i}" for i in range(3))
    parts.append(scales.log())
    names.extend(f"rot_{i}" for i in range(4))
    # gsplat and all three baseline renderers use wxyz, as does INRIA PLY.
    parts.append(rotations_wxyz)

    matrix = torch.cat(parts, dim=-1)
    _finite("serialized columns", matrix)
    array = matrix.numpy().astype("<f4", copy=False)
    dtype = np.dtype([(name, "<f4") for name in names])
    vertices = np.empty(n, dtype=dtype)
    for index, name in enumerate(names):
        vertices[name] = array[:, index]

    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"comment {comment}",
        f"element vertex {n}",
        *(f"property float {name}" for name in names),
        "end_header",
        "",
    ]
    header_bytes = "\n".join(header).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header_bytes)
        handle.write(vertices.tobytes(order="C"))

    expected_size = len(header_bytes) + n * dtype.itemsize
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"PLY size mismatch: {path}")
    if not REQUIRED_PROPERTIES.issubset(names):
        raise RuntimeError(f"PLY missing properties: {sorted(REQUIRED_PROPERTIES - set(names))}")
    return {
        "file": path.name,
        "num_gaussians": n,
        "num_sh_coefficients_per_channel": d_sh,
        "sh_degree": degree,
        "properties": names,
        "format": "binary_little_endian 1.0",
        "size_bytes": path.stat().st_size,
        "finite": True,
        "quaternion_order": "wxyz",
        "quaternion_norm_max_error": float((quat_norm - 1).abs().max()),
        "opacity_endpoint_clamp_count": endpoint_count,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_scene_dir(output_root: Path, model_name: str, scene: str) -> tuple[Path, Path]:
    scene_dir = output_root / model_name / f"iphone_{scene}"
    sequence_dir = scene_dir / "gaussian_sequence"
    if sequence_dir.exists():
        for old in sequence_dir.glob("frame_*.ply"):
            old.unlink()
    sequence_dir.mkdir(parents=True, exist_ok=True)
    return scene_dir, sequence_dir


def save_metadata(scene_dir: Path, metadata: dict[str, Any]) -> None:
    with (scene_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def add_paths(baseline_project: Path, deps: Path) -> None:
    sys.path.insert(0, str(baseline_project))
    # Append rather than prepend: the temporary target accidentally contains
    # alternate torch/numpy builds; system packages must win.
    if deps.exists():
        sys.path.append(str(deps))


def iphone_batch(
    baseline_root: Path,
    deps: Path,
    data_root: Path,
    scene: str,
    *,
    image_res: int,
    num_context_frames: int,
    max_temporal: int,
    num_target_frames: int | None,
    normalize_pose: bool,
    load_covisible: bool = False,
) -> dict[str, Any]:
    add_paths(baseline_root / "4DGT", deps)
    from torch.utils.data.dataloader import default_collate
    from tlod.data_loader.iphone_dataset import IPhoneEvalDataset

    dataset = IPhoneEvalDataset(
        data_root=str(data_root), data_list=[scene],
        input_image_res=(image_res, image_res),
        output_image_res=(image_res, image_res),
        num_context_frames=num_context_frames,
        max_temporal=max_temporal,
        context_camera=0, target_cameras=[1],
        num_target_frames=num_target_frames, fps=30.0,
        load_covisible=load_covisible, normalize_pose=normalize_pose,
        normalize_scale=False,
    )
    if len(dataset) != 1:
        raise RuntimeError(f"Could not load iPhone scene {scene}")
    return default_collate([dataset[0]])


@torch.inference_mode()
def export_4dgt(args: argparse.Namespace) -> None:
    project = args.baseline_root / "4DGT"
    add_paths(project, args.deps)
    from tlod.demo import FourDGTDemo
    from tlod.easyvolcap.utils.quat_utils import angle_axis_to_quaternion, qmul
    from tlod.renderers.tlod_renderer import compute_marginal_t

    class FourDGTDemoNoApex(FourDGTDemo):
        def _load_model_config(self):
            cfg = super()._load_model_config()
            # The released config requests Apex FusedLayerNorm, which is not
            # installed in this server image. PyTorch LayerNorm has the same
            # parameters/state-dict and inference semantics.
            cfg.image_encoder.enable_layernorm_kernel = False
            # The repository detects that FlashAttention is unavailable but
            # does not clear this config flag before calling a None function.
            # Standard scaled-dot-product attention is the built-in fallback.
            cfg.image_encoder.enable_flash_attn = False
            return cfg

    demo = FourDGTDemoNoApex(
        config_path=str(project / "configs/models/tlod-l3.py"),
        checkpoint_path=str(args.checkpoint), device="cuda",
        output_dir="/tmp/c4g_4dgt_demo_runtime",
    )
    settings = {
        "image_res": 504, "num_context_frames": 32, "max_temporal": 63,
        "context_camera": 0, "target_cameras": [1], "num_target_frames": 63,
        "fps": 30.0, "normalize_pose": True, "normalize_scale": False,
        "source": "4DGT/run.sh",
    }
    for scene in args.scenes:
        start = time.time()
        batch = iphone_batch(
            args.baseline_root, args.deps, args.data_root, scene,
            image_res=504, num_context_frames=32, max_temporal=63,
            num_target_frames=63, normalize_pose=True,
            load_covisible=args.save_eval_artifacts,
        )
        enc = demo.encode(batch["rgb_input"], batch["cameras_input"], batch["rays_t_un_input"])
        gp = enc["gaussian_parameters"]
        flat_full = {
            key: value.reshape(value.shape[0], -1, value.shape[-1])
            for key, value in gp.items()
        }
        # Keep the renderer-ready prediction untouched for evaluation. The
        # lossy 2D decimator is only an export/storage transform for PLY files.
        flat = flat_full
        timestamps = batch["rays_t_un_output"][0]
        if args.limit_frames:
            timestamps = timestamps[: args.limit_frames]

        decimation = decimation_metadata(
            args.spatial_block_size,
            "native-order contiguous top-1 groups after 4DGT's official hierarchical magic-pattern sampling",
        )
        if args.spatial_block_size > 1:
            # 4DGT has already concatenated its multi-level, patch-sampled grids
            # at this point. Native rows remain spatially local, but are no
            # longer a single dense HxW image. Keep the most visible original
            # temporal Gaussian in each contiguous group of block_size**2.
            base_opacity = flat["opacity"][0, :, 0].float()
            max_effective_opacity = torch.zeros_like(base_opacity)
            for ts_score in timestamps:
                marginal_score = compute_marginal_t(
                    ts_score.to(flat["t"].device), flat["t"][0], flat["cov_t"][0]
                ).float()[:, 0]
                max_effective_opacity = torch.maximum(
                    max_effective_opacity, base_opacity * marginal_score
                )
            keep = contiguous_block_top1_mask(
                max_effective_opacity, args.spatial_block_size**2
            )
            decimation["input_gaussians"] = int(keep.numel())
            decimation["selected_gaussians"] = int(keep.sum())
            flat = {key: value[:, keep] for key, value in flat.items()}
            flat["opacity"], flat["scaling"] = compensate_block_decimation(
                flat["opacity"], flat["scaling"], args.spatial_block_size
            )
            decimation["compensation"] = {
                "opacity": "1 - (1 - alpha) ** (block_size ** 2)",
                "scale": "scale * block_size",
                "reason": "preserve approximate block coverage after selecting one representative",
            }

        xyz, rgb, scales, rotations, opacity = (
            flat["xyz"], flat["feature"], flat["scaling"], flat["rotation"], flat["opacity"]
        )
        t, cov_t, ms3, omega = flat["t"], flat["cov_t"], flat["ms3"], flat["omega"]
        scene_dir, seq_dir = make_scene_dir(args.output_root, "4DGT", scene)
        frames = []
        for frame_index, ts in enumerate(timestamps):
            dt = (ts.to(t.device) - t[0]).float()
            ms_degree = ms3.shape[-1] // 3
            dmeans = sum(ms3[0, :, 3*i:3*(i+1)] * dt ** (i+1) for i in range(ms_degree))
            means = xyz[0].float() + dmeans
            omega_degree = omega.shape[-1] // 3
            domega = sum(omega[0, :, 3*i:3*(i+1)] * dt ** (i+1) for i in range(omega_degree))
            quats = qmul(rotations[0].float(), angle_axis_to_quaternion(domega)).float()
            marginal = compute_marginal_t(ts.to(t.device), t[0], cov_t[0]).float()[:, 0]
            opacities = opacity[0, :, 0].float() * marginal
            # This is the model's renderer filter, not visualization pruning.
            mask = (marginal > 0.05) & (opacities > 0.0001)
            path = seq_dir / f"frame_{frame_index:04d}.ply"
            info = write_standard_3dgs_ply(
                path, means=means[mask], harmonics=rgb[0, mask, :3].reshape(-1, 1, 3),
                opacities=opacities[mask], scales=scales[0, mask],
                rotations_wxyz=quats[mask], harmonics_are_rgb_parameters=True,
                comment="4DGT renderer Gaussians in standard INRIA 3DGS convention",
            )
            info["timestamp_seconds"] = float(ts)
            info["timestamp_index"] = int(round(float(ts) * 30.0))
            frames.append(info)
            print(f"[4DGT/{scene}] {frame_index+1}/{len(timestamps)} N={info['num_gaussians']:,}", flush=True)
        metrics = None
        if args.save_eval_artifacts:
            render_data = {
                "gaussian_parameters": flat_full,
                "height": enc["height"],
                "width": enc["width"],
            }
            num_targets = len(timestamps)
            rendered = demo.render_batch(
                render_data,
                batch["cameras_output"][:, :num_targets],
                batch["rays_t_un_output"][:, :num_targets],
                render_mode="RGB",
                ratios=batch["ratios_output"][:, :num_targets],
            )
            visibility = batch.get("visibility")
            metrics = save_eval_artifacts(
                model_name="4DGT", scene_dir=scene_dir,
                predictions=(rendered["rgb"][0] + 1) / 2,
                ground_truth=(batch["rgb_output"][0, :num_targets] + 1) / 2,
                visibility=None if visibility is None else visibility[0, :num_targets, None],
                timestamp_indices=[frame["timestamp_index"] for frame in frames],
            )
        metadata = {
            "model": "4DGT", "dataset": "iphone", "scene": scene,
            "source_sample": f"iphone_{scene}", "num_frames": len(frames),
            "timestamps": [f["timestamp_index"] for f in frames],
            "num_gaussians_per_frame": [f["num_gaussians"] for f in frames],
            "checkpoint": str(args.checkpoint), "checkpoint_sha256": args.checkpoint_sha256,
            "official_settings": settings,
            "coordinate_system": "4DGT normalized world coordinates; identical across timestamps",
            "row_ordering": "Original flattened Gaussian order retained after renderer temporal mask",
            "renderer_filter": "marginal_t > 0.05 and effective opacity > 0.0001 (4DGT renderer semantics)",
            "spatial_decimation": decimation,
            "metric_gaussians": "uncompressed official renderer prediction (before PLY decimation)",
            "metrics": metrics,
            "ply_attribute_convention": "INRIA 3DGS: SH, logit opacity, log scale, wxyz quaternion",
            "frames": frames, "elapsed_seconds": time.time() - start,
        }
        save_metadata(scene_dir, metadata)
        del enc, gp, batch
        torch.cuda.empty_cache()


def install_neoverse_image_decimator(
    gs_renderer: Any, block_size: int
) -> tuple[dict[str, Any], Any | None]:
    """Mask dense source-grid splats before NeoVerse temporal separation.

    Rows are retained inside NeoVerse so every selected Gaussian keeps its
    original forward/backward velocity, rotation and lifespan tensors. Dropped
    rows get exactly zero opacity and are removed only when materializing PLY.
    """
    metadata = decimation_metadata(
        block_size,
        "source-frame image-grid top-1 by base opacity before static/dynamic classification",
    )
    if block_size == 1:
        return metadata, None

    original = gs_renderer.separate_splats

    def separate_with_mask(splats: dict[str, torch.Tensor], *args: Any, **kwargs: Any) -> Any:
        context_depth = kwargs.get("context_depth")
        if context_depth is None or context_depth.ndim != 4:
            raise ValueError("NeoVerse 2D decimation requires [B,S,H,W] context_depth")
        batch, sources, height, width = context_depth.shape
        opacities = splats["opacities"]
        if opacities.shape != (batch, sources, height * width):
            raise ValueError(
                "NeoVerse opacity/grid mismatch: "
                f"{tuple(opacities.shape)} vs {(batch, sources, height, width)}"
            )
        importance = opacities.reshape(batch, sources, height, width)
        keep = image_block_top1_mask(importance, block_size)
        flat_keep = keep.reshape(batch, sources, height * width)
        splats = dict(splats)
        compensated_opacity, compensated_scale = compensate_block_decimation(
            opacities, splats["scales"], block_size
        )
        splats["opacities"] = torch.where(
            flat_keep, compensated_opacity, torch.zeros_like(opacities)
        )
        splats["scales"] = torch.where(
            flat_keep[..., None], compensated_scale, splats["scales"]
        )
        metadata["compensation"] = {
            "opacity": "1 - (1 - alpha) ** (block_size ** 2)",
            "scale": "scale * block_size",
        }
        metadata["input_gaussians"] = int(flat_keep.numel())
        metadata["selected_gaussians"] = int(flat_keep.sum())
        metadata["source_grid"] = [height, width]
        return original(splats, *args, **kwargs)

    # This instance attribute is intentionally local to this inference wrapper;
    # the read-only baseline repository and model weights are never modified.
    gs_renderer.separate_splats = separate_with_mask
    return metadata, original

def optimize_neoverse_shared_camera(
    *,
    recon: Any,
    enc: dict[str, Any],
    gt_images: torch.Tensor,
    visibility: torch.Tensor | None,
    c2w_init: torch.Tensor,
    K: torch.Tensor,
    timestamps: torch.Tensor,
    apply_se3_delta: Any,
    source_height: int,
    source_width: int,
    num_steps: int,
    num_sample_frames: int,
    optimize_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit one shared SE(3) camera correction over a static-camera sequence.

    Unlike NeoVerse's released per-target TTO, this has exactly six learnable
    values for the entire sequence. A deterministic temporal subset is used
    for the fit, while the same resulting transform is applied to every frame.
    """
    from contextlib import nullcontext

    frame_count = len(c2w_init)
    sample_count = min(frame_count, max(1, num_sample_frames))
    sample_ids = torch.linspace(0, frame_count - 1, sample_count).round().long().unique()

    # export_neoverse is inference-mode wrapped. Explicitly leave inference
    # mode here because only the two shared camera deltas require gradients.
    with torch.inference_mode(False), torch.enable_grad():
        c2w_all = c2w_init.detach().cuda().float().clone()
        K_all = K.detach().cuda().float().clone()
        ts_all = timestamps.detach().cuda().clone()
        gt = gt_images.detach().cuda().float().clone()
        mask = None if visibility is None else visibility.detach().cuda().float().clone()

        ids = sample_ids.cuda()
        base_w2c = recon.homo_matrix_inverse(c2w_all[ids])
        K_fit = K_all[ids].clone()
        if (optimize_size, optimize_size) != (source_height, source_width):
            K_fit[:, 0, :] *= optimize_size / source_width
            K_fit[:, 1, :] *= optimize_size / source_height
        gt_fit = F.interpolate(
            gt[ids], size=(optimize_size, optimize_size),
            mode="bilinear", align_corners=False,
        )
        mask_fit = None
        if mask is not None:
            mask_fit = F.interpolate(
                mask[ids], size=(optimize_size, optimize_size), mode="nearest"
            ).gt(0.5).float()

        rot_delta = torch.nn.Parameter(torch.zeros(1, 3, device="cuda"))
        trans_delta = torch.nn.Parameter(torch.zeros(1, 3, device="cuda"))
        optimizer = torch.optim.Adam(
            [
                {"params": [rot_delta], "lr": 1e-3},
                {"params": [trans_delta], "lr": 1e-3},
            ]
        )

        def loss_for_delta() -> torch.Tensor:
            w2c = apply_se3_delta(
                base_w2c,
                rot_delta.expand(len(ids), -1),
                trans_delta.expand(len(ids), -1),
            )
            autocast = (
                torch.amp.autocast("cuda", dtype=recon.dtype)
                if torch.cuda.is_available() else nullcontext()
            )
            with autocast:
                rgb, _, _ = recon.model.gs_renderer.rasterizer.forward(
                    enc["splats"], render_viewmats=[w2c], render_Ks=[K_fit],
                    render_timestamps=[ts_all[ids]], sh_degree=0,
                    width=optimize_size, height=optimize_size,
                )
            pred = rgb[0].permute(0, 3, 1, 2).float().clamp(0, 1)
            squared = (pred - gt_fit).square().mean(dim=1, keepdim=True)
            if mask_fit is None:
                return squared.mean()
            return (squared * mask_fit).sum() / mask_fit.sum().clamp_min(1)

        best_loss = math.inf
        best_rot = rot_delta.detach().clone()
        best_trans = trans_delta.detach().clone()
        initial_loss = None
        completed_steps = 0
        for step in range(num_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_delta()
            if initial_loss is None:
                initial_loss = float(loss.detach())
            current = float(loss.detach())
            if current < best_loss:
                best_loss = current
                best_rot = rot_delta.detach().clone()
                best_trans = trans_delta.detach().clone()
            loss.backward()
            optimizer.step()
            completed_steps = step + 1
            if step % 50 == 0 or step + 1 == num_steps:
                print(
                    f"  shared static-camera TTO step {step:4d} | loss={current:.6f}",
                    flush=True,
                )

        # Include the post-update endpoint in the best-state comparison.
        final_loss = float(loss_for_delta().detach())
        if initial_loss is None:
            initial_loss = final_loss
        if final_loss < best_loss:
            best_loss = final_loss
            best_rot = rot_delta.detach().clone()
            best_trans = trans_delta.detach().clone()

        base_w2c_all = recon.homo_matrix_inverse(c2w_all)
        corrected_w2c = apply_se3_delta(
            base_w2c_all,
            best_rot.expand(frame_count, -1),
            best_trans.expand(frame_count, -1),
        )
        corrected_c2w = recon.homo_matrix_inverse(corrected_w2c).detach().cpu()
        details = {
            "policy": "one shared SE(3) correction for the complete static-camera sequence",
            "degrees_of_freedom": 6,
            "per_frame_pose_parameters": False,
            "fit_frame_indices": [int(value) for value in sample_ids],
            "fit_timestamps": [int(timestamps[value]) for value in sample_ids],
            "fit_resolution": [optimize_size, optimize_size],
            "fit_mask": "DyCheck covisibility mask" if visibility is not None else "none",
            "steps": completed_steps,
            "initial_loss": initial_loss,
            "best_loss": best_loss,
            "rotation_axis_angle": best_rot[0].cpu().tolist(),
            "translation": best_trans[0].cpu().tolist(),
        }
    return corrected_c2w, details


def neoverse_state_for_timestamp(
    gs_list: list[Any], timestamp: int, rasterizer: Any, drop_zero_opacity: bool = False
) -> dict[str, torch.Tensor]:
    transitioned = []
    for splats in gs_list:
        if splats.timestamp == -1 or splats.timestamp == timestamp:
            render_flag = True
        elif timestamp > splats.timestamp and splats.forward_timestamp is not None and timestamp < splats.forward_timestamp:
            render_flag = rasterizer.bidirection or abs(timestamp - splats.timestamp) <= abs(timestamp - splats.forward_timestamp)
        elif timestamp < splats.timestamp and splats.backward_timestamp is not None and timestamp > splats.backward_timestamp:
            render_flag = rasterizer.bidirection or abs(timestamp - splats.timestamp) < abs(timestamp - splats.backward_timestamp)
        else:
            render_flag = False
        if not render_flag:
            continue
        mask = torch.ones_like(splats.opacities, dtype=torch.bool)
        if drop_zero_opacity:
            mask &= splats.opacities > 0
        if rasterizer.opacity_prune_threshold >= 0:
            mask &= splats.opacities >= rasterizer.opacity_prune_threshold
        if rasterizer.confidence_prune_threshold >= 0 and splats.confidences is not None:
            mask &= splats.confidences >= rasterizer.confidence_prune_threshold
        transitioned.append(splats.transition(timestamp, mask=mask))
    if not transitioned:
        raise RuntimeError(f"NeoVerse produced no Gaussians for timestamp {timestamp}")
    return {
        "means": torch.cat([g.means for g in transitioned]),
        "harmonics": torch.cat([g.harmonics for g in transitioned]),
        "opacities": torch.cat([g.opacities for g in transitioned]),
        "scales": torch.cat([g.scales for g in transitioned]),
        "rotations": torch.cat([g.rotations for g in transitioned]),
    }


def movies_pc_from_outputs(
    renderer: Any,
    model_outputs: dict[str, torch.Tensor],
    input_c2w: torch.Tensor,
    input_intr: torch.Tensor,
    spatial_keep_mask: torch.Tensor | None = None,
    spatial_block_size: int = 1,
) -> Any:
    """Run MoVieS' exact renderer activation/filter path without rasterization."""
    from einops import rearrange
    from src.models.gs_render.gs_util import GaussianModel
    from src.utils import unproject_depth

    color, scale = model_outputs["color"], model_outputs["scale"]
    rotation, opacity = model_outputs["rotation"], model_outputs["opacity"]
    depth, xyz = model_outputs.get("depth"), model_outputs.get("xyz")
    motion_color, motion_scale = model_outputs.get("motion_color"), model_outputs.get("motion_scale")
    motion_rotation = model_outputs.get("motion_rotation")
    motion_opacity = model_outputs.get("motion_opacity")

    def flatten(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else rearrange(value, "b v c h w -> b (v h w) c")

    color, scale, rotation, opacity = map(flatten, (color, scale, rotation, opacity))
    motion_color, motion_scale, motion_rotation, motion_opacity = map(
        flatten, (motion_color, motion_scale, motion_rotation, motion_opacity)
    )
    if xyz is None:
        depth = renderer.depth_activation(depth)
        xyz = unproject_depth(depth.squeeze(2), input_c2w, input_intr)
    xyz = renderer.xyz_activation(xyz)
    offset = model_outputs.get("offset", torch.zeros_like(xyz))
    xyz = flatten(xyz + renderer.offset_activation(offset))

    color = renderer.color_activation(color if motion_color is None else motion_color)
    opacity = renderer.opacity_activation(opacity if motion_opacity is None else motion_opacity)
    scale = renderer.scale_activation(scale if motion_scale is None else motion_scale)
    rotation = renderer.rotation_activation(rotation if motion_rotation is None else motion_rotation)
    color = rearrange(color, "b n (k rgb) -> b n k rgb", rgb=3)
    color = color * renderer.sh_mask[None, None, :, None].to(color.device)
    color = rearrange(color, "b n k rgb -> b n (k rgb)")

    if spatial_keep_mask is not None:
        keep = spatial_keep_mask.reshape(-1).to(device=xyz.device)
        expected = xyz.shape[1]
        if keep.numel() != expected:
            raise ValueError(
                f"MoVieS spatial mask has {keep.numel()} rows, expected {expected}"
            )
        xyz, color, scale, rotation, opacity = (
            value[:, keep] for value in (xyz, color, scale, rotation, opacity)
        )
        opacity, scale = compensate_block_decimation(
            opacity, scale, spatial_block_size
        )

    pc = GaussianModel().set_data(
        xyz[0], color[0], scale[0], rotation[0], opacity[0], renderer.opt.sh_degree
    )
    # This is the official eval-time branch in GaussianRenderer.render.
    if renderer.opt.prune_ratio > 0:
        keep = (pc.opacity > renderer.opt.opacity_threshold).squeeze(-1)
        pc.set_data(pc.xyz[keep], pc.color[keep], pc.scale[keep], pc.rotation[keep], pc.opacity[keep])
    if renderer.opt.voxel_size > 0:
        raise NotImplementedError("MoVieS official config unexpectedly enables voxelization")
    return pc


@torch.inference_mode()
def export_neoverse(args: argparse.Namespace) -> None:
    project = args.baseline_root / "NeoVerse"
    add_paths(project, args.deps)
    from eval_iphone import (
        NeoVerseReconstructor, _apply_se3_delta, align_poses_umeyama,
        cameras_4dgt_to_c2w_K,
    )

    recon = NeoVerseReconstructor(str(args.checkpoint), device="cuda")
    settings = {
        "image_res": 504, "num_context_frames": 32, "max_temporal": 63,
        "context_camera": 0, "target_cameras": [1], "num_target_frames": 63,
        "fps": 30.0, "normalize_pose": False, "normalize_scale": False,
        "use_motion": True, "source": "NeoVerse/run.sh",
        "camera_policy": "Umeyama context alignment + one shared static-camera SE(3) correction",
        "shared_camera_tto": {
            "steps": args.neoverse_shared_camera_tto_steps,
            "sample_frames": args.neoverse_shared_camera_tto_frames,
            "fit_size": args.neoverse_shared_camera_tto_size,
        },
    }
    rasterizer = recon.model.gs_renderer.rasterizer
    for scene in args.scenes:
        start = time.time()
        batch = iphone_batch(
            args.baseline_root, args.deps, args.data_root, scene,
            image_res=504, num_context_frames=32, max_temporal=63,
            num_target_frames=63, normalize_pose=False,
            load_covisible=args.save_eval_artifacts,
        )
        ctx = (batch["rgb_input"][0] + 1) / 2
        ctx_indices = torch.round(batch["rays_t_un_input"][0] * 30.0).long()
        target_indices = torch.round(batch["rays_t_un_output"][0] * 30.0).long()
        if args.limit_frames:
            target_indices = target_indices[: args.limit_frames]
        # First encode is lossless and is the only representation used for metrics.
        enc = recon.encode(ctx, ctx_indices, use_motion=True)
        scene_dir, seq_dir = make_scene_dir(args.output_root, "NeoVerse", scene)

        # A second encode receives the lossy image-grid mask solely for PLY export.
        # Restore the renderer immediately so the next scene again starts lossless.
        decimation, original_separate = install_neoverse_image_decimator(
            recon.model.gs_renderer, args.spatial_block_size
        )
        if original_separate is None:
            enc_export = enc
        else:
            try:
                enc_export = recon.encode(ctx, ctx_indices, use_motion=True)
            finally:
                recon.model.gs_renderer.separate_splats = original_separate
        gs_list = enc_export["splats"][0]
        frames = []
        for frame_index, ts_tensor in enumerate(target_indices):
            ts = int(ts_tensor)
            state = neoverse_state_for_timestamp(
                gs_list, ts, rasterizer, args.spatial_block_size > 1
            )
            path = seq_dir / f"frame_{frame_index:04d}.ply"
            info = write_standard_3dgs_ply(
                path, means=state["means"], harmonics=state["harmonics"],
                opacities=state["opacities"], scales=state["scales"],
                rotations_wxyz=state["rotations"], harmonics_are_rgb_parameters=False,
                comment="NeoVerse renderer Gaussians in standard INRIA 3DGS convention",
            )
            info["timestamp_index"] = ts
            frames.append(info)
            print(f"[NeoVerse/{scene}] {frame_index+1}/{len(target_indices)} N={info['num_gaussians']:,}", flush=True)
        metrics = None
        camera_alignment = None
        if args.save_eval_artifacts:
            height, width = ctx.shape[-2:]
            ctx_c2w, _ = cameras_4dgt_to_c2w_K(batch["cameras_input"][0], height, width)
            tgt_c2w, tgt_K = cameras_4dgt_to_c2w_K(
                batch["cameras_output"][0, :len(target_indices)], height, width
            )
            scale_uma, transform_uma = align_poses_umeyama(
                ctx_c2w.cuda().float(), enc["c2w"].detach().float()
            )
            if not torch.isnan(scale_uma) and float(scale_uma.abs()) > 1e-8:
                # T_sim stores [sR, t]. A camera c2w must stay rigid: keep R
                # unit-length and apply s only to its translation. The upstream
                # helper multiplies the rotation block by s, after which its
                # rigid homo_matrix_inverse no longer represents an inverse.
                tgt_source = tgt_c2w.cuda().float()
                align_rotation = transform_uma[:3, :3] / scale_uma
                tgt_c2w_aligned = tgt_source.clone()
                tgt_c2w_aligned[:, :3, :3] = (
                    align_rotation @ tgt_source[:, :3, :3]
                )
                tgt_c2w_aligned[:, :3, 3] = (
                    scale_uma * (align_rotation @ tgt_source[:, :3, 3, None])[:, :, 0]
                    + transform_uma[:3, 3]
                )
                tgt_c2w_aligned = tgt_c2w_aligned.cpu()
            else:
                tgt_c2w_aligned = tgt_c2w
            gt_images = (batch["rgb_output"][0, :len(target_indices)] + 1) / 2
            visibility_data = batch.get("visibility")
            visibility = (
                None if visibility_data is None
                else visibility_data[0, :len(target_indices), None]
            )
            tgt_c2w_corrected, camera_alignment = optimize_neoverse_shared_camera(
                recon=recon, enc=enc, gt_images=gt_images, visibility=visibility,
                c2w_init=tgt_c2w_aligned, K=tgt_K, timestamps=target_indices,
                apply_se3_delta=_apply_se3_delta, source_height=height,
                source_width=width, num_steps=args.neoverse_shared_camera_tto_steps,
                num_sample_frames=args.neoverse_shared_camera_tto_frames,
                optimize_size=args.neoverse_shared_camera_tto_size,
            )
            pred_rgb, _, _ = recon.render(
                enc, c2w_target=tgt_c2w_corrected, K_target=tgt_K,
                ts_target=target_indices, height=height, width=width,
            )
            metrics = save_eval_artifacts(
                model_name="NeoVerse", scene_dir=scene_dir,
                predictions=pred_rgb.permute(0, 3, 1, 2),
                ground_truth=gt_images, visibility=visibility,
                timestamp_indices=[int(value) for value in target_indices],
            )
        metadata = {
            "model": "NeoVerse", "dataset": "iphone", "scene": scene,
            "source_sample": f"iphone_{scene}", "num_frames": len(frames),
            "timestamps": [f["timestamp_index"] for f in frames],
            "num_gaussians_per_frame": [f["num_gaussians"] for f in frames],
            "checkpoint": str(args.checkpoint), "checkpoint_sha256": args.checkpoint_sha256,
            "official_settings": settings,
            "coordinate_system": "NeoVerse predicted canonical world coordinates; identical across timestamps",
            "row_ordering": "Renderer Gaussian-list order, then original row order; never shuffled",
            "renderer_filter": {
                "opacity_threshold": rasterizer.opacity_prune_threshold,
                "confidence_threshold": rasterizer.confidence_prune_threshold,
                "bidirection": rasterizer.bidirection,
            },
            "spatial_decimation": dict(decimation),
            "metric_gaussians": "uncompressed official renderer prediction (before PLY decimation)",
            "camera_alignment": camera_alignment,
            "metrics": metrics,
            "ply_attribute_convention": "INRIA 3DGS: native SH, logit opacity, log scale, wxyz quaternion",
            "frames": frames, "elapsed_seconds": time.time() - start,
        }
        save_metadata(scene_dir, metadata)
        del enc, enc_export, batch, gs_list
        torch.cuda.empty_cache()


@torch.inference_mode()
def export_movies(args: argparse.Namespace) -> None:
    project = args.baseline_root / "MoVieS"
    # MoVieS validates its official './resources' path at import time.
    os.chdir(project)
    add_paths(project, args.deps)
    from safetensors.torch import load_file
    from src.eval import inverse_c2w, load_images_from_paths, load_iphone_scenes
    from src.models import SplatRecon
    from src.models.gs_render.gs_util import render as render_gaussians
    from src.options import opt_dict

    opt = opt_dict["movies"]
    model = SplatRecon(opt, load_lpips=False)
    model.load_state_dict(load_file(str(args.checkpoint)), strict=True)
    model.eval().cuda()
    settings = {
        "image_res": 224, "num_context_frames": 32, "max_temporal": 64,
        "context_camera": 0, "target_camera": 1,
        "camera_normalization": (
            "canonical first context only; no baseline scale normalization because "
            "the iPhone cameras are static and temporal COLMAP jitter is not a baseline"
        ),
        "frames_chunk_size": 16, "source": "MoVieS/scripts/eval_iphone.sh",
        "time_policy": args.movies_time_policy,
    }
    for scene in args.scenes:
        start = time.time()
        scenes = load_iphone_scenes(
            str(args.data_root), max_temporal=64, num_context_frames=32,
            num_target_frames=None, scenes=[scene],
        )
        if len(scenes) != 1:
            raise RuntimeError(f"Could not load iPhone scene {scene}")
        data = scenes[0]
        ctx_idx, tgt_idx = data["context_indices"], data["target_indices"]
        ctx_time_indices = list(range(0, 2 * len(ctx_idx), 2))
        tgt_time_indices = list(range(len(tgt_idx)))
        if args.movies_time_policy == "official_flat":
            ctx_conditioning_indices = list(ctx_idx)
            tgt_conditioning_indices = list(tgt_idx)
        else:
            ctx_conditioning_indices = ctx_time_indices
            tgt_conditioning_indices = tgt_time_indices
        ctx_images = load_images_from_paths([data["image_paths"][i] for i in ctx_idx], size=(224, 224))
        c2w = data["C2W"].clone()
        intr = data["fxfycxcy"].clone()
        transform = inverse_c2w(c2w[ctx_idx[0]])
        c2w = transform.unsqueeze(0) @ c2w
        ctx_c2w, tgt_c2w = c2w[ctx_idx], c2w[tgt_idx]
        ctx_intr, tgt_intr = intr[ctx_idx], intr[tgt_idx]
        duration = float(max(ctx_conditioning_indices + tgt_conditioning_indices))
        ctx_ts = torch.tensor([t / duration for t in ctx_conditioning_indices])
        tgt_ts = torch.tensor([t / duration for t in tgt_conditioning_indices])
        if args.limit_frames:
            tgt_idx = tgt_idx[: args.limit_frames]
            tgt_time_indices = tgt_time_indices[: args.limit_frames]
            tgt_c2w, tgt_intr, tgt_ts = tgt_c2w[: args.limit_frames], tgt_intr[: args.limit_frames], tgt_ts[: args.limit_frames]
        gt_images = None
        visibility = None
        if args.save_eval_artifacts:
            gt_images = load_images_from_paths(
                [data["image_paths"][i] for i in tgt_idx], size=(224, 224)
            )
            visibility = load_visibility_masks(
                data["covisible_mask_paths"][:len(tgt_idx)], size=(224, 224)
            )

        input_images = ctx_images[None].cuda().to(torch.bfloat16)
        input_c2w = ctx_c2w[None].cuda().to(torch.bfloat16)
        input_intr = ctx_intr[None].cuda().to(torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs, pred_motions, pred_motion_gs = model.backbone(
                input_images, input_c2w, input_intr,
                ctx_ts[None].cuda().to(torch.bfloat16),
                tgt_ts[None].cuda().to(torch.bfloat16), frames_chunk_size=16,
            )

        decimation = decimation_metadata(
            args.spatial_block_size,
            "source-frame image-grid top-1 by maximum predicted opacity over all target timestamps",
        )
        spatial_keep_mask = None
        if args.spatial_block_size > 1:
            # Select once in the source image grid. Reuse this exact mask for
            # every target-conditioned motion/attribute prediction so rows do
            # not pop in and out merely because the decimator changed its mind.
            importance = outputs["opacity"][:, :, 0].float().clone()
            if pred_motion_gs is not None:
                for target_attrs in pred_motion_gs:
                    if "motion_opacity" in target_attrs:
                        importance = torch.maximum(
                            importance,
                            target_attrs["motion_opacity"][:, :, 0].float(),
                        )
            spatial_keep_mask = image_block_top1_mask(
                importance, args.spatial_block_size
            )
            decimation["input_gaussians"] = int(spatial_keep_mask.numel())
            decimation["selected_gaussians"] = int(spatial_keep_mask.sum())
            decimation["compensation"] = {
                "opacity": "1 - (1 - alpha) ** (block_size ** 2)",
                "scale": "scale * block_size",
            }

        scene_dir, seq_dir = make_scene_dir(args.output_root, "MoVieS", scene)
        frames = []
        predictions = []
        for i, timestamp_index in enumerate(tgt_time_indices):
            if pred_motions is not None:
                outputs["offset"] = pred_motions[:, i, :, :3]
            if pred_motion_gs is not None:
                outputs.update(pred_motion_gs[i])
            pc_full = movies_pc_from_outputs(
                model.gs_renderer, outputs, input_c2w, input_intr, None, 1,
            )
            if args.save_eval_artifacts:
                rendered = render_gaussians(
                    pc_full, 224, 224, tgt_c2w[i].cuda().float(), tgt_intr[i].cuda().float(),
                    bg_color=torch.zeros(3, device=pc_full.xyz.device),
                )
                predictions.append(rendered["image"][0].cpu())
            pc = pc_full if spatial_keep_mask is None else movies_pc_from_outputs(
                model.gs_renderer, outputs, input_c2w, input_intr,
                spatial_keep_mask, args.spatial_block_size,
            )
            path = seq_dir / f"frame_{i:04d}.ply"
            info = write_standard_3dgs_ply(
                path, means=pc.xyz, harmonics=pc.color.reshape(len(pc.xyz), -1, 3),
                opacities=pc.opacity, scales=pc.scale, rotations_wxyz=pc.rotation,
                harmonics_are_rgb_parameters=True,
                comment="MoVieS renderer Gaussians in standard INRIA 3DGS convention",
            )
            info["timestamp_index"] = int(timestamp_index)
            info["model_conditioning_index"] = int(tgt_conditioning_indices[i])
            info["normalized_timestamp"] = float(tgt_ts[i])
            frames.append(info)
            print(f"[MoVieS/{scene}] {i+1}/{len(tgt_time_indices)} N={info['num_gaussians']:,}", flush=True)
        metrics = None
        if args.save_eval_artifacts:
            metrics = save_eval_artifacts(
                model_name="MoVieS", scene_dir=scene_dir,
                predictions=torch.stack(predictions),
                ground_truth=gt_images,
                visibility=visibility,
                timestamp_indices=tgt_time_indices,
            )
        metadata = {
            "model": "MoVieS", "dataset": "iphone", "scene": scene,
            "source_sample": f"iphone_{scene}", "num_frames": len(frames),
            "timestamps": [f["timestamp_index"] for f in frames],
            "num_gaussians_per_frame": [f["num_gaussians"] for f in frames],
            "checkpoint": str(args.checkpoint), "checkpoint_sha256": args.checkpoint_sha256,
            "official_settings": settings,
            "coordinate_system": "MoVieS first-context canonical world coordinates; identical across timestamps",
            "row_ordering": "Original view/pixel Gaussian order retained after official opacity filter",
            "renderer_filter": f"opacity > {opt.opacity_threshold} because prune_ratio={opt.prune_ratio} at eval",
            "spatial_decimation": decimation,
            "metric_gaussians": "uncompressed official renderer prediction (before PLY decimation)",
            "metrics": metrics,
            "ply_attribute_convention": "INRIA 3DGS: RGB parameters converted to SH, logit opacity, log scale, wxyz quaternion",
            "frames": frames, "elapsed_seconds": time.time() - start,
        }
        save_metadata(scene_dir, metadata)
        del outputs, pred_motions, pred_motion_gs, input_images, data
        torch.cuda.empty_cache()


def write_bundle_files(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    launcher = output_root / "Open_in_SuperSplat.command"
    launcher.write_text(
        "#!/bin/bash\nset -euo pipefail\n\n"
        "SCRIPT_DIR=\"$(cd -- \"$(dirname -- \"$0\")\" && pwd)\"\n"
        "SUPERSPLAT_URL=\"https://superspl.at/editor\"\n\n"
        "open \"$SCRIPT_DIR\"\nopen \"$SUPERSPLAT_URL\"\n\n"
        "osascript <<'APPLESCRIPT'\n"
        "display dialog \"SuperSplat을 열었습니다. 모델/scene 아래의 gaussian_sequence 폴더 하나를 Finder에서 브라우저로 drag & drop하세요. frame_0000.ply부터 lexical/temporal 순서로 Timeline에 로드됩니다.\" buttons {\"확인\"} default button \"확인\" with title \"Baseline Gaussian Sequences\"\n"
        "APPLESCRIPT\n",
        encoding="utf-8",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (output_root / "README.md").write_text(
        "# Baseline SuperSplat Gaussian sequences\n\n"
        "Mac에서 `Open_in_SuperSplat.command`를 실행한 뒤, 원하는 scene의 "
        "`gaussian_sequence` 폴더를 SuperSplat으로 drag & drop하세요. "
        "sequence 폴더 안에는 정렬된 `frame_XXXX.ply`만 있습니다. "
        "각 scene에는 pred_images, GT/PRED/DIFF compare, PSNR 파일도 있습니다. "
        "설정과 검증 통계는 `metadata.json`에 있습니다.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["4dgt", "movies", "neoverse"], required=True)
    parser.add_argument("--baseline-root", type=Path, default=Path("/music-3d-shared-disk/user/KAIST/MG/HG/4DGS_baseline"))
    parser.add_argument("--data-root", type=Path, default=Path("/music-3d-shared-disk/dataset/iphone/original_data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/baseline_supersplat_sequences"))
    parser.add_argument("--deps", type=Path, default=Path("/tmp/c4g_baseline_deps"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--limit-frames", type=int, default=None, help="Smoke-test only")
    parser.add_argument(
        "--spatial-block-size", type=int, choices=[1, 2, 3, 4], default=1,
        help=(
            "Lossy source/image-grid decimation: keep one original Gaussian "
            "per NxN block with a sequence-stable mask (default: disabled)"
        ),
    )
    parser.add_argument(
        "--neoverse-shared-camera-tto-steps", type=int, default=300,
        help="Optimisation steps for one sequence-shared static-camera SE(3) correction",
    )
    parser.add_argument(
        "--neoverse-shared-camera-tto-frames", type=int, default=8,
        help="Evenly spaced sequence frames used to fit the shared camera correction",
    )
    parser.add_argument(
        "--neoverse-shared-camera-tto-size", type=int, default=504,
        help="Square fitting resolution for the shared camera correction",
    )
    parser.add_argument(
        "--movies-time-policy", choices=["official_flat", "physical"],
        default="physical",
        help="MoVieS conditioning timestamps; physical matches the unified temporal setup",
    )
    parser.add_argument(
        "--save-eval-artifacts", action="store_true",
        help="Save target-view predictions, GT/PRED/DIFF panels and metric files",
    )
    args = parser.parse_args()
    unknown = sorted(set(args.scenes) - set(SCENES))
    if unknown:
        parser.error(f"Unsupported iPhone scenes: {unknown}")
    args.checkpoint = args.checkpoint.resolve()
    args.output_root = args.output_root.resolve()
    args.checkpoint_sha256 = sha256(args.checkpoint)
    return args


def main() -> None:
    args = parse_args()
    write_bundle_files(args.output_root)
    {"4dgt": export_4dgt, "movies": export_movies, "neoverse": export_neoverse}[args.model](args)


if __name__ == "__main__":
    main()
