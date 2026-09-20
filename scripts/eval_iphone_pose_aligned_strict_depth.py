#!/usr/bin/env python3
"""Compare exported C4G sequences with the training-time strict depth objective.

The C4G predictions come from the official 32-context iPhone inference.  VGGT
pseudo-labels use 12 context and 11 target views sampled uniformly in time,
matching the multi-view count used by the strict-depth training batches.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import utils3d
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-file", type=Path, default=None)
    parser.add_argument("--vggt-weights", type=Path, required=True)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/music-3d-shared-disk/user/KAIST/HG/c4g_mg"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stage",
        choices=("val", "test"),
        default="test",
        help="Dataset split to evaluate (default: test, preserving previous behavior).",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Evaluate at most this many selected scenes.",
    )
    parser.add_argument("--num-context", type=int, default=12)
    parser.add_argument("--num-target", type=int, default=11)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--dataset-label", default="iphone")
    parser.add_argument("--new-label", default="new_pose_strict_depth_from0")
    parser.add_argument("--old-label", default="previous_v4_3_temb")
    parser.add_argument(
        "--skip-invalid-alignment",
        action="store_true",
        help="Record and skip scenes whose VGGT/dataset pose-scale alignment is invalid.",
    )
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def uniform_positions(length: int, count: int) -> Tensor:
    if count <= 0 or length <= 0:
        raise ValueError(f"Invalid sampling request length={length}, count={count}")
    if count >= length:
        return torch.arange(length)
    positions = torch.linspace(0, length - 1, count).long()
    if len(torch.unique(positions)) != count:
        raise ValueError(f"Uniform sampling produced duplicate positions: {positions}")
    return positions


def load_ply_gaussians(
    sequence_dir: Path, timestamp: int, serializer, Gaussians, device
):
    metadata_path = sequence_dir / "metadata.json"
    if not metadata_path.is_file():
        metadata_path = sequence_dir.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    by_timestamp = dict(zip(metadata["timestamps"], metadata["frame_filenames"]))
    if timestamp not in by_timestamp:
        raise KeyError(f"{sequence_dir}: timestamp {timestamp} is unavailable")
    _, vertices = serializer.read_3dgs_ply(sequence_dir / by_timestamp[timestamp])

    def stack(names: list[str]) -> Tensor:
        return torch.from_numpy(
            np.column_stack([vertices[name] for name in names]).copy()
        ).float()

    means = stack(["x", "y", "z"])
    opacities = stack(["opacity"]).sigmoid().squeeze(-1)
    scales = stack([f"scale_{index}" for index in range(3)]).exp()
    rotations_wxyz = stack([f"rot_{index}" for index in range(4)])
    rotations_xyzw = rotations_wxyz[:, [1, 2, 3, 0]]
    dc = stack([f"f_dc_{index}" for index in range(3)]).unsqueeze(-1)
    rest_names = sorted(
        (name for name in vertices.dtype.names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if rest_names:
        rest = stack(rest_names).reshape(len(vertices), 3, -1)
        harmonics = torch.cat([dc, rest], dim=-1)
    else:
        harmonics = dc
    covariances = serializer._build_covariance_xyzw(scales, rotations_xyzw)
    gaussian_kwargs = {
        "means": means[None].to(device),
        "covariances": covariances[None].to(device),
        "harmonics": harmonics[None].to(device),
        "opacities": opacities[None].to(device),
        "scales": scales[None].to(device),
        "rotations": rotations_xyzw[None].to(device),
    }
    if "feature" in inspect.signature(Gaussians).parameters:
        gaussian_kwargs["feature"] = None
    return Gaussians(**gaussian_kwargs)


def depth_to_normal(depth_map: Tensor, intrinsics: Tensor, depthmap_to_pts3d) -> Tensor:
    batch, height, width = depth_map.shape
    intrinsics_px = intrinsics.clone()
    intrinsics_px[:, 0] *= width
    intrinsics_px[:, 1] *= height
    focal = torch.stack((intrinsics_px[:, 0, 0], intrinsics_px[:, 1, 1]), dim=-1)[
        :, :, None, None
    ].expand(batch, 2, height, width)
    points = depthmap_to_pts3d(depth_map[..., None], focal)
    return utils3d.pt.point_map_to_normal_map(points.squeeze(-1))


def per_timestamp_metrics(
    *,
    sequence_dir: Path,
    labels,
    context: dict[str, Tensor],
    target: dict[str, Tensor],
    decoder,
    serializer,
    Gaussians,
    StrictDepthLoss,
    normal_map_loss,
    depthmap_to_pts3d,
    device: torch.device,
) -> dict[str, Any]:
    strict_loss = StrictDepthLoss("log_l1")
    timestamps = (
        torch.unique(torch.cat([context["index"], target["index"]]))
        .sort()
        .values.tolist()
    )
    num_context = len(context["index"])
    rows = []
    weighted_log_sum = 0.0
    weighted_rel_sum = 0.0
    valid_pixel_count = 0

    for timestamp in timestamps:
        context_mask = context["index"] == timestamp
        target_mask = target["index"] == timestamp
        view_extrinsics = torch.cat(
            [context["extrinsics"][context_mask], target["extrinsics"][target_mask]], 0
        ).to(device)
        view_intrinsics = torch.cat(
            [context["intrinsics"][context_mask], target["intrinsics"][target_mask]], 0
        ).to(device)
        view_near = torch.cat(
            [context["near"][context_mask], target["near"][target_mask]], 0
        ).to(device)
        view_far = torch.cat(
            [context["far"][context_mask], target["far"][target_mask]], 0
        ).to(device)
        label_indices = torch.cat(
            [
                torch.arange(num_context)[context_mask],
                num_context + torch.arange(len(target["index"]))[target_mask],
            ]
        )
        label_depth = labels.depth[0, label_indices].to(device)
        label_mask = labels.mask[0, label_indices].to(device)

        gaussians = load_ply_gaussians(
            sequence_dir, int(timestamp), serializer, Gaussians, device
        )
        with torch.inference_mode():
            output = decoder.forward(
                gaussians,
                view_extrinsics[None],
                view_intrinsics[None],
                view_near[None],
                view_far[None],
                tuple(label_depth.shape[-2:]),
                global_step=20000,
            )
            prediction = output.depth[0].float()
            loss, diagnostics = strict_loss(prediction, label_depth, mask=label_mask)
            valid = (
                torch.isfinite(prediction)
                & torch.isfinite(label_depth)
                & (label_depth > 0)
                & label_mask
            )
            count = int(valid.sum())
            if count == 0:
                del gaussians, output, prediction
                continue
            log_error = (
                (prediction[valid] + 1e-6).clamp_min(1e-6).log()
                - (label_depth[valid] + 1e-6).log()
            ).abs()
            relative_error = (
                prediction[valid] - label_depth[valid]
            ).abs() / label_depth[valid].clamp_min(1e-6)
            pred_normal = depth_to_normal(
                prediction, view_intrinsics, depthmap_to_pts3d
            )
            label_normal = depth_to_normal(
                label_depth, view_intrinsics, depthmap_to_pts3d
            )
            normal_loss, normal_diag = normal_map_loss(
                pred_normal, label_normal, mask=label_mask
            )
        weighted_log_sum += float(log_error.sum())
        weighted_rel_sum += float(relative_error.sum())
        valid_pixel_count += count
        rows.append(
            {
                "timestamp": int(timestamp),
                "num_views": int(len(view_near)),
                "valid_pixels": count,
                "valid_fraction": float(diagnostics["valid_fraction"]),
                "log_l1": float(loss),
                "relative_l1": float(relative_error.mean()),
                "normal_loss_rad2": float(normal_loss),
                "normal_valid_fraction": float(normal_diag["valid_fraction"]),
                "median_prediction_to_label_ratio": float(
                    diagnostics["median_prediction_ratio"]
                ),
            }
        )
        del gaussians, output, prediction, pred_normal, label_normal

    if not rows:
        return None
    return {
        "num_timestamps": len(rows),
        "num_rendered_views": sum(row["num_views"] for row in rows),
        "valid_pixels": valid_pixel_count,
        "timestamp_macro_log_l1": float(np.mean([row["log_l1"] for row in rows])),
        "pixel_weighted_log_l1": weighted_log_sum / max(valid_pixel_count, 1),
        "timestamp_macro_relative_l1": float(
            np.mean([row["relative_l1"] for row in rows])
        ),
        "pixel_weighted_relative_l1": weighted_rel_sum / max(valid_pixel_count, 1),
        "timestamp_macro_normal_loss_rad2": float(
            np.mean([row["normal_loss_rad2"] for row in rows])
        ),
        "median_prediction_to_label_ratio": float(
            np.median([row["median_prediction_to_label_ratio"] for row in rows])
        ),
        "per_timestamp": rows,
    }


def main() -> None:
    args = parse_args()
    if args.new_label == args.old_label:
        raise ValueError("--new-label and --old-label must be distinct")
    for name in ("config", "new_root", "old_root", "vggt_weights", "eval_root"):
        setattr(args, name, getattr(args, name).resolve())
    args.output = args.output.resolve()
    if args.batch_file is not None:
        args.batch_file = args.batch_file.resolve()
        if not args.batch_file.is_file():
            raise FileNotFoundError(args.batch_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    depth_module = load_module(
        "_strict_depth_supervision", ROOT / "src/loss/depth_supervision.py"
    )
    loss_ss_module = load_module("_strict_depth_loss_ss", ROOT / "src/loss/loss_ss.py")
    serializer = load_module(
        "_strict_depth_gaussian_sequence", ROOT / "src/misc/gaussian_sequence.py"
    )

    os.chdir(args.eval_root)
    sys.path.insert(0, str(args.eval_root))
    from src.config import load_typed_root_config
    from src.dataset import get_dataset
    from src.global_cfg import set_cfg
    from src.geometry.ptc_geometry import depthmap_to_pts3d
    from src.model.decoder import get_decoder
    from src.model.encoder.backbone.vggt.vggt import VGGT
    from src.model.types import Gaussians

    cfg_dict = OmegaConf.load(args.config)
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    seed = int(cfg_dict.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Loading VGGT weights: {args.vggt_weights}", flush=True)
    vggt = VGGT()
    try:
        state = torch.load(
            str(args.vggt_weights), map_location="cpu", mmap=True, weights_only=True
        )
    except TypeError:
        state = torch.load(str(args.vggt_weights), map_location="cpu")
    vggt.load_state_dict(state)
    del state
    vggt.point_head = None
    vggt.track_head = None
    vggt = vggt.to(device).eval()
    for parameter in vggt.parameters():
        parameter.requires_grad_(False)

    decoder = get_decoder(cfg.model.decoder).to(device).eval()
    if args.batch_file is not None:
        print(
            f"Loading exact exported evaluation batch: {args.batch_file}",
            flush=True,
        )
        loader = [torch.load(str(args.batch_file), map_location="cpu")]
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        datasets = get_dataset(cfg.dataset, args.stage, None)
        matching_datasets = [
            dataset for dataset in datasets if dataset.cfg.name == args.dataset_label
        ]
        if len(matching_datasets) != 1:
            available = [dataset.cfg.name for dataset in datasets]
            raise ValueError(
                f"Expected exactly one dataset named {args.dataset_label!r}; "
                f"available datasets: {available}"
            )
        dataset = matching_datasets[0]
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    requested = set(args.scene)
    results: dict[str, Any] = {
        "protocol": {
            "dataset_label": args.dataset_label,
            "dataset_stage": args.stage,
            "prediction_input": (
                f"official {args.dataset_label} evaluation-input exported C4G Gaussians"
            ),
            "pseudo_label_views": f"{args.num_context} context + {args.num_target} target, uniformly sampled",
            "depth_target": "VGGT depth scaled once from VGGT/dataset camera-center baselines",
            "confidence_quantile": 0.2,
            "pose_min_baseline": 1e-4,
            "pose_min_pair_fraction": 0.2,
            "pose_max_relative_error": 0.3,
            "primary_metric": "StrictDepthLoss(log_l1), no fitted depth scale or shift",
            "renderer_scale_invariant": bool(decoder.make_scale_invariant),
            "config": str(args.config),
            "batch_file": str(args.batch_file) if args.batch_file is not None else None,
            "vggt_weights": str(args.vggt_weights),
            "invalid_alignment_policy": (
                "skip and record" if args.skip_invalid_alignment else "raise"
            ),
        },
        "models": {
            args.new_label: {
                "ply_root": str(args.new_root),
                "scenes": {},
                "skipped_scenes": {},
            },
            args.old_label: {
                "ply_root": str(args.old_root),
                "scenes": {},
                "skipped_scenes": {},
            },
        },
        "label_alignment": {},
    }

    selected_scene_count = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    for raw_batch in loader:
        scene = str(raw_batch["scene"][0])
        if requested and scene not in requested:
            continue
        if args.max_scenes is not None and selected_scene_count >= args.max_scenes:
            break
        selected_scene_count += 1
        context_pos = uniform_positions(
            raw_batch["context"]["image"].shape[1], args.num_context
        )
        target_pos = uniform_positions(
            raw_batch["target"]["image"].shape[1], args.num_target
        )

        def selected(group: str, positions: Tensor) -> dict[str, Tensor]:
            view_keys = (
                "image",
                "extrinsics",
                "intrinsics",
                "near",
                "far",
                "index",
                "camera",
            )
            return {
                key: value[0, positions].cpu()
                for key in view_keys
                if isinstance((value := raw_batch[group].get(key)), Tensor)
            }

        context = selected("context", context_pos)
        target = selected("target", target_pos)
        images = torch.cat([context["image"], target["image"]], 0)[None].to(device)
        dataset_c2w = torch.cat([context["extrinsics"], target["extrinsics"]], 0)[
            None
        ].to(device)
        near = torch.cat([context["near"], target["near"]], 0)[None].to(device)
        far = torch.cat([context["far"], target["far"]], 0)[None].to(device)

        print(
            f"[{scene}] VGGT labels from {len(context_pos)}+{len(target_pos)} views",
            flush=True,
        )
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = vggt(images.clamp(0, 1))
        labels = depth_module.build_pose_aligned_depth_labels(
            vggt_depth=prediction["depth"].float(),
            vggt_depth_confidence=prediction["depth_conf"].float(),
            vggt_pose_encoding=prediction["pose_enc"].float(),
            dataset_c2w=dataset_c2w,
            near=near,
            far=far,
            renderer_scale_invariant=bool(decoder.make_scale_invariant),
            confidence_quantile=0.2,
            min_baseline=1e-4,
            min_pair_fraction=0.2,
            max_pose_relative_error=0.3,
        )
        results["label_alignment"][scene] = {
            "scale": float(labels.scale[0]),
            "pose_relative_error": float(labels.pose_relative_error[0]),
            "alignment_valid": bool(labels.alignment_valid[0]),
            "valid_pixel_fraction": float(labels.mask.float().mean()),
            "selected_context_timestamps": context["index"].tolist(),
            "selected_target_timestamps": target["index"].tolist(),
            "vggt_peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        if not bool(labels.alignment_valid[0]):
            if args.skip_invalid_alignment:
                print(
                    f"[{scene}] SKIP: invalid VGGT/dataset pose alignment", flush=True
                )
                del labels, images, prediction, dataset_c2w, near, far
                gc.collect()
                torch.cuda.empty_cache()
                continue
            raise RuntimeError(f"{scene}: VGGT/dataset pose alignment is invalid")
        del images, prediction, dataset_c2w, near, far
        torch.cuda.empty_cache()

        for model_name, root in (
            (args.new_label, args.new_root),
            (args.old_label, args.old_root),
        ):
            sequence_dir = root / scene / "gaussian_sequence"
            metrics = per_timestamp_metrics(
                sequence_dir=sequence_dir,
                labels=labels,
                context=context,
                target=target,
                decoder=decoder,
                serializer=serializer,
                Gaussians=Gaussians,
                StrictDepthLoss=depth_module.StrictDepthLoss,
                normal_map_loss=loss_ss_module.normal_map_loss,
                depthmap_to_pts3d=depthmap_to_pts3d,
                device=device,
            )
            if metrics is None:
                results["models"][model_name]["skipped_scenes"][
                    scene
                ] = "no valid pseudo-depth pixels"
                print(
                    f"[{scene}/{model_name}] SKIP: no valid pseudo-depth pixels",
                    flush=True,
                )
                continue
            results["models"][model_name]["scenes"][scene] = metrics
            print(
                f"[{scene}/{model_name}] logL1={metrics['timestamp_macro_log_l1']:.6f} "
                f"relL1={metrics['timestamp_macro_relative_l1']:.6f} "
                f"normal={metrics['timestamp_macro_normal_loss_rad2']:.6f}",
                flush=True,
            )
        del labels
        gc.collect()
        torch.cuda.empty_cache()
        args.output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    for model in results["models"].values():
        scenes = list(model["scenes"].values())
        if not scenes:
            continue
        model["macro_average"] = {
            key: float(np.mean([scene[key] for scene in scenes]))
            for key in (
                "timestamp_macro_log_l1",
                "pixel_weighted_log_l1",
                "timestamp_macro_relative_l1",
                "pixel_weighted_relative_l1",
                "timestamp_macro_normal_loss_rad2",
                "median_prediction_to_label_ratio",
            )
        }
    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
