#!/usr/bin/env python3
"""Build per-scene iPhone GT/Pred/Diff videos and metric tables.

The input is the official evaluator's ``iphone_*_tgt_*.png`` comparison
output. Metrics are recomputed from the stored 8-bit GT/prediction panels
using the evaluator's own PSNR, SSIM, and LPIPS implementations, so two
saved evaluations can be compared consistently even when their console logs
are unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from make_eval_triptych_videos import (
    VIDEO_HEADER_HEIGHT,
    split_official_comparison,
    target_psnr,
    write_video,
)


PANEL_GAP = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="CONDITION=DIR",
        help="Official comparison output directory; may be repeated.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument(
        "--mask-config",
        type=Path,
        help="Hydra eval config used to reconstruct official iPhone visibility masks.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Compute metrics and manifests without writing MP4 files.",
    )
    return parser.parse_args()


def parse_sources(values: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected CONDITION=DIR, got: {value}")
        condition, raw_path = value.split("=", 1)
        if not condition or condition in sources:
            raise ValueError(f"Invalid or duplicate condition: {condition}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        sources[condition] = path
    return sources


def find_scene_pngs(source_dir: Path) -> dict[str, Path]:
    by_scene: dict[str, list[Path]] = {}
    for path in source_dir.glob("iphone_*_tgt_*.png"):
        match = re.match(r"^(iphone_.+)_tgt_[0-9.]+\.png$", path.name)
        if match:
            by_scene.setdefault(match.group(1), []).append(path)
    if not by_scene:
        raise FileNotFoundError(f"No iPhone target comparison PNGs in {source_dir}")
    return {
        scene: max(paths, key=lambda item: (item.stat().st_mtime_ns, item.name))
        for scene, paths in sorted(by_scene.items())
    }


def panel_tensors(frames: list[np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    array = np.stack(frames)
    panel_size = array.shape[1] - VIDEO_HEADER_HEIGHT
    gt = array[:, VIDEO_HEADER_HEIGHT:, :panel_size]
    pred_x = panel_size + PANEL_GAP
    pred = array[:, VIDEO_HEADER_HEIGHT:, pred_x : pred_x + panel_size]
    gt_tensor = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2).float().div_(255).to(device)
    pred_tensor = torch.from_numpy(pred.copy()).permute(0, 3, 1, 2).float().div_(255).to(device)
    return gt_tensor, pred_tensor


def compute_metrics(
    gt: torch.Tensor,
    pred: torch.Tensor,
    compute_psnr,
    compute_ssim,
    compute_lpips,
    mask: torch.Tensor | None = None,
    compute_psnr_masked=None,
    compute_ssim_masked=None,
    compute_lpips_masked=None,
    batch_size: int = 8,
) -> dict[str, float]:
    psnr_values = []
    ssim_values = []
    lpips_values = []
    for start in range(0, len(gt), batch_size):
        end = min(start + batch_size, len(gt))
        gt_chunk = gt[start:end]
        pred_chunk = pred[start:end]
        if mask is None:
            psnr_values.append(compute_psnr(gt_chunk, pred_chunk).cpu())
            ssim_values.append(compute_ssim(gt_chunk, pred_chunk).cpu())
            lpips_values.append(compute_lpips(gt_chunk, pred_chunk).cpu())
        else:
            mask_chunk = mask[start:end]
            psnr_values.append(compute_psnr_masked(gt_chunk, pred_chunk, mask_chunk).cpu())
            ssim_values.append(compute_ssim_masked(gt_chunk, pred_chunk, mask_chunk).cpu())
            lpips_values.append(compute_lpips_masked(gt_chunk, pred_chunk, mask_chunk).cpu())
    return {
        "psnr": float(torch.cat(psnr_values).mean()),
        "ssim": float(torch.cat(ssim_values).mean()),
        "lpips": float(torch.cat(lpips_values).mean()),
    }


def build_visibility_masks(config_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    from omegaconf import OmegaConf
    from src.config import load_typed_root_config
    from src.dataset import get_dataset

    root_cfg = load_typed_root_config(OmegaConf.load(config_path))
    datasets = get_dataset(root_cfg.dataset, "test", None)
    if len(datasets) != 1 or datasets[0].cfg.name != "iphone":
        raise ValueError(f"Expected one iPhone dataset in {config_path}")
    dataset = datasets[0]
    output_shape = tuple(dataset.cfg.input_image_shape)
    masks: dict[str, torch.Tensor] = {}
    for scene_id, example in dataset.scenes.items():
        num_frames = len(example[0])
        max_t = min(num_frames - 1, dataset.cfg.max_temporal - 1)
        target_indices = range(max_t + 1)
        target_cameras = dataset.cfg.target_cameras or list(range(1, len(example)))
        frame_masks = []
        for camera in target_cameras:
            for frame_index in target_indices:
                frame = example[camera][frame_index]
                covisible_path = frame["file_path"].replace("rgb/1x", "covisible/2x/val")
                if Path(covisible_path).exists():
                    array = np.asarray(Image.open(covisible_path).convert("L"), dtype=np.float32) / 255
                else:
                    rgb_shape = np.asarray(Image.open(frame["file_path"])).shape[:2]
                    array = np.ones(rgb_shape, dtype=np.float32)
                frame_masks.append(torch.from_numpy(array).unsqueeze(0))
        mask = torch.stack(frame_masks)
        if mask.shape[-2:] != output_shape:
            mask = F.interpolate(mask, size=output_shape, mode="nearest")
        masks[f"iphone_{scene_id}"] = (mask > 0.5).float().to(device)
    return masks


def write_rows(rows: list[dict[str, object]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "metrics_manifest.json"
    csv_path = output_root / "metrics_manifest.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_comparison(rows: list[dict[str, object]], output_root: Path) -> None:
    conditions = sorted({str(row["condition"]) for row in rows})
    if len(conditions) != 2:
        return
    first, second = conditions
    indexed = {(str(row["condition"]), str(row["scene"])): row for row in rows}
    scenes = sorted(set(str(row["scene"]) for row in rows))
    comparison = []
    for scene in scenes:
        if (first, scene) not in indexed or (second, scene) not in indexed:
            continue
        row_a = indexed[(first, scene)]
        row_b = indexed[(second, scene)]
        comparison.append(
            {
                "scene": scene,
                "condition_a": first,
                "condition_b": second,
                "psnr_a": row_a["psnr_recomputed"],
                "psnr_b": row_b["psnr_recomputed"],
                "delta_psnr_b_minus_a": float(row_b["psnr_recomputed"]) - float(row_a["psnr_recomputed"]),
                "ssim_a": row_a["ssim_recomputed"],
                "ssim_b": row_b["ssim_recomputed"],
                "delta_ssim_b_minus_a": float(row_b["ssim_recomputed"]) - float(row_a["ssim_recomputed"]),
                "lpips_a": row_a["lpips_recomputed"],
                "lpips_b": row_b["lpips_recomputed"],
                "delta_lpips_b_minus_a": float(row_b["lpips_recomputed"]) - float(row_a["lpips_recomputed"]),
            }
        )
    if not comparison:
        return

    averages = {"scene": "AVERAGE", "condition_a": first, "condition_b": second}
    for key in comparison[0]:
        if key not in {"scene", "condition_a", "condition_b"}:
            averages[key] = float(np.mean([float(row[key]) for row in comparison]))
    comparison.append(averages)
    with (output_root / "force_intrinsics_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    (output_root / "force_intrinsics_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()
    sources = parse_sources(args.source)
    sys.path.insert(0, str(args.eval_root.resolve()))
    from src.evaluation.metrics import (
        compute_lpips,
        compute_lpips_masked,
        compute_psnr,
        compute_psnr_masked,
        compute_ssim,
        compute_ssim_masked,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visibility_masks = (
        build_visibility_masks(args.mask_config.resolve(), device)
        if args.mask_config is not None
        else {}
    )
    rows: list[dict[str, object]] = []
    total = sum(len(find_scene_pngs(path)) for path in sources.values())
    index = 0
    for condition, source_dir in sources.items():
        for scene, source_png in find_scene_pngs(source_dir).items():
            index += 1
            frames, metadata = split_official_comparison(source_png)
            video_path = args.output_root / condition / f"{scene}.mp4"
            if not args.skip_videos and (args.overwrite or not video_path.exists()):
                write_video(frames, video_path, args.fps)
            gt, pred = panel_tensors(frames, device)
            mask = visibility_masks.get(scene)
            if mask is not None and len(mask) != len(gt):
                raise ValueError(
                    f"Visibility frame count mismatch for {scene}: {len(mask)} != {len(gt)}"
                )
            metrics = compute_metrics(
                gt,
                pred,
                compute_psnr,
                compute_ssim,
                compute_lpips,
                mask=mask,
                compute_psnr_masked=compute_psnr_masked,
                compute_ssim_masked=compute_ssim_masked,
                compute_lpips_masked=compute_lpips_masked,
            )
            row: dict[str, object] = {
                "condition": condition,
                "scene": scene,
                "source_png": str(source_png),
                "output_video": None if args.skip_videos else str(video_path.resolve()),
                "frame_count": metadata["frame_count"],
                "fps": args.fps,
                "psnr_from_filename": target_psnr(source_png),
                "psnr_recomputed": metrics["psnr"],
                "ssim_recomputed": metrics["ssim"],
                "lpips_recomputed": metrics["lpips"],
                "metric_mode": "visibility_masked" if mask is not None else "unmasked",
                "diff_validation_max": metadata["diff_validation_max"],
            }
            rows.append(row)
            print(
                f"[{index:02d}/{total:02d}] {condition}/{scene}: "
                f"PSNR={metrics['psnr']:.4f} SSIM={metrics['ssim']:.4f} "
                f"LPIPS={metrics['lpips']:.4f}"
            )
    write_rows(rows, args.output_root)
    write_comparison(rows, args.output_root)


if __name__ == "__main__":
    main()
