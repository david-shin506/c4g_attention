#!/usr/bin/env python3
"""Foreground-masked cross-view evaluation for the NVIDIA GT scenes.

The input/target protocol matches ``eval_nvidia_singlecam12_crossview.py``:
cam01 at all 12 timestamps is context and cam02--cam12 at those timestamps are
targets. Each target camera's provided foreground mask is repeated across the
12 timestamps and supplied through the official evaluator's ``visibility``
field, activating its masked PSNR, SSIM, and LPIPS implementations.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from PIL import Image

import eval_nvidia_singlecam12_crossview as base


PANEL_SIZE = 224


def official_mask_crop(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L")
    width_in, height_in = image.size
    scale = max(PANEL_SIZE / height_in, PANEL_SIZE / width_in)
    height_scaled = round(height_in * scale)
    width_scaled = round(width_in * scale)
    image = image.resize((width_scaled, height_scaled), Image.Resampling.NEAREST)
    row = (height_scaled - PANEL_SIZE) // 2
    col = (width_scaled - PANEL_SIZE) // 2
    image = image.crop((col, row, col + PANEL_SIZE, row + PANEL_SIZE))
    array = np.asarray(image, dtype=np.uint8).copy()
    if array.shape != (PANEL_SIZE, PANEL_SIZE):
        raise ValueError(f"Unexpected mask shape {array.shape}: {path}")
    return torch.from_numpy(array > 127).float().unsqueeze(0)


class NvidiaSingleCamera12ForegroundMaskedDataset(
    base.NvidiaSingleCamera12CrossviewDataset
):
    def __getitem__(self, item: int) -> dict:
        example = super().__getitem__(item)
        scene = self.scene_names[item]
        mask_root = self.multiview_root / scene / "foreground_mask"
        masks = [
            official_mask_crop(mask_root / f"cam{camera:02d}.png")
            for camera in range(1, 13)
        ]
        target_cameras = example["target"]["camera"].tolist()
        visibility = torch.stack([masks[camera] for camera in target_cameras])
        if not (visibility.sum(dim=(-3, -2, -1)) > 0).all():
            raise ValueError(f"Empty cropped foreground mask in {scene}")
        example["target"]["visibility"] = visibility
        return example


class ForegroundMaskedDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_cfgs,
        data_loader_cfg,
        step_tracker=None,
        dataset_shim=lambda dataset, _: dataset,
        global_rank: int = 0,
    ) -> None:
        super().__init__()
        (field,) = fields(type(dataset_cfgs[0]))
        cfg = getattr(dataset_cfgs[0], field.name)
        options = MASKED_OPTIONS
        self.dataset = NvidiaSingleCamera12ForegroundMaskedDataset(
            cfg,
            options["processed_root"],
            options["multiview_root"],
            options["scenes"],
            options["input_camera"],
        )
        self.dataset_shim = dataset_shim

    def test_dataloader(self):
        dataset = self.dataset_shim(self.dataset, "test")
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )


MASKED_OPTIONS: dict[str, object] = {}


def main() -> None:
    args = base.parse_args()
    eval_root = args.eval_root.resolve()
    config = args.config.resolve()
    processed_root = args.processed_root.resolve()
    multiview_root = args.multiview_root.resolve()
    output_root = args.output_root.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None

    if not (eval_root / "src" / "main.py").is_file():
        raise FileNotFoundError(eval_root / "src" / "main.py")
    if not config.is_file():
        raise FileNotFoundError(config)
    if output_root.exists() and any(output_root.iterdir()) and not args.allow_existing:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. "
            "Pass --allow-existing only to resume a known partial run."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    MASKED_OPTIONS.update(
        processed_root=processed_root,
        multiview_root=multiview_root,
        scenes=args.scenes,
        input_camera=args.input_camera,
    )
    run_spec = {
        "config": str(config),
        "checkpoint_override": str(checkpoint) if checkpoint else None,
        "processed_pose_root": str(processed_root),
        "multiview_root": str(multiview_root),
        "scenes": args.scenes,
        "context_camera_one_based": args.input_camera,
        "context_times_zero_based": list(base.TIMESTAMPS),
        "target_cameras_one_based": [
            camera for camera in range(1, 13) if camera != args.input_camera
        ],
        "target_times_zero_based": list(base.TIMESTAMPS),
        "targets_per_scene": 132,
        "metric_region": "provided per-camera foreground_mask > 127",
        "mask_temporal_policy": "repeat each camera mask at all 12 timestamps",
        "mask_resize": "official crop geometry with nearest-neighbor sampling",
        "scale_normalization": "maximum source-to-full-rig camera baseline",
        "force_davis_intrinsics": True,
        "align_pose": False,
    }
    (output_root / "run_spec.json").write_text(json.dumps(run_spec, indent=2) + "\n")

    os.chdir(eval_root)
    sys.path.insert(0, str(eval_root))
    os.environ["HYDRA_FULL_ERROR"] = "1"

    import src.main as official_main

    official_main.DataModule = ForegroundMaskedDataModule
    hydra_args = [
        f"--config-path={config.parent}",
        f"--config-name={config.stem}",
        "mode=test",
        f"wandb.name={args.run_name}",
        "wandb.mode=disabled",
        "force_davis_intrinsics=true",
        "test.align_pose=false",
        "test.share_target_pose=false",
        "test.use_vggt_target_pose=false",
        "test.compute_scores=true",
        "test.save_image=false",
        "test.save_compare=false",
        "test.save_video=false",
        "test.target_only_eval=true",
        "test.disable_test_visualization_dump=true",
        "data_loader.test.num_workers=0",
        f"test.output_path={output_root / 'sources'}",
        f"hydra.run.dir={output_root / 'hydra_run'}",
    ]
    if checkpoint is not None:
        hydra_args.append(f"checkpointing.load={checkpoint}")
    sys.argv = [sys.argv[0], *hydra_args]
    official_main.train()


if __name__ == "__main__":
    main()
