#!/usr/bin/env python3
"""Evaluate C4G with the original NVIDIA multiview ground truth.

The input protocol is kept identical to the existing NVIDIA evaluation:

* six diagonal monocular context samples at zero-based indices 0,2,...,10;
* the same input images, poses, image processing, normalization, and intrinsics;
* five interpolation timestamps at indices 1,3,5,7,9.

Only the targets are expanded. At every interpolation timestamp, all twelve
camera views from multiview_GT are rendered and scored (60 targets/scene).
The official evaluator's model construction, checkpoint loading, rendering,
and metric implementation are reused by replacing only its test DataModule.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from PIL import Image


DEFAULT_SCENES = (
    "Balloon1",
    "Balloon2",
    "DynamicFace",
    "Jumping",
    "Playground",
    "Skating",
    "Teadybear",
    "Truck",
    "Umbrella",
)
CONTEXT_INDICES = (0, 2, 4, 6, 8, 10)
TARGET_TIMES = (1, 3, 5, 7, 9)
CAMERA_INDICES = tuple(range(12))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--multiview-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", default="v4_3_temb_20k_nvidia_multiview_gt9")
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    return parser.parse_args()


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255)


def processed_poses(path: Path) -> np.ndarray:
    poses_bounds = np.load(path)
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    c2w_llff = poses[:, :3, :4]
    c2w = np.zeros((len(poses), 4, 4), dtype=np.float32)
    c2w[:, :3, 0] = c2w_llff[:, :, 1]
    c2w[:, :3, 1] = c2w_llff[:, :, 0]
    c2w[:, :3, 2] = -c2w_llff[:, :, 2]
    c2w[:, :3, 3] = c2w_llff[:, :, 3]
    c2w[:, 3, 3] = 1
    return c2w


def calibration_poses(scene_root: Path) -> np.ndarray:
    poses = []
    for camera in range(1, 13):
        values = np.loadtxt(
            scene_root / "calibration" / f"cam{camera:02d}" / "extrinsic.txt"
        )
        if values.shape != (4, 3):
            raise ValueError(
                f"Unexpected extrinsic shape {values.shape} for camera {camera}"
            )
        center = values[0]
        rotation_w2c = values[1:]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation_w2c.T
        pose[:3, 3] = center
        poses.append(pose)
    return np.stack(poses)


class NvidiaMultiviewGTDataset(torch.utils.data.Dataset):
    near = 0.1
    far = 100.0

    def __init__(
        self,
        cfg,
        processed_root: Path,
        multiview_root: Path,
        scene_names: list[str],
    ) -> None:
        from src.dataset.shims.crop_shim import apply_crop_shim
        from src.misc.cam_utils import camera_normalization

        self.cfg = cfg
        self.processed_root = processed_root
        self.multiview_root = multiview_root
        self.scene_names = scene_names
        self.apply_crop_shim = apply_crop_shim
        self.camera_normalization = camera_normalization
        self.context_indices = torch.tensor(CONTEXT_INDICES, dtype=torch.long)
        self.multiview_dirs: dict[str, Path] = {}

        for scene in self.scene_names:
            original = self.multiview_root / scene
            if not original.is_dir():
                raise FileNotFoundError(original)
            for component in ("input_images", "calibration", "multiview_GT"):
                if not (original / component).is_dir():
                    raise FileNotFoundError(original / component)
            multiview_dir = original / "multiview_GT"
            if (multiview_dir / "multiview_GT").is_dir():
                multiview_dir = multiview_dir / "multiview_GT"
            time_dirs = sorted(
                path.name
                for path in multiview_dir.iterdir()
                if path.is_dir()
            )
            if time_dirs != [f"{time:08d}" for time in range(1, 13)]:
                raise ValueError(f"Unexpected time directories for {scene}: {time_dirs}")
            self.multiview_dirs[scene] = multiview_dir

    def __len__(self) -> int:
        return len(self.scene_names)

    def _poses(self, scene: str) -> np.ndarray:
        pose_path = self.processed_root / scene / "poses_bounds.npy"
        if pose_path.exists():
            poses = processed_poses(pose_path)
        else:
            poses = calibration_poses(self.multiview_root / scene)
        if poses.shape != (12, 4, 4):
            raise ValueError(f"Expected 12 poses for {scene}, got {poses.shape}")
        return poses

    def _input_path(self, scene: str, index: int) -> Path:
        processed = (
            self.processed_root / scene / "images_original_size" / f"{index:03d}.png"
        )
        if processed.exists():
            return processed
        return (
            self.multiview_root
            / scene
            / "input_images"
            / f"cam{index + 1:02d}.jpg"
        )

    def _intrinsics(self, scene: str) -> torch.Tensor:
        if self.cfg.re10k_intrinsic:
            intrinsic = torch.tensor(
                [
                    [0.4836, 0.0000, 0.5000],
                    [0.0000, 0.8597, 0.5000],
                    [0.0000, 0.0000, 1.0000],
                ],
                dtype=torch.float32,
            )
            return intrinsic.unsqueeze(0).repeat(12, 1, 1)

        intrinsics = []
        for camera in range(1, 13):
            intrinsic = np.loadtxt(
                self.multiview_root
                / scene
                / "calibration"
                / f"cam{camera:02d}"
                / "intrinsic.txt"
            ).astype(np.float32)
            intrinsic[0] /= 960
            intrinsic[1] /= 540
            intrinsics.append(torch.from_numpy(intrinsic))
        return torch.stack(intrinsics)

    def __getitem__(self, item: int) -> dict:
        scene = self.scene_names[item]
        original_root = self.multiview_root / scene
        multiview_dir = self.multiview_dirs[scene]
        poses = torch.from_numpy(self._poses(scene)).float()
        intrinsics = self._intrinsics(scene)

        context_poses = poses[self.context_indices]
        anchor = context_poses[0, :3, 3]
        scale = (context_poses[1:, :3, 3] - anchor).norm(dim=1).max()
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid baseline scale for {scene}: {scale}")
        poses[:, :3, 3] /= scale
        if self.cfg.relative_pose:
            poses = self.camera_normalization(
                poses[self.context_indices][0:1],
                poses,
            )
        if not torch.isfinite(poses).all():
            raise ValueError(f"Non-finite poses for {scene}")

        context_images = torch.stack(
            [image_tensor(self._input_path(scene, int(i))) for i in CONTEXT_INDICES]
        )
        target_entries = [
            (time_index, camera_index)
            for time_index in TARGET_TIMES
            for camera_index in CAMERA_INDICES
        ]
        target_images = torch.stack(
            [
                image_tensor(
                    multiview_dir
                    / f"{time_index + 1:08d}"
                    / f"cam{camera_index + 1:02d}.jpg"
                )
                for time_index, camera_index in target_entries
            ]
        )
        target_times = torch.tensor(
            [time for time, _ in target_entries], dtype=torch.long
        )
        target_cameras = torch.tensor(
            [camera for _, camera in target_entries], dtype=torch.long
        )
        target_poses = torch.stack(
            [poses[camera] for _, camera in target_entries]
        )
        target_intrinsics = torch.stack(
            [intrinsics[camera] for _, camera in target_entries]
        )
        near_value = float((self.near / scale).item())
        far_value = float((self.far / scale).item())

        example = {
            "context": {
                "extrinsics": poses[self.context_indices],
                "intrinsics": intrinsics[self.context_indices],
                "image": context_images,
                "near": torch.full((len(CONTEXT_INDICES),), near_value),
                "far": torch.full((len(CONTEXT_INDICES),), far_value),
                "index": self.context_indices.clone(),
                "camera": torch.zeros(len(CONTEXT_INDICES), dtype=torch.long),
                "overlap": torch.tensor([1.0]),
            },
            "target": {
                "extrinsics": target_poses,
                "intrinsics": target_intrinsics,
                "image": target_images,
                "near": torch.full((len(target_entries),), near_value),
                "far": torch.full((len(target_entries),), far_value),
                "index": target_times,
                "camera": target_cameras,
                "is_ctx_time": torch.zeros(len(target_entries), dtype=torch.bool),
            },
            "scene": "nvidia_mv_" + scene,
        }
        return self.apply_crop_shim(example, (224, 224))


class MultiviewDataModule(LightningDataModule):
    """Drop-in replacement for the official evaluator's test DataModule."""

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
        options = MULTIVIEW_OPTIONS
        self.dataset = NvidiaMultiviewGTDataset(
            cfg,
            options["processed_root"],
            options["multiview_root"],
            options["scenes"],
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


MULTIVIEW_OPTIONS: dict[str, object] = {}


def main() -> None:
    args = parse_args()
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

    MULTIVIEW_OPTIONS.update(
        processed_root=processed_root,
        multiview_root=multiview_root,
        scenes=args.scenes,
    )
    run_spec = {
        "config": str(config),
        "checkpoint_override": str(checkpoint) if checkpoint else None,
        "processed_input_root": str(processed_root),
        "multiview_gt_root": str(multiview_root),
        "scenes": args.scenes,
        "context_indices_zero_based": list(CONTEXT_INDICES),
        "target_times_zero_based": list(TARGET_TIMES),
        "target_cameras_zero_based": list(CAMERA_INDICES),
        "targets_per_scene": len(TARGET_TIMES) * len(CAMERA_INDICES),
        "force_davis_intrinsics": True,
        "align_pose": False,
    }
    (output_root / "run_spec.json").write_text(
        json.dumps(run_spec, indent=2) + "\n"
    )

    os.chdir(eval_root)
    sys.path.insert(0, str(eval_root))
    os.environ["HYDRA_FULL_ERROR"] = "1"

    import src.main as official_main

    official_main.DataModule = MultiviewDataModule
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
        f"test.save_image={'true' if args.save_outputs else 'false'}",
        f"test.save_compare={'true' if args.save_outputs else 'false'}",
        "test.save_video=false",
        "test.target_only_eval=false",
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
