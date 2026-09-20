#!/usr/bin/env python3
"""Evaluate C4G cross-view synthesis from one camera's full 12-frame video.

For every scene, all twelve timestamps from one source camera are supplied as
context. The other eleven cameras at the same twelve timestamps are rendered
and scored, giving 132 cross-view targets per scene. Model construction,
checkpoint loading, rendering, and metrics come from the official evaluator;
only the test DataModule is replaced.

Because a fixed source camera has zero spatial context baseline, translation
scale is defined by the maximum baseline from the source camera to the full
12-camera rig. Poses are then expressed relative to the source camera.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule

from eval_nvidia_multiview_gt import (
    CAMERA_INDICES,
    DEFAULT_SCENES,
    calibration_poses,
    image_tensor,
    processed_poses,
)


TIMESTAMPS = tuple(range(12))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--multiview-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-camera", type=int, default=1, choices=range(1, 13))
    parser.add_argument("--run-name", default="v4_3_temb_20k_nvidia_cam01_12f_crossview")
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    return parser.parse_args()


class NvidiaSingleCamera12CrossviewDataset(torch.utils.data.Dataset):
    near = 0.1
    far = 100.0

    def __init__(
        self,
        cfg,
        processed_root: Path,
        multiview_root: Path,
        scene_names: list[str],
        input_camera: int,
    ) -> None:
        from src.dataset.shims.crop_shim import apply_crop_shim
        from src.misc.cam_utils import camera_normalization

        self.cfg = cfg
        self.processed_root = processed_root
        self.multiview_root = multiview_root
        self.scene_names = scene_names
        self.input_camera = input_camera - 1
        self.target_cameras = tuple(
            camera for camera in CAMERA_INDICES if camera != self.input_camera
        )
        self.apply_crop_shim = apply_crop_shim
        self.camera_normalization = camera_normalization
        self.multiview_dirs: dict[str, Path] = {}

        for scene in self.scene_names:
            original = self.multiview_root / scene
            if not original.is_dir():
                raise FileNotFoundError(original)
            for component in ("calibration", "multiview_GT"):
                if not (original / component).is_dir():
                    raise FileNotFoundError(original / component)
            multiview_dir = original / "multiview_GT"
            if (multiview_dir / "multiview_GT").is_dir():
                multiview_dir = multiview_dir / "multiview_GT"
            time_dirs = sorted(path.name for path in multiview_dir.iterdir() if path.is_dir())
            expected = [f"{time:08d}" for time in range(1, 13)]
            if time_dirs != expected:
                raise ValueError(f"Unexpected time directories for {scene}: {time_dirs}")
            for time in range(1, 13):
                for camera in range(1, 13):
                    path = multiview_dir / f"{time:08d}" / f"cam{camera:02d}.jpg"
                    if not path.is_file():
                        raise FileNotFoundError(path)
            self.multiview_dirs[scene] = multiview_dir

    def __len__(self) -> int:
        return len(self.scene_names)

    def _poses(self, scene: str) -> torch.Tensor:
        pose_path = self.processed_root / scene / "poses_bounds.npy"
        if pose_path.exists():
            poses = processed_poses(pose_path)
        else:
            poses = calibration_poses(self.multiview_root / scene)
        if poses.shape != (12, 4, 4):
            raise ValueError(f"Expected 12 poses for {scene}, got {poses.shape}")
        return torch.from_numpy(poses).float()

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
            values = torch.from_numpy(
                __import__("numpy").loadtxt(
                    self.multiview_root
                    / scene
                    / "calibration"
                    / f"cam{camera:02d}"
                    / "intrinsic.txt"
                ).astype("float32")
            )
            values[0] /= 960
            values[1] /= 540
            intrinsics.append(values)
        return torch.stack(intrinsics)

    def __getitem__(self, item: int) -> dict:
        scene = self.scene_names[item]
        multiview_dir = self.multiview_dirs[scene]
        poses = self._poses(scene)
        intrinsics = self._intrinsics(scene)

        anchor = poses[self.input_camera, :3, 3]
        scale = (poses[:, :3, 3] - anchor).norm(dim=1).max()
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid full-rig baseline scale for {scene}: {scale}")
        poses[:, :3, 3] /= scale
        if self.cfg.relative_pose:
            poses = self.camera_normalization(
                poses[self.input_camera : self.input_camera + 1],
                poses,
            )
        if not torch.isfinite(poses).all():
            raise ValueError(f"Non-finite poses for {scene}")

        context_images = torch.stack(
            [
                image_tensor(
                    multiview_dir
                    / f"{time + 1:08d}"
                    / f"cam{self.input_camera + 1:02d}.jpg"
                )
                for time in TIMESTAMPS
            ]
        )
        target_entries = [
            (time, camera)
            for time in TIMESTAMPS
            for camera in self.target_cameras
        ]
        target_images = torch.stack(
            [
                image_tensor(
                    multiview_dir
                    / f"{time + 1:08d}"
                    / f"cam{camera + 1:02d}.jpg"
                )
                for time, camera in target_entries
            ]
        )

        context_times = torch.tensor(TIMESTAMPS, dtype=torch.long)
        target_times = torch.tensor([time for time, _ in target_entries], dtype=torch.long)
        target_cameras = torch.tensor(
            [camera for _, camera in target_entries], dtype=torch.long
        )
        near_value = float((self.near / scale).item())
        far_value = float((self.far / scale).item())
        source_pose = poses[self.input_camera]
        source_intrinsic = intrinsics[self.input_camera]

        example = {
            "context": {
                "extrinsics": source_pose.unsqueeze(0).repeat(12, 1, 1),
                "intrinsics": source_intrinsic.unsqueeze(0).repeat(12, 1, 1),
                "image": context_images,
                "near": torch.full((12,), near_value),
                "far": torch.full((12,), far_value),
                "index": context_times,
                "camera": torch.full((12,), self.input_camera, dtype=torch.long),
                "overlap": torch.tensor([1.0]),
            },
            "target": {
                "extrinsics": torch.stack([poses[camera] for _, camera in target_entries]),
                "intrinsics": torch.stack(
                    [intrinsics[camera] for _, camera in target_entries]
                ),
                "image": target_images,
                "near": torch.full((len(target_entries),), near_value),
                "far": torch.full((len(target_entries),), far_value),
                "index": target_times,
                "camera": target_cameras,
                "is_ctx_time": torch.ones(len(target_entries), dtype=torch.bool),
            },
            "scene": f"nvidia_xv_cam{self.input_camera + 1:02d}_{scene}",
        }
        return self.apply_crop_shim(example, (224, 224))


class CrossviewDataModule(LightningDataModule):
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
        options = CROSSVIEW_OPTIONS
        self.dataset = NvidiaSingleCamera12CrossviewDataset(
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


CROSSVIEW_OPTIONS: dict[str, object] = {}


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

    CROSSVIEW_OPTIONS.update(
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
        "context_times_zero_based": list(TIMESTAMPS),
        "target_cameras_one_based": [
            camera for camera in range(1, 13) if camera != args.input_camera
        ],
        "target_times_zero_based": list(TIMESTAMPS),
        "targets_per_scene": 12 * 11,
        "scale_normalization": "maximum source-to-full-rig camera baseline",
        "force_davis_intrinsics": True,
        "align_pose": False,
    }
    (output_root / "run_spec.json").write_text(json.dumps(run_spec, indent=2) + "\n")

    os.chdir(eval_root)
    sys.path.insert(0, str(eval_root))
    os.environ["HYDRA_FULL_ERROR"] = "1"

    import src.main as official_main

    official_main.DataModule = CrossviewDataModule
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
