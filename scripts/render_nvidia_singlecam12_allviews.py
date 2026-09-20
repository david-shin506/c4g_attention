#!/usr/bin/env python3
"""Render all 12 NVIDIA cameras from cam01's 12-frame input video.

This is the visualization companion to
``eval_nvidia_singlecam12_crossview.py``. It includes the source camera in the
target set, so every timestamp produces cam01--cam12 predictions. The official
evaluator saves each rendered image; a separate utility assembles synchronized
24-view GT/pred videos.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import fields

import torch
from lightning.pytorch import LightningDataModule

import eval_nvidia_singlecam12_crossview as base
from eval_nvidia_multiview_gt import CAMERA_INDICES


class NvidiaSingleCamera12AllviewDataset(
    base.NvidiaSingleCamera12CrossviewDataset
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.target_cameras = CAMERA_INDICES


class AllviewRenderDataModule(LightningDataModule):
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
        options = ALLVIEW_OPTIONS
        self.dataset = NvidiaSingleCamera12AllviewDataset(
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


ALLVIEW_OPTIONS: dict[str, object] = {}


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

    ALLVIEW_OPTIONS.update(
        processed_root=processed_root,
        multiview_root=multiview_root,
        scenes=args.scenes,
        input_camera=args.input_camera,
    )
    run_spec = {
        "purpose": "24-view synchronized video source render",
        "config": str(config),
        "checkpoint_override": str(checkpoint) if checkpoint else None,
        "processed_pose_root": str(processed_root),
        "multiview_root": str(multiview_root),
        "scenes": args.scenes,
        "context_camera_one_based": args.input_camera,
        "context_times_zero_based": list(base.TIMESTAMPS),
        "target_cameras_one_based": list(range(1, 13)),
        "target_times_zero_based": list(base.TIMESTAMPS),
        "targets_per_scene": 144,
        "scale_normalization": "maximum source-to-full-rig camera baseline",
        "force_davis_intrinsics": True,
        "align_pose": False,
        "save_predictions": True,
    }
    (output_root / "run_spec.json").write_text(json.dumps(run_spec, indent=2) + "\n")

    os.chdir(eval_root)
    sys.path.insert(0, str(eval_root))
    os.environ["HYDRA_FULL_ERROR"] = "1"

    import src.main as official_main

    official_main.DataModule = AllviewRenderDataModule
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
        "test.compute_scores=false",
        "test.save_image=true",
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
