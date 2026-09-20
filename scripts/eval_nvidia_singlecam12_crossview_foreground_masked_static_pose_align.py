#!/usr/bin/env python3
"""Evaluate NVIDIA cross-view synthesis with one pose per static camera."""

from __future__ import annotations

import json
import os
import sys

import eval_nvidia_singlecam12_crossview as base
import eval_nvidia_singlecam12_crossview_foreground_masked as masked
from static_camera_pose_align import align_static_camera_poses


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

    masked.MASKED_OPTIONS.update(
        processed_root=processed_root,
        multiview_root=multiview_root,
        scenes=args.scenes,
        input_camera=args.input_camera,
    )
    target_cameras = [
        camera for camera in range(1, 13) if camera != args.input_camera
    ]
    run_spec = {
        "config": str(config),
        "checkpoint_override": str(checkpoint) if checkpoint else None,
        "processed_pose_root": str(processed_root),
        "multiview_root": str(multiview_root),
        "scenes": args.scenes,
        "context_camera_one_based": args.input_camera,
        "context_times_zero_based": list(base.TIMESTAMPS),
        "target_cameras_one_based": target_cameras,
        "target_times_zero_based": list(base.TIMESTAMPS),
        "targets_per_scene": 132,
        "optimized_camera_poses_per_scene": len(target_cameras),
        "pose_temporal_policy": "one extrinsic per physical target camera shared across all 12 timestamps",
        "metric_region": "provided per-camera foreground_mask > 127",
        "mask_temporal_policy": "repeat each camera mask at all 12 timestamps",
        "mask_resize": "official crop geometry with nearest-neighbor sampling",
        "scale_normalization": "maximum source-to-full-rig camera baseline",
        "force_davis_intrinsics": True,
        "align_pose": True,
        "share_target_pose": True,
        "shared_pose_scope": "per physical camera, not across camera IDs",
        "use_vggt_target_pose": False,
        "pose_alignment_objective": "official full-frame MSE + LPIPS losses averaged across timestamps",
    }
    (output_root / "run_spec.json").write_text(json.dumps(run_spec, indent=2) + "\n")

    os.chdir(eval_root)
    sys.path.insert(0, str(eval_root))
    os.environ["HYDRA_FULL_ERROR"] = "1"

    import src.main as official_main
    from src.model.model_wrapper import ModelWrapper

    ModelWrapper.test_step_align_shared_target_pose = align_static_camera_poses
    official_main.DataModule = masked.ForegroundMaskedDataModule
    hydra_args = [
        f"--config-path={config.parent}",
        f"--config-name={config.stem}",
        "mode=test",
        f"wandb.name={args.run_name}",
        "wandb.mode=disabled",
        "force_davis_intrinsics=true",
        "test.align_pose=true",
        "test.share_target_pose=true",
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
