#!/usr/bin/env python3
"""Add per-frame C4G predictions, comparisons, and metrics to a PLY bundle."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from export_baseline_supersplat import save_eval_artifacts
from make_eval_triptych_videos import split_official_comparison
from make_iphone_eval_bundle import panel_tensors


def load_scene_visibility(
    data_root: Path, scene: str, num_frames: int
) -> torch.Tensor:
    scene_id = scene.removeprefix("iphone_")
    scene_root = data_root / scene_id / scene_id
    metadata = json.loads((scene_root / "metadata.json").read_text())

    by_camera: dict[int, dict[int, str]] = {}
    for frame_name, values in metadata.items():
        camera_id = int(values["camera_id"])
        appearance_id = int(values["appearance_id"])
        by_camera.setdefault(camera_id, {})[appearance_id] = frame_name

    common_appearances = set(next(iter(by_camera.values())))
    for camera_frames in by_camera.values():
        common_appearances &= set(camera_frames)
    if 1 not in by_camera:
        raise ValueError(f"{scene}: target camera 1 is unavailable")
    selected = sorted(common_appearances)[:num_frames]
    if len(selected) != num_frames:
        raise ValueError(f"{scene}: masks={len(selected)} frames={num_frames}")

    masks = []
    for appearance_id in selected:
        frame_name = by_camera[1][appearance_id]
        path = scene_root / "covisible" / "2x" / "val" / f"{frame_name}.png"
        if path.exists():
            image = Image.open(path).convert("L")
        else:
            image = Image.new("L", (224, 224), 255)
        image = image.resize((224, 224), Image.Resampling.NEAREST)
        masks.append(torch.from_numpy(np.asarray(image).copy()).float().div(255).gt(0.5))
    return torch.stack(masks).unsqueeze(1).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-name", default="C4G_v4_3_TEmb_20k")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = sorted(args.source_dir.resolve().glob("iphone_*_tgt_*.png"))
    if not sources:
        raise FileNotFoundError(f"No iPhone comparison sheets in {args.source_dir}")

    for source in sources:
        match = re.match(r"^(iphone_.+)_tgt_[0-9.]+\.png$", source.name)
        if match is None:
            continue
        scene = match.group(1)
        scene_dir = args.output_root.resolve() / args.model_name / scene
        if not (scene_dir / "gaussian_sequence").is_dir():
            raise FileNotFoundError(scene_dir / "gaussian_sequence")

        frames, source_metadata = split_official_comparison(source)
        ground_truth, predictions = panel_tensors(frames, torch.device("cpu"))
        visibility = load_scene_visibility(args.data_root.resolve(), scene, len(predictions))
        if visibility is not None and len(visibility) != len(predictions):
            raise ValueError(
                f"{scene}: visibility={len(visibility)} predictions={len(predictions)}"
            )
        metrics = save_eval_artifacts(
            model_name=args.model_name,
            scene_dir=scene_dir,
            predictions=predictions,
            ground_truth=ground_truth,
            visibility=visibility,
            timestamp_indices=list(range(len(predictions))),
        )

        metadata_path = scene_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        metadata["evaluation_artifacts"] = {
            "source_comparison": str(source),
            "source_validation": source_metadata,
            "metrics": metrics,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{args.model_name}/{scene}] {len(predictions)} frames "
            f"PSNR={metrics['psnr']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
