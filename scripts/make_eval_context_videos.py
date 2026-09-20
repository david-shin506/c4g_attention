#!/usr/bin/env python3
"""Make time-synchronized CTX | CTX | GT | Pred | Diff evaluation videos.

Target frames in the interpolation benchmarks lie between two input context
frames.  For every official target-comparison frame, this script finds the
context timestamps that bracket the target timestamp and places both input
images beside the evaluator's GT, prediction, and error panels.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL_SIZE = 224
PANEL_GAP = 8
OFFICIAL_HEADER_HEIGHT = 34
VIDEO_HEADER_HEIGHT = 48
PANEL_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        action="append",
        required=True,
        metavar="DATASET=DIR",
        help="Official evaluator scene directory; repeat for each video.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_scenes(values: list[str]) -> dict[str, Path]:
    scenes: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected DATASET=DIR, got {value!r}")
        dataset, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not dataset or dataset in scenes:
            raise ValueError(f"Invalid or duplicate dataset name: {dataset!r}")
        if not path.is_dir():
            raise FileNotFoundError(path)
        scenes[dataset] = path
    return scenes


def timestamp(path: Path) -> int:
    match = re.search(r"(?:^|_)t(\d{4})(?:_|\.)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse timestamp from {path}")
    return int(match.group(1))


def find_comparison(scene_dir: Path) -> Path:
    candidates = list(scene_dir.parent.glob(f"{scene_dir.name}_tgt_*.png"))
    if not candidates:
        raise FileNotFoundError(
            f"No official target-comparison PNG for {scene_dir.name}"
        )
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def load_official_panels(path: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    image = np.asarray(Image.open(path).convert("RGB"))
    expected_width = 3 * PANEL_SIZE + 2 * PANEL_GAP
    if image.shape[1] != expected_width:
        raise ValueError(f"Unexpected comparison width {image.shape[1]} in {path}")
    stacked_height = image.shape[0] - OFFICIAL_HEADER_HEIGHT
    frame_count, remainder = divmod(
        stacked_height + PANEL_GAP, PANEL_SIZE + PANEL_GAP
    )
    if remainder or frame_count <= 0:
        raise ValueError(f"Unexpected comparison height {image.shape[0]} in {path}")

    x_offsets = (0, PANEL_SIZE + PANEL_GAP, 2 * (PANEL_SIZE + PANEL_GAP))
    frames = []
    for index in range(frame_count):
        y0 = OFFICIAL_HEADER_HEIGHT + index * (PANEL_SIZE + PANEL_GAP)
        panels = tuple(
            image[y0 : y0 + PANEL_SIZE, x0 : x0 + PANEL_SIZE]
            for x0 in x_offsets
        )
        if any(panel.shape != (PANEL_SIZE, PANEL_SIZE, 3) for panel in panels):
            raise ValueError(f"Incomplete official frame {index} in {path}")
        frames.append(panels)
    return frames


def load_rgb(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"))
    if image.shape != (PANEL_SIZE, PANEL_SIZE, 3):
        raise ValueError(f"Expected 224x224 RGB image, got {image.shape}: {path}")
    return image


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        Path("assets/Inter-Regular.otf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_header(
    frame: np.ndarray,
    labels: tuple[str, str, str, str, str],
    font: ImageFont.ImageFont,
) -> None:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    for index, label in enumerate(labels):
        x0 = index * (PANEL_SIZE + PANEL_GAP)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = x0 + (PANEL_SIZE - text_width) // 2
        y = (VIDEO_HEADER_HEIGHT - text_height) // 2 - bbox[1]
        draw.text((x, y), label, font=font, fill=(255, 255, 255))
    frame[:] = np.asarray(image)


def compose_frame(
    previous_context: np.ndarray,
    next_context: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    diff: np.ndarray,
    previous_time: int,
    next_time: int,
    target_time: int,
    font: ImageFont.ImageFont,
) -> np.ndarray:
    width = PANEL_COUNT * PANEL_SIZE + (PANEL_COUNT - 1) * PANEL_GAP
    frame = np.full(
        (VIDEO_HEADER_HEIGHT + PANEL_SIZE, width, 3),
        (20, 20, 20),
        dtype=np.uint8,
    )
    for index, panel in enumerate(
        (previous_context, next_context, gt, pred, diff)
    ):
        x0 = index * (PANEL_SIZE + PANEL_GAP)
        frame[VIDEO_HEADER_HEIGHT:, x0 : x0 + PANEL_SIZE] = panel
    draw_header(
        frame,
        (
            f"CTX< t{previous_time:04d}",
            f"CTX> t{next_time:04d}",
            f"GT t{target_time:04d}",
            f"Pred t{target_time:04d}",
            "Diff",
        ),
        font,
    )
    return frame


def write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=2,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def process_scene(
    dataset: str,
    scene_dir: Path,
    output_dir: Path,
    fps: int,
    overwrite: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    comparison_path = find_comparison(scene_dir)
    official_panels = load_official_panels(comparison_path)
    gt_paths = sorted((scene_dir / "tgt_gt").glob("t*_c0000.png"), key=timestamp)
    pred_by_time = {
        timestamp(path): path
        for path in (scene_dir / "tgt_pred").glob("t*_c0000.png")
    }
    context_by_time = {
        timestamp(path): path for path in scene_dir.glob("context_t*_cam00.png")
    }
    target_times = [timestamp(path) for path in gt_paths]
    context_times = sorted(context_by_time)
    if len(official_panels) != len(target_times):
        raise ValueError(
            f"Official/target frame mismatch for {scene_dir.name}: "
            f"{len(official_panels)} != {len(target_times)}"
        )
    if set(target_times) != set(pred_by_time):
        raise ValueError(f"GT/pred timestamp mismatch for {scene_dir.name}")
    if len(context_times) < 2:
        raise ValueError(f"Need at least two context frames in {scene_dir}")

    font = load_font(17)
    video_frames = []
    sync_rows = []
    diff_validation_max = 0
    gt_consistency_max = 0
    pred_consistency_max = 0
    for index, (target_time, gt_path, panels) in enumerate(
        zip(target_times, gt_paths, official_panels)
    ):
        position = bisect.bisect_left(context_times, target_time)
        if position == 0 or position == len(context_times):
            raise ValueError(
                f"Target t{target_time:04d} is not bracketed by context in "
                f"{scene_dir.name}"
            )
        previous_time = context_times[position - 1]
        next_time = context_times[position]
        if not previous_time < target_time < next_time:
            raise ValueError(
                f"Target t{target_time:04d} is not strictly between "
                f"t{previous_time:04d} and t{next_time:04d}"
            )

        official_gt, official_pred, official_diff = panels
        saved_gt = load_rgb(gt_path)
        saved_pred = load_rgb(pred_by_time[target_time])
        gt_consistency_max = max(
            gt_consistency_max,
            int(np.abs(saved_gt.astype(np.int16) - official_gt).max()),
        )
        pred_consistency_max = max(
            pred_consistency_max,
            int(np.abs(saved_pred.astype(np.int16) - official_pred).max()),
        )
        expected_diff = np.abs(
            official_gt.astype(np.int16) - official_pred.astype(np.int16)
        )
        diff_validation_max = max(
            diff_validation_max,
            int(np.abs(expected_diff - official_diff.astype(np.int16)).max()),
        )
        video_frames.append(
            compose_frame(
                load_rgb(context_by_time[previous_time]),
                load_rgb(context_by_time[next_time]),
                official_gt,
                official_pred,
                official_diff,
                previous_time,
                next_time,
                target_time,
                font,
            )
        )
        sync_rows.append(
            {
                "dataset": dataset,
                "scene": scene_dir.name,
                "video_frame": index,
                "previous_context_time": previous_time,
                "target_time": target_time,
                "next_context_time": next_time,
                "previous_context_path": str(context_by_time[previous_time]),
                "gt_path": str(gt_path),
                "pred_path": str(pred_by_time[target_time]),
                "next_context_path": str(context_by_time[next_time]),
            }
        )

    video_path = output_dir / f"{dataset}.mp4"
    if overwrite or not video_path.exists():
        write_video(video_frames, video_path, fps)
    manifest_row: dict[str, object] = {
        "dataset": dataset,
        "scene": scene_dir.name,
        "source_comparison_png": str(comparison_path),
        "output_video": str(video_path.resolve()),
        "frame_count": len(video_frames),
        "context_frame_count": len(context_times),
        "fps": fps,
        "width": video_frames[0].shape[1],
        "height": video_frames[0].shape[0],
        "layout": "CTX previous | CTX next | GT | Pred | Diff",
        "strictly_bracketed": True,
        "gt_consistency_max": gt_consistency_max,
        "pred_consistency_max": pred_consistency_max,
        "diff_validation_max": diff_validation_max,
    }
    return manifest_row, sync_rows


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    scenes = parse_scenes(args.scene)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    sync_rows = []
    for index, (dataset, scene_dir) in enumerate(scenes.items(), start=1):
        row, rows = process_scene(
            dataset,
            scene_dir,
            args.output_dir,
            args.fps,
            args.overwrite,
        )
        manifest.append(row)
        sync_rows.extend(rows)
        print(
            f"[{index:02d}/{len(scenes):02d}] {dataset}/{scene_dir.name}: "
            f"{row['frame_count']} frames -> {row['output_video']}"
        )

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    write_table(args.output_dir / "manifest.csv", manifest)
    write_table(args.output_dir / "frame_sync.csv", sync_rows)


if __name__ == "__main__":
    main()
