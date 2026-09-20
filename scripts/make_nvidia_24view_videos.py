#!/usr/bin/env python3
"""Build synchronized 12-camera GT + 12-camera prediction videos.

Each video frame is one timestamp. The 24 views are arranged as four rows:
GT cam01--cam06, Pred cam01--cam06, GT cam07--cam12, and Pred cam07--cam12.
All GT images undergo the same Lanczos rescale and center crop as the official
224x224 evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL_SIZE = 224
PANEL_HEADER = 26
GLOBAL_HEADER = 40
GAP = 8
ROWS = 4
COLS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--multiview-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def official_crop(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width_in, height_in = image.size
    scale = max(PANEL_SIZE / height_in, PANEL_SIZE / width_in)
    height_scaled = round(height_in * scale)
    width_scaled = round(width_in * scale)
    image = image.resize((width_scaled, height_scaled), Image.Resampling.LANCZOS)
    row = (height_scaled - PANEL_SIZE) // 2
    col = (width_scaled - PANEL_SIZE) // 2
    image = image.crop((col, row, col + PANEL_SIZE, row + PANEL_SIZE))
    array = np.asarray(image, dtype=np.uint8)
    if array.shape != (PANEL_SIZE, PANEL_SIZE, 3):
        raise ValueError(f"Unexpected cropped GT shape {array.shape}: {path}")
    return array


def load_prediction(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if array.shape != (PANEL_SIZE, PANEL_SIZE, 3):
        raise ValueError(f"Unexpected prediction shape {array.shape}: {path}")
    return array


def prediction_path(scene_dir: Path, time: int, camera: int) -> Path | None:
    """Return a render path, or ``None`` when cam01 should repeat the input."""
    target = scene_dir / "color" / f"tgt_t{time:04d}_cam{camera:02d}.png"
    if target.is_file() or camera != 0:
        return target
    context = scene_dir / "color" / f"ctx_t{time:04d}_00.png"
    return context if context.is_file() else None


def multiview_dir(multiview_root: Path, scene: str) -> Path:
    directory = multiview_root / scene / "multiview_GT"
    nested = directory / "multiview_GT"
    return nested if nested.is_dir() else directory


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = x0 + (x1 - x0 - width) // 2
    y = y0 + (y1 - y0 - height) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=fill)


def compose_frame(
    scene: str,
    time: int,
    gt: list[np.ndarray],
    pred: list[np.ndarray],
    title_font: ImageFont.ImageFont,
    panel_font: ImageFont.ImageFont,
    pred_cam01_is_input_copy: bool = False,
) -> np.ndarray:
    cell_height = PANEL_HEADER + PANEL_SIZE
    width = COLS * PANEL_SIZE + (COLS - 1) * GAP
    height = GLOBAL_HEADER + ROWS * cell_height + (ROWS - 1) * GAP
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    centered_text(
        draw,
        (0, 0, width, GLOBAL_HEADER),
        f"{scene}   synchronized time {time + 1:02d}/12",
        title_font,
        (255, 255, 255),
    )

    rows = (
        ("GT", range(0, 6), gt),
        ("Pred", range(0, 6), pred),
        ("GT", range(6, 12), gt),
        ("Pred", range(6, 12), pred),
    )
    for row_index, (kind, cameras, panels) in enumerate(rows):
        y0 = GLOBAL_HEADER + row_index * (cell_height + GAP)
        for col_index, camera in enumerate(cameras):
            x0 = col_index * (PANEL_SIZE + GAP)
            label = f"{kind} cam{camera + 1:02d}"
            if camera == 0:
                if kind == "GT":
                    label += " [INPUT]"
                elif pred_cam01_is_input_copy:
                    label += " [INPUT COPY]"
                else:
                    label += " [RECON]"
            header_fill = (34, 72, 108) if kind == "GT" else (92, 48, 92)
            draw.rectangle(
                (x0, y0, x0 + PANEL_SIZE - 1, y0 + PANEL_HEADER - 1),
                fill=header_fill,
            )
            centered_text(
                draw,
                (x0, y0, x0 + PANEL_SIZE, y0 + PANEL_HEADER),
                label,
                panel_font,
                (255, 255, 255),
            )
            canvas.paste(
                Image.fromarray(panels[camera]),
                (x0, y0 + PANEL_HEADER),
            )
    return np.asarray(canvas)


def video_writer(path: Path, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=2,
    )


def main() -> None:
    args = parse_args()
    prediction_root = args.prediction_root.resolve()
    multiview_root = args.multiview_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted(
        path
        for path in prediction_root.glob("nvidia_xv_cam01_*")
        if (path / "color").is_dir()
    )
    if not scene_dirs:
        raise FileNotFoundError(f"No rendered scene directories in {prediction_root}")

    title_font = load_font(20)
    panel_font = load_font(15)
    combined_path = output_dir / "all_9_scenes_24view.mp4"
    if combined_path.exists() and not args.overwrite:
        raise FileExistsError(combined_path)

    manifest = []
    sync_rows = []
    with video_writer(combined_path, args.fps) as combined_writer:
        for scene_dir in scene_dirs:
            scene = scene_dir.name.removeprefix("nvidia_xv_cam01_")
            gt_root = multiview_dir(multiview_root, scene)
            video_path = output_dir / f"{scene}_24view.mp4"
            if video_path.exists() and not args.overwrite:
                raise FileExistsError(video_path)

            frames = []
            for time in range(12):
                gt_paths = [
                    gt_root / f"{time + 1:08d}" / f"cam{camera + 1:02d}.jpg"
                    for camera in range(12)
                ]
                pred_paths = [
                    prediction_path(scene_dir, time, camera)
                    for camera in range(12)
                ]
                missing = [
                    path
                    for path in gt_paths + pred_paths
                    if path is not None and not path.is_file()
                ]
                if missing:
                    raise FileNotFoundError(missing[0])
                gt_panels = [official_crop(path) for path in gt_paths]
                pred_cam01_is_input_copy = pred_paths[0] is None
                pred_panels = [
                    gt_panels[camera]
                    if path is None
                    else load_prediction(path)
                    for camera, path in enumerate(pred_paths)
                ]
                frame = compose_frame(
                    scene,
                    time,
                    gt_panels,
                    pred_panels,
                    title_font,
                    panel_font,
                    pred_cam01_is_input_copy,
                )
                frames.append(frame)
                combined_writer.append_data(frame)
                sync_rows.append(
                    {
                        "scene": scene,
                        "video_frame": time,
                        "time_zero_based": time,
                        "gt_time_directory": f"{time + 1:08d}",
                        "num_gt_views": 12,
                        "num_pred_views": 12,
                    }
                )

            with video_writer(video_path, args.fps) as writer:
                for frame in frames:
                    writer.append_data(frame)
            manifest.append(
                {
                    "scene": scene,
                    "video": str(video_path),
                    "frames": 12,
                    "fps": args.fps,
                    "width": int(frames[0].shape[1]),
                    "height": int(frames[0].shape[0]),
                    "layout": "GT cam01-06 / Pred cam01-06 / GT cam07-12 / Pred cam07-12",
                }
            )
            print(f"{scene}: 12 synchronized frames -> {video_path}")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (output_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    with (output_dir / "frame_sync.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sync_rows[0]))
        writer.writeheader()
        writer.writerows(sync_rows)
    print(f"Combined video -> {combined_path}")


if __name__ == "__main__":
    main()
