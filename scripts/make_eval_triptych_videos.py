#!/usr/bin/env python3
"""Convert official C4G evaluation comparison images into triptych videos.

The official evaluator stores every target frame for a scene in one PNG:

    Target GT | Target Rendered | Target Error

Each column contains vertically stacked 224x224 frames.  This utility reverses
that layout without re-running or altering evaluation and writes one
``GT | Pred | Diff`` MP4 per model and dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL_GAP = 8
OFFICIAL_HEADER_HEIGHT = 34
VIDEO_HEADER_HEIGHT = 32


@dataclass(frozen=True)
class DatasetSpec:
    scene: str
    source_pattern: str


DATASETS = {
    "hs_iphone": DatasetSpec("iphone_apple", "iphone_apple_tgt_*.png"),
    "hs_nvidia": DatasetSpec("nvidia_Balloon1", "nvidia_Balloon1_tgt_*.png"),
    "hs_tum": DatasetSpec(
        "tum_rgbd_dataset_freiburg2_desk_with_person",
        "tum_rgbd_dataset_freiburg2_desk_with_person_tgt_*.png",
    ),
    "mg_adt": DatasetSpec(
        "adt_Apartment_release_multiuser_cook_seq141_M1292",
        "adt_Apartment_release_multiuser_cook_seq141_M1292_tgt_*.png",
    ),
}


EXPERIMENTS = {
    "c4g_v4_rks_20k": {
        "hs_iphone": "exp_c4g_v4_rks_hs_iphone_v4_step20000",
        "hs_nvidia": "exp_c4g_v4_rks_hs_nvidia_v4_step20000",
        "hs_tum": "exp_c4g_v4_rks_hs_tum_v4_step20000",
        "mg_adt": "exp_c4g_v4_rks_mg_adt_v4_step20000",
    },
    "base_20k": {
        "hs_iphone": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_hs_iphone_step20000",
        "hs_nvidia": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_hs_nvidia_step20000",
        "hs_tum": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_hs_tum_step20000",
        "mg_adt": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_mg_adt_step20000",
    },
    "base_30k": {
        "hs_iphone": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_30k_resume_from29k_4gpu_hs_iphone_step30000",
        "hs_nvidia": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_30k_resume_from29k_4gpu_hs_nvidia_step30000",
        "hs_tum": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_30k_resume_from29k_4gpu_hs_tum_step30000",
        "mg_adt": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_30k_resume_from29k_4gpu_mg_adt_step30000",
    },
    "egoexo_30k": {
        "hs_iphone": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_30k_resume_from29k_4gpu_hs_iphone_step30000",
        "hs_nvidia": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_30k_resume_from29k_4gpu_hs_nvidia_step30000",
        "hs_tum": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_30k_resume_from29k_4gpu_hs_tum_step30000",
        "mg_adt": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_30k_resume_from29k_4gpu_mg_adt_step30000",
    },
    "egoexo_mono_40k": {
        "hs_iphone": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_40k_resume_from36k_4gpu_hs_iphone_step40000",
        "hs_nvidia": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_40k_resume_from36k_4gpu_hs_nvidia_step40000",
        "hs_tum": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_40k_resume_from36k_4gpu_hs_tum_step40000",
        "mg_adt": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_40k_resume_from36k_4gpu_mg_adt_step40000",
    },
    "egoexo_mono_45k": {
        "hs_iphone": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_45k_resume_from40k_4gpu_hs_iphone_step45000",
        "hs_nvidia": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_45k_resume_from40k_4gpu_hs_nvidia_step45000",
        "hs_tum": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_45k_resume_from40k_4gpu_hs_tum_step45000",
        "mg_adt": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_45k_resume_from40k_4gpu_mg_adt_step45000",
    },
    "v4_3_temb_20k": {
        "hs_iphone": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_v4-3_temb_20k_4gpu_hs_iphone_v4-3_step20000",
        "hs_nvidia": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_v4-3_temb_20k_4gpu_hs_nvidia_v4-3_step20000",
        "hs_tum": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_v4-3_temb_20k_4gpu_hs_tum_v4-3_step20000",
        "mg_adt": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_v4-3_temb_20k_4gpu_mg_adt_v4-3_step20000",
    },
    "v4_4_temb_20k": {
        "hs_iphone": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_v4-4_temb_40k_4gpu_hs_iphone_v4-4_step20000",
        "hs_nvidia": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_v4-4_temb_40k_4gpu_hs_nvidia_v4-4_step20000",
        "hs_tum": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_v4-4_temb_40k_4gpu_hs_tum_v4-4_step20000",
        "mg_adt": "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_egoexo_mono_v4-4_temb_40k_4gpu_mg_adt_v4-4_step20000",
    },
    "release_tokens1024_20k": {
        "hs_iphone": "exp_intrinsic_test_release20k_tokens1024_4gpu_hs_iphone_step20000",
        "hs_nvidia": "exp_intrinsic_test_release20k_tokens1024_4gpu_hs_nvidia_step20000",
        "hs_tum": "exp_intrinsic_test_release20k_tokens1024_4gpu_hs_tum_step20000",
        "mg_adt": "exp_intrinsic_test_release20k_tokens1024_4gpu_mg_adt_step20000",
    },
    "release_tokens4096_noise_20k": {
        "hs_iphone": "exp_intrinsic_test_release20k_tokens4096_noise_4gpu_hs_iphone_step20000",
        "hs_nvidia": "exp_intrinsic_test_release20k_tokens4096_noise_4gpu_hs_nvidia_step20000",
        "hs_tum": "exp_intrinsic_test_release20k_tokens4096_noise_4gpu_hs_tum_step20000",
        "mg_adt": "exp_intrinsic_test_release20k_tokens4096_noise_4gpu_mg_adt_step20000",
    },
}


def find_source(source_roots: list[Path], output_name: str, pattern: str) -> Path | None:
    candidates = [
        path
        for root in source_roots
        for path in (root / output_name).glob(pattern)
    ]
    if not candidates:
        return None
    # Some old output directories contain multiple reruns.  The newest PNG is
    # the one corresponding to the most recent official evaluation log.
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def split_official_comparison(path: Path) -> tuple[list[np.ndarray], dict[str, float | int]]:
    image = np.asarray(Image.open(path).convert("RGB"))
    height, width, _ = image.shape
    panel_size, remainder = divmod(width - 2 * PANEL_GAP, 3)
    if remainder or panel_size <= 0:
        raise ValueError(f"Unexpected comparison width {width}: {path}")

    frame_stride = panel_size + PANEL_GAP
    stacked_height = height - OFFICIAL_HEADER_HEIGHT
    frame_count, trailing = divmod(stacked_height + PANEL_GAP, frame_stride)
    if trailing or frame_count <= 0:
        raise ValueError(
            f"Unexpected comparison height {height} for {panel_size}px panels: {path}"
        )

    x_offsets = [0, panel_size + PANEL_GAP, 2 * (panel_size + PANEL_GAP)]
    font = load_font(18)
    frames = []
    diff_mae = []
    diff_max = []
    for index in range(frame_count):
        y0 = OFFICIAL_HEADER_HEIGHT + index * frame_stride
        panels = [
            image[y0 : y0 + panel_size, x0 : x0 + panel_size]
            for x0 in x_offsets
        ]
        if any(panel.shape != (panel_size, panel_size, 3) for panel in panels):
            raise ValueError(f"Incomplete frame {index} in {path}")

        expected_diff = np.abs(panels[0].astype(np.int16) - panels[1].astype(np.int16))
        delta = np.abs(expected_diff - panels[2].astype(np.int16))
        diff_mae.append(float(delta.mean()))
        diff_max.append(int(delta.max()))

        frame = np.full(
            (VIDEO_HEADER_HEIGHT + panel_size, width, 3),
            (20, 20, 20),
            dtype=np.uint8,
        )
        frame[VIDEO_HEADER_HEIGHT:, :panel_size] = panels[0]
        frame[
            VIDEO_HEADER_HEIGHT:,
            panel_size + PANEL_GAP : 2 * panel_size + PANEL_GAP,
        ] = panels[1]
        frame[
            VIDEO_HEADER_HEIGHT:,
            2 * (panel_size + PANEL_GAP) :,
        ] = panels[2]
        draw_headers(frame, x_offsets, panel_size, font)
        frames.append(frame)

    metadata: dict[str, float | int] = {
        "frame_count": frame_count,
        "panel_size": panel_size,
        "diff_validation_mae": float(np.mean(diff_mae)),
        "diff_validation_max": int(max(diff_max)),
    }
    return frames, metadata


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("assets/Inter-Regular.otf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_headers(
    frame: np.ndarray,
    x_offsets: list[int],
    panel_size: int,
    font: ImageFont.ImageFont,
) -> None:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    for text, x0 in zip(("GT", "Pred", "Diff"), x_offsets):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = x0 + (panel_size - text_width) // 2
        y = (VIDEO_HEADER_HEIGHT - text_height) // 2 - bbox[1]
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
    frame[:] = np.asarray(image)


def target_psnr(path: Path) -> float | None:
    match = re.search(r"_tgt_([0-9]+(?:\.[0-9]+)?)\.png$", path.name)
    return float(match.group(1)) if match else None


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        action="append",
        type=Path,
        default=[],
        help="Root containing official evaluator output directories (repeatable)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval_triptych_videos"),
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_roots = args.source_root or [
        Path(
            "/music-3d-shared-disk/user/KAIST/MG/InterpolatedC3G/"
            "c4g/eval/outputs/test"
        ),
        Path("outputs/eval_triptych_sources"),
    ]

    found: list[tuple[str, str, Path]] = []
    missing: list[tuple[str, str, str]] = []
    for experiment, output_names in EXPERIMENTS.items():
        for dataset, output_name in output_names.items():
            source = find_source(
                source_roots,
                output_name,
                DATASETS[dataset].source_pattern,
            )
            if source is None:
                missing.append((experiment, dataset, output_name))
            else:
                found.append((experiment, dataset, source))

    print(f"Found {len(found)}/{len(EXPERIMENTS) * len(DATASETS)} official sources")
    for experiment, dataset, output_name in missing:
        print(f"MISSING {experiment:32s} {dataset:10s} {output_name}")

    if args.check_only:
        raise SystemExit(1 if missing else 0)
    if missing:
        raise SystemExit("Generate the missing official comparison PNGs before conversion")

    manifest = []
    for run_index, (experiment, dataset, source) in enumerate(found, start=1):
        output_path = args.output_dir / experiment / f"{dataset}.mp4"
        frames, metadata = split_official_comparison(source)
        if args.overwrite or not output_path.exists():
            write_video(frames, output_path, args.fps)
        row = {
            "experiment": experiment,
            "dataset": dataset,
            "scene": DATASETS[dataset].scene,
            "source_png": str(source.resolve()),
            "output_video": str(output_path.resolve()),
            "fps": args.fps,
            "target_psnr_from_filename": target_psnr(source),
            **metadata,
        }
        manifest.append(row)
        print(
            f"[{run_index:02d}/{len(found):02d}] {experiment}/{dataset}: "
            f"{metadata['frame_count']} frames -> {output_path}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "manifest.json", "w") as file:
        json.dump(manifest, file, indent=2)
    with open(args.output_dir / "manifest.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)


if __name__ == "__main__":
    main()
