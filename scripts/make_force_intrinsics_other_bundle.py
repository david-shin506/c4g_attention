#!/usr/bin/env python3
"""Bundle non-iPhone force-intrinsics comparisons for v4-3 TEmb."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from make_eval_triptych_videos import (
    split_official_comparison,
    target_psnr,
    write_video,
)


DATASETS = {
    "hs_nvidia": {
        "scene": "nvidia_Balloon1",
        "pattern": "nvidia_Balloon1_tgt_*.png",
        "output_dir": (
            "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_"
            "egoexo_mono_v4-3_temb_20k_4gpu_hs_nvidia_v4-3_step20000"
        ),
        "scene_count": 7,
    },
    "hs_tum": {
        "scene": "tum_rgbd_dataset_freiburg2_desk_with_person",
        "pattern": "tum_rgbd_dataset_freiburg2_desk_with_person_tgt_*.png",
        "output_dir": (
            "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_"
            "egoexo_mono_v4-3_temb_20k_4gpu_hs_tum_v4-3_step20000"
        ),
        "scene_count": 9,
    },
    "mg_adt": {
        "scene": "adt_Apartment_release_multiuser_cook_seq141_M1292",
        "pattern": "adt_Apartment_release_multiuser_cook_seq141_M1292_tgt_*.png",
        "output_dir": (
            "exp_intrinsic_test_davisK_no_cube_l40s_spring_allfalse_"
            "egoexo_mono_v4-3_temb_20k_4gpu_mg_adt_v4-3_step20000"
        ),
        "scene_count": 4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--false-source-root", type=Path, required=True)
    parser.add_argument("--true-source-root", type=Path, required=True)
    parser.add_argument("--false-results", type=Path, required=True)
    parser.add_argument("--true-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=RESULTS_JSON",
        help="Add a force=true model to the cross-model comparison.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def newest_source(root: Path, output_dir: str, pattern: str) -> Path:
    candidates = list((root / output_dir).glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No {pattern} in {root / output_dir}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def write_table(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_model_comparison(values: list[str], output_root: Path) -> None:
    if not values:
        return
    rows = []
    for value in values:
        name, raw_path = value.split("=", 1)
        results = json.loads(Path(raw_path).read_text())
        for dataset in DATASETS:
            step = max(results[dataset], key=int)
            metrics = results[dataset][step]
            rows.append(
                {
                    "dataset": dataset,
                    "model": name,
                    "step": int(step),
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "lpips": metrics["lpips"],
                }
            )
    for dataset in DATASETS:
        selected = [row for row in rows if row["dataset"] == dataset]
        for rank, row in enumerate(sorted(selected, key=lambda item: -float(item["psnr"])), 1):
            row["psnr_rank"] = rank
        for rank, row in enumerate(sorted(selected, key=lambda item: -float(item["ssim"])), 1):
            row["ssim_rank"] = rank
        for rank, row in enumerate(sorted(selected, key=lambda item: float(item["lpips"])), 1):
            row["lpips_rank"] = rank
    rows.sort(key=lambda row: (str(row["dataset"]), int(row["psnr_rank"])))
    (output_root / "model_comparison_force_true.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    write_table(rows, output_root / "model_comparison_force_true.csv")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    conditions = {
        "force_false": args.false_source_root,
        "force_true": args.true_source_root,
    }
    manifest = []
    for condition, source_root in conditions.items():
        for dataset, spec in DATASETS.items():
            source = newest_source(source_root, str(spec["output_dir"]), str(spec["pattern"]))
            frames, metadata = split_official_comparison(source)
            output = args.output_root / condition / f"{dataset}.mp4"
            if args.overwrite or not output.exists():
                write_video(frames, output, args.fps)
            manifest.append(
                {
                    "condition": condition,
                    "dataset": dataset,
                    "scene": spec["scene"],
                    "source_png": str(source.resolve()),
                    "output_video": str(output.resolve()),
                    "fps": args.fps,
                    "frame_count": metadata["frame_count"],
                    "target_psnr_from_filename": target_psnr(source),
                    "diff_validation_max": metadata["diff_validation_max"],
                }
            )
            print(f"{condition}/{dataset}: {len(frames)} frames -> {output}")

    false_results = json.loads(args.false_results.read_text())
    true_results = json.loads(args.true_results.read_text())
    comparison = []
    for dataset, spec in DATASETS.items():
        false = false_results[dataset]["20000"]
        true = true_results[dataset]["20000"]
        comparison.append(
            {
                "dataset": dataset,
                "scene_count": spec["scene_count"],
                "psnr_false": false["psnr"],
                "psnr_true": true["psnr"],
                "delta_psnr": true["psnr"] - false["psnr"],
                "ssim_false": false["ssim"],
                "ssim_true": true["ssim"],
                "delta_ssim": true["ssim"] - false["ssim"],
                "lpips_false": false["lpips"],
                "lpips_true": true["lpips"],
                "delta_lpips": true["lpips"] - false["lpips"],
            }
        )

    (args.output_root / "video_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    write_table(manifest, args.output_root / "video_manifest.csv")
    (args.output_root / "force_intrinsics_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )
    write_table(comparison, args.output_root / "force_intrinsics_comparison.csv")
    write_model_comparison(args.model, args.output_root)
    readme = """# v4-3 TEmb 20k: non-iPhone force-intrinsics comparison

- Official evaluation datasets: NVIDIA (7 scenes), TUM (9 scenes), ADT (4 scenes)
- Videos use the same representative scenes as the prior all-model bundle.
- Each video is reconstructed from the official evaluator's `GT | Pred | Diff` PNG.
- Metrics are full-dataset official Target averages from the evaluation logs.
- Positive PSNR/SSIM deltas and negative LPIPS deltas indicate improvement.
"""
    (args.output_root / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
