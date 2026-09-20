#!/usr/bin/env python3
"""Verify exported SuperSplat Gaussian sequence directories and metadata."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_serializer():
    path = REPOSITORY_ROOT / "src/misc/gaussian_sequence.py"
    spec = importlib.util.spec_from_file_location("_c4g_gaussian_sequence", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    serializer = load_serializer()
    reports = []
    failures = []

    metadata_paths = sorted(root.rglob("gaussian_sequence/metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No gaussian_sequence/metadata.json below {root}")

    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        sequence_dir = metadata_path.parent
        expected = metadata["frame_filenames"]
        actual = sorted(path.name for path in sequence_dir.glob("frame_*.ply"))
        checks = {
            "sequence_dir": str(sequence_dir),
            "num_frames": metadata["num_frames"],
            "expected_filenames_match": actual == expected,
            "timestamps_match_num_frames": len(metadata["timestamps"]) == len(expected),
            "counts_match_num_frames": (
                len(metadata["num_gaussians_per_frame"]) == len(expected)
            ),
            "frame_checks": [],
        }
        if actual != expected:
            failures.append(f"{sequence_dir}: filename/order mismatch")
        for index, filename in enumerate(expected):
            path = sequence_dir / filename
            try:
                result = serializer.validate_3dgs_ply(path)
                source_validation = metadata["frames"][index].get("validation", {})
                roundtrip = source_validation.get("roundtrip_matches_model")
                frame_check = {
                    "filename": filename,
                    "num_gaussians": result["num_gaussians"],
                    "finite": result["finite"],
                    "standard_properties": True,
                    "source_roundtrip_matches_model": roundtrip,
                    "size_bytes": path.stat().st_size,
                }
                if result["num_gaussians"] != metadata["num_gaussians_per_frame"][index]:
                    failures.append(f"{path}: Gaussian count mismatch")
                if roundtrip is not True:
                    failures.append(f"{path}: source round-trip validation missing/failed")
                checks["frame_checks"].append(frame_check)
            except Exception as error:
                failures.append(f"{path}: {error}")
        reports.append(checks)
        print(
            f"{sequence_dir}: {len(actual)} frames, "
            f"{sum(item['size_bytes'] for item in checks['frame_checks']) / 1024**2:.1f} MiB"
        )

    report = {
        "root": str(root),
        "num_sequences": len(reports),
        "passed": not failures,
        "failures": failures,
        "sequences": reports,
    }
    if args.write_report:
        output = root / "verification_report.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {output}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"[OK] verified {len(reports)} Gaussian sequences")


if __name__ == "__main__":
    main()
