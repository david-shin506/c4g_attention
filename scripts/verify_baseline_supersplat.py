#!/usr/bin/env python3
"""Exhaustively verify exported baseline 3DGS PLY sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED = {
    "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1",
    "rot_2", "rot_3",
}


def inspect_ply(path: Path, check_finite: bool) -> dict:
    header = []
    count = None
    properties = []
    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"Missing end_header: {path}")
            line = raw.decode("ascii").rstrip("\n")
            header.append(line)
            tokens = line.split()
            if tokens[:2] == ["element", "vertex"]:
                count = int(tokens[2])
            elif tokens[:2] == ["property", "float"]:
                properties.append(tokens[2])
            if line == "end_header":
                offset = handle.tell()
                break
    if header[:2] != ["ply", "format binary_little_endian 1.0"]:
        raise ValueError(f"Nonstandard PLY format: {path}")
    if count is None or count <= 0:
        raise ValueError(f"Invalid vertex count: {path}")
    missing = REQUIRED - set(properties)
    if missing:
        raise ValueError(f"Missing properties {sorted(missing)}: {path}")
    expected_size = offset + count * len(properties) * 4
    if path.stat().st_size != expected_size:
        raise ValueError(f"Payload size mismatch: {path}")
    if check_finite:
        payload = np.memmap(path, mode="r", dtype="<f4", offset=offset, shape=(count, len(properties)))
        if not np.isfinite(payload).all():
            raise ValueError(f"NaN/Inf payload: {path}")
        del payload
    return {
        "num_gaussians": count,
        "properties": properties,
        "size_bytes": path.stat().st_size,
        "header": header,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--skip-finite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = {"root": str(root), "models": {}}
    examples = {}
    total_files = total_bytes = total_gaussians = 0

    # Final bundle layout: MODEL/iphone_SCENE/gaussian_sequence/frame_XXXX.ply.
    metadata_paths = sorted(root.glob("*/iphone_*/metadata.json"))
    if not metadata_paths:
        raise ValueError(f"No scene metadata files found below {root}")
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = metadata.get("model", metadata_path.parents[1].name)
        scene = metadata.get("scene", metadata_path.parent.name.removeprefix("iphone_"))
        sequence_dir = metadata_path.parent / "gaussian_sequence"
        all_entries = sorted(sequence_dir.iterdir())
        if any(p.suffix != ".ply" for p in all_entries):
            raise ValueError(f"Non-PLY file inside sequence directory: {sequence_dir}")
        files = sorted(sequence_dir.glob("frame_*.ply"))
        expected_names = [f"frame_{i:04d}.ply" for i in range(metadata["num_frames"])]
        if [p.name for p in files] != expected_names:
            raise ValueError(f"Filename ordering mismatch: {sequence_dir}")
        if len(files) != len(metadata["num_gaussians_per_frame"]):
            raise ValueError(f"Metadata frame count mismatch: {sequence_dir}")

        counts, sizes = [], []
        for index, path in enumerate(files):
            info = inspect_ply(path, check_finite=not args.skip_finite)
            if info["num_gaussians"] != metadata["num_gaussians_per_frame"][index]:
                raise ValueError(f"Vertex count differs from metadata: {path}")
            counts.append(info["num_gaussians"])
            sizes.append(info["size_bytes"])
            if model not in examples:
                examples[model] = {"path": str(path.relative_to(root)), "header": info["header"]}
            total_files += 1
            total_gaussians += info["num_gaussians"]
            total_bytes += info["size_bytes"]
        model_entry = manifest["models"].setdefault(model, {"scenes": {}})
        model_entry["scenes"][scene] = {
            "num_frames": len(files),
            "timestamps": metadata["timestamps"],
            "num_gaussians_min": min(counts),
            "num_gaussians_max": max(counts),
            "num_gaussians_total": sum(counts),
            "size_bytes": sum(sizes),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "settings": metadata.get("official_settings", {
                "source": metadata.get("source"),
                "checkpoint_step": metadata.get("checkpoint_step"),
            }),
        }

    for model_entry in manifest["models"].values():
        scenes = model_entry["scenes"].values()
        model_entry["num_frames"] = sum(s["num_frames"] for s in scenes)
        model_entry["num_gaussians_total"] = sum(s["num_gaussians_total"] for s in scenes)
        model_entry["size_bytes"] = sum(s["size_bytes"] for s in scenes)
    manifest.update({
        "num_models": len(manifest["models"]),
        "num_scenes": len(metadata_paths),
        "num_ply_files": total_files,
        "num_gaussians_total": total_gaussians,
        "size_bytes": total_bytes,
        "sequence_directories_contain_only_ply": True,
        "lexical_order_matches_temporal_order": True,
    })
    report = {
        "status": "passed",
        "checks": {
            "binary_little_endian_3dgs_header": True,
            "required_properties_present": True,
            "filename_ordering": True,
            "metadata_frame_and_vertex_counts": True,
            "payload_file_sizes": True,
            "all_payload_values_finite": not args.skip_finite,
            "quaternion_order": "wxyz (native renderer and INRIA convention)",
        },
        "totals": {
            "models": len(manifest["models"]), "scenes": len(metadata_paths),
            "ply_files": total_files, "gaussians": total_gaussians,
            "bytes": total_bytes,
        },
        "header_examples": examples,
    }
    (root / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "verification_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["totals"], indent=2))


if __name__ == "__main__":
    main()
