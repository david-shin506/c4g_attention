#!/usr/bin/env python3
"""Export one official-evaluation sample as a SuperSplat PLY sequence.

This script intentionally runs inference without the renderer: the encoder's
exact timestamp-indexed Gaussians are serialized before any visualization
filtering or coloring. It can load dataset/config code from a separate C4G
evaluation checkout while reusing this repository's PLY serializer.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import importlib.util
import os
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_ROOT = Path("/music-3d-shared-disk/user/KAIST/HG/c4g_mg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--comparison-checkpoint",
        type=Path,
        default=None,
        help="Optional second checkpoint evaluated on the exact same raw batch.",
    )
    parser.add_argument(
        "--comparison-output-root",
        type=Path,
        default=None,
        help="Output root paired with --comparison-checkpoint.",
    )
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--batch-file", type=Path, default=None)
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Scene to export. Repeat to export multiple scenes.",
    )
    parser.add_argument(
        "--all-scenes",
        action="store_true",
        help="Export every scene in the configured test dataset.",
    )
    parser.add_argument(
        "--clamp-zero-scales-for-eval",
        action="store_true",
        help=(
            "Evaluation-only: replace exact zero renderer scales with the smallest "
            "positive value of the same dtype so finite log-scales can be serialized."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stage",
        choices=("val", "test"),
        default="test",
        help="Dataset split to export (default: test, preserving previous behavior).",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Stop after this many exported scenes.",
    )
    return parser.parse_args()


def load_serializer():
    path = REPOSITORY_ROOT / "src/misc/gaussian_sequence.py"
    spec = importlib.util.spec_from_file_location("_c4g_gaussian_sequence", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def load_encoder_state(encoder, checkpoint_path: Path) -> int:
    load_kwargs = {"map_location": "cpu"}
    try:
        checkpoint = torch.load(str(checkpoint_path), mmap=True, **load_kwargs)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), **load_kwargs)
    state_dict = checkpoint.get("state_dict", checkpoint)
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(f"No encoder.* weights in {checkpoint_path}")
    result = encoder.load_state_dict(encoder_state, strict=False)
    if result.unexpected_keys:
        raise ValueError(
            f"Unexpected encoder checkpoint keys: {result.unexpected_keys[:20]}"
        )
    # Frozen/non-persistent helper modules can be absent, but learned model tensors cannot.
    material_missing = [
        key for key in result.missing_keys if not key.startswith(("dpt_head.",))
    ]
    if material_missing:
        raise ValueError(f"Missing encoder checkpoint keys: {material_missing[:20]}")
    step = int(checkpoint.get("global_step", 0))
    del encoder_state, state_dict, checkpoint
    gc.collect()
    return step


def clamp_zero_scales_for_eval(gaussians_per_timestamp):
    """Make exact-zero scales finitely log-serializable with a one-ULP clamp."""
    adjusted = {}
    counts: dict[str, int] = {}
    for timestamp, gaussians in gaussians_per_timestamp.items():
        scales = gaussians.scales
        if scales is None:
            raise ValueError("Gaussian scales are required for 3DGS export")
        if torch.any(scales < 0):
            raise ValueError(f"Negative renderer scale at timestamp {timestamp}")
        zero_mask = scales == 0
        count = int(zero_mask.sum())
        counts[str(timestamp)] = count
        if count:
            smallest_positive = torch.nextafter(
                torch.zeros((), dtype=scales.dtype, device=scales.device),
                torch.ones((), dtype=scales.dtype, device=scales.device),
            )
            scales = torch.where(zero_mask, smallest_positive, scales)
            gaussians = replace(gaussians, scales=scales)
        adjusted[timestamp] = gaussians
    return adjusted, counts


def main() -> None:
    args = parse_args()
    if args.batch_file is None and not args.scene and not args.all_scenes:
        raise ValueError("Provide --batch-file, at least one --scene, or --all-scenes")
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.batch_file is not None:
        args.batch_file = args.batch_file.resolve()
        if not args.batch_file.is_file():
            raise FileNotFoundError(args.batch_file)
    comparison_requested = (
        args.comparison_checkpoint is not None
        or args.comparison_output_root is not None
    )
    if comparison_requested and (
        args.comparison_checkpoint is None or args.comparison_output_root is None
    ):
        raise ValueError(
            "Use --comparison-checkpoint and --comparison-output-root together"
        )
    args.eval_root = args.eval_root.resolve()
    args.output_root = args.output_root.resolve()
    if comparison_requested:
        args.comparison_checkpoint = args.comparison_checkpoint.resolve()
        args.comparison_output_root = args.comparison_output_root.resolve()
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if comparison_requested and not args.comparison_checkpoint.is_file():
        raise FileNotFoundError(args.comparison_checkpoint)
    if not args.eval_root.is_dir():
        raise FileNotFoundError(args.eval_root)

    serializer = load_serializer()

    # The official evaluation checkout builds CUDA extensions from relative
    # submodule paths, so mirror its normal launch working directory.
    os.chdir(args.eval_root)

    # Import the official dataset/model implementation before importing any local src.
    sys.path.insert(0, str(args.eval_root))
    from src.config import load_typed_root_config
    from src.dataset import get_dataset
    from src.dataset.data_module import get_data_shim
    from src.global_cfg import set_cfg
    from src.model.encoder import get_encoder

    cfg_dict = OmegaConf.load(args.config)
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    seed = int(cfg_dict.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    checkpoint_specs = [(args.checkpoint, args.output_root)]
    if comparison_requested:
        checkpoint_specs.append(
            (args.comparison_checkpoint, args.comparison_output_root)
        )
    loaded_models = []
    for checkpoint_path, output_root in checkpoint_specs:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        encoder, _ = get_encoder(cfg.model.encoder)
        checkpoint_step = load_encoder_state(encoder, checkpoint_path)
        encoder = encoder.to(device).eval()
        encoder_forward_kwargs = {}
        if "render_feature" in inspect.signature(encoder.forward).parameters:
            encoder_forward_kwargs["render_feature"] = False
        loaded_models.append(
            (
                encoder,
                checkpoint_step,
                checkpoint_path,
                output_root,
                encoder_forward_kwargs,
            )
        )
    encoder = loaded_models[0][0]

    if args.batch_file is not None:
        loader = [torch.load(str(args.batch_file), map_location="cpu")]
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        datasets = get_dataset(cfg.dataset, args.stage, None)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        matching_datasets = [
            dataset for dataset in datasets if dataset.cfg.name == args.dataset_label
        ]
        if len(matching_datasets) != 1:
            available = [dataset.cfg.name for dataset in datasets]
            raise ValueError(
                f"Expected exactly one dataset named {args.dataset_label!r}; "
                f"available datasets: {available}"
            )
        loader = DataLoader(
            matching_datasets[0], batch_size=1, shuffle=False, num_workers=0
        )
    data_shim = get_data_shim(encoder)

    requested = set(args.scene)
    exported: set[str] = set()
    available: list[str] = []
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    for dataset_index, batch in enumerate(loader):
        source_sample = str(batch["scene"][0])
        available.append(source_sample)
        if requested and source_sample not in requested:
            continue
        evaluation_batch_path = args.batch_file
        if comparison_requested and evaluation_batch_path is None:
            evaluation_batch_path = (
                args.output_root
                / args.dataset_label
                / source_sample
                / "evaluation_batch.pt"
            )
            evaluation_batch_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(batch, evaluation_batch_path)
            print(
                f"[OK] saved exact evaluation batch -> {evaluation_batch_path}",
                flush=True,
            )
        batch = move_to_device(batch, device)
        batch = data_shim(batch)
        context_timestamps = torch.unique(batch["context"]["index"]).sort().values
        target_timestamps = torch.unique(batch["target"]["index"]).sort().values
        timestamps = (
            torch.unique(torch.cat([context_timestamps, target_timestamps]))
            .sort()
            .values
        )

        for (
            encoder,
            checkpoint_step,
            checkpoint_path,
            output_root,
            encoder_forward_kwargs,
        ) in loaded_models:
            with torch.inference_mode():
                gaussians_per_timestamp = encoder(
                    batch["context"],
                    checkpoint_step,
                    target_timestamps=timestamps,
                    visualization_dump=None,
                    **encoder_forward_kwargs,
                )
            zero_scale_clamp_counts: dict[str, int] = {}
            if args.clamp_zero_scales_for_eval:
                gaussians_per_timestamp, zero_scale_clamp_counts = (
                    clamp_zero_scales_for_eval(gaussians_per_timestamp)
                )

            sequence_dir = (
                output_root / args.dataset_label / source_sample / "gaussian_sequence"
            )
            metadata = serializer.export_gaussian_sequence(
                gaussians_per_timestamp,
                sequence_dir,
                source_sample=source_sample,
                checkpoint_step=checkpoint_step,
                extra_metadata={
                    "dataset_label": args.dataset_label,
                    "dataset_stage": args.stage,
                    "dataset_index": dataset_index,
                    "config": str(args.config),
                    "checkpoint": str(checkpoint_path),
                    "eval_root": str(args.eval_root),
                    "evaluation_batch": (
                        str(evaluation_batch_path)
                        if evaluation_batch_path is not None
                        else None
                    ),
                    "context_timestamps": context_timestamps.tolist(),
                    "target_timestamps": target_timestamps.tolist(),
                    "force_davis_intrinsics": bool(
                        cfg_dict.get("force_davis_intrinsics", False)
                    ),
                    "zero_scale_eval_clamp": {
                        "enabled": bool(args.clamp_zero_scales_for_eval),
                        "replacement": "smallest positive value representable in the source dtype",
                        "counts_by_timestamp": zero_scale_clamp_counts,
                        "total_count": sum(zero_scale_clamp_counts.values()),
                    },
                },
                validate=True,
            )
            print(
                f"[OK] {args.dataset_label}/{source_sample}: "
                f"{metadata['num_frames']} frames, "
                f"{metadata['num_gaussians_per_frame'][0]} Gaussians/frame -> "
                f"{sequence_dir}",
                flush=True,
            )
        exported.add(source_sample)
        del gaussians_per_timestamp, batch
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if requested and exported == requested:
            break
        if args.max_scenes is not None and len(exported) >= args.max_scenes:
            break

    missing = requested - exported
    if missing:
        raise ValueError(
            f"Scenes {sorted(missing)!r} were not found in {args.config}; "
            f"available samples: {available}"
        )
    if not exported:
        raise ValueError(f"No scenes were exported from {args.config}")


if __name__ == "__main__":
    main()
