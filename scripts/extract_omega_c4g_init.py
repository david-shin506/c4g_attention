#!/usr/bin/env python3
"""Create minimal, weights-only initialization artifacts for the Omega C4G run."""

from argparse import ArgumentParser
from pathlib import Path

import torch


def load_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
        str(path),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no state dict: {path}")
    return {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if isinstance(key, str) and isinstance(value, torch.Tensor)
    }


def save_atomic(state: dict[str, torch.Tensor], output: Path, source: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(
        {
            "source": str(source),
            "state_dict": state,
        },
        temporary,
    )
    temporary.replace(output)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--head-source", type=Path, required=True)
    parser.add_argument("--omega-source", type=Path, required=True)
    parser.add_argument("--head-output", type=Path, required=True)
    parser.add_argument("--omega-output", type=Path, required=True)
    args = parser.parse_args()

    head = {
        key: value
        for key, value in load_state(args.head_source).items()
        if not key.startswith(("backbone.", "dpt_head."))
    }
    for required in ("gaussian_tokens", "gmae_decoder.", "gmae_to_gaussians."):
        if not any(key == required or key.startswith(required) for key in head):
            raise ValueError(f"Head source is missing {required!r}")

    omega_prefixes = ("backbone.aggregator.", "backbone.dense_head.")
    omega = {
        key: value
        for key, value in load_state(args.omega_source).items()
        if key.startswith(omega_prefixes)
    }
    for required in omega_prefixes:
        if not any(key.startswith(required) for key in omega):
            raise ValueError(f"Omega source is missing {required!r}")

    save_atomic(head, args.head_output, args.head_source)
    save_atomic(omega, args.omega_output, args.omega_source)
    print(f"head tensors: {len(head)} -> {args.head_output}")
    print(f"omega tensors: {len(omega)} -> {args.omega_output}")


if __name__ == "__main__":
    main()
