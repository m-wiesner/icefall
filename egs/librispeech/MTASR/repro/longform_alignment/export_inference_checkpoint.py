#!/usr/bin/env python3
"""Strip optimizer/sampler state from a trusted icefall training checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    if args.destination.exists():
        raise FileExistsError(f"Refusing to overwrite {args.destination}")
    checkpoint = torch.load(
        args.source, map_location="cpu", weights_only=False, mmap=True
    )
    if "model" not in checkpoint:
        raise KeyError(f"{args.source} does not contain a model state dict")

    # average_checkpoints(), used by decode/alignment with avg=1, requires only
    # the `model` entry. Preserve compact scalar provenance useful to humans.
    release = {
        "model": checkpoint["model"],
        "source_checkpoint": args.source.name,
        "batch_idx_train": checkpoint.get("batch_idx_train", 72000),
        "release_format": "icefall-model-state-v1",
    }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release, args.destination)
    print(args.destination)


if __name__ == "__main__":
    main()
