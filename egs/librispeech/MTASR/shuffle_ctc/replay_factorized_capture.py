#!/usr/bin/env python3
"""Replay a captured factorized dense intersection without model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from shuffle_ctc.factorized import _load_factorized_extension


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    saved = torch.load(args.capture, map_location="cpu", weights_only=False)
    if saved.get("format_version") != 1:
        raise ValueError(f"Unsupported capture format: {saved.get('format_version')}")
    extension = _load_factorized_extension()
    tensor_keys = (
        "arc_row_splits1",
        "arc_row_splits2",
        "arc_values",
        "token_row_splits",
        "token_scores",
        "speaker_scores",
    )
    inputs = {key: saved[key].to(args.device) for key in tensor_keys}
    torch.cuda.synchronize()
    for iteration in range(args.repeat):
        output = extension.intersect_dense_factorized(
            inputs["arc_row_splits1"],
            inputs["arc_row_splits2"],
            inputs["arc_values"],
            inputs["token_row_splits"],
            inputs["token_scores"],
            inputs["speaker_scores"],
            saved["token_vocab_size"],
            saved["output_beam"],
            saved["max_states"],
            saved["max_arcs"],
        )
        torch.cuda.synchronize()
        fsa, map_a, map_token, map_speaker = output
        print(
            f"iteration={iteration + 1} arcs={fsa.shape[0]} "
            f"map_a={map_a.numel()} map_token={map_token.numel()} "
            f"map_speaker={map_speaker.numel()}"
        )


if __name__ == "__main__":
    main()
