#!/usr/bin/env python3
"""Decode a Whisper-encoder SPaSHL checkpoint."""

import argparse
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("greedy",), default="greedy")
    parser.add_argument(
        "--whisper-model-id", default="openai/whisper-large-v3-turbo"
    )
    parser.add_argument("--whisper-revision")
    args, remainder = parser.parse_known_args()

    from train import configure_whisper_encoder

    configure_whisper_encoder(
        model_id=args.whisper_model_id,
        revision=args.whisper_revision,
    )
    if remainder[:1] == ["--"]:
        remainder = remainder[1:]
    if "--frame-duration" not in remainder:
        remainder.extend(("--frame-duration", "0.02"))
    sys.argv = ["wavlm_spashl_ctc.decode_greedy", *remainder]
    runpy.run_module("wavlm_spashl_ctc.decode_greedy", run_name="__main__")


if __name__ == "__main__":
    main()
