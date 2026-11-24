#!/usr/bin/env python3
"""Graph-align a token-only Parakeet shuffle-CTC checkpoint."""

import argparse
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("graph",), default="graph")
    parser.add_argument("--parakeet-model-id", default="nvidia/parakeet-tdt-0.6b-v3")
    parser.add_argument("--parakeet-revision")
    parser.add_argument("--encoder-upsample-factor", type=int, default=4)
    parser.add_argument("--speaker-layer", type=int, default=12)
    args, remainder = parser.parse_known_args()

    from train import configure_parakeet_encoder

    configure_parakeet_encoder(
        model_id=args.parakeet_model_id,
        revision=args.parakeet_revision,
        upsample_factor=args.encoder_upsample_factor,
        speaker_layer=args.speaker_layer,
    )
    if remainder[:1] == ["--"]:
        remainder = remainder[1:]
    if "--frame-duration" not in remainder:
        remainder.extend(("--frame-duration", "0.02"))
    sys.argv = ["wavlm_shuffle_ctc.align_graph", *remainder]
    runpy.run_module("wavlm_shuffle_ctc.align_graph", run_name="__main__")


if __name__ == "__main__":
    main()
