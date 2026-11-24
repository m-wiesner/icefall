#!/usr/bin/env python3
"""Align with a Parakeet speaker-attributed shuffle-CTC checkpoint."""

import argparse
import runpy

METHODS = {
    "graph": "wavlm_spashl_ctc.align_graph",
    "token": "wavlm_spashl_ctc.align_token",
    "oracle-mono": "wavlm_spashl_ctc.align_oracle_mono",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="token")
    parser.add_argument("--parakeet-model-id", default="nvidia/parakeet-tdt-0.6b-v3")
    parser.add_argument("--parakeet-revision")
    parser.add_argument("--encoder-upsample-factor", type=int, default=4)
    parser.add_argument("--encoder-upsample-mode", choices=("linear", "nearest"), default="linear")
    parser.add_argument("--speaker-layer", type=int, default=12)
    args, remainder = parser.parse_known_args()
    import sys
    from train import configure_parakeet_encoder
    configure_parakeet_encoder(
        model_id=args.parakeet_model_id,
        revision=args.parakeet_revision,
        upsample_factor=args.encoder_upsample_factor,
        upsample_mode=args.encoder_upsample_mode,
        speaker_layer=args.speaker_layer,
    )
    if remainder[:1] == ["--"]:
        remainder = remainder[1:]
    if "--frame-duration" not in remainder:
        remainder.extend(("--frame-duration", "0.02"))
    sys.argv = [METHODS[args.method], *remainder]
    runpy.run_module(METHODS[args.method], run_name="__main__")


if __name__ == "__main__":
    main()
