#!/usr/bin/env python3
"""Align with a Whisper-encoder SPaSHL checkpoint."""

import argparse
import importlib
import sys


METHODS = {
    "graph": "wavlm_spashl_ctc.align_graph",
    "token": "wavlm_spashl_ctc.align_token",
    "oracle-mono": "wavlm_spashl_ctc.align_oracle_mono",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="token")
    parser.add_argument(
        "--whisper-model-id", default="openai/whisper-large-v3-turbo"
    )
    parser.add_argument("--whisper-revision")
    args, remainder = parser.parse_known_args()

    from whisper_spashl_ctc import train as whisper_train

    sys.modules["train"] = whisper_train
    whisper_train.configure_whisper_encoder(
        model_id=args.whisper_model_id,
        revision=args.whisper_revision,
    )
    if remainder[:1] == ["--"]:
        remainder = remainder[1:]
    if "--frame-duration" not in remainder:
        remainder.extend(("--frame-duration", "0.02"))
    sys.argv = [METHODS[args.method], *remainder]
    target = importlib.import_module(METHODS[args.method])
    target.get_mdctc_model = whisper_train.get_mdctc_model
    target.main()


if __name__ == "__main__":
    main()
