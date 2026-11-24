#!/usr/bin/env python3
"""Align with a native-resolution Parakeet SPaSHL checkpoint."""

import argparse
import importlib
import sys

METHODS = {
    "graph": "wavlm_spashl_ctc.align_graph",
    "token": "wavlm_spashl_ctc.align_token",
    "oracle-mono": "wavlm_spashl_ctc.align_oracle_mono",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="token")
    parser.add_argument(
        "--parakeet-model-id", default="nvidia/parakeet-tdt-0.6b-v3"
    )
    parser.add_argument("--parakeet-revision")
    parser.add_argument("--adapter-dim", type=int, default=256)
    parser.add_argument("--adapter-num-heads", type=int, default=4)
    parser.add_argument("--adapter-ff-dim", type=int, default=1024)
    parser.add_argument("--adapter-layer-indices", default="7,15,23")
    parser.add_argument("--adapter-local-attention-window", type=int, default=257)
    parser.add_argument("--adapter-cross-attention-window", type=int, default=65)
    parser.add_argument("--adapter-attention-block-size", type=int, default=256)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    args, remainder = parser.parse_known_args()
    from parakeet_native_spashl_ctc import train as parakeet_train

    # The reused alignment module imports the recipe as top-level ``train``.
    # Pin that alias explicitly; otherwise its path manipulation can resolve
    # wavlm_spashl_ctc/train.py and silently construct the wrong encoder.
    sys.modules["train"] = parakeet_train

    parakeet_train.configure_parakeet_encoder(
        model_id=args.parakeet_model_id,
        revision=args.parakeet_revision,
        adapter_dim=args.adapter_dim,
        adapter_num_heads=args.adapter_num_heads,
        adapter_ff_dim=args.adapter_ff_dim,
        adapter_layer_indices=args.adapter_layer_indices,
        adapter_local_attention_window=args.adapter_local_attention_window,
        adapter_cross_attention_window=args.adapter_cross_attention_window,
        adapter_attention_block_size=args.adapter_attention_block_size,
        adapter_dropout=args.adapter_dropout,
    )
    if remainder[:1] == ["--"]:
        remainder = remainder[1:]
    if "--frame-duration" not in remainder:
        remainder.extend(("--frame-duration", "0.08"))
    sys.argv = [METHODS[args.method], *remainder]
    target = importlib.import_module(METHODS[args.method])
    # Unlike the decoding implementation, the legacy aligners define their
    # WavLM model factory locally. Replace it with the selected recipe factory.
    target.get_mdctc_model = parakeet_train.get_mdctc_model
    target.main()


if __name__ == "__main__":
    main()
