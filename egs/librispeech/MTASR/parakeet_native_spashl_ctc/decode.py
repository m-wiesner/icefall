#!/usr/bin/env python3
"""Decode a native-resolution Parakeet SPaSHL checkpoint."""

import argparse
import runpy
import sys

METHODS = {
    "greedy": "wavlm_spashl_ctc.decode_greedy",
    "hlg": "wavlm_spashl_ctc.decode_hlg_algorithm",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="greedy")
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
    from train import configure_parakeet_encoder

    configure_parakeet_encoder(
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
        # The parallel adapter and fused output retain the convolutional tap's
        # 50 Hz resolution, even though the Parakeet backbone itself is 12.5 Hz.
        remainder.extend(("--frame-duration", "0.02"))
    sys.argv = [METHODS[args.method], *remainder]
    runpy.run_module(METHODS[args.method], run_name="__main__")


if __name__ == "__main__":
    main()
