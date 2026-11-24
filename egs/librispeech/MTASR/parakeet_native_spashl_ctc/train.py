#!/usr/bin/env python3
"""Train high-resolution adapter Parakeet with SPaSHL token/speaker heads."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch.nn as nn

_recipe_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_recipe_root))
sys.path.insert(0, str(_recipe_root / "wavlm_spashl_ctc"))

import wavlm_spashl_ctc.train as base
from wavlm_spashl_ctc.train import *  # noqa: F401,F403
from icefall.utils import AttributeDict
from parakeet_native_spashl_ctc.encoder import ParakeetEncoder
from parakeet_native_spashl_ctc.model import MDCTCModel


_base_get_parser = base.get_parser
_encoder_overrides = {}


def configure_parakeet_encoder(**kwargs) -> None:
    """Supply encoder identity when reusing WavLM decode/alignment programs."""
    _encoder_overrides.update(
        {key: value for key, value in kwargs.items() if value is not None}
    )


def _encoder_option(params: AttributeDict, name: str, default):
    return _encoder_overrides.get(name, getattr(params, name, default))


def get_parser():
    parser = _base_get_parser()
    group = parser.add_argument_group("Native Parakeet encoder")
    group.add_argument(
        "--parakeet-model-id", default="nvidia/parakeet-tdt-0.6b-v3"
    )
    group.add_argument("--parakeet-revision", default=None)
    group.add_argument("--adapter-dim", type=int, default=256)
    group.add_argument("--adapter-num-heads", type=int, default=4)
    group.add_argument("--adapter-ff-dim", type=int, default=1024)
    group.add_argument("--adapter-layer-indices", type=str, default="7,15,23")
    group.add_argument("--adapter-local-attention-window", type=int, default=257)
    group.add_argument("--adapter-cross-attention-window", type=int, default=65)
    group.add_argument("--adapter-attention-block-size", type=int, default=256)
    group.add_argument("--adapter-dropout", type=float, default=0.1)
    return parser


def get_encoder_model(params: AttributeDict) -> tuple[nn.Module, int]:
    encoder = ParakeetEncoder.from_pretrained(
        model_id=_encoder_overrides.get(
            "model_id",
            getattr(params, "parakeet_model_id", "nvidia/parakeet-tdt-0.6b-v3"),
        ),
        revision=_encoder_overrides.get(
            "revision", getattr(params, "parakeet_revision", None)
        ),
        adapter_dim=_encoder_option(params, "adapter_dim", 256),
        adapter_num_heads=_encoder_option(params, "adapter_num_heads", 4),
        adapter_ff_dim=_encoder_option(params, "adapter_ff_dim", 1024),
        adapter_layer_indices=tuple(
            int(index)
            for index in _encoder_option(
                params, "adapter_layer_indices", "7,15,23"
            ).split(",")
        ),
        adapter_local_attention_window=_encoder_option(
            params, "adapter_local_attention_window", 257
        ),
        adapter_cross_attention_window=_encoder_option(
            params, "adapter_cross_attention_window", 65
        ),
        adapter_attention_block_size=_encoder_option(
            params, "adapter_attention_block_size", 256
        ),
        adapter_dropout=_encoder_option(params, "adapter_dropout", 0.1),
    )
    return encoder, encoder.output_dim


def get_mdctc_model(params: AttributeDict) -> nn.Module:
    encoder, encoder_dim = get_encoder_model(params)
    model = MDCTCModel(
        encoder=encoder,
        encoder_dim=encoder_dim,
        vocab_size1=params.vocab_size,
        vocab_size2=params.max_num_spks,
        factorized_token_speaker_output=params.factorized_token_speaker_output,
        layer_norm=params.use_layer_norm,
        downsample=params.downsample,
    )
    pretrained_parameters = sum(
        parameter.numel()
        for module in (encoder.frontend, encoder.encoder)
        for parameter in module.parameters()
    )
    adapter_parameters = sum(
        parameter.numel()
        for name, parameter in encoder.named_parameters()
        if not name.startswith("frontend.") and not name.startswith("encoder.")
    )
    head_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("ctc_output1.")
        or name.startswith("ctc_output2.")
        or name.startswith("layer_norm.")
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    logging.info(
        "Parameter breakdown: pretrained Parakeet=%s; high-resolution "
        "adapter/fusion=%s; SPaSHL heads=%s; total=%s",
        f"{pretrained_parameters:,}",
        f"{adapter_parameters:,}",
        f"{head_parameters:,}",
        f"{total_parameters:,}",
    )
    return model


base.get_parser = get_parser
base.get_encoder_model = get_encoder_model
base.get_mdctc_model = get_mdctc_model
base.MDCTCModel = MDCTCModel


if __name__ == "__main__":
    base.main()
