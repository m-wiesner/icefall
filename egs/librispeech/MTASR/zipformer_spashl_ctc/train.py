#!/usr/bin/env python3
"""Train a waveform Zipformer from scratch with token and speaker CTC heads."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch.nn as nn

_recipe_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_recipe_root))
sys.path.insert(0, str(_recipe_root / "wavlm_spashl_ctc"))

import wavlm_spashl_ctc.train as base
from icefall.utils import AttributeDict
from wavlm_spashl_ctc.train import *  # noqa: F401,F403
from zipformer_spashl_ctc.encoder import ZipformerSslEncoder, int_tuple
from zipformer_spashl_ctc.model import MDCTCModel


_base_get_parser = base.get_parser


def get_parser():
    parser = _base_get_parser()
    group = parser.add_argument_group("From-scratch SSL Zipformer encoder")
    group.add_argument("--zipformer-num-encoder-layers", default="2,2,3,4,3,2")
    group.add_argument("--zipformer-downsampling-factor", default="1,2,4,8,4,2")
    group.add_argument(
        "--zipformer-feedforward-dim", default="512,768,1024,1536,1024,768"
    )
    group.add_argument("--zipformer-num-heads", default="4,4,4,8,4,4")
    group.add_argument("--zipformer-encoder-dim", default="192,256,384,512,384,256")
    group.add_argument(
        "--zipformer-encoder-unmasked-dim", default="192,192,256,256,256,192"
    )
    group.add_argument("--zipformer-query-head-dim", default="32")
    group.add_argument("--zipformer-value-head-dim", default="12")
    group.add_argument("--zipformer-pos-head-dim", default="4")
    group.add_argument("--zipformer-pos-dim", type=int, default=48)
    group.add_argument(
        "--zipformer-cnn-module-kernel", default="31,31,15,15,15,31"
    )
    group.add_argument(
        "--zipformer-conv-feature-layers",
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
    )
    group.add_argument(
        "--zipformer-extractor-mode",
        choices=("default", "layer_norm"),
        default="default",
    )
    group.add_argument("--zipformer-conv-bias", type=str2bool, default=False)
    group.add_argument("--zipformer-dropout-start", type=float, default=0.3)
    group.add_argument("--zipformer-dropout-end", type=float, default=0.1)
    group.add_argument("--zipformer-warmup-updates", type=float, default=4000.0)
    # There is no pretrained backbone to freeze.
    parser.set_defaults(freeze_iters=0, downsample=False, use_layer_norm=False)
    return parser


def get_encoder_model(params: AttributeDict) -> tuple[nn.Module, int]:
    encoder = ZipformerSslEncoder(
        conv_feature_layers=params.zipformer_conv_feature_layers,
        extractor_mode=params.zipformer_extractor_mode,
        conv_bias=params.zipformer_conv_bias,
        num_encoder_layers=int_tuple(params.zipformer_num_encoder_layers),
        downsampling_factor=int_tuple(params.zipformer_downsampling_factor),
        feedforward_dim=int_tuple(params.zipformer_feedforward_dim),
        num_heads=int_tuple(params.zipformer_num_heads),
        encoder_dim=int_tuple(params.zipformer_encoder_dim),
        encoder_unmasked_dim=int_tuple(params.zipformer_encoder_unmasked_dim),
        query_head_dim=int_tuple(params.zipformer_query_head_dim),
        value_head_dim=int_tuple(params.zipformer_value_head_dim),
        pos_head_dim=int_tuple(params.zipformer_pos_head_dim),
        pos_dim=params.zipformer_pos_dim,
        cnn_module_kernel=int_tuple(params.zipformer_cnn_module_kernel),
        dropout_start=params.zipformer_dropout_start,
        dropout_end=params.zipformer_dropout_end,
        warmup_updates=params.zipformer_warmup_updates,
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
    encoder_parameters = sum(p.numel() for p in encoder.parameters())
    head_parameters = sum(
        p.numel()
        for name, p in model.named_parameters()
        if name.startswith(("ctc_output1.", "ctc_output2.", "layer_norm."))
    )
    total_parameters = sum(p.numel() for p in model.parameters())
    logging.info(
        "Parameter breakdown: from-scratch SSL Zipformer=%s; SPaSHL heads=%s; total=%s",
        f"{encoder_parameters:,}",
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
