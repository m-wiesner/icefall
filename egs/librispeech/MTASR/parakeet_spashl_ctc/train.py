#!/usr/bin/env python3
"""Train speaker-attributed shuffle CTC with a Parakeet TDT v3 encoder."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch.nn as nn

_recipe_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_recipe_root))
# The inherited icefall recipe still uses sibling-style imports internally.
sys.path.insert(0, str(_recipe_root / "wavlm_spashl_ctc"))

import wavlm_spashl_ctc.train as base
from wavlm_spashl_ctc.train import *  # noqa: F401,F403
from icefall.utils import AttributeDict
# Direct execution needs this directory ahead of the inherited recipe for the
# local model/encoder fallback imports below.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from .model import MDCTCModel
    from .parakeet_encoder import ParakeetWaveformEncoder
except ImportError:  # Executed directly as ``python parakeet_spashl_ctc/train.py``.
    from parakeet_spashl_ctc.model import MDCTCModel
    from parakeet_spashl_ctc.parakeet_encoder import ParakeetWaveformEncoder


_base_get_parser = base.get_parser
_encoder_overrides = {}


def configure_parakeet_encoder(**kwargs) -> None:
    """Set encoder options for reused decode/alignment programs."""
    _encoder_overrides.update({key: value for key, value in kwargs.items() if value is not None})


def get_parser():
    parser = _base_get_parser()
    group = parser.add_argument_group("Parakeet encoder")
    group.add_argument("--parakeet-model-id", default="nvidia/parakeet-tdt-0.6b-v3")
    group.add_argument("--parakeet-revision", default=None)
    group.add_argument(
        "--parakeet-random-init",
        type=str2bool,
        default=False,
        help="Construct Parakeet from its config without loading pretrained weights.",
    )
    group.add_argument("--encoder-upsample-factor", type=int, default=4)
    group.add_argument(
        "--high-resolution-fusion",
        type=str2bool,
        default=False,
        help="Fuse a residual recovery path tapped at the 20-ms subsampler layer.",
    )
    group.add_argument("--recovery-adapter-blocks", type=int, default=2)
    group.add_argument(
        "--encoder-upsample-mode", choices=("linear", "nearest"), default="linear"
    )
    group.add_argument(
        "--speaker-layer",
        type=int,
        default=12,
        help="One-based FastConformer block used by the speaker branch.",
    )
    group.add_argument(
        "--encoder-layerdrop",
        type=float,
        default=0.0,
        help="Encoder block-drop probability; zero keeps intermediate taps stable.",
    )
    group.add_argument(
        "--parakeet-convolution-norm",
        choices=("masked_batch_norm", "layer_norm"),
        default="masked_batch_norm",
        help="Normalization used inside each FastConformer convolution block.",
    )
    group.add_argument(
        "--parakeet-finite-diagnostics",
        type=str2bool,
        default=False,
        help="Check waveform, frontend, convolution, normalization, and block outputs.",
    )
    return parser


def get_encoder_model(params: AttributeDict) -> tuple[nn.Module, int]:
    model_id = _encoder_overrides.get(
        "model_id", getattr(params, "parakeet_model_id", "nvidia/parakeet-tdt-0.6b-v3")
    )
    revision = _encoder_overrides.get(
        "revision", getattr(params, "parakeet_revision", None)
    )
    upsample_factor = _encoder_overrides.get(
        "upsample_factor", getattr(params, "encoder_upsample_factor", 4)
    )
    upsample_mode = _encoder_overrides.get(
        "upsample_mode", getattr(params, "encoder_upsample_mode", "linear")
    )
    speaker_layer = _encoder_overrides.get(
        "speaker_layer", getattr(params, "speaker_layer", 12)
    )
    encoder_layerdrop = _encoder_overrides.get(
        "encoder_layerdrop", getattr(params, "encoder_layerdrop", 0.0)
    )
    convolution_norm = _encoder_overrides.get(
        "convolution_norm",
        getattr(params, "parakeet_convolution_norm", "masked_batch_norm"),
    )
    finite_diagnostics = _encoder_overrides.get(
        "finite_diagnostics", getattr(params, "parakeet_finite_diagnostics", False)
    )
    random_init = _encoder_overrides.get(
        "random_init", getattr(params, "parakeet_random_init", False)
    )
    encoder = ParakeetWaveformEncoder.from_pretrained(
        model_id=model_id,
        revision=revision,
        upsample_factor=upsample_factor,
        upsample_mode=upsample_mode,
        speaker_layer=speaker_layer,
        encoder_layerdrop=encoder_layerdrop,
        convolution_norm=convolution_norm,
        finite_diagnostics=finite_diagnostics,
        random_init=random_init,
    )
    return encoder, encoder.output_dim


def get_mdctc_model(params: AttributeDict) -> nn.Module:
    encoder, encoder_dim = get_encoder_model(params)
    return MDCTCModel(
        encoder=encoder,
        encoder_dim=encoder_dim,
        vocab_size1=params.vocab_size,
        vocab_size2=params.max_num_spks,
        factorized_token_speaker_output=params.factorized_token_speaker_output,
        layer_norm=params.use_layer_norm,
        downsample=params.downsample,
        speaker_branch=getattr(params, "speaker_branch", True),
        upsample_factor=getattr(params, "encoder_upsample_factor", 4),
        high_resolution_fusion=getattr(params, "high_resolution_fusion", False),
        recovery_adapter_blocks=getattr(params, "recovery_adapter_blocks", 2),
    )


base.get_parser = get_parser
base.get_encoder_model = get_encoder_model
base.get_mdctc_model = get_mdctc_model
base.MDCTCModel = MDCTCModel


if __name__ == "__main__":
    base.main()
