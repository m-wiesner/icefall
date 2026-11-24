"""From-scratch waveform Zipformer encoder based on the Icefall SSL recipe."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

_icefall_root = Path(__file__).resolve().parents[4]
_ssl_zipformer_dir = _icefall_root / "egs" / "librispeech" / "SSL" / "zipformer"
sys.path.insert(0, str(_ssl_zipformer_dir))

from scaling import ScheduledFloat  # noqa: E402
from wav2vec2_module import ConvFeatureExtractionModel  # noqa: E402
from zipformer import Zipformer2  # noqa: E402


def int_tuple(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(item) for item in value.split(","))
    return tuple(value)


class ZipformerSslEncoder(nn.Module):
    """The SSL convolutional frontend and Zipformer, without the SSL objective."""

    def __init__(
        self,
        conv_feature_layers: str,
        extractor_mode: str,
        conv_bias: bool,
        num_encoder_layers: Sequence[int],
        downsampling_factor: Sequence[int],
        feedforward_dim: Sequence[int],
        num_heads: Sequence[int],
        encoder_dim: Sequence[int],
        encoder_unmasked_dim: Sequence[int],
        query_head_dim: Sequence[int],
        value_head_dim: Sequence[int],
        pos_head_dim: Sequence[int],
        pos_dim: int,
        cnn_module_kernel: Sequence[int],
        dropout_start: float = 0.3,
        dropout_end: float = 0.1,
        warmup_updates: float = 4000.0,
    ) -> None:
        super().__init__()
        # This is the same trusted recipe string accepted by the upstream SSL
        # implementation (a list of dim/kernel/stride tuples).
        feature_layers = eval(conv_feature_layers, {"__builtins__": {}}, {})
        self.feature_layers = tuple(tuple(int(x) for x in layer) for layer in feature_layers)
        feature_dim = self.feature_layers[-1][0]
        encoder_dim = tuple(encoder_dim)
        self.output_dim = max(encoder_dim)

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=self.feature_layers,
            dropout=0.0,
            mode=extractor_mode,
            conv_bias=conv_bias,
        )
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.input_projection = (
            nn.Linear(feature_dim, encoder_dim[0])
            if feature_dim != encoder_dim[0]
            else nn.Identity()
        )
        self.encoder = Zipformer2(
            output_downsampling_factor=1,
            downsampling_factor=tuple(downsampling_factor),
            num_encoder_layers=tuple(num_encoder_layers),
            encoder_dim=encoder_dim,
            encoder_unmasked_dim=tuple(encoder_unmasked_dim),
            query_head_dim=tuple(query_head_dim),
            pos_head_dim=tuple(pos_head_dim),
            value_head_dim=tuple(value_head_dim),
            pos_dim=pos_dim,
            num_heads=tuple(num_heads),
            feedforward_dim=tuple(feedforward_dim),
            cnn_module_kernel=tuple(cnn_module_kernel),
            dropout=ScheduledFloat((0.0, dropout_start), (20000.0, dropout_end)),
            warmup_batches=warmup_updates,
        )

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths.to(torch.int64)
        for _, kernel, stride in self.feature_layers:
            lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
        return lengths.clamp_min(0)

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor):
        if waveforms.ndim == 3:
            if waveforms.size(-1) != 1:
                raise ValueError(f"Expected mono waveform input, got {waveforms.shape}")
            waveforms = waveforms.squeeze(-1)
        if waveforms.ndim != 2:
            raise ValueError(f"Expected (batch, samples), got {waveforms.shape}")

        states = self.feature_extractor(waveforms).transpose(1, 2)
        states = self.input_projection(self.layer_norm(states))
        output_lengths = self.output_lengths(lengths).to(states.device)
        states, output_lengths = self.encoder(
            states.transpose(0, 1), output_lengths
        )
        return states.transpose(0, 1), output_lengths.to(torch.int32)
