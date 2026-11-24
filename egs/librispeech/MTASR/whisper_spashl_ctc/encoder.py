"""A raw-waveform wrapper around the fixed-context Whisper encoder."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WhisperAudioEncoder(nn.Module):
    """Convert 16-kHz waveforms to Whisper's 50-Hz encoder states.

    Whisper has learned positional embeddings for exactly 1500 encoder frames.
    We therefore pad every input to 30 seconds, but return lengths derived from
    the unpadded waveform so downstream graph losses ignore padded frames.
    """

    sample_rate = 16000
    n_fft = 400
    hop_length = 160
    max_audio_samples = 30 * sample_rate

    def __init__(
        self,
        encoder: nn.Module,
        mel_filters: torch.Tensor,
        max_source_positions: int = 1500,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.max_source_positions = int(max_source_positions)
        mel_filters = torch.as_tensor(mel_filters, dtype=torch.float32)
        if mel_filters.ndim != 2:
            raise ValueError(f"Expected 2-D Mel filters, got {mel_filters.shape}")
        # HF stores filters as (n_fft // 2 + 1, n_mels).
        if mel_filters.shape[0] != self.n_fft // 2 + 1:
            mel_filters = mel_filters.T
        if mel_filters.shape[0] != self.n_fft // 2 + 1:
            raise ValueError(f"Unexpected Mel filter shape: {mel_filters.shape}")
        self.register_buffer("mel_filters", mel_filters, persistent=False)
        self.register_buffer(
            "window", torch.hann_window(self.n_fft), persistent=False
        )
        self.output_dim = int(encoder.config.d_model)

    @classmethod
    def from_pretrained(
        cls, model_id: str, revision: Optional[str] = None
    ) -> "WhisperAudioEncoder":
        from transformers import WhisperFeatureExtractor, WhisperModel

        # The Hub config advertises float16 weights. Keep trainable master
        # parameters in float32: training autocast still executes eligible
        # operations in float16, while validation (which intentionally runs
        # without autocast) receives dtype-compatible parameters.
        model = WhisperModel.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch.float32,
        )
        extractor = WhisperFeatureExtractor.from_pretrained(
            model_id, revision=revision
        )
        return cls(
            encoder=model.encoder,
            mel_filters=torch.from_numpy(np.asarray(extractor.mel_filters)),
            max_source_positions=model.config.max_source_positions,
        )

    def _log_mel(self, waveforms: torch.Tensor) -> torch.Tensor:
        # Match WhisperFeatureExtractor._torch_extract_fbank_features.  Feature
        # extraction is intentionally float32 even under AMP.
        with torch.autocast(device_type=waveforms.device.type, enabled=False):
            stft = torch.stft(
                waveforms.float(),
                self.n_fft,
                self.hop_length,
                window=self.window.float(),
                return_complex=True,
            )
            magnitudes = stft[..., :-1].abs().square()
            mel_spec = self.mel_filters.T @ magnitudes
            log_spec = torch.clamp(mel_spec, min=1.0e-10).log10()
            log_spec = torch.maximum(
                log_spec,
                log_spec.amax(dim=(-2, -1), keepdim=True) - 8.0,
            )
            return (log_spec + 4.0) / 4.0

    def output_lengths(self, sample_lengths: torch.Tensor) -> torch.Tensor:
        mel_lengths = torch.div(
            sample_lengths, self.hop_length, rounding_mode="floor"
        )
        # Whisper conv1 has stride 1 and conv2 has stride 2, both with padding.
        lengths = torch.div(mel_lengths + 1, 2, rounding_mode="floor")
        return lengths.clamp(max=self.max_source_positions).to(torch.int32)

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor):
        if waveforms.ndim != 2:
            raise ValueError(f"Expected (batch, samples), got {waveforms.shape}")
        if torch.any(lengths > self.max_audio_samples):
            longest = int(lengths.max().item())
            raise ValueError(
                f"Whisper inputs must be <=30 s ({self.max_audio_samples} samples); "
                f"found {longest} samples ({longest / self.sample_rate:.3f} s)."
            )
        waveforms = waveforms[:, : self.max_audio_samples]
        waveforms = F.pad(
            waveforms, (0, self.max_audio_samples - waveforms.shape[-1])
        )
        input_features = self._log_mel(waveforms)
        expected_mel_frames = 2 * self.max_source_positions
        if input_features.shape[-1] != expected_mel_frames:
            raise RuntimeError(
                f"Expected {expected_mel_frames} Mel frames, got "
                f"{input_features.shape[-1]}"
            )
        states = self.encoder(
            input_features=input_features, return_dict=True
        ).last_hidden_state
        return states, self.output_lengths(lengths)

    def freeze_pretrained_backbone(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_pretrained_backbone(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True
