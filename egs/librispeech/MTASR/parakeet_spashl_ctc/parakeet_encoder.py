"""Hugging Face Parakeet waveform frontend and FastConformer adapter."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, ParakeetFeatureExtractor, ParakeetForTDT


def _tensor_diagnostic(name: str, value: torch.Tensor) -> str:
    x = value.detach().float()
    finite = torch.isfinite(x)
    num_finite = int(finite.sum().item())
    result = f"{name}: shape={tuple(x.shape)}, finite={num_finite}/{x.numel()}"
    if num_finite:
        valid = x[finite]
        result += (
            f", min={valid.min().item():.8g}, max={valid.max().item():.8g}, "
            f"mean={valid.mean().item():.8g}"
        )
    return result


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm over valid ``(batch, time)`` positions, accumulated in FP32."""

    @classmethod
    def from_batch_norm(cls, source: nn.BatchNorm1d) -> "MaskedBatchNorm1d":
        target = cls(
            source.num_features,
            eps=source.eps,
            momentum=source.momentum,
            affine=source.affine,
            track_running_stats=source.track_running_stats,
            device=source.weight.device if source.affine else source.running_mean.device,
            dtype=torch.float32,
        )
        target.load_state_dict(source.state_dict())
        target.train(source.training)
        return target

    def forward(
        self, input: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if valid_mask is None:
            # Explicitly leave autocast so BatchNorm and its buffer update use
            # FP32 even when the surrounding encoder runs in FP16.
            output = F.batch_norm(
                input.float(),
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                self.training or not self.track_running_stats,
                self.momentum if self.momentum is not None else 0.0,
                self.eps,
            )
            return output.to(input.dtype)

        if input.ndim != 3:
            raise ValueError(f"Expected BatchNorm input (B, C, T), got {input.shape}")
        mask = valid_mask.to(device=input.device, dtype=torch.bool)
        if mask.ndim != 2 or mask.shape != (input.size(0), input.size(2)):
            raise ValueError(
                f"Expected valid mask {(input.size(0), input.size(2))}, got {mask.shape}"
            )

        x = input.float()
        mask_f = mask[:, None, :].float()
        count = mask_f.sum()
        if count.item() == 0:
            raise ValueError("MaskedBatchNorm1d received no valid time positions")

        if self.training or not self.track_running_stats:
            mean = (x * mask_f).sum(dim=(0, 2)) / count
            centered = x - mean[None, :, None]
            variance = (centered.square() * mask_f).sum(dim=(0, 2)) / count

            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked.add_(1)
                    if self.momentum is None:
                        factor = 1.0 / float(self.num_batches_tracked)
                    else:
                        factor = self.momentum
                    # PyTorch normalizes with biased variance but stores the
                    # unbiased estimate in running_var.
                    if count.item() > 1:
                        running_variance = variance * count / (count - 1.0)
                    else:
                        running_variance = variance
                    self.running_mean.lerp_(mean.detach(), factor)
                    self.running_var.lerp_(running_variance.detach(), factor)
        else:
            mean = self.running_mean
            variance = self.running_var

        output = (x - mean[None, :, None]) * torch.rsqrt(
            variance[None, :, None] + self.eps
        )
        if self.affine:
            output = output * self.weight[None, :, None] + self.bias[None, :, None]
        return output.to(input.dtype)


class ChannelLayerNorm1d(nn.Module):
    """FP32 LayerNorm over channels at each time step for ``(B, C, T)`` input."""

    def __init__(self, source: nn.BatchNorm1d):
        super().__init__()
        self.eps = source.eps
        self.weight = nn.Parameter(source.weight.detach().float().clone())
        self.bias = nn.Parameter(source.bias.detach().float().clone())

    def forward(
        self, input: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output = F.layer_norm(
            input.float().transpose(1, 2),
            (input.size(1),),
            self.weight,
            self.bias,
            self.eps,
        ).transpose(1, 2)
        if valid_mask is not None:
            output = output.masked_fill(~valid_mask[:, None, :], 0.0)
        return output.to(input.dtype)


class MaskedParakeetConvolutionModule(nn.Module):
    """Parakeet convolution module whose BatchNorm excludes padded frames."""

    def __init__(
        self, source: nn.Module, convolution_norm: str, layer_index: int,
        finite_diagnostics: bool = False,
    ):
        super().__init__()
        self.layer_index = layer_index
        self.finite_diagnostics = finite_diagnostics
        self.padding = source.padding
        self.pointwise_conv1 = source.pointwise_conv1
        self.depthwise_conv = source.depthwise_conv
        if convolution_norm == "masked_batch_norm":
            self.norm = MaskedBatchNorm1d.from_batch_norm(source.norm)
        elif convolution_norm == "layer_norm":
            self.norm = ChannelLayerNorm1d(source.norm)
        else:
            raise ValueError(f"Unsupported convolution norm: {convolution_norm}")
        self.activation = source.activation
        self.pointwise_conv2 = source.pointwise_conv2

    @staticmethod
    def _valid_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.dtype == torch.bool:
            all_masked_rows = torch.all(~attention_mask, dim=2)
        else:
            all_masked_rows = torch.all(~(attention_mask == 0.0), dim=2)
        valid = ~all_masked_rows
        while valid.ndim > 2 and valid.size(1) == 1:
            valid = valid.squeeze(1)
        if valid.ndim != 2:
            raise ValueError(f"Unexpected Parakeet attention-mask shape: {attention_mask.shape}")
        return valid

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        convolution_input = hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = F.glu(hidden_states, dim=1)

        valid_mask = None
        if attention_mask is not None:
            valid_mask = self._valid_mask(attention_mask)
            hidden_states = hidden_states.masked_fill(~valid_mask[:, None, :], 0.0)

        hidden_states = self.depthwise_conv(hidden_states)
        if self.finite_diagnostics and not torch.isfinite(hidden_states).all():
            raise FloatingPointError(
                f"First non-finite convolution/BatchNorm input in encoder block "
                f"{self.layer_index + 1}.\n"
                + _tensor_diagnostic("convolution module input", convolution_input)
                + "\n"
                + _tensor_diagnostic("BatchNorm input", hidden_states)
            )
        hidden_states = self.norm(hidden_states, valid_mask)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        return hidden_states.transpose(1, 2)


def install_convolution_norm(
    encoder: nn.Module, convolution_norm: str, finite_diagnostics: bool = False
) -> None:
    """Install padding-safe normalization in every Parakeet convolution block."""
    for layer_index, layer in enumerate(encoder.layers):
        layer.conv = MaskedParakeetConvolutionModule(
            layer.conv, convolution_norm, layer_index, finite_diagnostics
        )


@dataclass(frozen=True)
class ParakeetTiming:
    sample_rate: int
    hop_length: int
    encoder_subsampling: int
    upsample_factor: int

    @property
    def output_stride_samples(self) -> float:
        return self.hop_length * self.encoder_subsampling / self.upsample_factor

    @property
    def output_stride_seconds(self) -> float:
        return self.output_stride_samples / self.sample_rate


class TorchParakeetFrontend(nn.Module):
    """Differentiation-free GPU implementation of the HF feature extractor.

    Unlike ``ParakeetFeatureExtractor.__call__``, this accepts an already
    padded tensor together with its true lengths. This avoids a GPU-to-CPU
    conversion and preserves the lengths supplied by Lhotse.
    """

    def __init__(self, extractor: ParakeetFeatureExtractor):
        super().__init__()
        self.sample_rate = int(extractor.sampling_rate)
        self.n_fft = int(extractor.n_fft)
        self.hop_length = int(extractor.hop_length)
        self.win_length = int(extractor.win_length)
        self.preemphasis = extractor.preemphasis
        self.register_buffer("mel_filters", extractor.mel_filters.clone(), persistent=False)

    @classmethod
    def from_pretrained(
        cls, model_id: str, revision: Optional[str] = None
    ) -> "TorchParakeetFrontend":
        config_path = hf_hub_download(
            model_id, "processor_config.json", revision=revision
        )
        with open(config_path, encoding="utf-8") as config_file:
            processor_config = json.load(config_file)
        extractor = ParakeetFeatureExtractor.from_dict(
            processor_config["feature_extractor"]
        )
        return cls(extractor)

    def forward(
        self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Calling ``.float()`` is not sufficient inside an outer autocast
        # region: eligible operations such as the mel matrix multiplication
        # are cast back to FP16. Large reverberated/loudness-normalized audio
        # can then overflow its STFT power. Keep the complete frontend in FP32.
        with torch.autocast(device_type=waveforms.device.type, enabled=False):
            return self._forward_fp32(waveforms, waveform_lengths)

    def _forward_fp32(
        self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if waveforms.ndim != 2:
            raise ValueError(f"Expected (batch, samples), got {waveforms.shape}")
        lengths = waveform_lengths.to(device=waveforms.device, dtype=torch.long)
        sample_mask = torch.arange(waveforms.size(1), device=waveforms.device)[None] < lengths[:, None]
        x = waveforms.float().masked_fill(~sample_mask, 0.0)
        if self.preemphasis is not None:
            x = torch.cat([x[:, :1], x[:, 1:] - self.preemphasis * x[:, :-1]], dim=1)
            x = x.masked_fill(~sample_mask, 0.0)

        window = torch.hann_window(
            self.win_length, periodic=False, device=x.device, dtype=x.dtype
        )
        stft = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
        )
        power = torch.view_as_real(stft).square().sum(-1)
        mel = torch.matmul(self.mel_filters.to(dtype=power.dtype), power)
        features = torch.log(mel + 2**-24).transpose(1, 2)

        feature_lengths = torch.div(lengths, self.hop_length, rounding_mode="floor")
        feature_lengths = feature_lengths.clamp(min=1, max=features.size(1))
        feature_mask = torch.arange(features.size(1), device=x.device)[None] < feature_lengths[:, None]
        mask = feature_mask.unsqueeze(-1)
        masked = features * mask
        mean = masked.sum(1) / feature_lengths.unsqueeze(-1)
        variance = ((masked - mean.unsqueeze(1)).square() * mask).sum(1)
        variance = variance / (feature_lengths - 1).clamp(min=1).unsqueeze(-1)
        features = (features - mean.unsqueeze(1)) / (variance.sqrt().unsqueeze(1) + 1e-5)
        features = features * mask
        return features, feature_mask


class ParakeetWaveformEncoder(nn.Module):
    """Convert padded 16-kHz waveforms to upsampled FastConformer states."""

    def __init__(
        self,
        encoder: nn.Module,
        frontend: TorchParakeetFrontend,
        encoder_projector: Optional[nn.Module] = None,
        speaker_layer: int = 12,
        upsample_factor: int = 4,
        upsample_mode: str = "linear",
        encoder_subsampling: int = 8,
        finite_diagnostics: bool = False,
    ):
        super().__init__()
        if upsample_factor < 1:
            raise ValueError("upsample_factor must be at least 1")
        if upsample_mode not in ("linear", "nearest"):
            raise ValueError("upsample_mode must be 'linear' or 'nearest'")
        self.encoder = encoder
        self.frontend = frontend
        self.encoder_projector = (
            encoder_projector if encoder_projector is not None else nn.Identity()
        )
        num_layers = len(encoder.layers) if hasattr(encoder, "layers") else 1
        if not hasattr(encoder, "layers"):
            speaker_layer = 1
        if not 1 <= speaker_layer <= num_layers:
            raise ValueError(
                f"speaker_layer must be in [1, {num_layers}], got {speaker_layer}"
            )
        self.speaker_layer = speaker_layer
        self.finite_diagnostics = finite_diagnostics
        self.upsample_factor = upsample_factor
        self.upsample_mode = upsample_mode
        self.encoder_dim = int(encoder.config.hidden_size)
        self.output_dim = int(
            getattr(self.encoder_projector, "out_features", self.encoder_dim)
        )
        self.speaker_output_dim = self.encoder_dim
        self.timing = ParakeetTiming(
            sample_rate=frontend.sample_rate,
            hop_length=frontend.hop_length,
            encoder_subsampling=encoder_subsampling,
            upsample_factor=upsample_factor,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "nvidia/parakeet-tdt-0.6b-v3",
        revision: Optional[str] = None,
        upsample_factor: int = 4,
        upsample_mode: str = "linear",
        speaker_layer: int = 12,
        encoder_layerdrop: float = 0.0,
        convolution_norm: str = "masked_batch_norm",
        finite_diagnostics: bool = False,
        random_init: bool = False,
    ) -> "ParakeetWaveformEncoder":
        config = AutoConfig.from_pretrained(model_id, revision=revision)
        if random_init:
            # Reuse only the published architecture/configuration.  This does
            # not load any pretrained encoder or projector parameters.
            tdt_model = ParakeetForTDT(config)
        else:
            tdt_model = ParakeetForTDT.from_pretrained(
                model_id, config=config, revision=revision
            )
        encoder = tdt_model.encoder
        encoder.layerdrop = encoder_layerdrop
        encoder_projector = tdt_model.encoder_projector
        install_convolution_norm(encoder, convolution_norm, finite_diagnostics)
        frontend = TorchParakeetFrontend.from_pretrained(model_id, revision=revision)
        return cls(
            encoder=encoder,
            frontend=frontend,
            encoder_projector=encoder_projector,
            speaker_layer=speaker_layer,
            upsample_factor=upsample_factor,
            upsample_mode=upsample_mode,
            encoder_subsampling=config.encoder_config.subsampling_factor,
            finite_diagnostics=finite_diagnostics,
        )

    def forward(
        self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states, _, lengths = self.forward_branches(waveforms, waveform_lengths)
        if self.upsample_factor > 1:
            interpolation_args = {}
            if self.upsample_mode == "linear":
                interpolation_args["align_corners"] = False
            states = F.interpolate(
                states.transpose(1, 2),
                scale_factor=self.upsample_factor,
                mode=self.upsample_mode,
                **interpolation_args,
            ).transpose(1, 2)
            lengths = lengths * self.upsample_factor
        return states, lengths.clamp(max=states.size(1)).to(torch.int32)

    def forward_branches(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        return_high_resolution: bool = False,
    ):
        """Return projected final and intermediate coarse encoder states."""
        features, feature_mask = self.frontend(waveforms, waveform_lengths)
        captured = {}
        hooks = []
        if return_high_resolution:
            # The frontend produces 10-ms log-mel frames.  The first
            # stride-2 convolution therefore has a genuine 20-ms frame rate.
            hooks.append(
                self.encoder.subsampling.layers[1].register_forward_hook(
                    lambda _module, _inputs, output: captured.__setitem__(
                        "high_resolution", output
                    )
                )
            )
        if hasattr(self.encoder, "layers") and not return_high_resolution:
            def capture_states(_module, _inputs, output):
                captured["states"] = output

            hooks.append(self.encoder.layers[self.speaker_layer - 1].register_forward_hook(capture_states))
        if hasattr(self.encoder, "layers") and self.finite_diagnostics:
            for layer_index, layer in enumerate(self.encoder.layers):
                def check_block(_module, _inputs, output, index=layer_index):
                    value = output[0] if isinstance(output, tuple) else output
                    if torch.is_tensor(value) and not torch.isfinite(value).all():
                        raise FloatingPointError(
                            f"First non-finite encoder block output: block {index + 1}.\n"
                            + _tensor_diagnostic("block output", value)
                        )
                hooks.append(layer.register_forward_hook(check_block))
        try:
            outputs = self.encoder(input_features=features, attention_mask=feature_mask)
        except FloatingPointError as error:
            raise FloatingPointError(
                "Parakeet finite diagnostic failure:\n"
                + _tensor_diagnostic("waveform", waveforms)
                + "\n"
                + _tensor_diagnostic("frontend features", features)
                + "\n"
                + str(error)
            ) from error
        finally:
            for hook in hooks:
                hook.remove()
        final_states = outputs.last_hidden_state
        speaker_states = captured.get("states", final_states)
        token_states = self.encoder_projector(final_states)
        if outputs.attention_mask is None:
            lengths = torch.full(
                (final_states.size(0),),
                final_states.size(1),
                dtype=torch.long,
                device=final_states.device,
            )
        else:
            lengths = outputs.attention_mask.sum(1).long()
        if not return_high_resolution:
            return token_states, speaker_states, lengths.to(torch.int32)

        high_states = captured["high_resolution"]
        # Reuse the pretrained frequency-processing convolutions and linear
        # projection, but suppress their temporal stride.  This preserves the
        # 20-ms tap while producing the same 1024-dimensional representation
        # expected by the FastConformer encoder.
        for layer in self.encoder.subsampling.layers[2:]:
            if isinstance(layer, nn.Conv2d) and layer.kernel_size != (1, 1):
                high_states = F.conv2d(
                    high_states,
                    layer.weight,
                    layer.bias,
                    stride=(1, layer.stride[1]),
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                )
            else:
                high_states = layer(high_states)
        batch, channels, time, frequency = high_states.shape
        high_states = high_states.permute(0, 2, 1, 3).reshape(
            batch, time, channels * frequency
        )
        high_states = self.encoder.subsampling.linear(high_states)
        high_states = high_states * self.encoder.input_scale

        feature_lengths = feature_mask.sum(1).long()
        first_conv = self.encoder.subsampling.layers[0]
        high_lengths = self.encoder.subsampling._get_output_length(
            feature_lengths, first_conv
        ).clamp(max=high_states.size(1))
        valid = torch.arange(high_states.size(1), device=high_states.device)[None, :]
        high_states = high_states.masked_fill(
            ~(valid < high_lengths[:, None]).unsqueeze(-1), 0.0
        )
        return (
            token_states,
            speaker_states,
            lengths.to(torch.int32),
            high_states,
            high_lengths.to(torch.int32),
        )
