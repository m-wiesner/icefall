"""High-resolution parallel adapter for NVIDIA Parakeet TDT."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoConfig, ParakeetForTDT

from parakeet_spashl_ctc.parakeet_encoder import TorchParakeetFrontend


def _padding_mask(lengths: torch.Tensor, size: int) -> torch.Tensor:
    return torch.arange(size, device=lengths.device)[None, :] >= lengths[:, None]


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device, dtype=torch.float32)
        frequencies = torch.exp(
            torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32)
            * (-math.log(10000.0) / self.dim)
        )
        angles = positions[:, None] * frequencies[None, :]
        encoding = torch.zeros(x.size(1), self.dim, device=x.device, dtype=torch.float32)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles.cos()[:, : encoding[:, 1::2].size(1)]
        return x + encoding.to(dtype=x.dtype)[None, :, :]


class LocalSelfAttention(nn.Module):
    """Blockwise sliding-window attention with bounded activation memory."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("Local-attention window size must be a positive odd number")
        if dim % num_heads:
            raise ValueError("Attention dimension must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.block_size = block_size
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        radius = self.window_size // 2
        valid = ~padding_mask
        blocks = []
        for query_start in range(0, length, self.block_size):
            query_end = min(query_start + self.block_size, length)
            key_start = max(0, query_start - radius)
            key_end = min(length, query_end + radius)
            query_positions = torch.arange(query_start, query_end, device=x.device)
            key_positions = torch.arange(key_start, key_end, device=x.device)
            local_mask = (
                query_positions[:, None] - key_positions[None, :]
            ).abs() <= radius
            attention_mask = (
                local_mask[None, None, :, :]
                & valid[:, None, None, key_start:key_end]
            )
            blocks.append(
                nn.functional.scaled_dot_product_attention(
                    q[:, :, query_start:query_end],
                    k[:, :, key_start:key_end],
                    v[:, :, key_start:key_end],
                    attn_mask=attention_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
            )
        attended = torch.cat(blocks, dim=2).transpose(1, 2).reshape(batch, length, dim)
        return self.output(attended).masked_fill(padding_mask.unsqueeze(-1), 0.0)


class LocalCrossAttention(nn.Module):
    """Blockwise attention from 50 Hz queries to aligned 12.5 Hz states."""

    def __init__(
        self,
        query_dim: int,
        key_value_dim: int,
        num_heads: int,
        window_size: int,
        block_size: int,
        query_to_key_ratio: int,
        dropout: float,
    ):
        super().__init__()
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("Cross-attention window must be a positive odd number")
        if query_dim % num_heads:
            raise ValueError("Attention dimension must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.window_size = window_size
        self.block_size = block_size
        self.query_to_key_ratio = query_to_key_ratio
        self.dropout = dropout
        self.query = nn.Linear(query_dim, query_dim)
        self.key = nn.Linear(key_value_dim, query_dim)
        self.value = nn.Linear(key_value_dim, query_dim)
        self.output = nn.Linear(query_dim, query_dim)

    def forward(
        self,
        query_states: torch.Tensor,
        key_value_states: torch.Tensor,
        query_padding_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_length, dim = query_states.shape
        key_length = key_value_states.size(1)
        q = self.query(query_states).view(
            batch, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.key(key_value_states).view(
            batch, key_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.value(key_value_states).view(
            batch, key_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        radius = self.window_size // 2
        valid_keys = ~key_padding_mask
        blocks = []
        for query_start in range(0, query_length, self.block_size):
            query_end = min(query_start + self.block_size, query_length)
            first_key = query_start // self.query_to_key_ratio
            last_key = (query_end - 1) // self.query_to_key_ratio
            key_start = max(0, first_key - radius)
            key_end = min(key_length, last_key + radius + 1)
            query_positions = torch.arange(
                query_start, query_end, device=query_states.device
            ) // self.query_to_key_ratio
            key_positions = torch.arange(
                key_start, key_end, device=query_states.device
            )
            local_mask = (
                query_positions[:, None] - key_positions[None, :]
            ).abs() <= radius
            attention_mask = (
                local_mask[None, None, :, :]
                & valid_keys[:, None, None, key_start:key_end]
            )
            blocks.append(
                nn.functional.scaled_dot_product_attention(
                    q[:, :, query_start:query_end],
                    k[:, :, key_start:key_end],
                    v[:, :, key_start:key_end],
                    attn_mask=attention_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
            )
        attended = torch.cat(blocks, dim=2).transpose(1, 2).reshape(
            batch, query_length, dim
        )
        return self.output(attended).masked_fill(
            query_padding_mask.unsqueeze(-1), 0.0
        )


class HighResolutionAdapterLayer(nn.Module):
    """Pre-norm Transformer layer plus low-resolution cross-attention."""

    def __init__(
        self,
        adapter_dim: int,
        backbone_dim: int,
        num_heads: int,
        ff_dim: int,
        local_attention_window: int,
        cross_attention_window: int,
        attention_block_size: int,
        query_to_key_ratio: int,
        dropout: float,
    ):
        super().__init__()
        self.self_norm = nn.LayerNorm(adapter_dim)
        self.self_attn = LocalSelfAttention(
            adapter_dim,
            num_heads,
            window_size=local_attention_window,
            block_size=attention_block_size,
            dropout=dropout,
        )
        self.cross_norm = nn.LayerNorm(adapter_dim)
        self.cross_attn = LocalCrossAttention(
            query_dim=adapter_dim,
            key_value_dim=backbone_dim,
            num_heads=num_heads,
            window_size=cross_attention_window,
            block_size=attention_block_size,
            query_to_key_ratio=query_to_key_ratio,
            dropout=dropout,
        )
        self.ff_norm = nn.LayerNorm(adapter_dim)
        self.ff = nn.Sequential(
            nn.Linear(adapter_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, adapter_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        high: torch.Tensor,
        low: torch.Tensor,
        high_padding_mask: torch.Tensor,
        low_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.self_norm(high)
        update = self.self_attn(query, high_padding_mask)
        high = high + self.dropout(update)

        update = self.cross_attn(
            self.cross_norm(high),
            low,
            high_padding_mask,
            low_padding_mask,
        )
        high = high + self.dropout(update)
        high = high + self.dropout(self.ff(self.ff_norm(high)))
        return high.masked_fill(high_padding_mask.unsqueeze(-1), 0.0)


class ParakeetEncoder(nn.Module):
    """Fuse a 50 Hz parallel adapter with native 12.5 Hz Parakeet states.

    The adapter starts at the activation after the first stride-2 convolution.
    Its layers query selected FastConformer block outputs.  Final Parakeet
    states are interleaved with three learned phase/fill vectors, added to the
    projected adapter states, and layer-normalized at 50 Hz.
    """

    def __init__(
        self,
        encoder: nn.Module,
        frontend: TorchParakeetFrontend,
        encoder_subsampling: int,
        num_mel_bins: int,
        conv_channels: int,
        adapter_dim: int = 256,
        adapter_num_heads: int = 4,
        adapter_ff_dim: int = 1024,
        adapter_layer_indices: Sequence[int] = (7, 15, 23),
        adapter_local_attention_window: int = 257,
        adapter_cross_attention_window: int = 65,
        adapter_attention_block_size: int = 256,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()
        if encoder_subsampling != 8:
            raise ValueError(f"Expected Parakeet subsampling 8, got {encoder_subsampling}")
        if adapter_dim % adapter_num_heads:
            raise ValueError("adapter_dim must be divisible by adapter_num_heads")
        indices = tuple(int(i) for i in adapter_layer_indices)
        if not indices:
            raise ValueError("At least one adapter/backbone layer correspondence is required")
        if len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= len(encoder.layers):
            raise ValueError(f"Invalid Parakeet layer indices: {indices}")

        self.encoder = encoder
        self.frontend = frontend
        self.output_dim = int(encoder.config.hidden_size)
        self.frame_shift = 0.02
        self.adapter_layer_indices = indices
        self.upsampling_factor = encoder_subsampling // 2

        # The first convolution has stride 2 in time and frequency.
        tap_frequency_bins = (num_mel_bins + 1) // 2
        self.adapter_input = nn.Linear(conv_channels * tap_frequency_bins, adapter_dim)
        self.adapter_position = SinusoidalPositionEncoding(adapter_dim)
        self.adapter_layers = nn.ModuleList(
            HighResolutionAdapterLayer(
                adapter_dim=adapter_dim,
                backbone_dim=self.output_dim,
                num_heads=adapter_num_heads,
                ff_dim=adapter_ff_dim,
                local_attention_window=adapter_local_attention_window,
                cross_attention_window=adapter_cross_attention_window,
                attention_block_size=adapter_attention_block_size,
                query_to_key_ratio=self.upsampling_factor,
                dropout=adapter_dropout,
            )
            for _ in indices
        )
        self.adapter_output = nn.Linear(adapter_dim, self.output_dim)
        # Separate phase vectors prevent the three inserted locations from
        # beginning as indistinguishable queries/representations.
        self.upsampling_fill = nn.Parameter(
            torch.empty(self.upsampling_factor - 1, self.output_dim)
        )
        nn.init.normal_(self.upsampling_fill, mean=0.0, std=0.02)
        self.fusion_norm = nn.LayerNorm(self.output_dim)

        self._captures: dict[str | int, torch.Tensor] = {}
        self.memory_reporter = None
        self._hook_handles = [
            self.encoder.subsampling.layers[1].register_forward_hook(
                self._capture_high_resolution
            )
        ]
        for index in indices:
            self._hook_handles.append(
                self.encoder.layers[index].register_forward_hook(
                    self._capture_backbone_layer(index)
                )
            )
        # Hooks must fire for every selected block during training.
        self.encoder.layerdrop = 0.0

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "nvidia/parakeet-tdt-0.6b-v3",
        revision: Optional[str] = None,
        **adapter_kwargs,
    ) -> "ParakeetEncoder":
        config = AutoConfig.from_pretrained(model_id, revision=revision)
        model = ParakeetForTDT.from_pretrained(
            model_id, config=config, revision=revision
        )
        frontend = TorchParakeetFrontend.from_pretrained(
            model_id, revision=revision
        )
        encoder_config = config.encoder_config
        return cls(
            encoder=model.encoder,
            frontend=frontend,
            encoder_subsampling=encoder_config.subsampling_factor,
            num_mel_bins=encoder_config.num_mel_bins,
            conv_channels=encoder_config.subsampling_conv_channels,
            **adapter_kwargs,
        )

    def _capture_high_resolution(self, _module, _inputs, output) -> None:
        # Conv output is (batch, channels, time, frequency).
        self._captures["high"] = output

    def _capture_backbone_layer(self, index: int):
        def hook(_module, _inputs, output) -> None:
            self._captures[index] = output

        return hook

    def _interleave(self, states: torch.Tensor, target_length: int) -> torch.Tensor:
        batch, low_length, dim = states.shape
        phases = torch.cat(
            [states.unsqueeze(2), self.upsampling_fill[None, None].expand(batch, low_length, -1, -1)],
            dim=2,
        )
        upsampled = phases.reshape(batch, low_length * self.upsampling_factor, dim)
        if upsampled.size(1) < target_length:
            upsampled = nn.functional.pad(
                upsampled, (0, 0, 0, target_length - upsampled.size(1))
            )
        return upsampled[:, :target_length]

    def forward(
        self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._captures.clear()
        features, feature_mask = self.frontend(waveforms, waveform_lengths)
        if self.memory_reporter is not None:
            self.memory_reporter("model/parakeet_frontend")
        outputs = self.encoder(input_features=features, attention_mask=feature_mask)
        if self.memory_reporter is not None:
            self.memory_reporter("model/parakeet_backbone")
        low = outputs.last_hidden_state
        if outputs.attention_mask is None:
            low_lengths = torch.full(
                (low.size(0),), low.size(1), dtype=torch.int32, device=low.device
            )
        else:
            low_lengths = outputs.attention_mask.sum(1).to(torch.int32)

        high_conv = self._captures.get("high")
        if high_conv is None:
            raise RuntimeError("The 50 Hz Parakeet convolutional tap did not run")
        high = high_conv.transpose(1, 2).flatten(2)
        high = self.adapter_position(self.adapter_input(high))
        if self.memory_reporter is not None:
            self.memory_reporter("model/adapter_input")
        feature_lengths = feature_mask.sum(1).to(torch.int64)
        high_lengths = torch.div(feature_lengths + 1, 2, rounding_mode="floor")
        high_lengths = high_lengths.clamp(max=high.size(1))
        high_padding_mask = _padding_mask(high_lengths, high.size(1))
        low_padding_mask = _padding_mask(low_lengths, low.size(1))

        for index, layer in zip(self.adapter_layer_indices, self.adapter_layers):
            backbone = self._captures.get(index)
            if backbone is None:
                raise RuntimeError(f"Parakeet backbone layer {index} was not captured")
            high = layer(high, backbone, high_padding_mask, low_padding_mask)
            if self.memory_reporter is not None:
                self.memory_reporter(f"model/adapter_layer_{index}")

        adapter = self.adapter_output(high)
        upsampled = self._interleave(low, target_length=high.size(1))
        fused = self.fusion_norm(upsampled + adapter)
        if self.memory_reporter is not None:
            self.memory_reporter("model/fusion")
        fused = fused.masked_fill(high_padding_mask.unsqueeze(-1), 0.0)
        return fused, high_lengths.to(torch.int32)

    def freeze_pretrained_backbone(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.frontend.parameters():
            parameter.requires_grad = False

    def unfreeze_pretrained_backbone(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True
        for parameter in self.frontend.parameters():
            parameter.requires_grad = True
