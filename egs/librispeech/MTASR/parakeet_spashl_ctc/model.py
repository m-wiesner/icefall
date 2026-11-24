"""Token-only and speaker-attributed shuffle-CTC heads for Parakeet."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalResidualBlock(nn.Module):
    """A lightweight depthwise-separable temporal refinement block."""

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.pointwise = nn.Conv1d(dim, dim, 1)
        self.activation = nn.SiLU()
        # Start as an exact residual identity. The depthwise filter and
        # pointwise projection become active gradually during training.
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        residual = states
        states = self.norm(states).transpose(1, 2)
        states = self.depthwise(states)
        states = self.activation(states)
        states = self.pointwise(states).transpose(1, 2)
        return residual + states


class LearnedPhaseUpsampler(nn.Module):
    """Generate ``factor`` learned fine-time phases from each coarse state."""

    def __init__(self, dim: int, factor: int = 4, kernel_size: int = 3):
        super().__init__()
        if factor < 1:
            raise ValueError(f"Upsampling factor must be positive, got {factor}")
        self.dim = dim
        self.factor = factor
        self.coarse_refinement = TemporalResidualBlock(dim, kernel_size)
        self.phase_projection = nn.Linear(dim, factor * dim)
        self.fine_refinement = TemporalResidualBlock(dim, kernel_size)
        self._initialize_as_repeat()

    def _initialize_as_repeat(self) -> None:
        """Initialize all phases to copy the input coarse state."""
        with torch.no_grad():
            self.phase_projection.weight.zero_()
            self.phase_projection.bias.zero_()
            identity = torch.eye(self.dim)
            for phase in range(self.factor):
                start = phase * self.dim
                self.phase_projection.weight[start : start + self.dim].copy_(identity)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = self.coarse_refinement(states)
        batch, time, _ = states.shape
        states = self.phase_projection(states)
        states = states.reshape(batch, time * self.factor, self.dim)
        return self.fine_refinement(states)


class HighResolutionRecoveryAdapter(nn.Module):
    """Recover detail present at 20 ms but suppressed by the main encoder."""

    def __init__(self, early_dim: int, output_dim: int, num_blocks: int = 2):
        super().__init__()
        self.early_projection = nn.Linear(early_dim, output_dim)
        self.early_norm = nn.LayerNorm(output_dim)
        self.semantic_norm = nn.LayerNorm(output_dim)
        self.residual_blocks = nn.ModuleList(
            [TemporalResidualBlock(output_dim) for _ in range(num_blocks)]
        )
        self.residual_output = nn.Linear(output_dim, output_dim)
        self.gate = nn.Linear(2 * output_dim, output_dim)
        # Begin as the pretrained semantic path.  The adapter must learn both
        # a missing-speech residual and where it should be enabled.
        nn.init.zeros_(self.residual_output.weight)
        nn.init.zeros_(self.residual_output.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)

    def forward(
        self,
        semantic: torch.Tensor,
        early: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        early = self.early_projection(early)
        if valid_mask is not None:
            semantic = semantic.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
            early = early.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        difference = self.early_norm(early) - self.semantic_norm(semantic)
        for block in self.residual_blocks:
            difference = block(difference)
            if valid_mask is not None:
                difference = difference.masked_fill(
                    ~valid_mask.unsqueeze(-1), 0.0
                )
        residual = self.residual_output(difference)
        gate = torch.sigmoid(self.gate(torch.cat([semantic, early], dim=-1)))
        fused = semantic + gate * residual
        if valid_mask is not None:
            fused = fused.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return fused


class MDCTCModel(nn.Module):
    """Parakeet shuffle CTC with optional intermediate-layer speaker branch."""

    def __init__(
        self,
        encoder,
        encoder_dim: int,
        vocab_size1: int,
        vocab_size2: int = 0,
        freeze_feat_extractor: bool = True,
        factorized_token_speaker_output: bool = True,
        layer_norm: bool = False,
        downsample: bool = False,
        speaker_branch: bool = True,
        upsample_factor: int = 4,
        high_resolution_fusion: bool = False,
        recovery_adapter_blocks: int = 2,
    ):
        super().__init__()
        self.encoder = encoder
        self.speaker_branch = speaker_branch
        self.factorized_token_speaker_output = factorized_token_speaker_output
        if speaker_branch and not factorized_token_speaker_output:
            raise ValueError(
                "The redesigned Parakeet speaker branch requires "
                "--factorized-token-speaker-output=true"
            )
        self.downsample = downsample
        self.freeze_feat_extractor = freeze_feat_extractor
        self.frozen = False
        self.high_resolution_fusion = high_resolution_fusion
        self.upsample_factor = upsample_factor

        token_dim = encoder_dim
        speaker_dim = int(getattr(encoder, "speaker_output_dim", token_dim))
        # At native Parakeet rate (factor 1), keep this as a genuinely
        # encoder-only model: no learned temporal refinement/projection sits
        # between the backbone and its output heads.
        self.token_upsampler = (
            nn.Identity()
            if upsample_factor == 1
            else LearnedPhaseUpsampler(token_dim, upsample_factor)
        )
        if high_resolution_fusion:
            high_dim = int(getattr(encoder, "encoder_dim", 1024))
            self.recovery_adapter = HighResolutionRecoveryAdapter(
                high_dim, token_dim, recovery_adapter_blocks
            )
        else:
            self.recovery_adapter = None
        self.token_layer_norm = nn.LayerNorm(token_dim) if layer_norm else nn.Identity()
        self.ctc_output1 = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(token_dim, vocab_size1)
        )

        if speaker_branch:
            if vocab_size2 < 1:
                raise ValueError("vocab_size2 must be positive with a speaker branch")
            self.speaker_upsampler = (
                None
                if high_resolution_fusion
                else nn.Identity()
                if upsample_factor == 1
                else LearnedPhaseUpsampler(speaker_dim, upsample_factor)
            )
            speaker_head_dim = token_dim if high_resolution_fusion else speaker_dim
            self.speaker_layer_norm = nn.LayerNorm(speaker_head_dim) if layer_norm else nn.Identity()
            self.ctc_output2 = nn.Sequential(
                nn.Dropout(0.1), nn.Linear(speaker_head_dim, vocab_size2)
            )
        else:
            self.speaker_upsampler = None
            self.speaker_layer_norm = None
            self.ctc_output2 = None

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def _encode_branches(self, x: torch.Tensor, x_lens: torch.Tensor):
        encoded = self.encoder.forward_branches(
            x, x_lens, return_high_resolution=self.high_resolution_fusion
        )
        token_states, speaker_states, lengths = encoded[:3]
        token_states = self.token_upsampler(token_states)
        lengths = lengths * self.upsample_factor
        if self.high_resolution_fusion:
            high_states, high_lengths = encoded[3:]
            if token_states.size(1) != high_states.size(1):
                token_states = F.interpolate(
                    token_states.transpose(1, 2),
                    size=high_states.size(1),
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            lengths = high_lengths
            valid = torch.arange(high_states.size(1), device=high_states.device)[None, :] < lengths[:, None]
            token_states = self.recovery_adapter(token_states, high_states, valid)
        elif self.speaker_branch:
            speaker_states = self.speaker_layer_norm(
                self.speaker_upsampler(speaker_states)
            )
        token_states = self.token_layer_norm(token_states)
        if self.high_resolution_fusion and self.speaker_branch:
            # Both heads consume exactly the same fused output representation.
            speaker_states = token_states

        if self.downsample:
            token_states = F.avg_pool1d(
                token_states.transpose(1, 2), 2, 2
            ).transpose(1, 2)
            if self.speaker_branch:
                speaker_states = F.avg_pool1d(
                    speaker_states.transpose(1, 2), 2, 2
                ).transpose(1, 2)
            lengths = torch.div(lengths, 2, rounding_mode="floor")
        if not torch.all(lengths > 0):
            raise ValueError(f"Non-positive encoder lengths: {lengths}")
        return token_states, speaker_states, lengths.to(torch.int32)

    def forward(
        self,
        x,
        x_lens,
        speaker_weight: float = 1.0,
        return_factors: bool = False,
    ):
        token_states, speaker_states, lengths = self._encode_branches(x, x_lens)
        token_log_probs = self.log_softmax(self.ctc_output1(token_states))
        if not self.speaker_branch:
            return token_log_probs, lengths

        speaker_log_probs = speaker_weight * self.log_softmax(
            self.ctc_output2(speaker_states)
        )
        if return_factors:
            return token_log_probs, speaker_log_probs, lengths

        blank = token_log_probs[..., 0]
        joint = token_log_probs[..., 1:].unsqueeze(-1) + speaker_log_probs.unsqueeze(-2)
        joint = joint.permute(0, 1, 3, 2).flatten(2)
        output = torch.cat([blank.unsqueeze(-1), joint], dim=-1)
        return output.clamp(max=0.0), lengths

    def forward_tokens_only(self, x, x_lens):
        token_states, _, lengths = self._encode_branches(x, x_lens)
        return self.log_softmax(self.ctc_output1(token_states)), lengths

    def forward_target_speaker(self, x, x_lens, speakers=(0,)):
        if not self.speaker_branch:
            raise ValueError("Target-speaker decoding requires the speaker branch")
        token, speaker, lengths = self.forward(
            x, x_lens, speaker_weight=1.0, return_factors=True
        )
        blank = token[..., 0]
        joint = token[..., 1:].unsqueeze(-1) + speaker.unsqueeze(-2)
        joint = joint.permute(0, 1, 3, 2)
        outputs = []
        for speaker_index in speakers:
            other = [i for i in range(joint.size(2)) if i != speaker_index]
            other_weight = joint[:, :, other].logsumexp((2, 3))
            speaker_blank = torch.logaddexp(blank, other_weight)
            outputs.append(
                torch.cat(
                    [speaker_blank.unsqueeze(-1), joint[:, :, speaker_index]], -1
                ).clamp(max=0.0)
            )
        return lengths, outputs

    def freeze_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.frozen = True

    def unfreeze_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True
        self.frozen = False
