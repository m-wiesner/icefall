"""SPaSHL token and speaker heads on fused 50 Hz Parakeet states."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MDCTCModel(nn.Module):
    """Match the WavLM SPaSHL model while accepting encoder-provided lengths."""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int,
        vocab_size1: int,
        vocab_size2: int,
        factorized_token_speaker_output: bool = True,
        layer_norm: bool = False,
        downsample: bool = False,
    ):
        super().__init__()
        if not factorized_token_speaker_output:
            raise ValueError(
                "Native Parakeet SPaSHL requires factorized token/speaker output."
            )
        self.encoder = encoder
        self.factorized_token_speaker_output = True
        self.use_layer_norm = layer_norm
        self.downsample = downsample
        self.frozen = False
        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.ctc_output1 = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(encoder_dim, vocab_size1)
        )
        self.ctc_output2 = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(encoder_dim, vocab_size2)
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def _encode(self, x: torch.Tensor, x_lens: torch.Tensor):
        states, lengths = self.encoder(x, x_lens)
        if self.downsample:
            states = F.avg_pool1d(states.transpose(1, 2), 2, 2).transpose(1, 2)
            lengths = torch.div(lengths, 2, rounding_mode="floor")
        if self.use_layer_norm:
            states = self.layer_norm(states)
        if not torch.all(lengths > 0):
            raise ValueError(f"Non-positive encoder lengths: {lengths}")
        return states, lengths.to(torch.int32)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        speaker_weight: float = 1.0,
        return_factors: bool = False,
    ):
        states, lengths = self._encode(x, x_lens)
        reporter = getattr(self.encoder, "memory_reporter", None)
        token = self.log_softmax(self.ctc_output1(states))
        speaker = speaker_weight * self.log_softmax(self.ctc_output2(states))
        if reporter is not None:
            reporter("model/token_speaker_heads")
        if return_factors:
            return token, speaker, lengths
        blank = token[..., 0]
        joint = token[..., 1:].unsqueeze(-1) + speaker.unsqueeze(-2)
        joint = joint.permute(0, 1, 3, 2).flatten(2)
        output = torch.cat([blank.unsqueeze(-1), joint], dim=-1).clamp(max=0)
        if reporter is not None:
            reporter("model/joint_logits")
        return output, lengths

    def forward_tokens_only(self, x: torch.Tensor, x_lens: torch.Tensor):
        states, lengths = self._encode(x, x_lens)
        return self.log_softmax(self.ctc_output1(states)), lengths

    def forward_target_speaker(self, x, x_lens, speakers=(0,)):
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
                ).clamp(max=0)
            )
        return lengths, outputs

    def freeze_encoder(self):
        if hasattr(self.encoder, "freeze_pretrained_backbone"):
            self.encoder.freeze_pretrained_backbone()
        else:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        self.frozen = True

    def unfreeze_encoder(self):
        if hasattr(self.encoder, "unfreeze_pretrained_backbone"):
            self.encoder.unfreeze_pretrained_backbone()
        else:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True
        self.frozen = False
