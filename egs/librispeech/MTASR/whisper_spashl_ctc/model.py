"""Shuffle-CTC output heads on Whisper encoder states."""

import torch.nn as nn

from parakeet_native_spashl_ctc.model import MDCTCModel


class TokenOnlyMDCTCModel(nn.Module):
    """A single token head for speaker-independent Shuffle training."""

    def __init__(
        self,
        encoder,
        encoder_dim: int,
        vocab_size: int,
        layer_norm=False,
        max_num_spks: int = 8,
    ):
        super().__init__()
        self.encoder = encoder
        self.frozen = False
        self.use_layer_norm = layer_norm
        self.max_num_spks = max_num_spks
        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.ctc_output1 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, vocab_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(
        self,
        x,
        x_lens,
        speaker_weight: float = 1.0,
        return_factors: bool = False,
    ):
        states, lengths = self.encoder(x, x_lens)
        if self.use_layer_norm:
            states = self.layer_norm(states)
        token_log_probs = self.ctc_output1(states)
        output_lens = lengths.to(dtype=x_lens.dtype)
        if return_factors:
            # Generic long-form alignment code carries a speaker-factor tensor
            # through chunking even for an explicitly speaker-independent
            # graph.  Supply neutral log scores; with ignore_speaker=True they
            # are bookkeeping only and contribute exactly zero to every path.
            neutral_speakers = token_log_probs.new_zeros(
                (*token_log_probs.shape[:2], self.max_num_spks)
            )
            return token_log_probs, neutral_speakers, output_lens
        return token_log_probs, output_lens

    def freeze_encoder(self):
        self.encoder.freeze_pretrained_backbone()
        self.frozen = True

    def unfreeze_encoder(self):
        self.encoder.unfreeze_pretrained_backbone()
        self.frozen = False


__all__ = ["MDCTCModel", "TokenOnlyMDCTCModel"]
