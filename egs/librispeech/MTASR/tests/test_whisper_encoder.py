from types import SimpleNamespace

import torch
import torch.nn as nn
import torchaudio

from whisper_spashl_ctc.encoder import WhisperAudioEncoder


class _DummyEncoder(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.config = SimpleNamespace(d_model=dim)
        self.proj = nn.Linear(80, dim)

    def forward(self, input_features, return_dict=True):
        states = self.proj(input_features[:, :, ::2].transpose(1, 2))
        return SimpleNamespace(last_hidden_state=states)


def _encoder():
    mel = torchaudio.functional.melscale_fbanks(
        n_freqs=201,
        f_min=0,
        f_max=8000,
        n_mels=80,
        sample_rate=16000,
        norm="slaney",
        mel_scale="slaney",
    )
    return WhisperAudioEncoder(_DummyEncoder(), mel)


def test_whisper_lengths_are_50_hz():
    encoder = _encoder()
    lengths = torch.tensor([16000, 32000, 480000])
    assert encoder.output_lengths(lengths).tolist() == [50, 100, 1500]


def test_whisper_pads_short_audio_to_fixed_context():
    encoder = _encoder()
    states, lengths = encoder(torch.randn(2, 32000), torch.tensor([16000, 32000]))
    assert states.shape == (2, 1500, 8)
    assert lengths.tolist() == [50, 100]


def test_whisper_rejects_audio_over_30_seconds():
    encoder = _encoder()
    try:
        encoder(torch.zeros(1, 480001), torch.tensor([480001]))
    except ValueError as exc:
        assert "<=30 s" in str(exc)
    else:
        raise AssertionError("Expected overlength input to be rejected")
