#!/usr/bin/env python3

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from parakeet_spashl_ctc.parakeet_encoder import (
    ChannelLayerNorm1d,
    MaskedBatchNorm1d,
    ParakeetWaveformEncoder,
    TorchParakeetFrontend,
)
from parakeet_spashl_ctc.model import LearnedPhaseUpsampler, MDCTCModel


class DummyFrontend(nn.Module):
    sample_rate = 16000
    hop_length = 160

    def forward(self, waveforms, lengths):
        features = waveforms[:, :24].reshape(waveforms.size(0), 6, 4)
        mask = torch.arange(6)[None] < torch.tensor([6, 4])[:, None]
        return features, mask


class DummyEncoder(nn.Module):
    config = SimpleNamespace(hidden_size=4)

    def forward(self, input_features, attention_mask):
        return SimpleNamespace(
            last_hidden_state=input_features[:, ::2],
            attention_mask=attention_mask[:, ::2].int(),
        )


class DummyBranchEncoder(nn.Module):
    speaker_output_dim = 6

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward_branches(self, waveforms, lengths):
        token = waveforms[:, :12].reshape(waveforms.size(0), 3, 4) * self.scale
        speaker = torch.cat([token, token[..., :2]], dim=-1)
        return token, speaker, torch.tensor([3] * waveforms.size(0))


class ParakeetAdapterTest(unittest.TestCase):
    def test_frontend_disables_outer_autocast(self):
        extractor = SimpleNamespace(
            sampling_rate=16000,
            n_fft=32,
            hop_length=8,
            win_length=32,
            preemphasis=0.97,
            mel_filters=torch.rand(8, 17),
        )
        frontend = TorchParakeetFrontend(extractor)
        waveform = torch.randn(2, 256) * 25.0
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            features, mask = frontend(waveform, torch.tensor([256, 192]))
        self.assertEqual(features.dtype, torch.float32)
        self.assertTrue(torch.isfinite(features).all())
        self.assertEqual(mask.dtype, torch.bool)

    def test_phase_upsampler_starts_as_exact_repeat(self):
        upsampler = LearnedPhaseUpsampler(3, factor=4)
        inputs = torch.randn(2, 5, 3)
        expected = inputs.repeat_interleave(4, dim=1)
        torch.testing.assert_close(upsampler(inputs), expected)

    def test_token_only_and_speaker_attributed_shapes(self):
        inputs = torch.randn(2, 12)
        lengths = torch.tensor([12, 12])
        token_only = MDCTCModel(
            DummyBranchEncoder(),
            encoder_dim=4,
            vocab_size1=11,
            speaker_branch=False,
        )
        token_probs, token_lengths = token_only(inputs, lengths)
        self.assertEqual(token_probs.shape, (2, 12, 11))
        self.assertEqual(token_lengths.tolist(), [12, 12])

        attributed = MDCTCModel(
            DummyBranchEncoder(),
            encoder_dim=4,
            vocab_size1=11,
            vocab_size2=3,
            speaker_branch=True,
        )
        joint_probs, joint_lengths = attributed(inputs, lengths)
        self.assertEqual(joint_probs.shape, (2, 12, 31))
        self.assertEqual(joint_lengths.tolist(), [12, 12])

    def test_masked_batch_norm_excludes_padding(self):
        batch_norm = MaskedBatchNorm1d(2, affine=False, momentum=1.0)
        inputs = torch.tensor(
            [[[1.0, 3.0, 100.0, 100.0], [2.0, 4.0, -100.0, -100.0]]]
        )
        valid = torch.tensor([[True, True, False, False]])

        outputs = batch_norm(inputs, valid)

        torch.testing.assert_close(batch_norm.running_mean, torch.tensor([2.0, 3.0]))
        # Stored variance follows PyTorch BatchNorm's unbiased convention.
        torch.testing.assert_close(batch_norm.running_var, torch.tensor([2.0, 2.0]))
        torch.testing.assert_close(
            outputs[0, :, :2],
            torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_channel_layer_norm_is_padding_independent(self):
        source = nn.BatchNorm1d(2)
        layer_norm = ChannelLayerNorm1d(source)
        inputs = torch.tensor(
            [[[1.0, 3.0, 100.0], [2.0, 5.0, -100.0]]]
        )
        valid = torch.tensor([[True, True, False]])
        outputs = layer_norm(inputs, valid)
        expected = layer_norm(inputs[:, :, :2], valid[:, :2])
        torch.testing.assert_close(outputs[:, :, :2], expected)
        torch.testing.assert_close(outputs[:, :, 2], torch.zeros(1, 2))

    def test_fourfold_upsampling_restores_20ms_stride(self):
        encoder = ParakeetWaveformEncoder(
            DummyEncoder(), DummyFrontend(), upsample_factor=4, encoder_subsampling=8
        )
        states, lengths = encoder(torch.randn(2, 24), torch.tensor([24, 16]))
        self.assertEqual(states.shape, (2, 12, 4))
        self.assertEqual(lengths.tolist(), [12, 8])
        self.assertEqual(encoder.timing.output_stride_samples, 320)
        self.assertAlmostEqual(encoder.timing.output_stride_seconds, 0.02)


if __name__ == "__main__":
    unittest.main()
