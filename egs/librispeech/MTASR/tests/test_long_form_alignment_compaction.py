import sys
from pathlib import Path

import k2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wavlm_spashl_ctc"))

from align_token import (  # noqa: E402
    add_segment_window_feasibility_frames,
    apply_windowed_token_speaker_mask,
    chunked_factor_posteriors,
    remap_decoding_graph_labels,
    restore_compact_units,
)


def test_segment_window_feasibility_adds_distinct_local_frames():
    kept = [torch.tensor([0, 9])]
    lengths = torch.tensor([10])
    # Two tokens in [2, 4] and two more in overlapping [3, 5].
    windows = torch.tensor([[[2, 4], [3, 5]]])
    augmented, additions = add_segment_window_feasibility_frames(
        kept, lengths, windows, [[2, 2]]
    )
    assert additions == [4]
    assert augmented[0].tolist() == [0, 2, 3, 4, 5, 9]


class _TinyPieceModel:
    def encode(self, text):
        return {"FIRST": [2], "SECOND": [3]}[text]


def test_windowed_token_speaker_mask_uses_original_frame_time():
    # Compact columns: blank, A/2, A/3, B/2, B/3.
    scores = torch.zeros(1, 5, 5)
    # Blank removal returns one variable-length tensor per cut.
    retained_original_frames = [torch.tensor([0, 50, 100, 150, 200])]
    masked, stats = apply_windowed_token_speaker_mask(
        scores=scores,
        lengths=torch.tensor([5]),
        original_frame_indices=retained_original_frames,
        texts=[["FIRST", "SECOND"]],
        speakers=[["A", "B"]],
        supervision_start_samples=[[8000, 40000]],  # 0.5s and 2.5s
        supervision_num_samples=[[8000, 8000]],
        compact_token_ids=[[0, 2, 3]],
        sp=_TinyPieceModel(),
        frame_duration=0.02,
        window_duration=2.0,
        boundary_collar=0.25,
    )
    assert torch.isfinite(masked[:, :, 0]).all()  # blank is always allowed
    assert torch.isfinite(masked[0, :2, 1]).all()  # speaker A, token 2
    assert torch.isneginf(masked[0, :2, 4]).all()
    assert torch.isfinite(masked[0, 2:4, 4]).all()  # speaker B, token 3
    assert torch.isneginf(masked[0, 2:4, 1]).all()
    assert torch.isneginf(masked[0, 4, 1:]).all()  # no nearby supervision
    assert stats["windows"] == 3
    assert stats["blank_only_windows"] == 1


def test_joint_graph_label_round_trip():
    graph = k2.Fsa.from_str(
        "0 1 7 1 0.0\n"
        "1 2 5010 2 0.0\n"
        "2 3 -1 -1 0.0\n"
        "3",
        acceptor=False,
    )
    graphs = k2.create_fsa_vec([graph])
    inverse = remap_decoding_graph_labels(
        graphs, [[0, 7, 11]], [2], original_vocab_size=5000,
        use_speaker=True,
    )
    assert graphs.labels.tolist() == [1, 4, -1]
    units = k2.RaggedTensor([[1, 4]])
    restored = restore_compact_units(units, inverse)
    assert restored.values.tolist() == [7, 5010]


class _ConstantPosteriorModel:
    def __call__(self, waveform, lengths, speaker_weight, return_factors):
        length = int(lengths[0])
        time = torch.arange(length, device=waveform.device).float()
        token = torch.log_softmax(
            torch.stack([time * 0, time * 0 + 1, time * 0 + 2, time * 0 + 3], -1),
            -1,
        )[None]
        speaker = torch.log_softmax(
            torch.stack([time * 0, time * 0 + 1, time * 0 + 2], -1), -1
        )[None]
        return token, speaker, torch.tensor([length], device=waveform.device)


def test_chunk_crossfade_preserves_full_softmax_denominators():
    token, speaker = chunked_factor_posteriors(
        model=_ConstantPosteriorModel(),
        waveform=torch.zeros(1, 450),
        waveform_length=450,
        token_ids=[0, 2],
        speaker_count=2,
        speaker_weight=1.0,
        chunk_duration=25.0,
        chunk_overlap=5.0,
        frame_duration=0.1,
        sample_rate=10,
    )
    assert token.shape == (450, 2)
    assert speaker.shape == (450, 2)
    assert torch.isfinite(token).all() and torch.isfinite(speaker).all()
    # Gather happens after normalization, so retained subset mass stays < 1.
    assert token[0].exp().sum() < 1
    assert speaker[0].exp().sum() < 1


class _RecordingPosteriorModel(_ConstantPosteriorModel):
    def __init__(self):
        self.waveforms = []

    def __call__(self, waveform, lengths, speaker_weight, return_factors):
        self.waveforms.append(waveform.clone())
        return super().__call__(
            waveform, lengths, speaker_weight, return_factors
        )


def test_chunk_context_keeps_centers_and_pads_boundaries():
    model = _RecordingPosteriorModel()
    waveform = torch.arange(1, 11).float().unsqueeze(0)
    token, speaker = chunked_factor_posteriors(
        model=model,
        waveform=waveform,
        waveform_length=10,
        token_ids=[0, 2],
        speaker_count=2,
        speaker_weight=1.0,
        chunk_duration=6.0,
        chunk_overlap=0.0,
        chunk_context=1.0,
        frame_duration=1.0,
        sample_rate=1,
    )
    assert token.shape == (10, 2)
    assert speaker.shape == (10, 2)
    assert len(model.waveforms) == 3  # centers cover [0:4], [4:8], [8:10]
    assert model.waveforms[0].tolist() == [[0, 1, 2, 3, 4, 5]]
    assert model.waveforms[1].tolist() == [[4, 5, 6, 7, 8, 9]]
    assert model.waveforms[2].tolist() == [[8, 9, 10, 0, 0, 0]]
