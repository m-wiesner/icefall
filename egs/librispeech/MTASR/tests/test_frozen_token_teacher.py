import torch

from wavlm_spashl_ctc.train import apply_teacher_local_token_mask


class _ToySentencePiece:
    def encode(self, text):
        return {"FIRST": [1], "SECOND": [2]}[text]


def test_teacher_local_mask_uses_nonoverlapping_windows_and_preserves_scores():
    scores = torch.log_softmax(torch.arange(12, dtype=torch.float32).reshape(1, 4, 3), dim=-1)
    masked, stats = apply_teacher_local_token_mask(
        token_log_probs=scores,
        output_lengths=torch.tensor([4]),
        input_sample_lengths=torch.tensor([80000]),
        texts=[["FIRST", "SECOND"]],
        supervision_start_samples=[[0, 50000]],
        supervision_num_samples=[[10000, 10000]],
        sp=_ToySentencePiece(),
        window_duration=2.0,
        boundary_collar=0.0,
    )

    # Frame centers are 10, 30, 50, and 70 ks. The first two frames share
    # window [0,32k), the third is in [32,64k), and the final window is silent.
    assert torch.isfinite(masked[0, :, 0]).all()  # blank is always legal
    assert torch.isfinite(masked[0, :2, 1]).all()
    assert torch.isneginf(masked[0, 2:, 1]).all()
    assert torch.isfinite(masked[0, 2, 2])
    assert torch.isneginf(masked[0, [0, 1, 3], 2]).all()
    assert torch.equal(masked[torch.isfinite(masked)], scores[torch.isfinite(masked)])
    assert stats["windows"] == 3
    assert stats["blank_only_windows"] == 1


def test_teacher_local_mask_boundary_collar_adds_neighboring_transcript_tokens():
    scores = torch.zeros(1, 2, 3)
    masked, _ = apply_teacher_local_token_mask(
        token_log_probs=scores,
        output_lengths=torch.tensor([2]),
        input_sample_lengths=torch.tensor([64000]),
        texts=[["FIRST", "SECOND"]],
        supervision_start_samples=[[0, 48000]],
        supervision_num_samples=[[8000, 8000]],
        sp=_ToySentencePiece(),
        window_duration=2.0,
        boundary_collar=2.0,
    )

    # A two-second collar around both 2-second windows makes both nearby
    # supervision token types admissible in each window.
    assert torch.isfinite(masked).all()
