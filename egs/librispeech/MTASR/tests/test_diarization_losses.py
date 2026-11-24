import math

import torch

from wavlm_spashl_ctc.train import (
    compute_diarization_objective,
    make_fifo_speaker_targets,
    powerset_class_bitmasks,
)


def test_fifo_targets_preserve_first_onset_order_and_overlap():
    targets, valid, counts = make_fifo_speaker_targets(
        speakers=[["alice", "bob", "alice"]],
        supervision_start_samples=[[0, 20, 60]],
        supervision_num_samples=[[40, 60, 20]],
        input_sample_lengths=torch.tensor([100]),
        output_lengths=torch.tensor([5]),
        max_output_frames=6,
        max_num_speakers=3,
        device=torch.device("cpu"),
    )
    # Frame centers are 10, 30, 50, 70, and 90 samples.  Alice is FIFO slot
    # zero and Bob is slot one, including their overlap at frame center 30.
    expected = torch.tensor(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(targets[0], expected)
    assert torch.equal(valid[0], torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool))
    assert counts == (2,)


def test_bce_masks_padded_frames():
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    valid = torch.tensor([[True, True, False]])
    loss, stats = compute_diarization_objective(
        logits, targets, valid, loss_type="bce", max_overlap=2
    )
    assert torch.allclose(loss, torch.tensor(math.log(2.0)))
    assert stats["powerset_classes"] == 0
    loss.backward()
    assert torch.equal(logits.grad[0, 2], torch.zeros(2))


def test_powerset_classes_and_targets():
    # Empty, 3 singletons, and 3 pairs: 7 classes total.
    assert len(powerset_class_bitmasks(3, 2)) == 7
    logits = torch.zeros(1, 2, 7, requires_grad=True)
    targets = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]])
    valid = torch.tensor([[True, True]])
    loss, stats = compute_diarization_objective(
        logits, targets, valid, loss_type="powerset", max_overlap=2
    )
    assert torch.allclose(loss, torch.tensor(math.log(7.0)))
    assert stats["powerset_classes"] == 7
    loss.backward()


def test_powerset_rejects_more_active_speakers_than_configured():
    logits = torch.zeros(1, 1, 7)
    targets = torch.ones(1, 1, 3)
    valid = torch.ones(1, 1, dtype=torch.bool)
    try:
        compute_diarization_objective(
            logits, targets, valid, loss_type="powerset", max_overlap=2
        )
    except ValueError as error:
        assert "more than --max-overlap=2" in str(error)
    else:
        raise AssertionError("Expected an over-capacity powerset target to fail")
