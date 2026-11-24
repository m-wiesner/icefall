import torch

from wavlm_spashl_ctc.train import make_uniform_segment_log_probs


def test_soft_segment_log_probs_are_normalized_finite_and_decay_with_distance():
    scores = make_uniform_segment_log_probs(
        start_samples_list=[[100, 500]],
        num_samples_list=[[100, 100]],
        input_sample_lengths=torch.tensor([1000]),
        output_lengths=torch.tensor([10]),
        max_output_frames=10,
        collar=0,
        device=torch.device("cpu"),
        decay_scale=100.0,
        floor_probability=0.01,
    )
    assert scores.shape == (1, 10, 2)
    assert torch.allclose(scores.exp().sum(dim=-1), torch.ones(1, 10))
    assert torch.isfinite(scores).all()
    # The frame centered at sample 350 is equally far from both intervals.
    assert torch.allclose(scores[0, 3].exp(), torch.tensor([0.5, 0.5]))
    # Frames inside one supervision interval strongly prefer that segment, but
    # the other segment retains nonzero support.
    assert scores[0, 1, 0] > scores[0, 1, 1]
    assert scores[0, 5, 1] > scores[0, 5, 0]
    assert torch.all(scores.exp() >= 0.005)


def test_uniform_segment_log_probs_mask_padding_and_expand_by_collar():
    scores = make_uniform_segment_log_probs(
        start_samples_list=[[100], [100, 300]],
        num_samples_list=[[100], [100, 100]],
        input_sample_lengths=torch.tensor([500, 500]),
        output_lengths=torch.tensor([3, 5]),
        max_output_frames=5,
        collar=100,
        device=torch.device("cpu"),
        decay_scale=100.0,
        floor_probability=0.01,
    )
    assert scores.shape == (2, 5, 2)
    assert torch.allclose(scores[0, :3].exp().sum(dim=-1), torch.ones(3))
    assert torch.allclose(scores[1].exp().sum(dim=-1), torch.ones(5))
    assert torch.isfinite(scores[0, :3, :1]).all()
    assert torch.isfinite(scores[1]).all()
    assert torch.isneginf(scores[0, :3, 1]).all()
    # Padded rows are benign and normalized on segment zero.
    assert torch.equal(scores[0, 3:].exp(), torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
