"""Factorized token/speaker scoring for k2 dense graph intersection."""

from __future__ import annotations

from pathlib import Path
import sys
import logging
from typing import Callable, List, Optional

import _k2
import k2
import torch

def _load_factorized_extension():
    """Load the optional CUDA extension after it has been built locally."""
    try:
        import _k2_factorized
        return _k2_factorized
    except ImportError:
        pass
    extension_dir = Path(__file__).resolve().parents[1] / ".extensions/factorized_k2"
    sys.path.insert(0, str(extension_dir))
    try:
        import _k2_factorized
        return _k2_factorized
    except ImportError as error:
        raise RuntimeError(
            "The factorized k2 extension is not built. See the README section "
            "'Optional factorized CUDA intersection'."
        ) from error


def _speaker_dense_scores(
    speaker_log_probs: torch.Tensor,
    supervision_segments: torch.Tensor,
    allow_truncate: int,
) -> torch.Tensor:
    """Arrange speaker scores with the same rows as ``k2.DenseFsaVec``."""
    n, t, num_speakers = speaker_log_probs.shape
    indexes = []
    final_rows = []
    current = 0
    for sequence, start, duration in supervision_segments.tolist():
        assert 0 <= sequence < n and 0 <= start < t
        end = min(start + duration, t)
        assert start + duration <= t + allow_truncate
        indexes.append(torch.arange(start, end) + sequence * t)
        indexes.append(torch.tensor([0]))
        current += end - start
        final_rows.append(current)
        current += 1
    indexes = torch.cat(indexes).to(speaker_log_probs.device)
    selected = speaker_log_probs.reshape(-1, num_speakers).index_select(0, indexes)
    ans = torch.cat([selected.new_zeros(selected.size(0), 1), selected], dim=1)
    # Speaker score is absent on the terminating final arc.
    ans[torch.tensor(final_rows, device=ans.device)] = 0
    return ans.contiguous()


class _IntersectDenseFactorized(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a_fsas,
        token_fsas,
        token_row_splits,
        speaker_scores,
        token_vocab_size,
        out_fsa: List[k2.Fsa],
        output_beam,
        max_states,
        max_arcs,
        unused_scores_a,
        unused_scores_token,
        unused_scores_speaker,
    ):
        extension = _load_factorized_extension()
        fsa_tensor, map_a, map_token, map_speaker = (
            extension.intersect_dense_factorized(
                a_fsas.arcs.row_splits(1), a_fsas.arcs.row_splits(2),
                a_fsas.arcs.values(), token_row_splits, token_fsas.scores,
                speaker_scores, token_vocab_size, float(output_beam),
                max_states, max_arcs,
            )
        )
        ragged_arc = _k2.fsa_from_tensor(fsa_tensor)
        out_fsa[0] = k2.Fsa(ragged_arc)
        for name, value in a_fsas.named_tensor_attr(include_scores=False):
            if isinstance(value, torch.Tensor):
                value = _k2.index_select(value, map_a)
            else:
                value, _ = value.index(map_a, axis=0, need_value_indexes=False)
            setattr(out_fsa[0], name, value)
        for name, value in a_fsas.named_non_tensor_attr():
            setattr(out_fsa[0], name, value)
        ctx.maps = map_a, map_token, map_speaker
        ctx.save_for_backward(
            unused_scores_a, unused_scores_token, unused_scores_speaker
        )
        return out_fsa[0].scores

    @staticmethod
    def backward(ctx, output_grad):
        scores_a, scores_token, scores_speaker = ctx.saved_tensors
        map_a, map_token, map_speaker = ctx.maps
        grad_a = torch.zeros_like(scores_a)
        grad_token = torch.zeros_like(scores_token).contiguous()
        grad_speaker = torch.zeros_like(scores_speaker).contiguous()
        _k2.index_add(map_a, output_grad, grad_a)
        _k2.index_add(map_token, output_grad, grad_token.view(-1))
        _k2.index_add(map_speaker, output_grad, grad_speaker.view(-1))
        return (None, None, None, None, None, None, None, None, None,
                grad_a, grad_token, grad_speaker)


def intersect_dense_factorized(
    a_fsas: k2.Fsa,
    token_log_probs: torch.Tensor,
    speaker_log_probs: torch.Tensor,
    supervision_segments: torch.Tensor,
    *,
    output_beam: float,
    max_states: int = 15_000_000,
    max_arcs: int = 1_073_741_824,
    allow_truncate: int = 0,
) -> k2.Fsa:
    """Intersect joint-labelled graphs without constructing joint logits."""
    if token_log_probs.ndim != 3 or speaker_log_probs.ndim != 3:
        raise ValueError("token and speaker log probabilities must be 3-D")
    if token_log_probs.shape[:2] != speaker_log_probs.shape[:2]:
        raise ValueError("token and speaker batch/time dimensions differ")
    segments = supervision_segments.to(dtype=torch.int32, device="cpu")
    token_fsas = k2.DenseFsaVec(
        token_log_probs.float(), segments, allow_truncate=allow_truncate
    )
    speaker_scores = _speaker_dense_scores(
        speaker_log_probs.float(), segments, allow_truncate
    )
    durations = torch.minimum(
        segments[:, 2], token_log_probs.size(1) - segments[:, 1]
    )
    token_row_splits = torch.cat([
        torch.zeros(1, dtype=torch.int32),
        torch.cumsum(durations + 1, dim=0),
    ]).to(device=token_log_probs.device, dtype=torch.int32)
    out = [None]
    _IntersectDenseFactorized.apply(
        a_fsas, token_fsas, token_row_splits, speaker_scores,
        token_log_probs.size(-1), out,
        output_beam, max_states, max_arcs, a_fsas.scores,
        token_fsas.scores, speaker_scores,
    )
    return out[0]


def factorized_graph_ctc_loss(
    token_log_probs: torch.Tensor,
    speaker_log_probs: torch.Tensor,
    target_graphs: k2.Fsa,
    input_lengths: torch.Tensor,
    *,
    supervision_segments: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    output_beam: float = 10.0,
    max_states: int = 25_000_000,
    allow_truncate: int = 0,
    use_double_scores: bool = True,
    zero_infinity: bool = False,
    clamp_negative: bool = False,
    memory_reporter: Optional[Callable[[str], None]] = None,
) -> torch.Tensor:
    if supervision_segments is None:
        sequence = torch.arange(input_lengths.numel(), dtype=torch.int32)
        supervision_segments = torch.stack(
            [sequence, torch.zeros_like(sequence), input_lengths.cpu().int()], 1
        )
    graphs = target_graphs.to(token_log_probs.device)
    if memory_reporter is not None:
        memory_reporter("ctc/target_graphs_on_gpu")
        memory_reporter("ctc/factorized_dense_inputs")
    lattice = intersect_dense_factorized(
        graphs, token_log_probs,
        speaker_log_probs, supervision_segments, output_beam=output_beam,
        max_states=max_states, allow_truncate=allow_truncate,
    )
    if memory_reporter is not None:
        memory_reporter("ctc/intersect_dense_factorized")
    losses = -lattice.get_tot_scores(
        log_semiring=True, use_double_scores=use_double_scores
    )
    if memory_reporter is not None:
        memory_reporter("ctc/total_scores")
    if zero_infinity:
        losses = torch.where(torch.isinf(losses), torch.zeros_like(losses), losses)
    if clamp_negative and torch.any(losses < 0):
        logging.warning("Clamping negative graph CTC losses to zero")
        losses = losses.clamp_min(0.0)
    losses = losses.float()
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    if reduction != "none":
        raise ValueError(f"Unsupported reduction: {reduction}")
    return losses
