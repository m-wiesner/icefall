"""A small PyTorch-style CTC criterion for precompiled k2 target graphs.

The graph is the target: a linear graph recovers ordinary graph-based CTC,
while a shuffle graph accepts every token order permitted by the application.
This module deliberately knows nothing about Lhotse, WavLM, speakers, text
tokenizers, or the MTASR recipe.
"""

from __future__ import annotations

import logging
from typing import Callable, Literal, Optional

import k2
import torch
from torch import Tensor, nn


Reduction = Literal["none", "mean", "sum"]


def _supervision_segments(input_lengths: Tensor) -> Tensor:
    """Create full-utterance k2 supervision segments for a padded batch."""
    if input_lengths.ndim != 1:
        raise ValueError(
            f"input_lengths must have shape (N,), got {tuple(input_lengths.shape)}"
        )
    sequence_idx = torch.arange(input_lengths.numel(), dtype=torch.int32)
    start_frame = torch.zeros_like(sequence_idx)
    return torch.stack(
        (sequence_idx, start_frame, input_lengths.to(dtype=torch.int32, device="cpu")),
        dim=1,
    )


def graph_ctc_loss(
    log_probs: Tensor,
    target_graphs: k2.Fsa,
    input_lengths: Tensor,
    *,
    supervision_segments: Optional[Tensor] = None,
    reduction: Reduction = "mean",
    output_beam: float = 10.0,
    max_states: int = 25_000_000,
    allow_truncate: int = 0,
    use_double_scores: bool = True,
    zero_infinity: bool = False,
    clamp_negative: bool = False,
    memory_reporter: Optional[Callable[[str], None]] = None,
) -> Tensor:
    """Compute negative log total probability of each accepted target graph.

    Args:
      log_probs:
        Log probabilities shaped ``(batch, time, classes)``. Unlike
        :class:`torch.nn.CTCLoss`, time is the second dimension.
      target_graphs:
        A k2 FsaVec with one target graph per batch item. Each graph may accept
        one transcript or a set of legal shuffled transcripts.
      input_lengths:
        Valid time dimension for each batch item.
      supervision_segments:
        Optional explicit k2 segments shaped ``(batch, 3)``. When omitted,
        every sequence starts at frame zero and uses its full input length.
      zero_infinity:
        Match PyTorch CTCLoss terminology: replace infinite losses with zero.
      clamp_negative:
        Clamp small negative losses to zero. This compatibility option is
        enabled by the migrated research recipe but defaults to false so new
        callers notice invalid, non-normalized scores.
    """
    if log_probs.ndim != 3:
        raise ValueError(
            f"log_probs must have shape (N, T, C), got {tuple(log_probs.shape)}"
        )
    if input_lengths.numel() != log_probs.size(0):
        raise ValueError(
            "input_lengths and log_probs batch sizes differ: "
            f"{input_lengths.numel()} != {log_probs.size(0)}"
        )
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"Unsupported reduction: {reduction!r}")
    if supervision_segments is None:
        supervision_segments = _supervision_segments(input_lengths)
    else:
        supervision_segments = supervision_segments.to(dtype=torch.int32, device="cpu")

    graphs = target_graphs.to(log_probs.device)
    if memory_reporter is not None:
        memory_reporter("ctc/target_graphs_on_gpu")
    dense = k2.DenseFsaVec(
        log_probs.float(),
        supervision_segments,
        allow_truncate=allow_truncate,
    )
    if memory_reporter is not None:
        memory_reporter("ctc/dense_fsa_vec")
    lattice = k2.intersect_dense(
        a_fsas=graphs,
        b_fsas=dense,
        output_beam=output_beam,
        max_states=max_states,
        frame_idx_name=None,
    )
    if memory_reporter is not None:
        memory_reporter("ctc/intersect_dense")
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
    return losses


class GraphCTCLoss(nn.Module):
    """Module form of :func:`graph_ctc_loss`, analogous to ``nn.CTCLoss``."""

    def __init__(
        self,
        *,
        reduction: Reduction = "mean",
        output_beam: float = 10.0,
        max_states: int = 25_000_000,
        allow_truncate: int = 0,
        use_double_scores: bool = True,
        zero_infinity: bool = False,
        clamp_negative: bool = False,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.output_beam = output_beam
        self.max_states = max_states
        self.allow_truncate = allow_truncate
        self.use_double_scores = use_double_scores
        self.zero_infinity = zero_infinity
        self.clamp_negative = clamp_negative

    def forward(
        self,
        log_probs: Tensor,
        target_graphs: k2.Fsa,
        input_lengths: Tensor,
        supervision_segments: Optional[Tensor] = None,
    ) -> Tensor:
        return graph_ctc_loss(
            log_probs,
            target_graphs,
            input_lengths,
            supervision_segments=supervision_segments,
            reduction=self.reduction,
            output_beam=self.output_beam,
            max_states=self.max_states,
            allow_truncate=self.allow_truncate,
            use_double_scores=self.use_double_scores,
            zero_infinity=self.zero_infinity,
            clamp_negative=self.clamp_negative,
        )


class ShuffleCTCLoss(GraphCTCLoss):
    """Semantic name for :class:`GraphCTCLoss` when targets are shuffle FSAs.

    The implementation is identical: the supplied target graphs define
    whether the criterion represents ordinary linear CTC targets or a union of
    valid shuffled token paths.
    """
