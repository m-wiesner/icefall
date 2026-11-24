"""Factorized token/speaker scoring for k2 dense graph intersection."""

from __future__ import annotations

from pathlib import Path
import sys
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import _k2
import k2
import torch

_K2_DENSE_INT32_MAX = torch.iinfo(torch.int32).max

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
        capture_path = os.environ.get("SHUFFLE_CTC_CAPTURE_PATH")
        if capture_path:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 0
            )
            capture_path = capture_path.format(rank=rank)
            # This is deliberately synchronous and opt-in.  Copying CUDA
            # tensors to CPU before launching the extension ensures that an
            # illegal access leaves behind the inputs to the failing call,
            # rather than inputs from a later asynchronous call.
            capture = {
                "format_version": 1,
                "arc_row_splits1": a_fsas.arcs.row_splits(1).cpu(),
                "arc_row_splits2": a_fsas.arcs.row_splits(2).cpu(),
                "arc_values": a_fsas.arcs.values().cpu(),
                "token_row_splits": token_row_splits.cpu(),
                "token_scores": token_fsas.scores.detach().cpu(),
                "speaker_scores": speaker_scores.detach().cpu(),
                "token_vocab_size": int(token_vocab_size),
                "output_beam": float(output_beam),
                "max_states": int(max_states),
                "max_arcs": int(max_arcs),
            }
            destination = Path(capture_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(
                f".{destination.name}.tmp.{os.getpid()}"
            )
            torch.save(capture, temporary)
            os.replace(temporary, destination)
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
        # Preserve the acoustic frame selected for each output arc.  The CUDA
        # kernel returns a map into the flattened DenseFsaVec score tensor;
        # convert its dense row back to a frame relative to that supervision.
        # Final arcs have negative maps and receive frame -1.
        dense_row = torch.div(
            map_token.clamp_min(0), token_vocab_size, rounding_mode="floor"
        )
        arc_to_state = out_fsa[0].arcs.shape().row_ids(2)
        state_to_fsa = out_fsa[0].arcs.shape().row_ids(1)
        arc_to_fsa = state_to_fsa.index_select(0, arc_to_state)
        frame = dense_row - token_row_splits.index_select(0, arc_to_fsa)
        frame = torch.where(map_token >= 0, frame, frame.new_full((), -1))
        out_fsa[0].frame = frame.to(torch.int32)
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
    return_diagnostics: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    if supervision_segments is None:
        sequence = torch.arange(input_lengths.numel(), dtype=torch.int32)
        supervision_segments = torch.stack(
            [sequence, torch.zeros_like(sequence), input_lengths.cpu().int()], 1
        )
    # k2's dense intersection uses signed int32 indexes for an internal array
    # whose upper bound is (longest_duration + 1) * total_graph_states.  A
    # perfectly reasonable dynamic batch can overflow that bound even when
    # every cut in the batch is individually safe.  The C++ check is fatal
    # (K2_CHECK), so detect it before entering the extension and fall back to
    # independent intersections.  Summing/averaging the resulting losses is
    # mathematically identical to intersecting an FsaVec, and autograd still
    # accumulates into the original token and speaker tensors.
    segments_cpu = supervision_segments.to(dtype=torch.int32, device="cpu")
    graph_state_splits = target_graphs.arcs.shape().row_splits(1).cpu()
    num_graphs = int(graph_state_splits.numel() - 1)
    if num_graphs != int(segments_cpu.size(0)):
        raise ValueError(
            "Expected one target graph per supervision segment, got "
            f"{num_graphs} graphs and {segments_cpu.size(0)} segments"
        )
    effective_durations = torch.minimum(
        segments_cpu[:, 2],
        token_log_probs.size(1) - segments_cpu[:, 1],
    )
    total_graph_states = int(graph_state_splits[-1].item())
    longest_duration = int(effective_durations.max().item())
    int32_max = _K2_DENSE_INT32_MAX
    address_product = (longest_duration + 1) * total_graph_states
    if num_graphs > 1 and 1 + address_product > int32_max:
        logging.warning(
            "FACTORIZED_INTERSECTION_SPLIT graphs=%d longest_duration=%d "
            "total_graph_states=%d address_product=%d int32_limit=%d; "
            "intersecting cuts independently",
            num_graphs,
            longest_duration,
            total_graph_states,
            address_product,
            int32_max - 1,
        )
        per_cut_losses = []
        per_cut_diagnostics = []
        for i in range(num_graphs):
            sequence_index = int(segments_cpu[i, 0].item())
            segment_i = segments_cpu[i : i + 1].clone()
            segment_i[0, 0] = 0
            result_i = factorized_graph_ctc_loss(
                token_log_probs[sequence_index : sequence_index + 1],
                speaker_log_probs[sequence_index : sequence_index + 1],
                k2.create_fsa_vec([target_graphs[i]]),
                input_lengths[sequence_index : sequence_index + 1],
                supervision_segments=segment_i,
                reduction="none",
                output_beam=output_beam,
                max_states=max_states,
                allow_truncate=allow_truncate,
                use_double_scores=use_double_scores,
                zero_infinity=zero_infinity,
                clamp_negative=clamp_negative,
                memory_reporter=memory_reporter,
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                loss_i, diagnostics_i = result_i
                per_cut_diagnostics.append(diagnostics_i)
            else:
                loss_i = result_i
            per_cut_losses.append(loss_i.reshape(1))
        losses = torch.cat(per_cut_losses)
        if reduction == "sum":
            reduced = losses.sum()
        elif reduction == "mean":
            reduced = losses.mean()
        elif reduction == "none":
            reduced = losses
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
        if not return_diagnostics:
            return reduced
        diagnostics = {
            key: torch.cat([item[key] for item in per_cut_diagnostics])
            for key in per_cut_diagnostics[0]
        }
        return reduced, diagnostics
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
    raw_losses = -lattice.get_tot_scores(
        log_semiring=True, use_double_scores=use_double_scores
    )
    losses = raw_losses
    if memory_reporter is not None:
        memory_reporter("ctc/total_scores")
    if zero_infinity:
        losses = torch.where(torch.isinf(losses), torch.zeros_like(losses), losses)
    if clamp_negative and torch.any(losses < 0):
        logging.warning("Clamping negative graph CTC losses to zero")
        losses = losses.clamp_min(0.0)
    losses = losses.float()
    if reduction == "sum":
        reduced = losses.sum()
    elif reduction == "mean":
        reduced = losses.mean()
    elif reduction == "none":
        reduced = losses
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
    if not return_diagnostics:
        return reduced

    state_splits = lattice.arcs.shape().row_splits(1)
    arc_splits = lattice.arcs.shape().row_splits(2)
    num_states = state_splits[1:] - state_splits[:-1]
    num_arcs = (
        arc_splits.index_select(0, state_splits[1:].to(torch.int64))
        - arc_splits.index_select(0, state_splits[:-1].to(torch.int64))
    )
    diagnostics = {
        # Preserve the values before zero_infinity or negative-loss clamping.
        "raw_losses": raw_losses.detach(),
        "lattice_num_states": num_states.detach(),
        "lattice_num_arcs": num_arcs.detach(),
    }
    return reduced, diagnostics
