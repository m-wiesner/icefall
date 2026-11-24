"""Exact, lattice-free Viterbi alignment for factorized shuffle graphs.

This module is intentionally independent of :mod:`shuffle_ctc.factorized`.
The training loss continues to use the existing k2 CUDA intersection; this
code only borrows its joint-label convention and acoustic-score factorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import k2
import torch
from torch import Tensor


@dataclass
class ViterbiAlignment:
    score: Tensor
    graph_arc_indexes: Tensor
    labels: Tensor
    aux_labels: Tensor
    frames: Tensor
    active_state_counts: Optional[Tensor] = None


@torch.no_grad()
def token_viterbi_batched(
    graphs: k2.Fsa,
    token_log_probs: Tensor,
    *,
    token_allowed: Optional[Tensor] = None,
) -> list[ViterbiAlignment]:
    """Exact parallel Viterbi for a token-graph FsaVec.

    ``graphs`` contains one graph per target speaker and ``token_log_probs``
    has shape ``[S,T,V]``. All speakers share the encoder frames, while each
    graph and acoustic row advance independently in one batched GPU DP.
    """
    if token_log_probs.ndim != 3:
        raise ValueError("Expected target-speaker scores [S,T,V]")
    num_graphs, time, vocab = token_log_probs.shape
    if len(graphs.shape) != 3 or graphs.shape[0] != num_graphs:
        raise ValueError(
            f"Expected {num_graphs} graphs, got FSA shape {graphs.shape}"
        )
    if token_allowed is not None and token_allowed.shape != token_log_probs.shape:
        raise ValueError("token_allowed must have the same [S,T,V] shape as scores")
    graphs = graphs.to(token_log_probs.device)
    arcs = graphs.arcs.values()
    labels = arcs[:, 2].to(torch.long)
    graph_scores = graphs.scores.float()
    aux = graphs.aux_labels
    if not isinstance(aux, Tensor):
        raise ValueError("token_viterbi_batched requires tensor aux_labels")
    shape = graphs.arcs.shape()
    state_offsets = shape.row_splits(1)[:-1].to(torch.long)
    arc_state_rows = shape.row_ids(2).to(torch.long)
    state_graph = shape.row_ids(1).to(torch.long)
    arc_graph = state_graph.index_select(0, arc_state_rows)
    offsets = state_offsets.index_select(0, arc_graph)
    src = arcs[:, 0].to(torch.long) + offsets
    dest = arcs[:, 1].to(torch.long) + offsets
    num_states = int(shape.row_splits(1)[-1])

    consuming_ids = torch.nonzero(labels >= 0, as_tuple=False).squeeze(1)
    a_label = labels.index_select(0, consuming_ids)
    if a_label.numel() and int(a_label.max()) >= vocab:
        raise ValueError(
            f"Graph label {int(a_label.max())} exceeds token dimension {vocab}"
        )
    a_graph = arc_graph.index_select(0, consuming_ids)
    a_src = src.index_select(0, consuming_ids)
    a_dest = dest.index_select(0, consuming_ids)
    a_graph_scores = graph_scores.index_select(0, consuming_ids)

    backpointers = torch.full(
        (time, num_states), -1, dtype=torch.int32,
        device=token_log_probs.device,
    )
    current = token_log_probs.new_full((num_states,), -torch.inf)
    current[state_offsets] = 0.0
    invalid_arc = torch.iinfo(torch.int32).max
    for frame in range(time):
        acoustic = token_log_probs[a_graph, frame, a_label]
        candidate = current.index_select(0, a_src) + a_graph_scores + acoustic
        if token_allowed is not None:
            candidate = candidate.masked_fill(
                ~token_allowed[a_graph, frame, a_label], -torch.inf
            )
        following = current.new_full((num_states,), -torch.inf)
        following.scatter_reduce_(
            0, a_dest, candidate, reduce="amax", include_self=True
        )
        wins = torch.isfinite(candidate) & (
            candidate == following.index_select(0, a_dest)
        )
        choices = torch.where(
            wins, consuming_ids.to(torch.int32),
            torch.full_like(consuming_ids, invalid_arc, dtype=torch.int32),
        )
        chosen = torch.full(
            (num_states,), invalid_arc, dtype=torch.int32,
            device=token_log_probs.device,
        )
        chosen.scatter_reduce_(
            0, a_dest, choices, reduce="amin", include_self=True
        )
        chosen.masked_fill_(chosen == invalid_arc, -1)
        backpointers[frame] = chosen
        current = following

    results = []
    final_ids = torch.nonzero(labels == -1, as_tuple=False).squeeze(1)
    for graph_index in range(num_graphs):
        graph_finals = final_ids[arc_graph.index_select(0, final_ids) == graph_index]
        candidates = (
            current.index_select(0, src.index_select(0, graph_finals))
            + graph_scores.index_select(0, graph_finals)
        )
        best_pos = int(candidates.argmax().item())
        best_score = candidates[best_pos]
        if not torch.isfinite(best_score):
            results.append(ViterbiAlignment(
                best_score, labels.new_empty(0), labels.new_empty(0),
                aux.new_empty(0), labels.new_empty(0), None,
            ))
            continue
        state = int(src[graph_finals[best_pos]].item())
        path_arcs = []
        for frame in range(time - 1, -1, -1):
            arc = int(backpointers[frame, state].item())
            if arc < 0:
                raise RuntimeError(
                    f"Broken batched traceback for graph {graph_index} at frame {frame}"
                )
            path_arcs.append(arc)
            state = int(src[arc].item())
        if state != int(state_offsets[graph_index]):
            raise RuntimeError("Batched Viterbi traceback did not reach graph start")
        path_arcs.reverse()
        arc_tensor = torch.tensor(path_arcs, device=labels.device, dtype=torch.long)
        path_labels = labels.index_select(0, arc_tensor)
        keep = path_labels > 0
        frames = torch.arange(time, device=labels.device, dtype=torch.int32)[keep]
        results.append(ViterbiAlignment(
            best_score, arc_tensor[keep], path_labels[keep],
            aux.index_select(0, arc_tensor)[keep], frames, None,
        ))
    return results


def _single_fsa_arc_data(graph: k2.Fsa):
    if len(graph.shape) != 3 or graph.shape[0] != 1:
        raise ValueError("factorized_viterbi currently requires a one-FSA FsaVec")
    arcs = graph.arcs.values()
    src = arcs[:, 0].to(torch.long)
    dest = arcs[:, 1].to(torch.long)
    labels = arcs[:, 2].to(torch.long)
    num_states = graph.arcs.shape().row_splits(2).numel() - 1
    aux = graph.aux_labels
    if not isinstance(aux, Tensor):
        raise ValueError("factorized_viterbi requires dense tensor aux_labels")
    return src, dest, labels, graph.scores.float(), aux, num_states


@torch.no_grad()
def factorized_viterbi_batched(
    graphs: k2.Fsa,
    token_log_probs: Tensor,
    speaker_log_probs: Tensor,
    lengths: Tensor,
    *,
    joint_allowed: Optional[Tensor] = None,
    segment_windows: Optional[Tensor] = None,
    original_frame_indices: Optional[Tensor] = None,
    score_speakers: bool = True,
) -> list[ViterbiAlignment]:
    """Exact lattice-free factorized Viterbi for an FsaVec.

    Args:
      graphs:
        An arc-sorted FsaVec with one graph per acoustic sequence.
      token_log_probs:
        Padded token scores with shape ``[B, T, V]``.
      speaker_log_probs:
        Padded speaker scores with shape ``[B, T, S]``.
      lengths:
        The valid frame count for each sequence, shape ``[B]``.
      joint_allowed:
        Optional padded boolean mask ``[B, T, D]`` indexed by graph label.
      segment_windows:
        Optional inclusive frame windows ``[B, U, 2]`` for the ``U`` source
        supervisions in each graph. Positive ``aux_labels`` are interpreted as
        one-based supervision IDs. Blank arcs (aux label zero) remain legal at
        every frame.
      original_frame_indices:
        Optional ``[B, T]`` map from retained acoustic frames to original
        encoder-frame indexes. It is required for correct timing after blank
        removal; without it the retained frame index is used directly.
      score_speakers:
        Add speaker scores on nonblank arcs.  Set false for token-only graphs.

    The frame loop is shared by the whole minibatch, amortizing Python and
    CUDA-launch overhead.  A sequence stops advancing after ``lengths[b]``;
    traceback uses only its valid frames.  The algorithm remains exact and
    stores one int32 predecessor per padded frame and graph state.
    """
    if token_log_probs.ndim != 3 or speaker_log_probs.ndim != 3:
        raise ValueError("Expected token [B,T,V] and speaker [B,T,S] scores")
    if token_log_probs.shape[:2] != speaker_log_probs.shape[:2]:
        raise ValueError("Token and speaker batch/frame dimensions differ")
    if token_log_probs.device != speaker_log_probs.device:
        raise ValueError("Token and speaker scores must share a device")
    batch, time, vocab = token_log_probs.shape
    speakers = speaker_log_probs.size(2)
    lengths = lengths.to(device=token_log_probs.device, dtype=torch.long)
    if lengths.shape != (batch,):
        raise ValueError(f"Expected lengths [{batch}], got {tuple(lengths.shape)}")
    if bool((lengths <= 0).any()) or bool((lengths > time).any()):
        raise ValueError(f"Invalid lengths {lengths.tolist()} for padded time {time}")
    if len(graphs.shape) != 3 or graphs.shape[0] != batch:
        raise ValueError(
            f"Expected an FsaVec with {batch} graphs, got shape {graphs.shape}"
        )
    if joint_allowed is not None:
        if joint_allowed.ndim != 3 or joint_allowed.shape[:2] != (batch, time):
            raise ValueError(
                "joint_allowed must have padded shape [B,T,D], got "
                f"{tuple(joint_allowed.shape)}"
            )
        joint_allowed = joint_allowed.to(
            device=token_log_probs.device, dtype=torch.bool
        )
    if segment_windows is not None:
        if (
            segment_windows.ndim != 3
            or segment_windows.size(0) != batch
            or segment_windows.size(2) != 2
        ):
            raise ValueError(
                "segment_windows must have shape [B,U,2], got "
                f"{tuple(segment_windows.shape)}"
            )
        segment_windows = segment_windows.to(
            device=token_log_probs.device, dtype=torch.long
        )
        if bool((segment_windows[..., 0] > segment_windows[..., 1]).any()):
            raise ValueError("Every segment window must have left <= right")
        if original_frame_indices is None:
            original_frame_indices = torch.arange(
                time, device=token_log_probs.device, dtype=torch.long
            ).unsqueeze(0).expand(batch, -1)
        else:
            if original_frame_indices.shape != (batch, time):
                raise ValueError(
                    "original_frame_indices must have shape [B,T], got "
                    f"{tuple(original_frame_indices.shape)}"
                )
            original_frame_indices = original_frame_indices.to(
                device=token_log_probs.device, dtype=torch.long
            )
    elif original_frame_indices is not None:
        raise ValueError(
            "original_frame_indices is only used together with segment_windows"
        )

    graphs = graphs.to(token_log_probs.device)
    arcs = graphs.arcs.values()
    labels = arcs[:, 2].to(torch.long)
    graph_scores = graphs.scores.float()
    aux = graphs.aux_labels
    if not isinstance(aux, Tensor):
        raise ValueError("factorized_viterbi_batched requires tensor aux_labels")
    shape = graphs.arcs.shape()
    state_offsets = shape.row_splits(1)[:-1].to(torch.long)
    state_rows = shape.row_ids(2).to(torch.long)
    state_graph = shape.row_ids(1).to(torch.long)
    arc_graph = state_graph.index_select(0, state_rows)
    arc_offsets = state_offsets.index_select(0, arc_graph)
    src = arcs[:, 0].to(torch.long) + arc_offsets
    dest = arcs[:, 1].to(torch.long) + arc_offsets
    num_states = int(shape.row_splits(1)[-1].item())

    arc_ids = torch.nonzero(labels >= 0, as_tuple=False).squeeze(1)
    a_src = src.index_select(0, arc_ids)
    a_dest = dest.index_select(0, arc_ids)
    a_label = labels.index_select(0, arc_ids)
    a_graph = arc_graph.index_select(0, arc_ids)
    a_graph_scores = graph_scores.index_select(0, arc_ids)
    a_aux = aux.index_select(0, arc_ids).to(torch.long)
    joint_dim = 1 + speakers * (vocab - 1)
    if a_label.numel() and int(a_label.max()) >= joint_dim:
        raise ValueError(
            f"Graph label {int(a_label.max())} exceeds factorized dimension "
            f"{joint_dim}"
        )
    if (
        joint_allowed is not None
        and a_label.numel()
        and int(a_label.max()) >= joint_allowed.size(2)
    ):
        raise ValueError(
            f"Graph label {int(a_label.max())} exceeds mask dimension "
            f"{joint_allowed.size(2)}"
        )

    nonblank = a_label > 0
    has_nonblank = bool(nonblank.any())
    timed = nonblank & (a_aux > 0)
    has_timed_arcs = bool(timed.any())
    if segment_windows is not None and has_timed_arcs:
        max_segment = int((a_aux[timed] - 1).max().item())
        if max_segment >= segment_windows.size(1):
            raise ValueError(
                f"Graph segment ID {max_segment + 1} exceeds the "
                f"{segment_windows.size(1)} supplied windows"
            )
    token_col = torch.zeros_like(a_label)
    speaker_col = torch.zeros_like(a_label)
    token_col[nonblank] = 1 + (a_label[nonblank] - 1).remainder(vocab - 1)
    speaker_col[nonblank] = torch.div(
        a_label[nonblank] - 1, vocab - 1, rounding_mode="floor"
    )

    backpointers = torch.full(
        (time, num_states), -1, dtype=torch.int32,
        device=token_log_probs.device,
    )
    current = token_log_probs.new_full((num_states,), -torch.inf)
    current[state_offsets] = 0.0
    invalid_arc = torch.iinfo(torch.int32).max
    state_lengths = lengths.index_select(0, state_graph)
    arc_lengths = lengths.index_select(0, a_graph)

    for frame in range(time):
        acoustic = token_log_probs[a_graph, frame, token_col]
        if score_speakers and has_nonblank:
            acoustic[nonblank] += speaker_log_probs[
                a_graph[nonblank], frame, speaker_col[nonblank]
            ]
        candidate = current.index_select(0, a_src) + a_graph_scores + acoustic
        candidate.masked_fill_(frame >= arc_lengths, -torch.inf)
        if segment_windows is not None and has_timed_arcs:
            timed_graphs = a_graph[timed]
            timed_segments = a_aux[timed] - 1
            windows = segment_windows[timed_graphs, timed_segments]
            absolute_frames = original_frame_indices[timed_graphs, frame]
            permitted = (
                (absolute_frames >= windows[:, 0])
                & (absolute_frames <= windows[:, 1])
            )
            timed_candidates = candidate[timed].masked_fill(
                ~permitted, -torch.inf
            )
            candidate[timed] = timed_candidates
        if joint_allowed is not None:
            candidate.masked_fill_(
                ~joint_allowed[a_graph, frame, a_label], -torch.inf
            )
        following = current.new_full((num_states,), -torch.inf)
        following.scatter_reduce_(
            0, a_dest, candidate, reduce="amax", include_self=True
        )
        # Finished sequences retain their final-frame state vector while
        # longer members of the batch continue advancing.
        inactive_states = frame >= state_lengths
        following[inactive_states] = current[inactive_states]

        wins = torch.isfinite(candidate) & (
            candidate == following.index_select(0, a_dest)
        )
        choices = torch.where(
            wins, arc_ids.to(torch.int32),
            torch.full_like(arc_ids, invalid_arc, dtype=torch.int32),
        )
        chosen = torch.full(
            (num_states,), invalid_arc, dtype=torch.int32,
            device=token_log_probs.device,
        )
        chosen.scatter_reduce_(
            0, a_dest, choices, reduce="amin", include_self=True
        )
        chosen.masked_fill_(chosen == invalid_arc, -1)
        backpointers[frame] = chosen
        current = following

    final_ids = torch.nonzero(labels == -1, as_tuple=False).squeeze(1)
    if final_ids.numel() == 0:
        raise ValueError("Graph batch has no final arcs")
    results = []
    for graph_index in range(batch):
        graph_final_ids = final_ids[
            arc_graph.index_select(0, final_ids) == graph_index
        ]
        if graph_final_ids.numel() == 0:
            raise ValueError(f"Graph {graph_index} has no final arc")
        final_candidates = (
            current.index_select(0, src.index_select(0, graph_final_ids))
            + graph_scores.index_select(0, graph_final_ids)
        )
        best_position = int(final_candidates.argmax().item())
        best_score = final_candidates[best_position]
        if not torch.isfinite(best_score):
            results.append(ViterbiAlignment(
                best_score, labels.new_empty(0), labels.new_empty(0),
                aux.new_empty(0), labels.new_empty(0), None,
            ))
            continue
        state = int(src[graph_final_ids[best_position]].item())
        path_arcs = []
        path_frames = []
        graph_time = int(lengths[graph_index].item())
        for frame in range(graph_time - 1, -1, -1):
            arc = int(backpointers[frame, state].item())
            if arc < 0:
                raise RuntimeError(
                    f"Broken batched traceback for graph {graph_index} at "
                    f"frame {frame}, state {state}"
                )
            path_arcs.append(arc)
            path_frames.append(frame)
            state = int(src[arc].item())
        expected_start = int(state_offsets[graph_index].item())
        if state != expected_start:
            raise RuntimeError(
                f"Batched traceback for graph {graph_index} ended at state "
                f"{state}, not {expected_start}"
            )
        path_arcs.reverse()
        path_frames.reverse()
        arc_tensor = torch.tensor(
            path_arcs, device=labels.device, dtype=torch.long
        )
        frame_tensor = torch.tensor(
            path_frames, device=labels.device, dtype=torch.int32
        )
        path_labels = labels.index_select(0, arc_tensor)
        keep = path_labels > 0
        results.append(ViterbiAlignment(
            best_score,
            arc_tensor[keep],
            path_labels[keep],
            aux.index_select(0, arc_tensor)[keep],
            frame_tensor[keep],
            None,
        ))
    return results


@torch.no_grad()
def factorized_viterbi(
    graph: k2.Fsa,
    token_log_probs: Tensor,
    speaker_log_probs: Tensor,
    *,
    joint_allowed: Optional[Tensor] = None,
    segment_windows: Optional[Tensor] = None,
    original_frame_indices: Optional[Tensor] = None,
    score_speakers: bool = True,
    search_beam: Optional[float] = None,
    min_active_states: int = 0,
    max_active_states: int = 2_000_000_000,
) -> ViterbiAlignment:
    """Return the exact best path without constructing an output lattice.

    Args:
      graph:
        A one-element, arc-sorted k2 FsaVec. Joint labels follow
        ``token + speaker * (token_vocab_size - 1)``; label zero is blank.
      token_log_probs:
        ``[T, V]`` token log probabilities, including blank in column zero.
      speaker_log_probs:
        ``[T, S]`` speaker log probabilities. They are not applied to blank.
      joint_allowed:
        Optional boolean ``[T, D]`` mask indexed by graph labels. For an
        attributed graph, ``D = 1 + S * (V - 1)``. For a shuffle-only graph,
        it may instead be the token vocabulary size ``V``.
      segment_windows:
        Optional inclusive frame windows ``[U,2]``. Positive graph aux labels
        identify the one-based source supervision for each emitting arc.
      original_frame_indices:
        Optional length-``T`` map from retained frames to original frame
        indexes, used when blank frames were removed before alignment.
      score_speakers:
        If false, score nonblank arcs using only token log probabilities. This
        is used by shuffle-only graphs while retaining speaker posteriors for
        post-hoc speaker assignment.
      search_beam:
        If set, apply k2-style frame-synchronous search pruning. The dynamic
        beam approaches this value, widens by 1.25 below ``min_active_states``,
        and narrows by 0.8 above ``max_active_states``. No pruning is applied
        on the final frame.

    The dynamic program retains one score per graph state and one int32
    predecessor per time/state. It never materializes alternative lattice
    arcs. Every non-final graph arc consumes one acoustic frame; label -1 arcs
    terminate after the last frame, matching k2 DenseFsaVec semantics.
    """
    if token_log_probs.ndim != 2 or speaker_log_probs.ndim != 2:
        raise ValueError("Expected token [T,V] and speaker [T,S] scores")
    if token_log_probs.size(0) != speaker_log_probs.size(0):
        raise ValueError("Token and speaker frame counts differ")
    if token_log_probs.device != speaker_log_probs.device:
        raise ValueError("Token and speaker scores must share a device")
    graph = graph.to(token_log_probs.device)
    src, dest, labels, graph_scores, aux, num_states = _single_fsa_arc_data(graph)
    time, vocab = token_log_probs.shape
    speakers = speaker_log_probs.size(1)
    joint_dim = 1 + speakers * (vocab - 1)
    if joint_allowed is not None:
        if joint_allowed.ndim != 2 or joint_allowed.size(0) != time:
            raise ValueError(
                f"joint_allowed must have {time} rows, got "
                f"{tuple(joint_allowed.shape)}"
            )
        joint_allowed = joint_allowed.to(
            device=token_log_probs.device, dtype=torch.bool
        )
    if segment_windows is not None:
        if segment_windows.ndim != 2 or segment_windows.size(1) != 2:
            raise ValueError(
                "segment_windows must have shape [U,2], got "
                f"{tuple(segment_windows.shape)}"
            )
        segment_windows = segment_windows.to(
            device=token_log_probs.device, dtype=torch.long
        )
        if bool((segment_windows[:, 0] > segment_windows[:, 1]).any()):
            raise ValueError("Every segment window must have left <= right")
        if original_frame_indices is None:
            original_frame_indices = torch.arange(
                time, device=token_log_probs.device, dtype=torch.long
            )
        else:
            if original_frame_indices.shape != (time,):
                raise ValueError(
                    "original_frame_indices must have shape [T], got "
                    f"{tuple(original_frame_indices.shape)}"
                )
            original_frame_indices = original_frame_indices.to(
                device=token_log_probs.device, dtype=torch.long
            )
    elif original_frame_indices is not None:
        raise ValueError(
            "original_frame_indices is only used together with segment_windows"
        )
    if search_beam is not None:
        if search_beam <= 0:
            raise ValueError(f"search_beam must be positive, got {search_beam}")
        if not 0 <= min_active_states < max_active_states:
            raise ValueError(
                "Expected 0 <= min_active_states < max_active_states, got "
                f"{min_active_states}, {max_active_states}"
            )

    consuming = labels >= 0
    arc_ids = torch.nonzero(consuming, as_tuple=False).squeeze(1)
    a_src, a_dest, a_label = src[arc_ids], dest[arc_ids], labels[arc_ids]
    a_graph_scores = graph_scores[arc_ids]
    a_aux = aux.index_select(0, arc_ids).to(torch.long)
    if a_label.numel() and int(a_label.max()) >= joint_dim:
        raise ValueError(
            f"Graph label {int(a_label.max())} exceeds factorized dimension "
            f"{joint_dim}"
        )
    if (
        joint_allowed is not None
        and a_label.numel()
        and int(a_label.max()) >= joint_allowed.size(1)
    ):
        raise ValueError(
            f"Graph label {int(a_label.max())} exceeds mask dimension "
            f"{joint_allowed.size(1)}"
        )

    # Four bytes per graph-state/frame, rather than storing every competing
    # time-expanded arc as k2's output lattice does.
    backpointers = None
    sparse_states = []
    sparse_arcs = []
    if search_beam is None:
        backpointers = torch.full(
            (time, num_states), -1, dtype=torch.int32,
            device=token_log_probs.device,
        )
    current = token_log_probs.new_full((num_states,), -torch.inf)
    current[0] = 0.0
    invalid_arc = torch.iinfo(torch.int32).max

    nonblank = a_label > 0
    has_nonblank = bool(nonblank.any())
    timed = nonblank & (a_aux > 0)
    has_timed_arcs = bool(timed.any())
    if segment_windows is not None and has_timed_arcs:
        max_segment = int((a_aux[timed] - 1).max().item())
        if max_segment >= segment_windows.size(0):
            raise ValueError(
                f"Graph segment ID {max_segment + 1} exceeds the "
                f"{segment_windows.size(0)} supplied windows"
            )
    token_col = torch.zeros_like(a_label)
    speaker_col = torch.zeros_like(a_label)
    token_col[nonblank] = 1 + (a_label[nonblank] - 1).remainder(vocab - 1)
    speaker_col[nonblank] = torch.div(
        a_label[nonblank] - 1, vocab - 1, rounding_mode="floor"
    )

    default_beam = search_beam
    dynamic_beam = search_beam
    active_counts = []
    for frame in range(time):
        acoustic = token_log_probs[frame].index_select(0, token_col)
        if score_speakers and has_nonblank:
            acoustic[nonblank] += speaker_log_probs[frame].index_select(
                0, speaker_col[nonblank]
            )
        candidate = current.index_select(0, a_src) + a_graph_scores + acoustic
        if segment_windows is not None and has_timed_arcs:
            windows = segment_windows.index_select(0, a_aux[timed] - 1)
            absolute_frame = original_frame_indices[frame]
            permitted = (
                (absolute_frame >= windows[:, 0])
                & (absolute_frame <= windows[:, 1])
            )
            timed_candidates = candidate[timed].masked_fill(
                ~permitted, -torch.inf
            )
            candidate[timed] = timed_candidates
        if joint_allowed is not None:
            candidate = candidate.masked_fill(
                ~joint_allowed[frame].index_select(0, a_label), -torch.inf
            )
        following = current.new_full((num_states,), -torch.inf)
        following.scatter_reduce_(
            0, a_dest, candidate, reduce="amax", include_self=True
        )

        if search_beam is not None:
            active_count = int(torch.isfinite(current).sum().item())
            active_counts.append(active_count)
            current_min = min_active_states
            if frame + 5 >= time:
                current_min = max(min_active_states, max_active_states // 2)
            if active_count <= max_active_states:
                if active_count >= current_min or active_count == 0:
                    dynamic_beam = 0.8 * dynamic_beam + 0.2 * default_beam
                else:
                    dynamic_beam = max(dynamic_beam, default_beam) * 1.25
            elif frame + 5 < time:
                dynamic_beam = min(dynamic_beam, default_beam) * 0.8
            if frame == time - 1:
                dynamic_beam = 1.0e10
            best_arc_score = candidate.max()
            following.masked_fill_(
                following <= best_arc_score - dynamic_beam, -torch.inf
            )

        # Select a deterministic predecessor among exact-score ties. This is a
        # second inexpensive integer reduction and does not retain candidates.
        wins = torch.isfinite(candidate) & (
            candidate == following.index_select(0, a_dest)
        )
        choices = torch.where(
            wins, arc_ids.to(torch.int32),
            torch.full_like(arc_ids, invalid_arc, dtype=torch.int32),
        )
        chosen = torch.full(
            (num_states,), invalid_arc, dtype=torch.int32,
            device=token_log_probs.device,
        )
        chosen.scatter_reduce_(
            0, a_dest, choices, reduce="amin", include_self=True
        )
        chosen.masked_fill_(chosen == invalid_arc, -1)
        if search_beam is None:
            backpointers[frame] = chosen
        else:
            retained_states = torch.nonzero(
                torch.isfinite(following), as_tuple=False
            ).squeeze(1)
            sparse_states.append(retained_states.to(torch.int32))
            sparse_arcs.append(chosen.index_select(0, retained_states))
        current = following

    final_ids = torch.nonzero(labels == -1, as_tuple=False).squeeze(1)
    if final_ids.numel() == 0:
        raise ValueError("Graph has no final arcs")
    final_candidates = current.index_select(0, src[final_ids]) + graph_scores[final_ids]
    best_final_pos = int(final_candidates.argmax().item())
    best_score = final_candidates[best_final_pos]
    if not torch.isfinite(best_score):
        return ViterbiAlignment(
            score=best_score,
            graph_arc_indexes=final_ids.new_empty(0),
            labels=labels.new_empty(0),
            aux_labels=aux.new_empty(0),
            frames=labels.new_empty(0),
            active_state_counts=torch.tensor(active_counts, dtype=torch.int32),
        )

    state = int(src[final_ids[best_final_pos]].item())
    path_arcs = []
    path_frames = []
    for frame in range(time - 1, -1, -1):
        if search_beam is None:
            arc = int(backpointers[frame, state].item())
        else:
            states_at_frame = sparse_states[frame]
            position = int(torch.searchsorted(
                states_at_frame, states_at_frame.new_tensor(state)
            ).item())
            if (
                position == states_at_frame.numel()
                or int(states_at_frame[position].item()) != state
            ):
                arc = -1
            else:
                arc = int(sparse_arcs[frame][position].item())
        if arc < 0:
            raise RuntimeError(
                f"Broken Viterbi traceback at frame {frame}, state {state}"
            )
        path_arcs.append(arc)
        path_frames.append(frame)
        state = int(src[arc].item())
    if state != 0:
        raise RuntimeError(f"Viterbi traceback ended at state {state}, not 0")
    path_arcs.reverse()
    path_frames.reverse()
    arc_tensor = torch.tensor(path_arcs, device=labels.device, dtype=torch.long)
    frame_tensor = torch.tensor(path_frames, device=labels.device, dtype=torch.int32)
    path_labels = labels.index_select(0, arc_tensor)
    keep = path_labels > 0
    return ViterbiAlignment(
        score=best_score,
        graph_arc_indexes=arc_tensor[keep],
        labels=path_labels[keep],
        aux_labels=aux.index_select(0, arc_tensor)[keep],
        frames=frame_tensor[keep],
        active_state_counts=torch.tensor(active_counts, dtype=torch.int32),
    )
