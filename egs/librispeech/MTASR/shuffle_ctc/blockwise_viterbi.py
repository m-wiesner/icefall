"""Blockwise, lattice-free Viterbi over a timed transcript partial order.

Graph states are canonical global emitted-token bitmasks.  Each time block
materializes only states between its legal entry and exit frontiers.  Exact
boundary scores are carried forward, while local backpointers are retained on
CPU for one final backward trace (there is deliberately no recomputation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class TimedToken:
    token: int
    speaker: int
    segment: int
    center: float
    half_width: float
    predecessor_mask: int = 0
    # Optional occurrence-specific acoustic window.  The partial-order graph
    # still uses ``center``/``half_width`` to decide which token orders are
    # ambiguous, while these bounds control when this particular transcript
    # occurrence may consume a nonblank frame.  This lets blockwise Viterbi
    # reproduce the supervision-level --segment-time-window-margin constraint
    # used by the ordinary factorized Viterbi path.
    emission_start: Optional[float] = None
    emission_end: Optional[float] = None

    @property
    def earliest(self) -> float:
        if self.emission_start is not None:
            return self.emission_start
        return self.center - self.half_width

    @property
    def latest(self) -> float:
        if self.emission_end is not None:
            return self.emission_end
        return self.center + self.half_width


@dataclass
class BlockTrace:
    state_keys: List[Tuple[int, int]]
    src: Tensor
    acoustic_token_indexes: Tensor
    output_token_indexes: Tensor
    backpointers: Tensor
    absolute_frames: Tensor


@dataclass
class BlockwiseAlignment:
    score: float
    token_indexes: Tensor
    frames: Tensor
    traces: List[BlockTrace]
    boundary_state_counts: List[int]


def split_acoustic_blocks(
    blocks: Sequence[Dict[str, Tensor]],
    local_speaker_maps: Sequence[Sequence[int]],
    graph_chunk_duration: float,
) -> Tuple[List[Dict[str, Tensor]], List[Sequence[int]]]:
    """Split posterior cores into independently sized graph/Viterbi blocks.

    Acoustic inference may use a wider center-cropped core than the desired
    graph chunk. This partitions every frame-aligned tensor without rerunning
    the encoder and repeats the acoustic chunk's speaker map for each result.
    """
    if graph_chunk_duration <= 0:
        raise ValueError("graph_chunk_duration must be positive")
    if len(blocks) != len(local_speaker_maps):
        raise ValueError("Need one local speaker mapping per acoustic block")

    output_blocks: List[Dict[str, Tensor]] = []
    output_maps: List[Sequence[int]] = []
    for block, speaker_map in zip(blocks, local_speaker_maps):
        block_start = float(block["start"])
        block_end = float(block["end"])
        frame_duration = float(block["frame_duration"])
        if block_end <= block_start:
            raise ValueError(
                f"Invalid acoustic block [{block_start}, {block_end})"
            )

        sub_start = block_start
        while sub_start < block_end - 1e-9:
            sub_end = min(block_end, sub_start + graph_chunk_duration)
            first_frame = round(sub_start / frame_duration)
            final_frame = round(sub_end / frame_duration)

            absolute_frames = block["absolute_frames"]
            alignment_indexes = torch.nonzero(
                (absolute_frames >= first_frame)
                & (absolute_frames < final_frame),
                as_tuple=False,
            ).squeeze(1)
            sub_block = dict(block)
            for name in (
                "token_log_probs",
                "speaker_log_probs",
                "absolute_frames",
                "local_token_allowed",
            ):
                value = block.get(name)
                if value is not None:
                    sub_block[name] = value.index_select(0, alignment_indexes)

            diarization_frames = block.get("diarization_absolute_frames")
            if diarization_frames is not None:
                diarization_indexes = torch.nonzero(
                    (diarization_frames >= first_frame)
                    & (diarization_frames < final_frame),
                    as_tuple=False,
                ).squeeze(1)
                for name in (
                    "diarization_blank_probability",
                    "diarization_speaker_probability",
                    "diarization_absolute_frames",
                ):
                    value = block.get(name)
                    if value is not None:
                        sub_block[name] = value.index_select(
                            0, diarization_indexes
                        )

            sub_block["start"] = sub_start
            sub_block["end"] = sub_end
            output_blocks.append(sub_block)
            output_maps.append(speaker_map)
            sub_start = sub_end

    return output_blocks, output_maps


def add_partial_order_predecessors(
    tokens: Sequence[TimedToken], *, allow_self_overlap: bool = False
) -> List[TimedToken]:
    """Match the compiler's within-segment and pairwise timing constraints."""
    preds = [0] * len(tokens)
    previous_by_segment = {}
    for i, token in enumerate(tokens):
        if token.segment in previous_by_segment:
            preds[i] |= 1 << previous_by_segment[token.segment]
        previous_by_segment[token.segment] = i
    for i, left in enumerate(tokens):
        for j in range(i + 1, len(tokens)):
            right = tokens[j]
            if left.segment == right.segment:
                continue
            ambiguity = left.half_width + right.half_width
            if (
                abs(left.center - right.center) <= ambiguity
                and (allow_self_overlap or left.speaker != right.speaker)
            ):
                continue
            if left.center < right.center:
                preds[j] |= 1 << i
            else:
                preds[i] |= 1 << j
    return [
        TimedToken(
            token=x.token, speaker=x.speaker, segment=x.segment,
            center=x.center, half_width=x.half_width,
            predecessor_mask=preds[i],
            emission_start=x.emission_start, emission_end=x.emission_end,
        )
        for i, x in enumerate(tokens)
    ]


def _symbol(token: TimedToken, num_token_columns: int, score_speakers: bool) -> int:
    """Return a chunk-invariant CTC symbol (zero remains blank)."""
    if score_speakers:
        return 1 + token.speaker * num_token_columns + token.token
    return 1 + token.token


def _enumerate_block(
    tokens: Sequence[TimedToken],
    entry_keys: Sequence[Tuple[int, int]],
    block_end: float,
    *,
    topology: str,
    num_token_columns: int,
    score_speakers: bool,
):
    """Enumerate the induced frontier graph up to this block's right edge."""
    emit_mask = 0
    required_mask = 0
    for i, token in enumerate(tokens):
        if token.earliest < block_end:
            emit_mask |= 1 << i
        if token.latest < block_end:
            required_mask |= 1 << i

    state_keys = list(dict.fromkeys(entry_keys))
    state_to_id = {key: i for i, key in enumerate(state_keys)}
    queue_pos = 0
    arcs = []
    while queue_pos < len(state_keys):
        emitted, ctc_state = state_keys[queue_pos]
        src = queue_pos
        queue_pos += 1
        # Arc tuple: source, destination, acoustically scored token, output token.
        blank_key = (emitted, 0)
        if blank_key not in state_to_id:
            state_to_id[blank_key] = len(state_keys)
            state_keys.append(blank_key)
        arcs.append((src, state_to_id[blank_key], -1, -1))
        # In the original topology, repeating the current CTC symbol consumes
        # a frame but emits no new transcript label.  Store the originating
        # token index (+1) in the topology state so its time mask is retained.
        if topology == "original" and ctc_state:
            repeat_index = ctc_state - 1
            arcs.append((src, src, repeat_index, -1))

        remaining = emit_mask & ~emitted
        while remaining:
            bit = remaining & -remaining
            token_index = bit.bit_length() - 1
            remaining ^= bit
            if tokens[token_index].predecessor_mask & ~emitted:
                continue
            if topology == "training" and ctc_state != 0:
                continue
            if topology == "original" and ctc_state:
                previous = tokens[ctc_state - 1]
                if _symbol(previous, num_token_columns, score_speakers) == _symbol(
                    tokens[token_index], num_token_columns, score_speakers
                ):
                    # Equal consecutive labels require an intervening blank.
                    continue
            destination_key = (
                emitted | bit, token_index + 1 if topology != "single" else 0
            )
            output_index = token_index
            if destination_key not in state_to_id:
                state_to_id[destination_key] = len(state_keys)
                state_keys.append(destination_key)
            arcs.append((src, state_to_id[destination_key], token_index, output_index))
    exit_ids = [
        i for i, (mask, _) in enumerate(state_keys)
        if mask & required_mask == required_mask
    ]
    return state_keys, state_to_id, arcs, exit_ids


@torch.no_grad()
def blockwise_viterbi(
    tokens: Sequence[TimedToken],
    blocks: Sequence[Dict[str, Tensor]],
    *,
    local_speaker_maps: Sequence[Sequence[int]],
    apply_token_time_mask: bool = True,
    score_speakers: bool = True,
    topology: str = "single",
    search_beam: float | None = None,
    min_active_states: int = 0,
    max_active_states: int = 2_000_000_000,
) -> BlockwiseAlignment:
    """Run exact sequential Viterbi and retain each block's backpointers.

    Every block dictionary contains ``token_log_probs [T,V]``,
    ``speaker_log_probs [T,S]``, ``absolute_frames [T]``, ``start`` and
    ``end`` in seconds. ``local_speaker_maps[c][g]`` maps global speaker g to
    the duration-ranked output slot used for acoustic chunk c.
    """
    if topology not in ("single", "training", "original"):
        raise ValueError(f"Unknown CTC topology: {topology}")
    if search_beam is not None and search_beam <= 0:
        raise ValueError("search_beam must be positive")
    if not 0 <= min_active_states < max_active_states:
        raise ValueError("Expected 0 <= min_active_states < max_active_states")
    if len(blocks) != len(local_speaker_maps):
        raise ValueError("Need one local speaker mapping per block")
    if not blocks:
        raise ValueError("No Viterbi blocks")
    all_mask = (1 << len(tokens)) - 1
    boundary: Dict[Tuple[int, int], float] = {(0, 0): 0.0}
    traces = []
    boundary_counts = [1]

    for block_index, (block, speaker_map) in enumerate(
        zip(blocks, local_speaker_maps)
    ):
        token_scores = block["token_log_probs"]
        speaker_scores = block["speaker_log_probs"]
        frames = block["absolute_frames"].to(token_scores.device)
        state_keys, state_to_id, arcs, exit_ids = _enumerate_block(
            tokens, list(boundary), float(block["end"]), topology=topology,
            num_token_columns=token_scores.size(1), score_speakers=score_speakers,
        )
        if not exit_ids:
            raise RuntimeError(f"Block {block_index} has no legal exit frontier")
        device = token_scores.device
        num_states = len(state_keys)
        current = token_scores.new_full((num_states,), -torch.inf)
        for mask, score in boundary.items():
            current[state_to_id[mask]] = score

        arc_tensor = torch.tensor(arcs, device=device, dtype=torch.long)
        src, dest = arc_tensor[:, 0], arc_tensor[:, 1]
        acoustic_token_index, output_token_index = arc_tensor[:, 2], arc_tensor[:, 3]
        backpointers = torch.full(
            (token_scores.size(0), num_states), -1,
            dtype=torch.int32, device=device,
        )
        invalid = torch.iinfo(torch.int32).max
        nonblank = acoustic_token_index >= 0
        nonblank_token_indexes = acoustic_token_index[nonblank]
        if nonblank_token_indexes.numel():
            token_columns = torch.tensor(
                [tokens[int(i)].token for i in nonblank_token_indexes.tolist()],
                device=device, dtype=torch.long,
            )
            speaker_columns = torch.tensor(
                [speaker_map[tokens[int(i)].speaker]
                 for i in nonblank_token_indexes.tolist()],
                device=device, dtype=torch.long,
            )
            earliest = torch.tensor(
                [tokens[int(i)].earliest for i in nonblank_token_indexes.tolist()],
                device=device,
            )
            latest = torch.tensor(
                [tokens[int(i)].latest for i in nonblank_token_indexes.tolist()],
                device=device,
            )
        for frame_index in range(token_scores.size(0)):
            acoustic = token_scores.new_empty(src.numel())
            acoustic[~nonblank] = token_scores[frame_index, 0]
            if nonblank.any():
                values = (
                    token_scores[frame_index].index_select(0, token_columns)
                    + (speaker_scores[frame_index].index_select(0, speaker_columns)
                       if score_speakers else 0.0)
                )
                if apply_token_time_mask:
                    time = float(frames[frame_index]) * float(block["frame_duration"])
                    allowed = (earliest <= time) & (time <= latest)
                    values.masked_fill_(~allowed, -torch.inf)
                local_allowed = block.get("local_token_allowed")
                if local_allowed is not None:
                    values.masked_fill_(
                        ~local_allowed[frame_index].index_select(
                            0, nonblank_token_indexes
                        ),
                        -torch.inf,
                    )
                acoustic[nonblank] = values
            candidate = current.index_select(0, src) + acoustic
            following = current.new_full((num_states,), -torch.inf)
            following.scatter_reduce_(
                0, dest, candidate, reduce="amax", include_self=True
            )
            wins = torch.isfinite(candidate) & (
                candidate == following.index_select(0, dest)
            )
            arc_ids = torch.arange(src.numel(), device=device, dtype=torch.int32)
            choices = torch.where(wins, arc_ids, torch.full_like(arc_ids, invalid))
            chosen = torch.full(
                (num_states,), invalid, device=device, dtype=torch.int32
            )
            chosen.scatter_reduce_(0, dest, choices, reduce="amin", include_self=True)
            chosen.masked_fill_(chosen == invalid, -1)
            backpointers[frame_index] = chosen
            current = following
            if search_beam is not None and frame_index + 1 < token_scores.size(0):
                finite = torch.isfinite(current)
                count = int(finite.sum().item())
                beam = search_beam
                if count < min_active_states and count:
                    finite_scores = current[finite]
                    keep = min(min_active_states, finite_scores.numel())
                    beam = max(beam, float(finite_scores.max() - finite_scores.topk(keep).values[-1]))
                elif count > max_active_states:
                    finite_scores = current[finite]
                    beam = min(beam, float(finite_scores.max() - finite_scores.topk(max_active_states).values[-1]))
                current.masked_fill_(current < current.max() - beam, -torch.inf)

        finite_exits = [i for i in exit_ids if torch.isfinite(current[i])]
        if not finite_exits:
            raise RuntimeError(
                f"Block {block_index} has no reachable exit after "
                f"{token_scores.size(0)} frames"
            )
        normalizer = float(current[finite_exits].max().item())
        boundary = {
            state_keys[i]: float(current[i].item()) - normalizer
            for i in finite_exits
        }
        boundary_counts.append(len(boundary))
        traces.append(BlockTrace(
            state_keys=state_keys, src=src.cpu(),
            acoustic_token_indexes=acoustic_token_index.cpu(),
            output_token_indexes=output_token_index.cpu(),
            backpointers=backpointers.cpu(),
            absolute_frames=frames.cpu(),
        ))
        del backpointers, current

    final_keys = [key for key in boundary if key[0] == all_mask]
    if not final_keys:
        raise RuntimeError("Final transcript frontier is unreachable")
    selected_key = max(final_keys, key=boundary.__getitem__)
    emitted_tokens = []
    emitted_frames = []
    for trace in reversed(traces):
        state_to_id = {key: i for i, key in enumerate(trace.state_keys)}
        state = state_to_id[selected_key]
        for frame_index in range(trace.backpointers.size(0) - 1, -1, -1):
            arc = int(trace.backpointers[frame_index, state])
            if arc < 0:
                raise RuntimeError("Broken blockwise traceback")
            token_index = int(trace.output_token_indexes[arc])
            if token_index >= 0:
                emitted_tokens.append(token_index)
                emitted_frames.append(int(trace.absolute_frames[frame_index]))
            state = int(trace.src[arc])
        selected_key = trace.state_keys[state]
    if selected_key != (0, 0):
        raise RuntimeError("Blockwise traceback did not reach the global start")
    emitted_tokens.reverse()
    emitted_frames.reverse()
    return BlockwiseAlignment(
        score=boundary[max(final_keys, key=boundary.__getitem__)],
        token_indexes=torch.tensor(emitted_tokens, dtype=torch.long),
        frames=torch.tensor(emitted_frames, dtype=torch.int32),
        traces=traces, boundary_state_counts=boundary_counts,
    )
