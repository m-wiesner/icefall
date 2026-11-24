#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                  Wei Kang
#                                                  Mingshuang Luo)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Usage:
  export CUDA_VISIBLE_DEVICES="0,1,2,3"
  ./zipformer_ctc/train.py \
     --exp-dir ./zipformer_ctc/exp \
     --world-size 4 \
     --full-libri 1 \
     --max-duration 500 \
     --num-epochs 30
"""

import argparse
import copy
import logging
import math
import json

from collections import defaultdict
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union, List

import os
import time
from tqdm import tqdm
import torch
import k2
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from lhotse import CutSet
from model import (
    FrozenTokenTeacherSpeakerModel,
    MDCTCModel,
    split_joint_labels,
)
from shuffle_ctc import (
    TimedToken,
    add_partial_order_predecessors,
    blockwise_viterbi,
    split_acoustic_blocks,
    factorized_viterbi,
    factorized_viterbi_batched,
    intersect_dense_factorized,
    token_viterbi_batched,
)
from torch.optim import Adam
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from icefall import diagnostics
from mdctc_graph_compiler import MDCTCGraphCompiler
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.err import raise_grad_scale_is_too_small_error
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    get_parameter_groups_with_lrs,
    setup_logger,
    str2bool,
)

from lhotse.utils import compute_num_frames
import numpy as np
import torchaudio


LRSchedulerType = torch.optim.lr_scheduler._LRScheduler


def fallback_token_duration(
    piece: str,
    seconds_per_character: float,
    frame_duration: float,
    fixed_duration: Optional[float] = None,
) -> float:
    """Estimate a singleton token's duration.

    SentencePiece's U+2581 word-boundary marker is not a spoken character.
    Keep at least one frame for empty/special pieces so that every emitted
    token has a positive duration.  ``fixed_duration`` reproduces the former
    fixed-duration behavior and is useful for controlled ablations.
    """
    if fixed_duration is not None:
        if fixed_duration <= 0:
            raise ValueError(
                "single-token fixed duration must be positive, got "
                f"{fixed_duration}"
            )
        return max(frame_duration, fixed_duration)
    if seconds_per_character <= 0:
        raise ValueError(
            "single-token duration per character must be positive, got "
            f"{seconds_per_character}"
        )
    characters = piece.replace("\u2581", "")
    if piece.startswith("<") and piece.endswith(">"):
        character_count = 1
    else:
        character_count = max(
            1, sum(not character.isspace() for character in characters)
        )
    return max(frame_duration, character_count * seconds_per_character)


def collapse_max_speaker_scores(
    token_log_probs: Tensor,
    speaker_log_probs: Tensor,
    num_speakers: List[int],
) -> Tensor:
    """Score a speaker-free CTC graph without discarding speaker evidence.

    Blank keeps its token-branch score.  Every nonblank token at a frame gets
    the score of the most likely valid speaker at that frame.  This preserves
    the compact token-only graph while allowing speaker confidence to prune
    alternative CTC emission times.
    """
    assert token_log_probs.ndim == 3
    assert speaker_log_probs.ndim == 3
    assert token_log_probs.shape[:2] == speaker_log_probs.shape[:2]
    assert len(num_speakers) == token_log_probs.size(0)

    ans = token_log_probs.clone()
    for i, n in enumerate(num_speakers):
        if not 1 <= n <= speaker_log_probs.size(-1):
            raise ValueError(
                f"num_speakers[{i}]={n} is outside [1, "
                f"{speaker_log_probs.size(-1)}]"
            )
        best_speaker_score = speaker_log_probs[i, :, :n].max(
            dim=-1, keepdim=True
        ).values
        ans[i, :, 1:] += best_speaker_score
    return ans


def alignment_candidate_ids(texts, speakers, sp, token_encoder=None):
    """Return per-cut compact token alphabets and valid speaker counts.

    Token ID 0 (blank) is always first.  The returned scores must be gathered
    *after* the model's full-vocabulary log-softmax so branch denominators and
    token/speaker calibration are preserved.
    """
    encode = sp.encode if token_encoder is None else token_encoder
    token_ids = [
        [0] + sorted({token for text in cut for token in encode(text)})
        for cut in texts
    ]
    speaker_counts = [len(dict.fromkeys(cut_speakers)) for cut_speakers in speakers]
    if any(count < 1 for count in speaker_counts):
        raise ValueError(f"Every cut must have at least one speaker: {speaker_counts}")
    return token_ids, speaker_counts


def apply_windowed_token_speaker_mask(
    scores: Tensor,
    lengths: Tensor,
    original_frame_indices: Tensor,
    texts,
    speakers,
    supervision_start_samples,
    supervision_num_samples,
    compact_token_ids,
    sp,
    frame_duration: float,
    window_duration: float,
    boundary_collar: float,
    sample_rate: int = 16000,
    token_encoder=None,
):
    """Mask joint token/speaker scores using local reference activity.

    Windows are non-overlapping and are defined in the original encoder time
    axis.  This function is intentionally called only after blank-frame
    removal: ``original_frame_indices`` maps each retained frame back to that
    axis.  In a window [a, b), a supervision is active when it intersects
    [a - boundary_collar, b + boundary_collar].  Blank remains legal at every
    frame; nonblank labels are limited to token/speaker pairs from active
    supervisions.
    """
    if window_duration <= 0:
        raise ValueError(f"window_duration must be positive, got {window_duration}")
    if boundary_collar < 0:
        raise ValueError(
            f"boundary_collar must be non-negative, got {boundary_collar}"
        )
    frame_maps_are_tensor = isinstance(original_frame_indices, Tensor)
    if scores.ndim != 3 or (
        frame_maps_are_tensor and original_frame_indices.ndim != 2
    ):
        raise ValueError(
            "Expected scores [N,T,C] and original-frame maps as either a "
            "[N,T] tensor or a per-cut list"
        )
    if not frame_maps_are_tensor and len(original_frame_indices) != scores.size(0):
        raise ValueError(
            f"Got {len(original_frame_indices)} original-frame maps for "
            f"{scores.size(0)} cuts"
        )

    encode = sp.encode if token_encoder is None else token_encoder
    masked = scores.clone()
    total_valid_frames = 0
    total_allowed_nonblank = 0
    total_possible_nonblank = 0
    total_windows = 0
    blank_only_windows = 0

    for i in range(scores.size(0)):
        length = int(lengths[i].item())
        if length == 0:
            continue
        token_ids = list(compact_token_ids[i])
        if not token_ids or token_ids[0] != 0:
            raise ValueError(
                "compact_token_ids must begin with blank ID 0; got "
                f"{token_ids[:4]} for cut {i}"
            )
        num_nonblank_tokens = len(token_ids) - 1
        speaker_to_index = {
            name: index for index, name in enumerate(dict.fromkeys(speakers[i]))
        }
        expected_dim = 1 + len(speaker_to_index) * num_nonblank_tokens
        if scores.size(-1) != expected_dim:
            raise ValueError(
                "Joint score dimension does not match the compact token/speaker "
                f"alphabet for cut {i}: {scores.size(-1)} != {expected_dim}"
            )
        token_to_local = {
            token: index for index, token in enumerate(token_ids[1:])
        }
        supervisions = []
        for text, speaker, start, duration in zip(
            texts[i],
            speakers[i],
            supervision_start_samples[i],
            supervision_num_samples[i],
        ):
            labels = {
                1
                + speaker_to_index[speaker] * num_nonblank_tokens
                + token_to_local[token]
                for token in encode(text)
                if token in token_to_local
            }
            start_seconds = float(start) / sample_rate
            supervisions.append(
                (start_seconds, start_seconds + float(duration) / sample_rate, labels)
            )

        if frame_maps_are_tensor:
            frame_ids = original_frame_indices[i, :length]
        else:
            frame_ids = original_frame_indices[i][:length]
        if frame_ids.numel() != length:
            raise ValueError(
                f"Original-frame map for cut {i} has {frame_ids.numel()} "
                f"entries, expected {length}"
            )
        window_ids = torch.floor(
            frame_ids.to(torch.float64) * frame_duration / window_duration
        ).to(torch.long)
        allowed = torch.zeros(
            (length, scores.size(-1)), dtype=torch.bool, device=scores.device
        )
        allowed[:, 0] = True
        for window_id in torch.unique(window_ids).tolist():
            frame_mask = window_ids == window_id
            window_start = window_id * window_duration
            window_end = window_start + window_duration
            active_labels = set()
            for sup_start, sup_end, labels in supervisions:
                if (
                    sup_start < window_end + boundary_collar
                    and sup_end > window_start - boundary_collar
                ):
                    active_labels.update(labels)
            if active_labels:
                label_index = torch.tensor(
                    sorted(active_labels), device=scores.device, dtype=torch.long
                )
                frame_index = torch.nonzero(frame_mask, as_tuple=False).squeeze(1)
                allowed[frame_index[:, None], label_index[None, :]] = True
            else:
                blank_only_windows += 1
            total_windows += 1

        masked[i, :length] = masked[i, :length].masked_fill(~allowed, -torch.inf)
        total_valid_frames += length
        total_allowed_nonblank += int(allowed[:, 1:].sum().item())
        total_possible_nonblank += length * max(0, scores.size(-1) - 1)

    stats = {
        "valid_frames": total_valid_frames,
        "windows": total_windows,
        "blank_only_windows": blank_only_windows,
        "allowed_nonblank": total_allowed_nonblank,
        "possible_nonblank": total_possible_nonblank,
    }
    return masked, stats


def make_segment_time_windows(
    supervision_start_samples,
    supervision_num_samples,
    margin_seconds: float,
    frame_duration: float,
    sample_rate: int = 16000,
) -> Tensor:
    """Convert supervision intervals to inclusive encoder-frame windows.

    The row order is deliberately unchanged: positive graph ``aux_labels``
    are one-based indexes into these supervision lists.  These windows are
    therefore occurrence-specific, unlike a token mask: two occurrences of
    the same BPE token in different segments receive different legal times.
    """
    if margin_seconds < 0:
        raise ValueError(
            f"segment time-window margin must be non-negative, got "
            f"{margin_seconds}"
        )
    if frame_duration <= 0:
        raise ValueError(f"frame_duration must be positive, got {frame_duration}")
    if len(supervision_start_samples) != len(supervision_num_samples):
        raise ValueError("Start/duration batch sizes differ")

    max_segments = max((len(x) for x in supervision_start_samples), default=0)
    windows = torch.zeros(
        (len(supervision_start_samples), max_segments, 2), dtype=torch.long
    )
    for cut_index, (starts, durations) in enumerate(
        zip(supervision_start_samples, supervision_num_samples)
    ):
        if len(starts) != len(durations):
            raise ValueError(
                f"Cut {cut_index} has {len(starts)} starts but "
                f"{len(durations)} durations"
            )
        for segment_index, (start, duration) in enumerate(zip(starts, durations)):
            if duration < 0:
                raise ValueError(
                    f"Negative duration for cut {cut_index}, segment "
                    f"{segment_index}: {duration}"
                )
            start_seconds = float(start) / sample_rate
            end_seconds = float(start + duration) / sample_rate
            # A frame is legal when its timestamp lies in the closed expanded
            # interval.  Tiny epsilons make exact frame boundaries stable.
            left = math.ceil(
                (max(0.0, start_seconds - margin_seconds) / frame_duration)
                - 1.0e-9
            )
            right = math.floor(
                ((end_seconds + margin_seconds) / frame_duration) + 1.0e-9
            )
            windows[cut_index, segment_index, 0] = left
            windows[cut_index, segment_index, 1] = max(left, right)
    return windows


def make_uniform_segment_log_probs(
    start_samples_list,
    num_samples_list,
    input_sample_lengths: Tensor,
    output_lengths: Tensor,
    max_output_frames: int,
    collar: int,
    device: torch.device,
    decay_scale: float = 16000.0,
    floor_probability: float = 1.0e-4,
) -> Tensor:
    """Recreate the fixed temporal segment factor used during training.

    The distribution is normalized over the real supervision segments in a
    cut. It is flat inside each collar-expanded interval, decays
    exponentially outside that interval, and mixes a small uniform floor so
    every graph path remains finite. Blank arcs do not consume this factor.
    """
    if collar < 0:
        raise ValueError(f"segment-factor collar must be non-negative: {collar}")
    if decay_scale <= 0:
        raise ValueError(
            f"segment-factor decay scale must be positive: {decay_scale}"
        )
    if not 0.0 <= floor_probability < 1.0:
        raise ValueError(
            "segment-factor floor probability must be in [0, 1): "
            f"{floor_probability}"
        )
    if len(start_samples_list) != len(num_samples_list):
        raise ValueError("segment start/duration batch dimensions differ")
    batch_size = len(start_samples_list)
    if batch_size != int(output_lengths.numel()):
        raise ValueError(
            f"Expected {batch_size} output lengths, got {output_lengths.numel()}"
        )
    max_segments = max((len(starts) for starts in start_samples_list), default=0)
    if max_segments == 0:
        raise ValueError("Every cut must contain at least one supervision segment")

    ans = torch.full(
        (batch_size, max_output_frames, max_segments),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    for i, (starts, durations) in enumerate(
        zip(start_samples_list, num_samples_list)
    ):
        if len(starts) != len(durations) or not starts:
            raise ValueError(
                f"Cut {i} has {len(starts)} starts and {len(durations)} durations"
            )
        output_length = int(output_lengths[i].item())
        input_length = int(input_sample_lengths[i].item())
        if output_length <= 0 or input_length <= 0:
            raise ValueError(
                f"Cut {i} has input_length={input_length}, "
                f"output_length={output_length}"
            )
        frame_centers = (
            torch.arange(output_length, device=device, dtype=torch.float32) + 0.5
        ) * (input_length / output_length)
        starts_tensor = torch.tensor(starts, device=device, dtype=torch.float32)
        ends_tensor = starts_tensor + torch.tensor(
            durations, device=device, dtype=torch.float32
        )
        distance = torch.maximum(
            (starts_tensor - collar)[None, :] - frame_centers[:, None],
            frame_centers[:, None] - (ends_tensor + collar)[None, :],
        ).clamp_min_(0)
        local_log_probs = torch.log_softmax(-distance / decay_scale, dim=1)
        if floor_probability > 0.0:
            local_log_probs = torch.logaddexp(
                local_log_probs + math.log1p(-floor_probability),
                torch.full_like(
                    local_log_probs,
                    math.log(floor_probability / len(starts)),
                ),
            )
        ans[i, :output_length, :len(starts)] = local_log_probs
        if output_length < max_output_frames:
            ans[i, output_length:, 0] = 0.0
    return ans


def _crossfade_weights(
    length: int,
    overlap_frames: int,
    has_left_overlap: bool,
    has_right_overlap: bool,
    device: torch.device,
) -> Tensor:
    weights = torch.ones(length, device=device, dtype=torch.float32)
    width = min(overlap_frames, length)
    if width and has_left_overlap:
        weights[:width] *= torch.linspace(
            0.0, 1.0, width + 2, device=device, dtype=torch.float32
        )[1:-1]
    if width and has_right_overlap:
        weights[-width:] *= torch.linspace(
            1.0, 0.0, width + 2, device=device, dtype=torch.float32
        )[1:-1]
    return weights


def chunked_factor_posteriors(
    model,
    waveform: Tensor,
    waveform_length: int,
    token_ids: List[int],
    speaker_count: int,
    speaker_weight: float,
    chunk_duration: float,
    chunk_overlap: float,
    frame_duration: float,
    chunk_context: float = 0.0,
    sample_rate: int = 16000,
    return_nonblank_mass: bool = False,
    speaker_column_map_fn=None,
    speaker_column_map_from_scores_fn=None,
):
    """Run one waveform in chunks and combine recording-specific posteriors.

    Full token and speaker normalizers are evaluated inside every chunk.  Only
    then are transcript-specific token columns and valid speaker columns
    gathered, keeping the calibrated log-probability scale of both branches.

    If ``chunk_context`` is positive, ``chunk_duration`` is the complete input
    duration, including that much left and right context.  Consecutive chunks
    advance by ``chunk_duration - 2 * chunk_context`` and only their center
    frames are retained.  Missing context at recording boundaries is padded
    with silence.  If context is zero, the original overlap-and-crossfade mode
    is used.
    """
    if not 0.0 <= chunk_context < chunk_duration / 2:
        raise ValueError(
            "alignment chunk context must satisfy 0 <= 2*context < duration; "
            f"got context={chunk_context} duration={chunk_duration}"
        )
    if not 0.0 <= chunk_overlap < chunk_duration:
        raise ValueError(
            "alignment chunk overlap must satisfy 0 <= overlap < duration; "
            f"got {chunk_overlap} and {chunk_duration}"
        )
    chunk_samples = round(chunk_duration * sample_rate)
    context_samples = round(chunk_context * sample_rate)
    if chunk_context > 0:
        core_samples = chunk_samples - 2 * context_samples
        context_frames = round(chunk_context / frame_duration)
        total_frames = round(
            (waveform_length / sample_rate) / frame_duration
        )
        token_index = torch.tensor(
            token_ids, device=waveform.device, dtype=torch.long
        )
        token_pieces = []
        speaker_pieces = []
        nonblank_mass_pieces = []
        core_start = 0
        while core_start < waveform_length:
            requested_start = core_start - context_samples
            requested_end = requested_start + chunk_samples
            source_start = max(0, requested_start)
            source_end = min(waveform_length, requested_end)
            destination_start = source_start - requested_start
            chunk = waveform.new_zeros((waveform.size(0), chunk_samples))
            chunk[..., destination_start:destination_start + source_end - source_start] = (
                waveform[..., source_start:source_end]
            )
            chunk_len = torch.tensor([chunk_samples], device=waveform.device)
            token_full, speaker_full, output_lens = model(
                chunk,
                chunk_len,
                speaker_weight=1.0,
                return_factors=True,
            )
            valid = int(output_lens[0].item())
            core_global_frame = round(
                (core_start / sample_rate) / frame_duration
            )
            core_frames = min(
                round(core_samples / sample_rate / frame_duration),
                total_frames - core_global_frame,
            )
            core_end_frame = min(context_frames + core_frames, valid)
            if core_end_frame - context_frames != core_frames:
                raise RuntimeError(
                    "Context-cropped chunk produced too few frames: "
                    f"valid={valid}, crop=[{context_frames}, "
                    f"{context_frames + core_frames})"
                )
            if speaker_column_map_from_scores_fn is not None:
                column_map = speaker_column_map_from_scores_fn(
                    core_global_frame,
                    speaker_full[0, context_frames:core_end_frame].float(),
                )
                speaker_full = speaker_full.index_select(
                    -1,
                    torch.tensor(
                        column_map, device=speaker_full.device, dtype=torch.long
                    ),
                )
            elif speaker_column_map_fn is not None:
                column_map = speaker_column_map_fn(
                    requested_start / sample_rate, requested_end / sample_rate
                )
                speaker_full = speaker_full.index_select(
                    -1,
                    torch.tensor(
                        column_map, device=speaker_full.device, dtype=torch.long
                    ),
                )
            token_pieces.append(
                token_full[0, context_frames:core_end_frame]
                .index_select(-1, token_index).float()
            )
            speaker_pieces.append(
                speaker_full[0, context_frames:core_end_frame, :speaker_count]
                .float()
            )
            nonblank_mass_pieces.append(
                token_full[0, context_frames:core_end_frame, 1:]
                .logsumexp(-1).float()
            )
            del token_full, speaker_full
            core_start += core_samples

        token_log_probs = torch.cat(token_pieces, dim=0)[:total_frames]
        speaker_log_probs = torch.cat(speaker_pieces, dim=0)[:total_frames]
        nonblank_log_mass = torch.cat(nonblank_mass_pieces, dim=0)[:total_frames]
        if token_log_probs.size(0) != total_frames:
            raise RuntimeError(
                "Context-cropped chunks did not cover the recording: "
                f"got {token_log_probs.size(0)} frames, expected {total_frames}"
            )
        speaker_log_probs *= speaker_weight
        if return_nonblank_mass:
            return token_log_probs, speaker_log_probs, nonblank_log_mass
        return token_log_probs, speaker_log_probs

    overlap_samples = round(chunk_overlap * sample_rate)
    step_samples = chunk_samples - overlap_samples
    overlap_frames = round(chunk_overlap / frame_duration)
    starts = []
    start = 0
    while True:
        starts.append(start)
        if start + chunk_samples >= waveform_length:
            break
        start += step_samples

    pieces = []
    token_index = torch.tensor(token_ids, device=waveform.device, dtype=torch.long)
    for piece_index, start in enumerate(starts):
        end = min(start + chunk_samples, waveform_length)
        chunk = waveform[..., start:end]
        chunk_len = torch.tensor([end - start], device=waveform.device)
        token_full, speaker_full, output_lens = model(
            chunk,
            chunk_len,
            speaker_weight=1.0,
            return_factors=True,
        )
        valid = int(output_lens[0].item())
        if speaker_column_map_fn is not None:
            column_map = speaker_column_map_fn(
                start / sample_rate, end / sample_rate
            )
            speaker_full = speaker_full.index_select(
                -1,
                torch.tensor(
                    column_map, device=speaker_full.device, dtype=torch.long
                ),
            )
        token = token_full[0, :valid].index_select(-1, token_index).float()
        speaker = speaker_full[0, :valid, :speaker_count].float()
        nonblank_mass = token_full[0, :valid, 1:].exp().sum(-1).float()
        # Drop the full branch outputs immediately; long-form storage remains
        # proportional only to the recording-specific alphabet.
        del token_full, speaker_full
        global_start = round((start / sample_rate) / frame_duration)
        weights = _crossfade_weights(
            valid,
            overlap_frames,
            has_left_overlap=piece_index > 0,
            has_right_overlap=piece_index + 1 < len(starts),
            device=waveform.device,
        )
        pieces.append((global_start, token, speaker, nonblank_mass, weights))

    total_frames = max(start + token.size(0) for start, token, _, _, _ in pieces)
    token_sum = torch.zeros(
        total_frames, len(token_ids), device=waveform.device, dtype=torch.float32
    )
    speaker_sum = torch.zeros(
        total_frames, speaker_count, device=waveform.device, dtype=torch.float32
    )
    weight_sum = torch.zeros(total_frames, device=waveform.device, dtype=torch.float32)
    nonblank_mass_sum = torch.zeros_like(weight_sum)
    for start, token, speaker, nonblank_mass, weights in pieces:
        end = start + token.size(0)
        token_sum[start:end] += token.exp() * weights[:, None]
        speaker_sum[start:end] += speaker.exp() * weights[:, None]
        nonblank_mass_sum[start:end] += nonblank_mass * weights
        weight_sum[start:end] += weights
    if (weight_sum <= 0).any():
        raise RuntimeError("Chunk crossfade left one or more output frames uncovered")
    token_log_probs = (token_sum / weight_sum[:, None]).clamp_min_(1e-30).log_()
    speaker_log_probs = (speaker_sum / weight_sum[:, None]).clamp_min_(1e-30).log_()
    speaker_log_probs *= speaker_weight
    if return_nonblank_mass:
        nonblank_log_mass = (
            nonblank_mass_sum / weight_sum
        ).clamp_min_(1e-30).log_()
        return token_log_probs, speaker_log_probs, nonblank_log_mass
    return token_log_probs, speaker_log_probs


def remap_decoding_graph_labels(
    decoding_graphs: k2.Fsa,
    token_ids: List[List[int]],
    speaker_counts: List[int],
    original_vocab_size: int,
    use_speaker: bool,
):
    """Map graph input labels to compact, recording-specific dense columns."""
    inverse_maps = []
    for index, (tokens, num_speakers) in enumerate(zip(token_ids, speaker_counts)):
        token_to_local = {token: local for local, token in enumerate(tokens)}
        num_nonblank = len(tokens) - 1
        if use_speaker:
            inverse = [0] + [
                token + speaker * (original_vocab_size - 1)
                for speaker in range(num_speakers)
                for token in tokens[1:]
            ]
        else:
            inverse = tokens
        graph = decoding_graphs[index]
        labels = graph.labels
        positive = labels > 0
        old = labels[positive].to(torch.int64)
        mapped = torch.empty_like(old)
        if use_speaker:
            old_speaker = torch.div(
                old - 1, original_vocab_size - 1, rounding_mode="floor"
            )
            old_token = old - old_speaker * (original_vocab_size - 1)
            if old_speaker.numel() and old_speaker.max().item() >= num_speakers:
                raise ValueError("Graph contains a speaker outside the compact mask")
            mapped_values = []
            for token, speaker in zip(old_token.tolist(), old_speaker.tolist()):
                if token not in token_to_local:
                    raise ValueError(f"Graph token {token} is absent from transcript map")
                mapped_values.append(token_to_local[token] + speaker * num_nonblank)
            mapped.copy_(torch.tensor(mapped_values, device=mapped.device))
        else:
            mapped_values = []
            for token in old.tolist():
                if token not in token_to_local:
                    raise ValueError(f"Graph token {token} is absent from transcript map")
                mapped_values.append(token_to_local[token])
            mapped.copy_(torch.tensor(mapped_values, device=mapped.device))
        labels[positive] = mapped.to(labels.dtype)
        inverse_maps.append(torch.tensor(inverse, device=labels.device, dtype=torch.int64))
    return inverse_maps


def restore_compact_units(units: k2.RaggedTensor, inverse_maps):
    restored = []
    for index in range(units.shape.dim0):
        unit_indexes = units[index].to(torch.int64)
        inverse = inverse_maps[index].to(unit_indexes.device)
        restored.append(inverse[unit_indexes])
    values = torch.cat(restored) if restored else units.values
    return k2.RaggedTensor(units.shape, values.to(units.values.dtype))


def filter_blank_dominated_frames(
    scores: Tensor,
    lengths: Tensor,
    threshold: Optional[float],
    minimum_frames: Optional[List[int]] = None,
    keep_fraction: Optional[float] = None,
) -> Tuple[Tensor, Tensor, List[Tensor]]:
    """Remove frames whose token-blank posterior exceeds ``threshold``.

    Returns padded filtered scores, filtered lengths, and (for each sequence)
    a map from a filtered frame index to its original encoder-frame index.
    Padding is outside the supervision lengths and is never intersected.
    """
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"blank frame filter threshold must be in [0, 1], got {threshold}"
        )
    if keep_fraction is not None and not 0.0 < keep_fraction <= 1.0:
        raise ValueError(
            f"blank frame keep fraction must be in (0, 1], got {keep_fraction}"
        )
    if threshold is None and keep_fraction is None:
        raise ValueError("Either threshold or keep_fraction must be provided")
    if scores.ndim != 3 or lengths.ndim != 1:
        raise ValueError(
            f"Expected scores [N,T,C] and lengths [N], got "
            f"{tuple(scores.shape)} and {tuple(lengths.shape)}"
        )

    if minimum_frames is None:
        minimum_frames = [1] * scores.size(0)
    if len(minimum_frames) != scores.size(0):
        raise ValueError(
            f"minimum_frames has {len(minimum_frames)} entries for "
            f"a batch of size {scores.size(0)}"
        )

    kept_scores = []
    original_frame_indices = []
    for i, length_t in enumerate(lengths):
        length = int(length_t.item())
        blank_prob = scores[i, :length, 0].exp()
        minimum = min(length, max(1, int(minimum_frames[i])))
        if keep_fraction is not None:
            required = min(
                length, max(minimum, math.ceil(keep_fraction * length))
            )
            keep = torch.topk(blank_prob, k=required, largest=False).indices
            keep = keep.sort().values
        else:
            keep = torch.nonzero(
                blank_prob <= threshold, as_tuple=False
            ).squeeze(1)
        if keep.numel() < minimum:
            # Add the least-blank-like omitted frames, then restore temporal
            # order.  A transcript with N emissions cannot align to <N frames.
            keep = torch.topk(blank_prob, k=minimum, largest=False).indices
            keep = keep.sort().values
        kept_scores.append(scores[i, keep])
        original_frame_indices.append(keep)

    filtered_lengths = lengths.new_tensor(
        [item.size(0) for item in kept_scores]
    )
    max_length = int(filtered_lengths.max().item())
    filtered = scores.new_full(
        (scores.size(0), max_length, scores.size(2)), float("-inf")
    )
    for i, item in enumerate(kept_scores):
        filtered[i, : item.size(0)] = item
    return filtered, filtered_lengths, original_frame_indices


def select_frames_with_map(
    values: Tensor,
    frame_indices: List[Tensor],
    filtered_lengths: Tensor,
) -> Tensor:
    """Apply a blank-filter frame map to another frame-synchronous tensor."""
    ans = values.new_zeros(
        values.size(0), int(filtered_lengths.max().item()), values.size(2)
    )
    for i, indexes in enumerate(frame_indices):
        ans[i, : indexes.numel()] = values[i, indexes]
    return ans


def add_segment_window_feasibility_frames(
    frame_indices: List[Tensor],
    original_lengths: Tensor,
    segment_windows: Tensor,
    segment_token_counts: List[List[int]],
) -> Tuple[List[Tensor], List[int]]:
    """Ensure one distinct retained frame exists per reference token.

    Global blank-posterior thresholding can retain enough frames overall but
    leave a tightly constrained supervision with fewer frames than BPE tokens.
    For the one-state topology, a complete path exists only if all token
    occurrences can be matched to distinct frames inside their occurrence-
    specific timing windows.  Use earliest-deadline interval matching over the
    original frame axis, then add only matched frames omitted by blank pruning.
    """
    if len(frame_indices) != len(segment_token_counts):
        raise ValueError("Frame-map and segment-token batch sizes differ")
    augmented = []
    additions = []
    for cut_index, (kept, counts) in enumerate(
        zip(frame_indices, segment_token_counts)
    ):
        length = int(original_lengths[cut_index].item())
        if len(counts) > segment_windows.size(1):
            raise ValueError("Missing one or more segment timing windows")
        intervals = []
        for segment_index, count in enumerate(counts):
            left = max(0, int(segment_windows[cut_index, segment_index, 0]))
            right = min(
                length - 1,
                int(segment_windows[cut_index, segment_index, 1]),
            )
            if count and right < left:
                raise RuntimeError(
                    f"Segment {segment_index} in cut {cut_index} has no legal "
                    "encoder frame"
                )
            intervals.extend((left, right) for _ in range(count))
        intervals.sort(key=lambda item: (item[1], item[0]))

        # Disjoint-set successor structure: find(x) is the first unmatched
        # original frame at or after x.  This avoids quadratic list removal on
        # hour-long recordings.
        parent = list(range(length + 1))

        def find(value):
            root = value
            while parent[root] != root:
                root = parent[root]
            while parent[value] != value:
                following = parent[value]
                parent[value] = root
                value = following
            return root

        forced = []
        for left, right in intervals:
            frame = find(left)
            if frame > right:
                raise RuntimeError(
                    "Segment timing windows cannot assign distinct encoder "
                    f"frames in cut {cut_index}; failed interval [{left}, {right}]"
                )
            forced.append(frame)
            parent[frame] = find(frame + 1)

        forced_tensor = torch.tensor(
            forced, device=kept.device, dtype=kept.dtype
        )
        merged = torch.unique(torch.cat((kept, forced_tensor)), sorted=True)
        augmented.append(merged)
        additions.append(int(merged.numel() - kept.numel()))
    return augmented, additions


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=55,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    
    parser.add_argument(
        "--suffix",
        type=str,
        help="decoding suffix."
    )

    parser.add_argument(
        "--blank-weight",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer_mdctc/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_5000/bpe.model",
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_bpe_5000",
        help="""The lang dir
        It contains language related input files such as
        "lexicon.txt"
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--frame-duration",
        type=float,
        default=0.04,
    )

    parser.add_argument(
        "--single-token-duration-per-character",
        type=float,
        default=0.08,
        help=(
            "Fallback seconds per visible SentencePiece character when a "
            "transcript segment has only one emitted BPE token. The default "
            "is the approximate duration-weighted character rate measured "
            "from the NOTSOFAR reference word alignments."
        ),
    )

    parser.add_argument(
        "--single-token-fixed-duration",
        type=float,
        default=None,
        help=(
            "If set, use this fixed duration in seconds for singleton BPE "
            "segments instead of the character-proportional fallback. Set "
            "this to 0.2 to reproduce the previous behavior."
        ),
    )

    parser.add_argument(
        "--alignment-bpe-max-piece-characters",
        type=int,
        default=0,
        help=(
            "If positive, deterministically decompose every canonical BPE "
            "piece longer than this many visible characters into existing "
            "vocabulary pieces. Zero preserves the original SentencePiece "
            "encoding. The word-boundary marker is not counted."
        ),
    )

    parser.add_argument(
        "--collar",
        type=int,
        default=64000,
        help=(
            "The graph compiler collar in input waveform samples "
            "(64000 is 4 seconds at 16 kHz)."
        )
    )

    parser.add_argument(
        "--edge-collar",
        type=int,
        default=None,
        help="Graph edge collar in waveform samples; None follows --collar.",
    )
    parser.add_argument(
        "--interval-model",
        choices=("fixed", "brownian_bridge"),
        default="fixed",
        help="Partial-order token interval model used by the graph compiler.",
    )

    parser.add_argument(
        "--max-num-spks",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--oracle-num-spks",
        type=int,
        default=None,
        help=(
            "Restrict post-hoc speaker recovery to this many output slots. "
            "Use 0 to infer the known speaker count independently per cut."
        ),
    )
    parser.add_argument(
        "--blockwise-speaker-recovery",
        choices=("oracle_map", "identity_map", "embedding_cluster", "vbx"),
        default="oracle_map",
        help=(
            "For token-only blockwise alignment, either map local speaker "
            "slots with reference supervision timing (legacy), retain the "
            "raw local slot index across all chunks (identity_map), or stitch "
            "chunk-local slots with ECAPA/AHC, or use the bundled "
            "ResNet101/PLDA VBx pipeline. The latter two use no reference "
            "local/global map."
        ),
    )
    parser.add_argument(
        "--blockwise-speaker-attribution-unit",
        choices=("segment", "word"),
        default="segment",
        help=(
            "Unit over which token-emission speaker log posteriors are summed "
            "after token-only blockwise alignment. 'segment' makes one speaker "
            "decision per input supervision; 'word' makes one decision per "
            "reconstructed SentencePiece word."
        ),
    )
    parser.add_argument(
        "--speaker-embedding-model-dir",
        type=str,
        default=os.environ.get(
            "SPEECHBRAIN_ECAPA_DIR",
            str(
                Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
                / "hub/models--speechbrain--spkrec-ecapa-voxceleb/snapshots"
                / "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
            ),
        ),
    )
    parser.add_argument(
        "--speaker-embedding-min-seconds", type=float, default=0.0,
        help=("Minimum selected speech. Zero uses the embedding model's "
              "shortest valid input, matching pyannote."),
    )
    parser.add_argument(
        "--speaker-vbx-dir", type=str,
        default=str(Path(__file__).resolve().parent.parent / ".tools/VBx"),
        help="Official BUTSpeechFIT/VBx checkout containing bundled models.",
    )
    parser.add_argument(
        "--speaker-vbx-min-clean-seconds", type=float, default=1.44,
        help=("Per chunk-local slot, minimum preferred clean speech used for "
              "VBx VAD before adding less-clean and then overlapped token regions."),
    )
    parser.add_argument(
        "--speaker-embedding-max-seconds", type=float, default=0.0,
        help="Maximum selected speech. Zero means no cap, matching pyannote.",
    )
    parser.add_argument(
        "--speaker-clustering-threshold", type=float,
        default=0.7153814381597874,
    )
    parser.add_argument("--speaker-clustering-min-cluster-size", type=int, default=15)
    parser.add_argument(
        "--speaker-clustering-oracle-num-speakers",
        type=str2bool,
        default=True,
        help=("Force the number of AHC clusters to the number of transcript "
              "speakers. False estimates it from the clustering threshold."),
    )
    parser.add_argument(
        "--speaker-stitching-blank-threshold", type=float, default=0.4,
        help="Mark a full-rate frame as speech when p(blank) is below this.",
    )
    parser.add_argument(
        "--speaker-stitching-speaker-threshold", type=float, default=0.45,
        help="Activate each local speaker whose posterior exceeds this.",
    )
    parser.add_argument(
        "--speaker-stitching-smoothing", type=float, default=0.5,
        help="Seconds of max-pool dilation on either side of each speaker track.",
    )
    parser.add_argument(
        "--speaker-stitching-segmentation-source",
        choices=("posterior", "token_alignment"),
        default="posterior",
        help=("Build embedding masks from thresholded speaker posteriors or "
              "from locally attributed token-alignment spans."),
    )
    parser.add_argument(
        "--speaker-stitching-token-gap", type=float, default=0.25,
        help="Split token-induced speech spans at gaps larger than this many seconds.",
    )
    parser.add_argument(
        "--speaker-stitching-token-neighbors", type=int, default=2,
        help=("For token-derived embedding regions, prefer a token only when "
              "this many neighboring emissions on each available side have "
              "the same confident speaker argmax."),
    )
    parser.add_argument(
        "--speaker-stitching-token-speaker-confidence", type=float, default=None,
        help=("Minimum speaker posterior for a token and its neighbors to form "
              "a preferred clean embedding region. By default, reuse "
              "--speaker-stitching-speaker-threshold."),
    )
    parser.add_argument(
        "--speaker-stitching-timing-calibration", type=str, default=None,
        help="Optional saved dev timing-offset JSON applied to token-induced spans.",
    )

    parser.add_argument(
        "--max-overlap",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--total-max-overlaps",
        type=int,
        default=3600,
    )

    parser.add_argument(
        "--factorized-token-speaker-output",
        "--use-hat",
        dest="factorized_token_speaker_output",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--frozen-token-teacher-checkpoint",
        type=str,
        default=None,
        help=(
            "Reconstruct a FrozenTokenTeacherSpeakerModel for evaluation. "
            "The student checkpoint in --exp-dir intentionally excludes the "
            "immutable token teacher, so this must name the same token-only "
            "checkpoint used during training."
        ),
    )

    parser.add_argument(
        "--use-layer-norm",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--use-large",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--downsample",
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        "--beam-size",
        type=float,
        default=24,
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=25000000,
    )

    parser.add_argument(
        "--search-beam",
        type=float,
        default=20,
    )

    parser.add_argument(
        "--intersection-method",
        choices=(
            "dense", "dense_pruned", "online_dense_pruned",
            "factorized_viterbi", "factorized_viterbi_pruned",
        ),
        default="dense",
    )
    parser.add_argument(
        "--target-speaker-alignment",
        type=str2bool,
        default=False,
        help=(
            "Build one transcript graph per globally ordered speaker and run "
            "them as a batched exact Viterbi alignment. Other-speaker token "
            "mass is folded into each target speaker's blank score."
        ),
    )
    parser.add_argument(
        "--independent-segment-alignment",
        type=str2bool,
        default=False,
        help=(
            "Build one linear CTC graph per reference supervision and align "
            "all of them in one exact batched Viterbi call after running the "
            "encoder once. With --ignore-speaker=false, each graph is scored "
            "with that supervision's known globally ordered speaker slot; "
            "with --ignore-speaker=true, only token scores are used."
        ),
    )
    parser.add_argument(
        "--speaker-posterior-diarization",
        type=str2bool,
        default=False,
        help="Score graph-free frame-level diarization from the speaker posteriors.",
    )
    parser.add_argument(
        "--diarization-blank-thresholds",
        type=str,
        default="0.5,0.9,0.99,0.999",
    )
    parser.add_argument(
        "--diarization-speaker-thresholds",
        type=str,
        default="0.1,0.2,0.3,0.4",
    )
    parser.add_argument("--diarization-collar", type=float, default=0.25)
    parser.add_argument(
        "--diarization-speaker-mapping",
        choices=("expected_fifo", "oracle_chunk"),
        default="expected_fifo",
        help=(
            "Map chunk-local speaker slots either from the reference-derived "
            "FIFO convention expected by the model, or with a true independent "
            "Hungarian oracle permutation for every retained acoustic chunk."
        ),
    )
    parser.add_argument(
        "--diarization-smoothing",
        type=float,
        default=0.0,
        help=(
            "Maximum seconds over which a detected speaker label is propagated "
            "into neighboring blank frames. Multilabel tracks are propagated "
            "independently."
        ),
    )
    parser.add_argument(
        "--diarization-smoothing-values",
        type=str,
        default=None,
        help=(
            "Optional comma-separated smoothing sweep in seconds. When set, "
            "it supersedes --diarization-smoothing and reuses the same neural "
            "network posteriors for every value."
        ),
    )

    parser.add_argument(
        "--online-intersection-block-frames",
        type=int,
        default=500,
        help=(
            "Number of retained acoustic frames per call to k2's online "
            "dense intersecter. Used only with --intersection-method "
            "online_dense_pruned. This experimental path logs the growth of "
            "the carried decode history."
        ),
    )

    parser.add_argument(
        "--online-intersection-profile-only",
        type=str2bool,
        default=False,
        help=(
            "Stop after the online intersection pilot instead of extracting "
            "timestamps. The current k2 online API does not return the dense "
            "arc map needed to attach frame indexes."
        ),
    )

    parser.add_argument(
        "--pruned-min-active-states",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--pruned-max-active-states",
        type=int,
        default=200000,
    )

    parser.add_argument(
        "--blank-frame-filter-threshold",
        type=float,
        default=None,
        help=(
            "Optional first-pass pruning: discard encoder frames whose token "
            "blank posterior is greater than this threshold before graph "
            "intersection. Original frame indices are retained for timestamps."
        ),
    )

    parser.add_argument(
        "--blank-frame-filter-min-token-ratio",
        type=float,
        default=1.0,
        help=(
            "When blank-frame filtering is enabled, retain at least this many "
            "frames per reference BPE token by adding the least-blank-like "
            "discarded frames. Must be at least 1.0."
        ),
    )

    parser.add_argument(
        "--blank-frame-filter-keep-fraction",
        type=float,
        default=None,
        help=(
            "Retain this fraction of each cut's valid encoder frames, choosing "
            "the frames with the lowest blank posterior. Overrides the blank "
            "probability threshold, while respecting the token-count minimum."
        ),
    )
    parser.add_argument(
        "--blank-frame-filter-target-average-fraction",
        type=float,
        default=None,
        help=(
            "Calibrate one global blank-posterior threshold so the arithmetic "
            "mean of per-recording retained-frame fractions on the calibration "
            "set equals this target. The same threshold is then applied to all "
            "test sets; unlike --blank-frame-filter-keep-fraction, it does not "
            "force each recording to retain the same fraction."
        ),
    )
    parser.add_argument(
        "--blank-frame-filter-calibration-set",
        type=str,
        default=None,
        help=(
            "Name of the test set used to calibrate the global threshold. "
            "Required with --blank-frame-filter-target-average-fraction when "
            "more than one test set is supplied."
        ),
    )
    parser.add_argument(
        "--blank-frame-filter-threshold-output",
        type=str,
        default=None,
        help=(
            "Optional JSON path at which to store a dataset-calibrated blank "
            "threshold and its calibration metadata."
        ),
    )

    parser.add_argument(
        "--token-speaker-mask-window-duration",
        type=float,
        default=0.0,
        help=(
            "If positive, after blank-frame removal divide original recording "
            "time into non-overlapping windows of this many seconds and mask "
            "nonblank joint scores to locally active reference token/speaker "
            "pairs. Requires transcript-specific dense remapping."
        ),
    )

    parser.add_argument(
        "--token-speaker-mask-boundary-collar",
        type=float,
        default=2.0,
        help=(
            "Seconds added before and after each token/speaker-mask window "
            "when determining which reference supervisions are locally active."
        ),
    )

    parser.add_argument(
        "--segment-time-window-margin",
        type=float,
        default=None,
        help=(
            "If set, hard-constrain every nonblank graph arc to the source "
            "supervision interval expanded by this many seconds on each side. "
            "This is occurrence-specific (via graph aux labels), keeps blank "
            "legal everywhere, and requires a factorized_viterbi method."
        ),
    )

    parser.add_argument(
        "--original-topo",
        type=str2bool,
        default=False
    )

    parser.add_argument(
        "--single-state-ctc-topo",
        type=str2bool,
        default=False,
        help=(
            "Use a one-state topology with blank/epsilon and token/token "
            "self-loops. The transcript FSA supplies all token ordering."
        ),
    )

    parser.add_argument(
        "--factorized-alignment",
        type=str2bool,
        default=False,
        help=(
            "For speaker-attributed graphs, use the custom factorized CUDA "
            "intersection without materializing speaker-by-token logits."
        ),
    )

    parser.add_argument(
        "--token-only-segment-factor-alignment",
        type=str2bool,
        default=False,
        help=(
            "Compile the token-only partial-order graph with one unique factor "
            "per supervision and score that factor with the same fixed temporal "
            "prior used by --token-only-segment-factor training."
        ),
    )
    parser.add_argument(
        "--segment-factor-collar",
        type=int,
        default=48000,
        help="Flat segment-prior collar in waveform samples.",
    )
    parser.add_argument(
        "--segment-factor-decay-scale",
        type=float,
        default=16000.0,
        help="Exponential segment-prior decay scale in waveform samples.",
    )
    parser.add_argument(
        "--segment-factor-floor-prob",
        type=float,
        default=1.0e-4,
        help="Uniform probability mass mixed across all real segments.",
    )

    parser.add_argument(
        "--alignment-chunk-duration",
        type=float,
        default=0.0,
        help=(
            "If positive, run waveform encoder inference in chunks of this "
            "many seconds and crossfade posterior probabilities in overlaps."
        ),
    )
    parser.add_argument(
        "--alignment-chunk-overlap",
        type=float,
        default=5.0,
        help="Overlap in seconds between consecutive alignment inference chunks.",
    )
    parser.add_argument(
        "--alignment-chunk-context",
        type=float,
        default=0.0,
        help=(
            "If positive, use this much left/right context inside each chunk "
            "and retain only the center. For example, duration=30 and "
            "context=5 emits consecutive 20-second center regions without "
            "crossfading. Recording boundaries are silence-padded."
        ),
    )
    parser.add_argument(
        "--blockwise-alignment",
        type=str2bool,
        default=False,
        help=(
            "Tie center-cropped acoustic chunks to timed partial-order graph "
            "blocks, carry exact frontier scores between blocks, and retain "
            "each block's backpointers for one final traceback."
        ),
    )
    parser.add_argument(
        "--blockwise-whole-recording-inference",
        type=str2bool,
        default=False,
        help=(
            "Run the encoder once on the whole recording, then slice its "
            "posteriors into graph/Viterbi blocks. This keeps one global "
            "speaker-slot ordering across all blocks."
        ),
    )
    parser.add_argument(
        "--blockwise-graph-chunk-duration",
        type=float,
        default=20.0,
        help=(
            "Graph/Viterbi block duration used with "
            "--blockwise-whole-recording-inference."
        ),
    )
    parser.add_argument(
        "--blockwise-token-time-mask",
        type=str2bool,
        default=True,
        help="Restrict each token emission to its graph-collar time interval.",
    )
    parser.add_argument(
        "--blockwise-speaker-ordering",
        choices=("training", "duration", "fifo", "provided"),
        default="training",
        help=(
            "How model speaker slots are assigned independently in each "
            "center-cropped acoustic input. 'training' follows --sort-strategy "
            "(speech_time=duration, start_time=fifo)."
        ),
    )
    parser.add_argument(
        "--blockwise-speaker-order",
        type=str,
        default=None,
        help=(
            "Comma-separated speaker IDs defining a fixed model-slot order. "
            "Useful for a single recording; selects 'provided' semantics."
        ),
    )
    parser.add_argument(
        "--blockwise-speaker-order-file",
        type=str,
        default=None,
        help=(
            "JSON object mapping cut IDs either to one list of speaker IDs "
            "for the whole recording or to one such list per acoustic chunk."
        ),
    )
    parser.add_argument(
        "--transcript-specific-dense-remap",
        type=str2bool,
        default=False,
        help=(
            "After full token/speaker normalization, retain only transcript "
            "tokens and speakers present in each recording and remap graph "
            "labels to the resulting compact dense alphabet."
        ),
    )

    parser.add_argument(
        "--test-sets",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test-cut-manifest",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Add a named evaluation CutSet manifest; may be repeated.",
    )
    parser.add_argument(
        "--test-cut-id",
        type=str,
        default=None,
        help="If set, align only the cut with this exact ID.",
    )

    parser.add_argument(
        "--skip-empty-alignments",
        type=str2bool,
        default=False,
        help=(
            "Log and skip a minibatch when intersection returns an empty or "
            "non-finite lattice, allowing full-set diagnostics to complete."
        ),
    )

    parser.add_argument(
        "--alignment-diagnostics",
        type=str2bool,
        default=False,
        help=(
            "Log graph/lattice final scores, sizes, blank-only reachability, "
            "and shortest-path label counts for alignment debugging."
        ),
    )
    parser.add_argument(
        "--diagnostic-uniform-scores",
        type=str2bool,
        default=False,
        help="Replace dense acoustic scores by finite zeros for structural tests.",
    )
    parser.add_argument(
        "--diagnostic-frame-delta",
        type=int,
        default=0,
        help=(
            "Diagnostic-only number of zero-score frames to append (positive) "
            "or frames to remove from the end (negative)."
        ),
    )
    
    parser.add_argument(
        "--sort-strategy",
        type=str,
        default="start_time",
    )

    parser.add_argument(
        "--speaker-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--ignore-speaker",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--max-speaker-score-for-token-graph",
        type=str2bool,
        default=False,
        help=(
            "With --ignore-speaker, add the maximum valid speaker log "
            "probability to every nonblank token score before intersection."
        ),
    )
    parser.add_argument(
        "--latent-speaker-alignment",
        type=str2bool,
        default=False,
        help="Let every reference segment choose one speaker inside the graph.",
    )
    parser.add_argument(
        "--latent-speaker-max-graph-states",
        type=int,
        default=2000000,
    )
    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

        - use_feat_batchnorm: Normalization for the input features, can be a
                              boolean indicating whether to do batch
                              normalization, or a float which means just scaling
                              the input features with this float value.
                              If given a float value, we will remove batchnorm
                              layer in `ConvolutionModule` as well.

        - attention_dim: Hidden dim for multi-head attention model.

        - head: Number of heads of multi-head attention model.

        - beam_size: It is used in k2.ctc_loss

        - reduction: It is used in k2.ctc_loss

        - use_double_scores: It is used in k2.ctc_loss

        - weight_decay:  The weight_decay for the optimizer.

        - warm_step: The warm_step for Noam optimizer.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            #"log_interval": 100, #100,
            "decode_hyp_interval": 100, #100,
            "reset_interval": 200,
            "valid_interval": 3000,  # For the 100h subset, use 800
            # parameters for zipformer
            "subsampling_factor": 2,  # not passed in, this is fixed.
            "warm_step": 2000,
            "env_info": get_env_info(),
            # parameters for loss
            #"beam_size": 24,
            "reduction": "sum",
            "use_double_scores": True,
            # parameters for decoding
            "search_beam": 20,
            "output_beam": 24,
            "min_active_states": 60,
            "max_active_states": 100000,
            "max_avg_arcs": 100000,
        }
    )

    return params


def get_encoder_model(params: AttributeDict) -> nn.Module:
    if params.use_large:
        bundle = torchaudio.pipelines.WAVLM_LARGE
    else:
        bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
    model = bundle.get_model()
    x = torch.rand(1, 400)
    odim = model(x)[0].size(-1)
    return model, odim


def get_mdctc_model(
    params: AttributeDict,
) -> nn.Module:
    encoder, encoder_dim = get_encoder_model(params)

    if params.frozen_token_teacher_checkpoint:
        teacher_checkpoint = Path(params.frozen_token_teacher_checkpoint)
        if not teacher_checkpoint.is_file():
            raise FileNotFoundError(
                f"Frozen token-teacher checkpoint not found: {teacher_checkpoint}"
            )
        teacher_encoder, teacher_dim = get_encoder_model(params)
        if teacher_dim != encoder_dim:
            raise RuntimeError(
                "Teacher/student encoder dimensions differ: "
                f"{teacher_dim} != {encoder_dim}"
            )
        token_teacher = MDCTCModel(
            encoder=teacher_encoder,
            encoder_dim=teacher_dim,
            vocab_size1=params.vocab_size,
            vocab_size2=params.max_num_spks,
            factorized_token_speaker_output=False,
            layer_norm=params.use_layer_norm,
            downsample=params.downsample,
        )
        logging.info("Loading immutable token teacher from %s", teacher_checkpoint)
        load_checkpoint(teacher_checkpoint, model=token_teacher)
        return FrozenTokenTeacherSpeakerModel(
            token_teacher=token_teacher,
            speaker_encoder=encoder,
            encoder_dim=encoder_dim,
            vocab_size=params.vocab_size,
            num_speakers=params.max_num_spks,
            layer_norm=params.use_layer_norm,
            downsample=params.downsample,
        )

    model = MDCTCModel(
        encoder=encoder,
        encoder_dim=encoder_dim,
        vocab_size1=params.vocab_size,
        vocab_size2=params.max_num_spks,
        factorized_token_speaker_output=params.factorized_token_speaker_output,
        layer_norm=params.use_layer_norm,
        downsample=params.downsample,
    )
    
    return model


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

        if "cur_batch_idx" in saved_params:
            params["cur_batch_idx"] = saved_params["cur_batch_idx"]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def compute_avg_speaker_density_per_example(start_frames_list, num_frames_list):
    """
    Args:
        start_frames_list: List of lists of segment start frames per example
        num_frames_list: List of lists of segment durations per example

    Returns:
        List of average speaker densities, one per example
    """
    avg_densities = []

    for start_frames, num_frames in zip(start_frames_list, num_frames_list):
        assert len(start_frames) == len(num_frames)
        if len(start_frames) == 0:
            avg_densities.append(0.0)
            continue

        end_frame = max(s + n for s, n in zip(start_frames, num_frames))
        density = np.zeros(end_frame, dtype=np.int32)

        for s, n in zip(start_frames, num_frames):
            density[s:s+n] += 1

        avg_density = density.mean()
        avg_densities.append(avg_density)

    return avg_densities


def _target_speaker_scores(
    token: Tensor, speaker: Tensor, nonblank_log_mass: Tensor
) -> Tensor:
    """Create [speaker,time,token] scores used by target-speaker decoding."""
    outputs = []
    for speaker_index in range(speaker.size(-1)):
        if speaker.size(-1) == 1:
            other_mass = speaker.new_full((speaker.size(0),), -torch.inf)
        else:
            other = torch.cat([
                speaker[:, :speaker_index], speaker[:, speaker_index + 1:]
            ], dim=-1)
            other_mass = nonblank_log_mass + other.logsumexp(-1)
        target_blank = torch.logaddexp(token[:, 0], other_mass)
        target_nonblank = token[:, 1:] + speaker[:, speaker_index, None]
        outputs.append(torch.cat([target_blank[:, None], target_nonblank], -1))
    return torch.stack(outputs).clamp_max_(0.0)


def _chunk_speaker_map(
    ordered_speakers, speakers, starts, lengths, input_start, input_end,
    *, strategy="duration", provided_order=None, sample_rate=16000,
):
    """Map global speaker identities to model output slots for one chunk."""
    durations = {speaker: 0.0 for speaker in ordered_speakers}
    onsets = {speaker: math.inf for speaker in ordered_speakers}
    for speaker, start, length in zip(speakers, starts, lengths):
        supervision_start = start / sample_rate
        supervision_end = (start + length) / sample_rate
        overlap = max(
            0.0, min(input_end, supervision_end) - max(input_start, supervision_start)
        )
        durations[speaker] += overlap
        if overlap > 0:
            onsets[speaker] = min(
                onsets[speaker], max(input_start, supervision_start)
            )
    global_index = {speaker: i for i, speaker in enumerate(ordered_speakers)}
    if strategy == "duration":
        local_order = sorted(
            ordered_speakers, key=lambda x: (-durations[x], global_index[x])
        )
    elif strategy == "fifo":
        # Speakers already active at the left boundary tie at input_start;
        # use their stable global order to reproduce deterministic batching.
        local_order = sorted(
            ordered_speakers, key=lambda x: (onsets[x], global_index[x])
        )
    elif strategy == "provided":
        local_order = list(provided_order or [])
        if len(local_order) != len(ordered_speakers) or set(local_order) != set(
            ordered_speakers
        ):
            raise ValueError(
                "Provided speaker order must contain every recording speaker "
                f"exactly once; expected {ordered_speakers}, got {local_order}"
            )
    else:
        raise ValueError(f"Unsupported blockwise speaker ordering: {strategy}")
    rank = {speaker: i for i, speaker in enumerate(local_order)}
    return [rank[speaker] for speaker in ordered_speakers], durations, onsets


def _resolve_blockwise_speaker_order(params, cut):
    """Resolve ordering strategy and an optional per-recording explicit order."""
    strategy = params.blockwise_speaker_ordering
    explicit = None
    if params.blockwise_speaker_order and params.blockwise_speaker_order_file:
        raise ValueError(
            "Use only one of --blockwise-speaker-order and "
            "--blockwise-speaker-order-file"
        )
    if params.blockwise_speaker_order:
        explicit = [x.strip() for x in params.blockwise_speaker_order.split(",")]
        explicit = [x for x in explicit if x]
    elif params.blockwise_speaker_order_file:
        with open(params.blockwise_speaker_order_file) as f:
            orders = json.load(f)
        if not isinstance(orders, dict) or cut.id not in orders:
            raise ValueError(
                f"Speaker-order file has no list for cut {cut.id!r}"
            )
        explicit = orders[cut.id]
    elif strategy == "provided":
        custom = getattr(cut, "custom", None) or {}
        explicit = custom.get("speaker_order")

    if explicit is not None:
        strategy = "provided"
    elif strategy == "training":
        if params.sort_strategy == "speech_time":
            strategy = "duration"
        elif params.sort_strategy == "start_time":
            strategy = "fifo"
        else:
            raise ValueError(
                "Cannot infer blockwise speaker ordering from sort strategy "
                f"{params.sort_strategy!r}; choose duration/fifo/provided explicitly"
            )
    if strategy == "provided" and explicit is None:
        raise ValueError(
            "Provided speaker ordering requires --blockwise-speaker-order, "
            "--blockwise-speaker-order-file, or cut.custom['speaker_order']"
        )
    if explicit is not None and not isinstance(explicit, list):
        raise ValueError("Provided speaker order must be a JSON/list sequence")
    return strategy, explicit


@torch.no_grad()
def _center_cropped_alignment_blocks(
    params, model, waveform, waveform_length, token_ids,
    texts, ordered_speakers, speakers, starts, lengths, timed_tokens,
    speaker_ordering="duration", provided_speaker_order=None, sample_rate=16000,
):
    """Return non-overlapping posterior cores with chunk-local speaker maps."""
    chunk_duration = params.alignment_chunk_duration
    context = params.alignment_chunk_context
    whole_recording = params.blockwise_whole_recording_inference
    if whole_recording:
        if params.blockwise_graph_chunk_duration <= 0:
            raise ValueError("Whole-recording inference needs a positive graph chunk duration")
        core_samples = round(params.blockwise_graph_chunk_duration * sample_rate)
        context_samples = 0
        core_frames_nominal = round(
            params.blockwise_graph_chunk_duration / params.frame_duration
        )
        full_token, full_speaker, full_output_lens = model(
            waveform[..., :waveform_length],
            torch.tensor([waveform_length], device=waveform.device),
            speaker_weight=params.speaker_weight,
            return_factors=True,
        )
        full_valid = int(full_output_lens[0])
        logging.info(
            "Whole-recording encoder inference samples=%d frames=%d graph_block=%.3fs",
            waveform_length, full_valid, params.blockwise_graph_chunk_duration,
        )
    elif chunk_duration <= 0 or context <= 0 or 2 * context >= chunk_duration:
        raise ValueError(
            "Blockwise alignment requires a positive chunk duration and "
            "0 < 2*context < duration"
        )
    else:
        chunk_samples = round(chunk_duration * sample_rate)
        context_samples = round(context * sample_rate)
        core_samples = chunk_samples - 2 * context_samples
        core_frames_nominal = round(core_samples / sample_rate / params.frame_duration)
    total_frames = (
        full_valid if whole_recording else
        round(waveform_length / sample_rate / params.frame_duration)
    )
    token_index = torch.tensor(token_ids, device=waveform.device, dtype=torch.long)
    speaker_to_global = {
        speaker: index for index, speaker in enumerate(ordered_speakers)
    }
    blocks = []
    speaker_maps = []
    core_start = 0
    while core_start < waveform_length:
        block_index = len(blocks)
        global_start_frame = round(
            core_start / sample_rate / params.frame_duration
        )
        remaining_frames = total_frames - global_start_frame
        if remaining_frames <= 0:
            # The waveform can extend a few samples past an exact core
            # boundary while the convolutional encoder produces no additional
            # output frame.  In that case the preceding core already contains
            # every usable posterior.  Do not manufacture an empty/negative
            # trailing block (which would make torch.arange() fail).
            logging.info(
                "Skipping trailing graph block at sample %d: encoder output "
                "is exhausted at frame %d/%d",
                core_start, global_start_frame, total_frames,
            )
            break
        core_frames = min(core_frames_nominal, remaining_frames)
        if whole_recording:
            requested_start = 0
            requested_end = waveform_length
            crop_start = global_start_frame
            crop_end = crop_start + core_frames
            token = full_token[0, crop_start:crop_end].index_select(
                -1, token_index
            ).float()
            speaker = full_speaker[0, crop_start:crop_end].float()
        else:
            requested_start = core_start - context_samples
            requested_end = requested_start + chunk_samples
            source_start = max(0, requested_start)
            source_end = min(waveform_length, requested_end)
            destination_start = source_start - requested_start
            chunk = waveform.new_zeros((1, chunk_samples))
            chunk[..., destination_start:destination_start + source_end - source_start] = (
                waveform[..., source_start:source_end]
            )
            token_full, speaker_full, output_lens = model(
                chunk, torch.tensor([chunk_samples], device=waveform.device),
                speaker_weight=params.speaker_weight, return_factors=True,
            )
            valid = int(output_lens[0])
            context_frames = round(context / params.frame_duration)
            crop_end = context_frames + core_frames
            if crop_end > valid:
                raise RuntimeError(
                    f"Chunk output has {valid} frames but crop ends at {crop_end}"
                )
            token = token_full[0, context_frames:crop_end].index_select(
                -1, token_index
            ).float()
            speaker = speaker_full[0, context_frames:crop_end].float()
        absolute_frames = torch.arange(
            global_start_frame, global_start_frame + core_frames,
            device=waveform.device, dtype=torch.long,
        )
        # Preserve the full-rate posteriors for embedding-based diarization.
        # Alignment blank pruning below is deliberately independent: the
        # stitching segmentation must see silence and overlap in the original
        # 50-Hz output rather than only the retained alignment frames.
        diarization_blank_probability = token[:, 0].exp()
        diarization_speaker_probability = speaker.exp()
        diarization_absolute_frames = absolute_frames.clone()
        core_start_seconds = core_start / sample_rate
        core_end_seconds = min(
            waveform_length / sample_rate,
            (core_start + core_samples) / sample_rate,
        )
        if (
            params.blank_frame_filter_threshold is not None
            or params.blank_frame_filter_keep_fraction is not None
        ):
            local_token_count = sum(
                token.earliest < core_end_seconds
                and token.latest >= core_start_seconds
                for token in timed_tokens
            )
            minimum_frames = math.ceil(
                params.blank_frame_filter_min_token_ratio * local_token_count
            )
            filtered, filtered_lens, frame_maps = filter_blank_dominated_frames(
                token.unsqueeze(0),
                torch.tensor([core_frames], device=waveform.device, dtype=torch.int32),
                params.blank_frame_filter_threshold,
                minimum_frames=[minimum_frames],
                keep_fraction=params.blank_frame_filter_keep_fraction,
            )
            keep = frame_maps[0]
            # A corpus-calibrated blank threshold controls the ordinary set
            # of retained frames, but it cannot by itself guarantee that each
            # token whose deadline lies in this block still has a distinct
            # legal emission frame.  Construct an interval matching on the
            # unfiltered core and force those frames into the retained set.
            # Sorting by deadline and taking the earliest available frame is
            # the standard greedy maximum matching for interval constraints.
            is_final_core = core_start + core_samples >= waveform_length
            # Give each token to exactly one core according to its coarse
            # center.  Using every interval that merely touches the core is
            # incorrect for a wide collar: it redundantly reserves frames for
            # tokens that a previous block may already have emitted.
            required = [
                item for item in timed_tokens
                if core_start_seconds <= item.center < core_end_seconds
                or (is_final_core and item.center >= core_start_seconds)
            ]
            available = list(range(core_frames))
            forced = []
            for item in sorted(required, key=lambda x: (x.latest, x.earliest)):
                chosen_position = next((
                    pos for pos, local_frame in enumerate(available)
                    if item.earliest <= (
                        global_start_frame + local_frame
                    ) * params.frame_duration <= item.latest
                ), None)
                if chosen_position is None:
                    raise RuntimeError(
                        "Timed token intervals have no distinct frame matching "
                        f"in core [{core_start_seconds:.3f},"
                        f"{core_end_seconds:.3f}) before blank pruning"
                    )
                forced.append(available.pop(chosen_position))
            baseline_count = int(filtered_lens[0])
            if forced:
                forced_tensor = torch.tensor(
                    forced, device=keep.device, dtype=keep.dtype
                )
                keep = torch.unique(
                    torch.cat((keep, forced_tensor)), sorted=True
                )
            token = token.index_select(0, keep)
            speaker = speaker.index_select(0, keep)
            absolute_frames = absolute_frames.index_select(0, keep)
            if keep.numel() > baseline_count:
                logging.info(
                    "Blank pruning feasibility guard added %d frames in "
                    "core [%.3f,%.3f)", keep.numel() - baseline_count,
                    core_start_seconds, core_end_seconds,
                )
        chunk_provided_order = provided_speaker_order
        if (
            provided_speaker_order
            and isinstance(provided_speaker_order[0], list)
        ):
            if block_index >= len(provided_speaker_order):
                raise ValueError(
                    f"No supplied speaker order for acoustic chunk {block_index}"
                )
            chunk_provided_order = provided_speaker_order[block_index]
        speaker_map, durations, onsets = _chunk_speaker_map(
            ordered_speakers, speakers, starts, lengths,
            requested_start / sample_rate, requested_end / sample_rate,
            strategy=speaker_ordering,
            provided_order=chunk_provided_order,
            sample_rate=sample_rate,
        )
        local_token_allowed = None
        if params.token_speaker_mask_window_duration > 0:
            window_duration = params.token_speaker_mask_window_duration
            boundary_collar = params.token_speaker_mask_boundary_collar
            token_identities_by_segment = []
            token_to_compact = {value: index for index, value in enumerate(token_ids)}
            for text, speaker_name in zip(
                [x for x in texts], [x for x in speakers]
            ):
                token_identities_by_segment.append({
                    ((speaker_to_global[speaker_name], token_to_compact[value])
                     if not params.ignore_speaker else token_to_compact[value])
                    for value in params._alignment_token_encoder(text)
                    if value in token_to_compact
                })
            identities = [
                ((item.speaker, item.token) if not params.ignore_speaker else item.token)
                for item in timed_tokens
            ]
            window_ids = torch.floor(
                absolute_frames.to(torch.float64)
                * params.frame_duration / window_duration
            ).to(torch.long)
            local_token_allowed = torch.zeros(
                (absolute_frames.numel(), len(timed_tokens)),
                dtype=torch.bool, device=absolute_frames.device,
            )
            for window_id in torch.unique(window_ids).tolist():
                window_start = window_id * window_duration
                window_end = window_start + window_duration
                active_identities = set()
                for segment, (start, length) in enumerate(zip(starts, lengths)):
                    supervision_start = start / sample_rate
                    supervision_end = (start + length) / sample_rate
                    if (
                        supervision_start < window_end + boundary_collar
                        and supervision_end > window_start - boundary_collar
                    ):
                        active_identities.update(token_identities_by_segment[segment])
                selected_frames = torch.nonzero(
                    window_ids == window_id, as_tuple=False
                ).squeeze(1)
                selected_tokens = [
                    index for index, identity in enumerate(identities)
                    if identity in active_identities
                ]
                if selected_frames.numel() and selected_tokens:
                    selected_tokens_tensor = torch.tensor(
                        selected_tokens, device=absolute_frames.device,
                        dtype=torch.long,
                    )
                    local_token_allowed[
                        selected_frames[:, None], selected_tokens_tensor[None, :]
                    ] = True
        logging.info(
            "Block core=[%.3f,%.3f) input=[%.3f,%.3f) frames=%d/%d "
            "speaker_ordering=%s global_to_local_speaker=%s "
            "local_durations=%s local_onsets=%s",
            core_start_seconds, core_end_seconds,
            requested_start / sample_rate, requested_end / sample_rate,
            token.size(0), core_frames, speaker_ordering, speaker_map,
            [round(durations[s], 3) for s in ordered_speakers],
            [None if math.isinf(onsets[s]) else round(onsets[s], 3)
             for s in ordered_speakers],
        )
        blocks.append({
            "token_log_probs": token,
            "speaker_log_probs": speaker,
            "absolute_frames": absolute_frames,
            "start": core_start_seconds,
            "end": core_end_seconds,
            "frame_duration": params.frame_duration,
            "local_token_allowed": local_token_allowed,
            "diarization_blank_probability": diarization_blank_probability,
            "diarization_speaker_probability": diarization_speaker_probability,
            "diarization_absolute_frames": diarization_absolute_frames,
        })
        speaker_maps.append(speaker_map)
        if not whole_recording:
            del token_full, speaker_full
        core_start += core_samples
    if whole_recording:
        del full_token, full_speaker
    return blocks, speaker_maps


@torch.no_grad()
def align_one_batch_blockwise(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    graph_compiler: MDCTCGraphCompiler,
    batch: Dict,
):
    """Align a recording using center-cropped acoustic/partial-order blocks."""
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    feature = batch["inputs"].to(device)
    feature_lens = batch["num_frames"].to(device)
    if feature.size(0) != 1:
        # Recordings generally have different numbers and placements of graph
        # blocks.  Preserve ordinary dataloader batching at the API boundary,
        # while running those ragged block schedules independently.  This is
        # exact and keeps traceback ownership clear; distributed workers still
        # provide recording-level parallelism.
        supervisions = batch["supervisions"]
        sequence_idx = supervisions["sequence_idx"]
        for recording in range(feature.size(0)):
            selected = torch.nonzero(
                sequence_idx == recording, as_tuple=False
            ).squeeze(1)
            child_supervisions = {}
            for name, value in supervisions.items():
                if name == "sequence_idx":
                    child_supervisions[name] = torch.zeros_like(
                        sequence_idx.index_select(0, selected)
                    )
                elif isinstance(value, torch.Tensor):
                    child_supervisions[name] = value.index_select(0, selected)
                elif isinstance(value, (list, tuple)):
                    child_supervisions[name] = [value[int(i)] for i in selected]
                else:
                    child_supervisions[name] = value
            child = dict(batch)
            child.update({
                "inputs": batch["inputs"][recording:recording + 1],
                "num_frames": batch["num_frames"][recording:recording + 1],
                "texts": [batch["texts"][recording]],
                "speakers": [batch["speakers"][recording]],
                "sort_orders": [batch["sort_orders"][recording]],
                "supervisions": child_supervisions,
            })
            yield from align_one_batch_blockwise(
                params, model, graph_compiler, child
            )
        return
    texts = batch["texts"][0]
    # Used only while constructing the post-blank-removal local mask in the
    # block producer; keeping it on params avoids widening that already long
    # internal call signature.
    params._alignment_sp = graph_compiler.sp
    params._alignment_token_encoder = graph_compiler.encode
    speakers = batch["speakers"][0]
    cut = batch["supervisions"]["cut"][0]
    speaker_ordering, provided_speaker_order = _resolve_blockwise_speaker_order(
        params, cut
    )
    ordered_speakers = list(dict.fromkeys(speakers))
    if len(ordered_speakers) > params.max_num_spks:
        raise ValueError("Recording has more speakers than the speaker head")
    speaker_to_global = {s: i for i, s in enumerate(ordered_speakers)}
    seq_idx = batch["supervisions"]["sequence_idx"]
    sort_order = batch["sort_orders"][0]
    starts = [
        int(batch["supervisions"]["start_sample"][seq_idx == 0][j])
        for j in sort_order
    ]
    lengths = [
        int(batch["supervisions"]["num_samples"][seq_idx == 0][j])
        for j in sort_order
    ]
    if params.transcript_specific_dense_remap:
        compact_ids, _ = alignment_candidate_ids(
            [texts], [speakers], graph_compiler.sp,
            token_encoder=graph_compiler.encode,
        )
        token_ids = compact_ids[0]
    else:
        token_ids = list(range(graph_compiler.sp.vocab_size()))
    token_to_compact = {token: i for i, token in enumerate(token_ids)}

    timed_tokens = []
    for segment, (text, speaker, start, length) in enumerate(
        zip(texts, speakers, starts, lengths)
    ):
        encoded = graph_compiler.encode(text)
        if not encoded:
            continue
        emission_start = None
        emission_end = None
        if params.segment_time_window_margin is not None:
            emission_start = max(
                0.0,
                start / 16000 - params.segment_time_window_margin,
            )
            emission_end = (
                (start + length) / 16000
                + params.segment_time_window_margin
            )
        samples_per_token = length // len(encoded) + 1
        for position, token in enumerate(encoded):
            timed_tokens.append(TimedToken(
                token=token_to_compact[token],
                speaker=speaker_to_global[speaker],
                segment=segment,
                center=(start + position * samples_per_token) / 16000,
                half_width=graph_compiler.token_half_width(
                    position / len(encoded), params.collar
                ) / 16000,
                emission_start=emission_start,
                emission_end=emission_end,
            ))
    timed_tokens = add_partial_order_predecessors(
        timed_tokens, allow_self_overlap=params.allow_self_overlap
    )
    blocks, speaker_maps = _center_cropped_alignment_blocks(
        params, model, feature[0:1], int(feature_lens[0]), token_ids,
        texts, ordered_speakers, speakers, starts, lengths, timed_tokens,
        speaker_ordering=speaker_ordering,
        provided_speaker_order=provided_speaker_order,
    )
    acoustic_block_count = len(blocks)
    if any(
        float(block["end"]) - float(block["start"])
        > params.blockwise_graph_chunk_duration + 1e-9
        for block in blocks
    ):
        blocks, speaker_maps = split_acoustic_blocks(
            blocks,
            speaker_maps,
            params.blockwise_graph_chunk_duration,
        )
        logging.info(
            "Decoupled %d acoustic posterior cores into %d graph/Viterbi "
            "blocks of at most %.3fs",
            acoustic_block_count,
            len(blocks),
            params.blockwise_graph_chunk_duration,
        )
    start_time = time.perf_counter()
    topology = (
        "single" if params.single_state_ctc_topo
        else "original" if params.original_topo
        else "training"
    )
    pruned = params.intersection_method == "factorized_viterbi_pruned"
    result = blockwise_viterbi(
        timed_tokens, blocks, local_speaker_maps=speaker_maps,
        apply_token_time_mask=params.blockwise_token_time_mask,
        score_speakers=not params.ignore_speaker,
        topology=topology,
        search_beam=params.search_beam if pruned else None,
        min_active_states=params.pruned_min_active_states,
        max_active_states=params.pruned_max_active_states,
    )
    backpointer_bytes = sum(
        trace.backpointers.numel() * trace.backpointers.element_size()
        for trace in result.traces
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda = torch.cuda.max_memory_allocated(device) / 2**30
    else:
        peak_cuda = 0.0
    logging.info(
        "Blockwise Viterbi blocks=%d tokens=%d boundary_states=%s "
        "backpointer_states=%s backpointer_cpu=%.3fGiB peak_cuda=%.3fGiB "
        "topology=%s score_speakers=%s pruned=%s runtime=%.3fs",
        len(blocks), len(timed_tokens), result.boundary_state_counts,
        [trace.backpointers.numel() for trace in result.traces],
        backpointer_bytes / 2**30, peak_cuda, topology,
        not params.ignore_speaker, pruned,
        time.perf_counter() - start_time,
    )
    times = result.frames.float() * params.frame_duration
    token_indexes = result.token_indexes.tolist()
    recovered_segment_speakers = None
    recovered_token_speakers = None
    recovered_speaker_names = ordered_speakers
    if params.ignore_speaker:
        # The token-only graph retains segment IDs but does not score speaker
        # labels.  Accumulate the chunk-local speaker log posterior at every
        # token emission for a segment after mapping each local output slot
        # back to the recording-global speaker identity.
        frame_locations = {}
        for block_index, block in enumerate(blocks):
            for local_index, absolute_frame in enumerate(
                block["absolute_frames"].tolist()
            ):
                frame_locations[int(absolute_frame)] = (block_index, local_index)
        if params.blockwise_speaker_recovery in ("embedding_cluster", "vbx"):
            from speaker_stitching import (
                EcapaSpeakerEmbedder,
                cluster_chunk_speaker_slots,
                diarize_and_map_chunk_speaker_slots_vbx,
            )

            if params.speaker_stitching_segmentation_source == "token_alignment":
                _set_token_alignment_embedding_activity(
                    blocks=blocks,
                    result=result,
                    timed_tokens=timed_tokens,
                    frame_locations=frame_locations,
                    frame_duration=params.frame_duration,
                    gap_seconds=params.speaker_stitching_token_gap,
                    calibration_path=params.speaker_stitching_timing_calibration,
                    neighbor_radius=params.speaker_stitching_token_neighbors,
                    speaker_confidence=(
                        params.speaker_stitching_token_speaker_confidence
                        if params.speaker_stitching_token_speaker_confidence is not None
                        else params.speaker_stitching_speaker_threshold
                    ),
                )
            if params.blockwise_speaker_recovery == "vbx":
                recovery_maps, recovered_count, vbx_intervals = diarize_and_map_chunk_speaker_slots_vbx(
                    blocks=blocks,
                    waveform=feature[0, : int(feature_lens[0])],
                    frame_duration=params.frame_duration,
                    num_clusters=(
                        len(ordered_speakers)
                        if params.speaker_clustering_oracle_num_speakers else None
                    ),
                    vbx_dir=params.speaker_vbx_dir,
                    blank_threshold=params.speaker_stitching_blank_threshold,
                    speaker_threshold=params.speaker_stitching_speaker_threshold,
                    smoothing_seconds=params.speaker_stitching_smoothing,
                    min_clean_seconds=max(
                        params.speaker_vbx_min_clean_seconds,
                        params.speaker_embedding_min_seconds,
                    ),
                )
            else:
                if not hasattr(params, "_speaker_stitch_embedder"):
                    params._speaker_stitch_embedder = EcapaSpeakerEmbedder(
                        params.speaker_embedding_model_dir, device
                    )
                recovery_maps, recovered_count = cluster_chunk_speaker_slots(
                    blocks=blocks,
                    waveform=feature[0, : int(feature_lens[0])],
                    num_clusters=(
                        len(ordered_speakers)
                        if params.speaker_clustering_oracle_num_speakers else None
                    ),
                    frame_duration=params.frame_duration,
                    embedder=params._speaker_stitch_embedder,
                    min_seconds=params.speaker_embedding_min_seconds,
                    max_seconds=params.speaker_embedding_max_seconds,
                    blank_threshold=params.speaker_stitching_blank_threshold,
                    speaker_threshold=params.speaker_stitching_speaker_threshold,
                    smoothing_seconds=params.speaker_stitching_smoothing,
                    clustering_threshold=params.speaker_clustering_threshold,
                    min_cluster_size=params.speaker_clustering_min_cluster_size,
                )
            recovered_speaker_names = [
                f"SPEAKER_{index:02d}" for index in range(recovered_count)
            ]
        elif params.blockwise_speaker_recovery == "identity_map":
            num_slots = blocks[0]["speaker_log_probs"].size(-1)
            recovery_maps = [torch.arange(num_slots) for _ in blocks]
            recovered_count = num_slots
            recovered_speaker_names = [
                f"SPEAKER_{index:02d}" for index in range(recovered_count)
            ]
        else:
            recovery_maps = [
                torch.tensor(mapping, dtype=torch.long)
                for mapping in speaker_maps
            ]
            recovered_count = len(ordered_speakers)
        if (
            params.blockwise_speaker_attribution_unit == "word"
            and params.blockwise_speaker_recovery == "vbx"
        ):
            raise ValueError(
                "Word-level posterior attribution is not defined for direct "
                "VBx interval recovery. Use segment attribution with VBx."
            )
        if params.blockwise_speaker_recovery == "vbx":
            # VBx already produces recording-global, time-labeled speakers.
            # Assign each transcript segment directly by temporal overlap with
            # its aligned token span; do not route back through local Whisper
            # speaker slots or speaker-branch posteriors.
            recovered_segment_speakers = [0] * len(texts)
            for segment in range(len(texts)):
                selected = [
                    index for index, token_index in enumerate(token_indexes)
                    if timed_tokens[token_index].segment == segment
                ]
                if not selected:
                    continue
                segment_times = times[selected]
                start = float(segment_times[0])
                if segment_times.numel() > 1:
                    gaps = segment_times[1:] - segment_times[:-1]
                    tail = max(params.frame_duration, float(gaps.mean()))
                else:
                    tail = 0.2
                end = float(segment_times[-1]) + tail
                overlap = torch.zeros(recovered_count, dtype=torch.float64)
                for interval_start, interval_end, label in vbx_intervals:
                    overlap[label] += max(
                        0.0, min(end, interval_end) - max(start, interval_start)
                    )
                if float(overlap.max()) > 0:
                    recovered_segment_speakers[segment] = int(overlap.argmax())
                else:
                    midpoint = 0.5 * (start + end)
                    recovered_segment_speakers[segment] = min(
                        vbx_intervals,
                        key=lambda interval: (
                            max(interval[0] - midpoint, midpoint - interval[1], 0.0)
                        ),
                    )[2]
            assignment_counts = torch.bincount(
                torch.tensor(recovered_segment_speakers), minlength=recovered_count
            ).tolist()
            logging.info("VBx direct segment assignment counts=%s", assignment_counts)
        else:
            segment_scores = torch.zeros(
                (len(texts), recovered_count), dtype=torch.float64
            )
            emitted_speaker_scores = torch.empty(
                (len(token_indexes), recovered_count), dtype=torch.float64
            )
            for emitted_index, token_index_value in enumerate(token_indexes):
                segment = timed_tokens[token_index_value].segment
                absolute_frame = int(result.frames[emitted_index])
                block_index, local_index = frame_locations[absolute_frame]
                local_scores = blocks[block_index]["speaker_log_probs"][local_index]
                if params.blockwise_speaker_recovery == "embedding_cluster":
                    local_to_cluster = recovery_maps[block_index].to(local_scores.device)
                    clustered = local_scores.new_full((recovered_count,), -torch.inf)
                    for cluster in range(recovered_count):
                        selected = torch.nonzero(
                            local_to_cluster == cluster, as_tuple=False
                        ).squeeze(1)
                        if selected.numel():
                            clustered[cluster] = torch.logsumexp(
                                local_scores.index_select(0, selected), dim=0
                            )
                    finite = torch.isfinite(clustered)
                    global_scores = clustered.double().cpu()
                else:
                    global_to_local = recovery_maps[block_index].to(local_scores.device)
                    global_scores = local_scores.index_select(
                        0, global_to_local
                    ).double().cpu()
                emitted_speaker_scores[emitted_index] = global_scores
                segment_scores[segment] += global_scores
            if params.blockwise_speaker_attribution_unit == "segment":
                recovered_segment_speakers = segment_scores.argmax(dim=-1).tolist()
            else:
                recovered_token_speakers = [0] * len(token_indexes)
                for segment in range(len(texts)):
                    selected = [
                        emitted_index
                        for emitted_index, token_index in enumerate(token_indexes)
                        if timed_tokens[token_index].segment == segment
                    ]
                    word = []
                    words = []
                    for emitted_index in selected:
                        token_index = token_indexes[emitted_index]
                        piece = graph_compiler.sp.id_to_piece(
                            token_ids[timed_tokens[token_index].token]
                        )
                        if word and piece.startswith("\u2581"):
                            words.append(word)
                            word = []
                        word.append(emitted_index)
                    if word:
                        words.append(word)
                    for emitted_indexes in words:
                        word_scores = emitted_speaker_scores[emitted_indexes].sum(dim=0)
                        word_speaker = int(word_scores.argmax())
                        for emitted_index in emitted_indexes:
                            recovered_token_speakers[emitted_index] = word_speaker
        logging.info(
            "Blockwise token-only speaker recovery method=%s unit=%s "
            "segment_assignments=%s",
            params.blockwise_speaker_recovery,
            params.blockwise_speaker_attribution_unit,
            recovered_segment_speakers,
        )
    alignment = []
    recording_duration = float(feature_lens[0]) / 16000
    for segment in range(len(texts)):
        selected = [i for i, index in enumerate(token_indexes)
                    if timed_tokens[index].segment == segment]
        if not selected:
            continue
        segment_times = times[selected]
        pieces = [
            graph_compiler.sp.id_to_piece(
                token_ids[timed_tokens[token_indexes[i]].token]
            ) for i in selected
        ]
        if segment_times.numel() > 1:
            durations = (segment_times[1:] - segment_times[:-1]).tolist()
            final_duration = sum(durations) / len(durations)
        else:
            durations = []
            final_duration = fallback_token_duration(
                pieces[0],
                params.single_token_duration_per_character,
                params.frame_duration,
                params.single_token_fixed_duration,
            )
        durations.append(min(
            final_duration, recording_duration - float(segment_times[-1])
        ))
        if recovered_token_speakers is not None:
            output_speakers = [
                recovered_speaker_names[recovered_token_speakers[index]]
                for index in selected
            ]
        else:
            output_speaker = (
                recovered_speaker_names[recovered_segment_speakers[segment]]
                if recovered_segment_speakers is not None else speakers[segment]
            )
            output_speakers = [output_speaker] * len(selected)
        alignment.extend(zip(
            output_speakers, [segment] * len(selected),
            pieces, [round(float(x), 3) for x in segment_times],
            [round(float(x), 3) for x in durations],
        ))
    yield cut.id, list(alignment), len(texts)


def _set_token_alignment_embedding_activity(
    blocks, result, timed_tokens, frame_locations, frame_duration,
    gap_seconds=0.25, calibration_path=None, neighbor_radius=2,
    speaker_confidence=0.45,
):
    """Induce chunk-local speaker masks from token-only alignment spans."""
    start_bias = duration_bias = 0.0
    if calibration_path:
        with open(calibration_path) as f:
            calibration = json.load(f)
        start_bias = float(calibration["start_bias_seconds"])
        duration_bias = float(calibration["duration_bias_seconds"])

    num_slots = blocks[0]["diarization_speaker_probability"].size(-1)
    for block in blocks:
        block["embedding_activity"] = torch.zeros(
            (block["diarization_absolute_frames"].numel(), num_slots),
            dtype=torch.bool,
            device=block["diarization_absolute_frames"].device,
        )
        block["embedding_clean_activity"] = torch.zeros_like(
            block["embedding_activity"]
        )

    by_segment = defaultdict(list)
    for emitted_index, token_index in enumerate(result.token_indexes.tolist()):
        frame = int(result.frames[emitted_index])
        by_segment[timed_tokens[token_index].segment].append(frame)

    spans = 0
    for frames in by_segment.values():
        frames.sort()
        runs, begin = [], 0
        for index in range(1, len(frames)):
            if (frames[index] - frames[index - 1]) * frame_duration > gap_seconds:
                runs.append(frames[begin:index])
                begin = index
        runs.append(frames[begin:])
        for run in runs:
            differences = [
                (right - left) * frame_duration
                for left, right in zip(run[:-1], run[1:])
            ]
            tail = sum(differences) / len(differences) if differences else 0.2
            span_start = run[0] * frame_duration - start_bias
            span_end = run[-1] * frame_duration - start_bias + max(
                frame_duration, tail - duration_bias
            )
            emissions_by_block = defaultdict(list)
            for frame in run:
                if frame in frame_locations:
                    block_index, local_index = frame_locations[frame]
                    emissions_by_block[block_index].append(local_index)
            for block_index, local_indexes in emissions_by_block.items():
                block = blocks[block_index]
                selected = torch.tensor(
                    local_indexes, device=block["speaker_log_probs"].device,
                    dtype=torch.long,
                )
                slot = int(block["speaker_log_probs"].index_select(
                    0, selected
                ).sum(0).argmax())
                absolute = block["diarization_absolute_frames"]
                inside = (
                    absolute.float() * frame_duration >= span_start
                ) & (
                    absolute.float() * frame_duration < span_end
                )
                block["embedding_activity"][inside, slot] = True

                # Prefer locally stable, high-confidence token neighborhoods.
                # Speaker slots are chunk-local, so this test is deliberately
                # performed independently inside each acoustic block.
                local_indexes = sorted(set(local_indexes))
                scores = block["speaker_log_probs"].index_select(
                    0, torch.tensor(
                        local_indexes, device=block["speaker_log_probs"].device,
                        dtype=torch.long,
                    )
                )
                probabilities = scores.exp()
                confidence, labels = probabilities.max(dim=-1)
                clean = []
                for index in range(len(local_indexes)):
                    left = max(0, index - max(0, neighbor_radius))
                    right = min(len(local_indexes), index + max(0, neighbor_radius) + 1)
                    clean.append(bool(
                        (confidence[left:right] >= speaker_confidence).all()
                        and (labels[left:right] == labels[index]).all()
                    ))
                absolute = block["diarization_absolute_frames"].float()
                for index, is_clean in enumerate(clean):
                    if not is_clean:
                        continue
                    center = float(
                        block["absolute_frames"][local_indexes[index]]
                    ) * frame_duration - start_bias
                    if index:
                        previous = float(
                            block["absolute_frames"][local_indexes[index - 1]]
                        ) * frame_duration - start_bias
                        region_start = 0.5 * (previous + center)
                    else:
                        region_start = span_start
                    if index + 1 < len(local_indexes):
                        following = float(
                            block["absolute_frames"][local_indexes[index + 1]]
                        ) * frame_duration - start_bias
                        region_end = 0.5 * (center + following)
                    else:
                        region_end = span_end
                    clean_inside = (
                        absolute * frame_duration >= region_start
                    ) & (
                        absolute * frame_duration < region_end
                    )
                    block["embedding_clean_activity"][
                        clean_inside, int(labels[index])
                    ] = True
            spans += 1
    logging.info(
        "Token-induced embedding diarization spans=%d gap=%.3fs "
        "neighbors=%d confidence=%.3f calibration=(start=%.3fs,duration=%.3fs) "
        "active_slots_per_block=%s clean_seconds=%.2f",
        spans, gap_seconds, neighbor_radius, speaker_confidence,
        start_bias, duration_bias,
        [int(block["embedding_activity"].any(0).sum()) for block in blocks],
        sum(float(block["embedding_clean_activity"].any(-1).sum())
            for block in blocks) * frame_duration,
    )


@torch.no_grad()
def align_one_batch_independent_segments(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    graph_compiler: MDCTCGraphCompiler,
    batch: Dict,
):
    """Independently force-align each supervision in one batched exact DP.

    The encoder is evaluated once for the recording.  Every supervision gets
    its own linear transcript graph, but all graphs share the same acoustic
    score tensors through expanded views.  This isolates the effect of the
    partial-order/shuffle graph while leaving the model, CTC topology, timing
    window, and acoustic scores unchanged.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"].to(device)
    feature_lens = batch["num_frames"].to(device)
    if feature.size(0) != 1:
        raise ValueError("Independent-segment alignment requires --max-cuts=1")
    if params.intersection_method != "factorized_viterbi":
        raise ValueError(
            "Independent-segment alignment requires exact factorized_viterbi"
        )
    if params.segment_time_window_margin is None:
        raise ValueError(
            "Independent-segment alignment requires --segment-time-window-margin"
        )
    if params.transcript_specific_dense_remap:
        raise ValueError(
            "Independent-segment alignment does not yet support transcript remapping"
        )
    if params.alignment_chunk_duration > 0:
        raise ValueError(
            "Independent-segment alignment currently uses whole-recording inference"
        )
    if (
        params.blank_frame_filter_threshold is not None
        or params.blank_frame_filter_keep_fraction is not None
    ):
        raise ValueError(
            "Independent-segment alignment currently retains all encoder frames"
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    timing_start = time.perf_counter()

    texts = batch["texts"][0]
    speakers = batch["speakers"][0]
    if not texts:
        return
    ordered_speakers = list(dict.fromkeys(speakers))
    if len(ordered_speakers) > params.max_num_spks:
        raise ValueError(
            f"Recording has {len(ordered_speakers)} speakers but the model has "
            f"{params.max_num_spks} speaker slots"
        )
    speaker_to_slot = {
        speaker: index for index, speaker in enumerate(ordered_speakers)
    }

    seq_idx = batch["supervisions"]["sequence_idx"]
    sort_order = batch["sort_orders"][0]
    starts = [
        batch["supervisions"]["start_sample"][seq_idx == 0][j].item()
        for j in sort_order
    ]
    lengths = [
        batch["supervisions"]["num_samples"][seq_idx == 0][j].item()
        for j in sort_order
    ]

    token_scores, speaker_scores, output_lens = model(
        feature,
        feature_lens,
        speaker_weight=params.speaker_weight,
        return_factors=True,
    )
    valid_frames = int(output_lens[0].item())
    token_scores = token_scores[0, :valid_frames]
    speaker_scores = speaker_scores[0, :valid_frames]
    token_scores[..., 0] -= params.blank_weight
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timing_model = time.perf_counter()

    fsas = []
    vocab_size = graph_compiler.sp.vocab_size()
    for text, start, length, speaker in zip(texts, starts, lengths, speakers):
        graph = graph_compiler.compile_nospk(
            [[text]],
            [[start]],
            [[length]],
            collar=params.collar,
            max_overlaps=params.total_max_overlaps,
            original_topo=params.original_topo,
            single_state_topo=params.single_state_ctc_topo,
        )[0]
        if not params.ignore_speaker:
            positive = graph.labels > 0
            graph.labels[positive] += (
                speaker_to_slot[speaker] * (vocab_size - 1)
            )
        fsas.append(graph)
    graphs = k2.create_fsa_vec(fsas).to(device)
    segment_windows = make_segment_time_windows(
        [[start] for start in starts],
        [[length] for length in lengths],
        margin_seconds=params.segment_time_window_margin,
        frame_duration=params.frame_duration,
    ).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timing_graph = time.perf_counter()

    num_segments = len(fsas)
    # expand() creates views: the model scores are not physically duplicated.
    token_batch = token_scores.unsqueeze(0).expand(num_segments, -1, -1)
    speaker_batch = speaker_scores.unsqueeze(0).expand(num_segments, -1, -1)
    lengths_batch = output_lens.new_full((num_segments,), valid_frames)
    results = factorized_viterbi_batched(
        graphs,
        token_batch,
        speaker_batch,
        lengths_batch,
        segment_windows=segment_windows,
        score_speakers=not params.ignore_speaker,
    )
    bad = [index for index, result in enumerate(results) if not torch.isfinite(result.score)]
    if bad:
        raise RuntimeError(
            "Independent-segment factorized Viterbi found no complete path "
            f"for segment indexes {bad}"
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda_gib = torch.cuda.max_memory_allocated(device) / 2**30
    else:
        peak_cuda_gib = 0.0
    timing_viterbi = time.perf_counter()
    logging.info(
        "Independent-segment profile segments=%d score_speakers=%s "
        "model=%.3fs graphs=%.3fs viterbi=%.3fs total=%.3fs "
        "graph_states=%d graph_arcs=%d peak_cuda=%.3fGiB",
        num_segments,
        not params.ignore_speaker,
        timing_model - timing_start,
        timing_graph - timing_model,
        timing_viterbi - timing_graph,
        timing_viterbi - timing_start,
        graphs.arcs.shape().tot_size(1),
        graphs.arcs.num_elements(),
        peak_cuda_gib,
    )

    recording_duration = float(feature_lens[0].item()) / 16000
    alignment = []
    for segment_index, result in enumerate(results):
        labels = result.labels.long()
        if not params.ignore_speaker:
            labels = 1 + (labels - 1).remainder(vocab_size - 1)
        times = result.frames.float() * params.frame_duration
        if times.numel() == 0:
            continue
        pieces = [
            graph_compiler.sp.id_to_piece(int(label)) for label in labels
        ]
        if times.numel() > 1:
            durations = (times[1:] - times[:-1]).tolist()
            average_duration = sum(durations) / len(durations)
        else:
            durations = []
            average_duration = fallback_token_duration(
                pieces[0],
                params.single_token_duration_per_character,
                params.frame_duration,
                params.single_token_fixed_duration,
            )
        durations.append(
            max(
                0.001,
                min(average_duration, recording_duration - float(times[-1])),
            )
        )
        alignment.extend(
            zip(
                [speakers[segment_index]] * len(pieces),
                [segment_index] * len(pieces),
                pieces,
                [round(float(value), 3) for value in times],
                [round(float(value), 3) for value in durations],
            )
        )

    cut_id = batch["supervisions"]["cut"][0].id
    yield cut_id, list(alignment), num_segments


@torch.no_grad()
def align_one_batch_target_speaker(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    graph_compiler: MDCTCGraphCompiler,
    batch: Dict,
):
    """Align one recording with parallel speaker-specific transcript FSAs."""
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"].to(device)
    feature_lens = batch["num_frames"].to(device)
    if feature.size(0) != 1:
        raise ValueError("Target-speaker alignment currently requires max-cuts=1")
    if params.intersection_method != "factorized_viterbi":
        raise ValueError("Target-speaker alignment currently uses exact factorized_viterbi")

    texts = batch["texts"][0]
    speakers = batch["speakers"][0]
    ordered_speakers = list(dict.fromkeys(speakers))
    if len(ordered_speakers) > params.max_num_spks:
        raise ValueError("Recording has more speakers than the model speaker head")
    seq_idx = batch["supervisions"]["sequence_idx"]
    sort_order = batch["sort_orders"][0]
    starts = [
        batch["supervisions"]["start_sample"][seq_idx == 0][j].item()
        for j in sort_order
    ]
    lengths = [
        batch["supervisions"]["num_samples"][seq_idx == 0][j].item()
        for j in sort_order
    ]
    def target_speaker_column_map(input_start, input_end):
        global_to_local, _, _ = _chunk_speaker_map(
            ordered_speakers, speakers, starts, lengths,
            input_start, input_end,
            strategy=("duration" if params.sort_strategy == "speech_time" else "fifo"),
        )
        # Known recording speakers occupy the leading training targets after
        # chunk-local ordering.  Preserve all remaining head slots as
        # additional non-target mass for the target-specific blank.
        return global_to_local + [
            index for index in range(params.max_num_spks)
            if index not in set(global_to_local)
        ]
    if params.transcript_specific_dense_remap:
        compact_ids, _ = alignment_candidate_ids(
            [texts], [speakers], graph_compiler.sp,
            token_encoder=graph_compiler.encode,
        )
        token_ids = compact_ids[0]
    else:
        # Keep the full, original BPE alphabet for the early ablation rows.
        # The model still runs only once; each target-speaker graph consumes
        # the same full-vocabulary posterior tensor.
        token_ids = list(range(graph_compiler.sp.vocab_size()))

    start_time = time.perf_counter()
    if params.alignment_chunk_duration > 0:
        token, speaker, nonblank_mass = chunked_factor_posteriors(
            model=model,
            waveform=feature[0:1],
            waveform_length=int(feature_lens[0].item()),
            token_ids=token_ids,
            # Retain every output of the speaker head.  Even speaker slots
            # absent from the reference are non-target speakers, so their
            # joint token mass belongs in each target-specific blank.
            speaker_count=params.max_num_spks,
            speaker_weight=params.speaker_weight,
            chunk_duration=params.alignment_chunk_duration,
            chunk_overlap=params.alignment_chunk_overlap,
            chunk_context=params.alignment_chunk_context,
            frame_duration=params.frame_duration,
            return_nonblank_mass=True,
            speaker_column_map_fn=target_speaker_column_map,
        )
    else:
        token_full, speaker_full, output_lens = model(
            feature, feature_lens, speaker_weight=params.speaker_weight,
            return_factors=True,
        )
        valid = int(output_lens[0].item())
        nonblank_mass = token_full[0, :valid, 1:].logsumexp(-1)
        token_index = torch.tensor(token_ids, device=device, dtype=torch.long)
        token = token_full[0, :valid].index_select(-1, token_index)
        # Do not truncate to the oracle/reference speaker count here.  The
        # model may assign probability to any speaker-head slot, and all
        # non-target slots contribute to the target-specific blank.
        speaker = speaker_full[0, :valid]
    original_length = token.size(0)

    original_frame_indices = None
    if (
        params.blank_frame_filter_threshold is not None
        or params.blank_frame_filter_keep_fraction is not None
    ):
        tokens_per_speaker = []
        for name in ordered_speakers:
            tokens_per_speaker.append(sum(
                len(graph_compiler.encode(texts[j]))
                for j, speaker_name in enumerate(speakers) if speaker_name == name
            ))
        token_batch, filtered_lens, original_frame_indices = filter_blank_dominated_frames(
            token.unsqueeze(0), token.new_tensor([token.size(0)], dtype=torch.int32),
            params.blank_frame_filter_threshold,
            minimum_frames=[math.ceil(
                params.blank_frame_filter_min_token_ratio * max(tokens_per_speaker)
            )],
            keep_fraction=params.blank_frame_filter_keep_fraction,
        )
        token = token_batch[0, :int(filtered_lens[0])]
        frame_map = original_frame_indices[0]
        speaker = speaker.index_select(0, frame_map)
        nonblank_mass = nonblank_mass.index_select(0, frame_map)
    else:
        frame_map = torch.arange(token.size(0), device=device)
    target_scores = _target_speaker_scores(
        token, speaker, nonblank_mass
    )[:len(ordered_speakers)]
    target_scores[..., 0] -= params.blank_weight

    speaker_segment_indexes = [
        [j for j, name in enumerate(speakers) if name == target]
        for target in ordered_speakers
    ]
    fsas = []
    for indexes in speaker_segment_indexes:
        graph = graph_compiler.compile_nospk(
            [[texts[j] for j in indexes]],
            [[starts[j] for j in indexes]],
            [[lengths[j] for j in indexes]],
            collar=params.collar,
            max_overlaps=params.total_max_overlaps,
            original_topo=params.original_topo,
            single_state_topo=params.single_state_ctc_topo,
        )[0]
        local_to_global = torch.tensor(
            [0] + [j + 1 for j in indexes], device=graph.aux_labels.device,
            dtype=torch.long,
        )
        positive = graph.aux_labels > 0
        graph.aux_labels[positive] = local_to_global[
            graph.aux_labels[positive].long()
        ].to(graph.aux_labels.dtype)
        fsas.append(graph)
    graphs = k2.create_fsa_vec(fsas)
    if params.transcript_specific_dense_remap:
        inverse_maps = remap_decoding_graph_labels(
            graphs, [token_ids] * len(fsas), [1] * len(fsas),
            graph_compiler.sp.vocab_size(), use_speaker=False,
        )
    else:
        inverse_maps = [
            torch.arange(
                graph_compiler.sp.vocab_size(), device=graphs.labels.device,
                dtype=torch.int64,
            )
            for _ in fsas
        ]

    token_allowed = None
    if params.token_speaker_mask_window_duration > 0:
        dummy = token.new_zeros(
            1, token.size(0), 1 + len(ordered_speakers) * (token.size(-1) - 1)
        )
        masked, stats = apply_windowed_token_speaker_mask(
            scores=dummy,
            lengths=torch.tensor([token.size(0)], device=device, dtype=torch.int32),
            original_frame_indices=[frame_map],
            texts=[texts], speakers=[speakers],
            supervision_start_samples=[starts], supervision_num_samples=[lengths],
            compact_token_ids=[token_ids], sp=graph_compiler.sp,
            token_encoder=graph_compiler.encode,
            frame_duration=params.frame_duration,
            window_duration=params.token_speaker_mask_window_duration,
            boundary_collar=params.token_speaker_mask_boundary_collar,
        )
        joint_allowed = torch.isfinite(masked[0])
        nonblank_allowed = joint_allowed[:, 1:].reshape(
            token.size(0), len(ordered_speakers), token.size(-1) - 1
        ).permute(1, 0, 2)
        token_allowed = torch.cat([
            torch.ones(
                len(ordered_speakers), token.size(0), 1,
                device=device, dtype=torch.bool,
            ),
            nonblank_allowed,
        ], -1)
        logging.info(
            "Target-speaker local mask allowed_nonblank=%d/%d",
            stats["allowed_nonblank"], stats["possible_nonblank"],
        )

    graphs = graphs.to(device)
    graph_time = time.perf_counter()
    logging.info(
        "Target-speaker batched graph speakers=%d states=%d arcs=%d frames=%d/%d",
        len(fsas), graphs.arcs.shape().tot_size(1), graphs.arcs.num_elements(),
        token.size(0), original_length,
    )
    results = token_viterbi_batched(
        graphs, target_scores, token_allowed=token_allowed
    )
    if any(not torch.isfinite(result.score) for result in results):
        failed = [i for i, result in enumerate(results) if not torch.isfinite(result.score)]
        raise RuntimeError(f"Target-speaker Viterbi found no path for speakers {failed}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / 2**30
    else:
        peak = 0.0
    logging.info(
        "Target-speaker alignment profile model+graphs=%.3fs viterbi=%.3fs "
        "peak_cuda=%.3fGiB",
        graph_time - start_time, time.perf_counter() - graph_time, peak,
    )

    cut_id = batch["supervisions"]["cut"][0].id
    recording_duration = float(feature_lens[0].item()) / 16000
    for speaker_index, result in enumerate(results):
        inverse = inverse_maps[speaker_index].to(device)
        units = inverse[result.labels.long()]
        segment_ids = result.aux_labels.long()
        original_frames = frame_map[result.frames.long()]
        times = original_frames.float() * params.frame_duration
        alignment = []
        for global_segment in speaker_segment_indexes[speaker_index]:
            segment_id = global_segment + 1
            selected = segment_ids == segment_id
            segment_times = times[selected]
            if segment_times.numel() == 0:
                continue
            pieces = [graph_compiler.sp.id_to_piece(int(x)) for x in units[selected]]
            if segment_times.numel() > 1:
                durations = (segment_times[1:] - segment_times[:-1]).tolist()
                final_duration = sum(durations) / len(durations)
            else:
                durations = []
                final_duration = fallback_token_duration(
                    pieces[0],
                    params.single_token_duration_per_character,
                    params.frame_duration,
                    params.single_token_fixed_duration,
                )
            durations.append(min(final_duration, recording_duration - float(segment_times[-1])))
            alignment.extend(zip(
                [ordered_speakers[speaker_index]] * len(pieces),
                [global_segment] * len(pieces), pieces,
                [round(float(x), 3) for x in segment_times],
                [round(float(x), 3) for x in durations],
            ))
        yield cut_id, list(alignment), len(speaker_segment_indexes[speaker_index])


def align_one_batch(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    graph_compiler: MDCTCGraphCompiler,
) -> Dict[str, List[List[str]]]:
    """
    Compute CTC loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Conformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      graph_compiler:
        It is used to build a decoding graph from a ctc topo and training
        transcript. The training transcript is contained in the given `batch`,
        while the ctc topo is built when this compiler is instantiated.
    """
    if params.blockwise_alignment:
        yield from align_one_batch_blockwise(
            params=params, model=model, graph_compiler=graph_compiler, batch=batch
        )
        return
    if params.independent_segment_alignment:
        yield from align_one_batch_independent_segments(
            params=params, model=model, graph_compiler=graph_compiler, batch=batch
        )
        return
    if params.target_speaker_alignment:
        yield from align_one_batch_target_speaker(
            params=params, model=model, graph_compiler=graph_compiler, batch=batch
        )
        return
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    timing_start = time.perf_counter()
    feature = batch["inputs"]
    max_density = batch["max_density"]
    feature_lens = batch["num_frames"].to(device)
    # at entry, feature is (N, T, C)
    feature = feature.to(device)
    feature_lens = feature_lens.to(device)
    texts = batch["texts"]
    speakers = batch["speakers"]
    segments = [[i for i, s in enumerate(b)] for b in batch["speakers"]]
    sort_orders = batch["sort_orders"] 
    seq_idx = batch['supervisions']['sequence_idx']
    start_frames = [
        [batch['supervisions']['start_sample'][seq_idx == i][j].item() for j in sort_orders[i]]
        for i in range(seq_idx.max()+1)
    ]
    num_frames_init = [
        [batch['supervisions']['num_samples'][seq_idx == i][j].item() for j in sort_orders[i]]
        for i in range(seq_idx.max()+1)
    ]
    beam_factor = 1
    with torch.set_grad_enabled(False):
        compact_token_ids = None
        compact_speaker_counts = None
        if params.transcript_specific_dense_remap:
            compact_token_ids, compact_speaker_counts = alignment_candidate_ids(
                texts, speakers, graph_compiler.sp,
                token_encoder=graph_compiler.encode,
            )
            if any(count > params.max_num_spks for count in compact_speaker_counts):
                raise ValueError(
                    "A recording has more transcript speakers than the model head: "
                    f"{compact_speaker_counts} vs max_num_spks={params.max_num_spks}"
                )
        use_viterbi_alignment = params.intersection_method in (
            "factorized_viterbi", "factorized_viterbi_pruned"
        )
        use_pruned_viterbi = (
            params.intersection_method == "factorized_viterbi_pruned"
        )
        use_segment_factor_alignment = params.token_only_segment_factor_alignment
        use_factorized_alignment = (
            use_segment_factor_alignment
            or (
                (params.factorized_alignment or use_viterbi_alignment)
                and not params.ignore_speaker
                and not params.latent_speaker_alignment
            )
        )
        if use_segment_factor_alignment:
            if not params.ignore_speaker:
                raise ValueError(
                    "Segment-factor alignment is token-only and requires "
                    "--ignore-speaker=true"
                )
            if not use_viterbi_alignment:
                raise ValueError(
                    "Segment-factor alignment currently requires exact or "
                    "pruned factorized Viterbi"
                )
            if params.latent_speaker_alignment:
                raise ValueError(
                    "Segment-factor and latent-speaker alignment are mutually "
                    "exclusive"
                )
        if params.factorized_alignment and not use_factorized_alignment:
            raise ValueError(
                "Factorized alignment requires a directly speaker-attributed "
                "graph (ignore-speaker=false, latent-speaker-alignment=false)"
            )
        if use_viterbi_alignment and params.latent_speaker_alignment:
            raise ValueError(
                "Factorized Viterbi does not support latent-speaker graphs"
            )
        if (
            params.segment_time_window_margin is not None
            and not use_viterbi_alignment
        ):
            raise ValueError(
                "--segment-time-window-margin requires "
                "--intersection-method=factorized_viterbi or "
                "factorized_viterbi_pruned"
            )
        factor_outputs = (
            params.ignore_speaker
            or params.latent_speaker_alignment
            or use_factorized_alignment
            or params.transcript_specific_dense_remap
            or params.alignment_chunk_duration > 0
        )
        if params.alignment_chunk_duration > 0:
            if not params.transcript_specific_dense_remap:
                raise ValueError(
                    "Chunked long-form inference requires "
                    "--transcript-specific-dense-remap=true so full-vocabulary "
                    "chunk outputs are released immediately"
                )
            if feature.size(0) != 1:
                raise ValueError(
                    "Chunked long-form alignment currently requires max-cuts=1; "
                    f"got batch size {feature.size(0)}"
                )
            speaker_column_map_fn = None
            if not params.ignore_speaker:
                ordered_speakers = list(dict.fromkeys(speakers[0]))
                if params.sort_strategy == "speech_time":
                    chunk_speaker_ordering = "duration"
                elif params.sort_strategy == "start_time":
                    chunk_speaker_ordering = "fifo"
                else:
                    raise ValueError(
                        "Chunked speaker-attributed alignment cannot infer "
                        "the local speaker ordering from sort strategy "
                        f"{params.sort_strategy!r}"
                    )

                def speaker_column_map_fn(input_start, input_end):
                    mapping, _, _ = _chunk_speaker_map(
                        ordered_speakers,
                        speakers[0],
                        start_frames[0],
                        num_frames_init[0],
                        input_start,
                        input_end,
                        strategy=chunk_speaker_ordering,
                    )
                    return mapping

            token_piece, speaker_piece = chunked_factor_posteriors(
                model=model,
                waveform=feature[0:1],
                waveform_length=int(feature_lens[0].item()),
                token_ids=compact_token_ids[0],
                speaker_count=compact_speaker_counts[0],
                speaker_weight=params.speaker_weight,
                chunk_duration=params.alignment_chunk_duration,
                chunk_overlap=params.alignment_chunk_overlap,
                chunk_context=params.alignment_chunk_context,
                frame_duration=params.frame_duration,
                speaker_column_map_fn=speaker_column_map_fn,
            )
            ctc_output = token_piece.unsqueeze(0)
            out2 = speaker_piece.unsqueeze(0)
            x_lens = torch.tensor(
                [token_piece.size(0)], device=device, dtype=torch.int32
            )
            logging.info(
                "Chunked inference used %.3fs chunks with %.3fs overlap and "
                "%.3fs left/right context; "
                "stitched_frames=%s compact_tokens=%s compact_speakers=%s",
                params.alignment_chunk_duration,
                params.alignment_chunk_overlap,
                params.alignment_chunk_context,
                x_lens.tolist(),
                [len(x) for x in compact_token_ids],
                compact_speaker_counts,
            )
        elif factor_outputs:
            factor_output = model(
                feature, feature_lens,
                speaker_weight = params.speaker_weight,
                return_factors = True,
            )
            if not isinstance(factor_output, tuple) or len(factor_output) != 3:
                raise ValueError(
                    "This alignment mode requires the model to expose token "
                    "and speaker/factor outputs separately. For WavLM, set "
                    "--factorized-token-speaker-output=true."
                )
            ctc_output, out2, x_lens = factor_output
            if params.transcript_specific_dense_remap:
                if feature.size(0) != 1:
                    raise ValueError(
                        "Transcript-specific dense remapping currently requires "
                        f"max-cuts=1; got batch size {feature.size(0)}"
                    )
                token_index = torch.tensor(
                    compact_token_ids[0], device=device, dtype=torch.long
                )
                ctc_output = ctc_output.index_select(-1, token_index)
                out2 = out2[..., : compact_speaker_counts[0]]
            if params.ignore_speaker and params.max_speaker_score_for_token_graph:
                valid_speaker_counts = [
                    len(set(batch_speakers))
                    if params.oracle_num_spks == 0
                    else params.oracle_num_spks
                    if params.oracle_num_spks is not None
                    else params.max_num_spks
                    for batch_speakers in speakers
                ]
                ctc_output = collapse_max_speaker_scores(
                    ctc_output, out2, valid_speaker_counts
                )
        else:
            ctc_output, x_lens = model(
                feature, feature_lens,
                speaker_weight = params.speaker_weight,
            )

        if use_segment_factor_alignment:
            # Replace the token-only model's neutral compatibility factor with
            # the same normalized, nonlearned segment prior used in training.
            out2 = make_uniform_segment_log_probs(
                start_frames,
                num_frames_init,
                feature_lens,
                x_lens,
                ctc_output.size(1),
                params.segment_factor_collar,
                device,
                decay_scale=params.segment_factor_decay_scale,
                floor_probability=params.segment_factor_floor_prob,
            )
            logging.info(
                "Segment-factor alignment prior: segments=%s collar=%d "
                "decay_scale=%.1f floor_probability=%.8g",
                [len(x) for x in start_frames],
                params.segment_factor_collar,
                params.segment_factor_decay_scale,
                params.segment_factor_floor_prob,
            )

        original_frame_indices = None
        if (
            params.blank_frame_filter_threshold is not None
            or params.blank_frame_filter_keep_fraction is not None
        ):
            if params.blank_frame_filter_min_token_ratio < 1.0:
                raise ValueError(
                    "--blank-frame-filter-min-token-ratio must be >= 1.0, got "
                    f"{params.blank_frame_filter_min_token_ratio}"
                )
            original_lengths = x_lens.clone()
            unfiltered_ctc_output = ctc_output
            unfiltered_out2 = out2 if factor_outputs else None
            minimum_frames = [
                math.ceil(
                    params.blank_frame_filter_min_token_ratio
                    * sum(
                        len(graph_compiler.encode(text))
                        for text in cut_texts
                    )
                )
                for cut_texts in texts
            ]
            ctc_output, x_lens, original_frame_indices = (
                filter_blank_dominated_frames(
                    ctc_output,
                    x_lens,
                    params.blank_frame_filter_threshold,
                    minimum_frames=minimum_frames,
                    keep_fraction=params.blank_frame_filter_keep_fraction,
                )
            )
            if (
                params.segment_time_window_margin is not None
                and params.single_state_ctc_topo
            ):
                feasibility_windows = make_segment_time_windows(
                    supervision_start_samples=start_frames,
                    supervision_num_samples=num_frames_init,
                    margin_seconds=params.segment_time_window_margin,
                    frame_duration=params.frame_duration,
                )
                segment_token_counts = [
                    [len(graph_compiler.encode(text)) for text in cut_texts]
                    for cut_texts in texts
                ]
                original_frame_indices, additions = (
                    add_segment_window_feasibility_frames(
                        original_frame_indices,
                        original_lengths,
                        feasibility_windows,
                        segment_token_counts,
                    )
                )
                if any(additions):
                    x_lens = original_lengths.new_tensor(
                        [indexes.numel() for indexes in original_frame_indices]
                    )
                    ctc_output = select_frames_with_map(
                        unfiltered_ctc_output, original_frame_indices, x_lens
                    )
                    logging.info(
                        "Blank pruning segment-window feasibility guard added "
                        "frames per cut=%s",
                        additions,
                    )
            if factor_outputs:
                out2 = select_frames_with_map(
                    unfiltered_out2, original_frame_indices, x_lens
                )
            kept = int(x_lens.sum().item())
            total = int(original_lengths.sum().item())
            logging.info(
                "Blank-frame first pass retained %s/%s frames (%.1f%%) "
                "with threshold=%s keep_fraction=%s; per-cut retained=%s/%s",
                kept,
                total,
                100.0 * kept / max(1, total),
                params.blank_frame_filter_threshold,
                params.blank_frame_filter_keep_fraction,
                x_lens.tolist(),
                original_lengths.tolist(),
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing_model_filter = time.perf_counter()

        # I need to expand the speaker dimension back out
        subsampling_factor = params.subsampling_factor
        beam_size = params.beam_size
        reduction = params.reduction
        use_double_scores = params.use_double_scores
        
        # Align with WFST supervisions
        sequence_idx = torch.arange(
            0, x_lens.size(0),
        ).unsqueeze(0).t().to(torch.int32)

        start_frame = torch.zeros(
            [x_lens.size(0)], dtype=torch.int32,
        ).unsqueeze(0).t()

        num_frames = x_lens.unsqueeze(1).to(torch.int32).cpu()

        supervision_segments = torch.cat(
            [sequence_idx, start_frame, num_frames],
            dim=1,
        )
        supervision_segments = supervision_segments.to(torch.int32)
        
        ctc_output[..., 0] -= params.blank_weight
        collar = params.collar

        if params.latent_speaker_alignment:
            valid_speaker_counts = [
                len(set(batch_speakers))
                if params.oracle_num_spks == 0
                else params.oracle_num_spks
                if params.oracle_num_spks is not None
                else params.max_num_spks
                for batch_speakers in speakers
            ]
            decoding_graphs = graph_compiler.compile_latent_speaker(
                texts, start_frames, num_frames_init, valid_speaker_counts,
                collar=collar,
                max_overlaps=params.total_max_overlaps,
                original_topo=params.original_topo,
                max_graph_states=params.latent_speaker_max_graph_states,
            )
        elif use_segment_factor_alignment:
            segment_ids = [list(range(len(cut_texts))) for cut_texts in texts]
            decoding_graphs = graph_compiler.compile(
                texts,
                start_frames,
                num_frames_init,
                segment_ids,
                collar=collar,
                edge_collar=params.edge_collar,
                interval_model=params.interval_model,
                max_overlaps=params.total_max_overlaps,
                original_topo=params.original_topo,
                single_state_topo=params.single_state_ctc_topo,
                # Segment identities, unlike physical speakers, are unique.
                # Keep the same cross-segment partial order as training.
                allow_self_overlap=True,
            )
        elif params.ignore_speaker:
            decoding_graphs = graph_compiler.compile_nospk(
                texts, start_frames, num_frames_init,
                collar=collar,
                max_overlaps=params.total_max_overlaps,
                original_topo=params.original_topo,
                single_state_topo=params.single_state_ctc_topo,
            )
        else: 
            decoding_graphs = graph_compiler.compile(
                texts, start_frames, num_frames_init, speakers,
                collar=collar,
                max_overlaps=params.total_max_overlaps,
                original_topo=params.original_topo,
                single_state_topo=params.single_state_ctc_topo,
                allow_self_overlap=params.allow_self_overlap,
            )

        compact_inverse_maps = None
        if params.transcript_specific_dense_remap:
            remap_factor_counts = (
                [len(cut_texts) for cut_texts in texts]
                if use_segment_factor_alignment
                else compact_speaker_counts
            )
            compact_inverse_maps = remap_decoding_graph_labels(
                decoding_graphs=decoding_graphs,
                token_ids=compact_token_ids,
                speaker_counts=remap_factor_counts,
                original_vocab_size=graph_compiler.sp.vocab_size(),
                use_speaker=(
                    not params.ignore_speaker or use_segment_factor_alignment
                ),
            )
        if params.alignment_diagnostics:
            graph_properties = decoding_graphs.properties
            blank_graph = decoding_graphs.clone()
            blank_graph.scores[blank_graph.labels > 0] = -torch.inf
            try:
                blank_totals = blank_graph.get_tot_scores(
                    log_semiring=True, use_double_scores=True
                )
                blank_totals_value = blank_totals.tolist()
                blank_reachable = torch.isfinite(blank_totals).tolist()
            except RuntimeError:
                # CTC decoding graphs contain cycles.  k2 total scores require
                # a topologically sortable FSA, so this graph-only probe is
                # unavailable; the intersected time-expanded lattice below is
                # acyclic and remains diagnostic.
                blank_totals_value = "unavailable_for_cyclic_graph"
                blank_reachable = "unknown"
            logging.info(
                "Alignment diagnostics graph: states=%d arcs=%d "
                "positive_input_arcs=%d positive_aux_arcs=%d "
                "properties=%d valid=%s nonempty=%s accessible=%s "
                "coaccessible=%s arc_sorted=%s blank_only_totals=%s "
                "blank_only_reachable=%s",
                decoding_graphs.arcs.shape().tot_size(1),
                decoding_graphs.arcs.num_elements(),
                int((decoding_graphs.labels > 0).sum().item()),
                int((decoding_graphs.aux_labels > 0).sum().item()),
                graph_properties,
                bool(graph_properties & k2.fsa_properties.VALID),
                bool(graph_properties & k2.fsa_properties.NONEMPTY),
                bool(graph_properties & k2.fsa_properties.ACCESSIBLE),
                bool(graph_properties & k2.fsa_properties.COACCESSIBLE),
                bool(graph_properties & k2.fsa_properties.ARC_SORTED),
                blank_totals_value,
                blank_reachable,
            )
        if factor_outputs and not params.ignore_speaker and not use_factorized_alignment:
            # Expand only the compact recording-specific token/speaker product.
            # Both branches retain their original full-softmax denominators.
            blank = ctc_output[..., :1]
            joint = ctc_output[..., 1:].unsqueeze(-1) + out2.unsqueeze(-2)
            joint = joint.permute(0, 1, 3, 2).reshape(
                joint.size(0), joint.size(1), -1
            )
            ctc_output = torch.cat([blank, joint], dim=-1).clamp_max(0.0)

        joint_allowed = None
        if params.token_speaker_mask_window_duration > 0:
            if original_frame_indices is None:
                raise ValueError(
                    "--token-speaker-mask-window-duration is deliberately a "
                    "post-blank-removal operation; enable blank-frame filtering"
                )
            if not params.transcript_specific_dense_remap:
                raise ValueError(
                    "--token-speaker-mask-window-duration currently requires "
                    "--transcript-specific-dense-remap=true"
                )
            if params.latent_speaker_alignment:
                raise ValueError(
                    "The token/speaker mask does not support latent-speaker graphs"
                )
            mask_input = ctc_output
            if use_viterbi_alignment:
                # The mask is joint but the scores remain factorized. Build
                # only a temporary boolean admissibility matrix; no joint
                # floating-point logits are passed to the Viterbi core.
                joint_dim = 1 + out2.size(-1) * (ctc_output.size(-1) - 1)
                mask_input = ctc_output.new_zeros(
                    ctc_output.size(0), ctc_output.size(1), joint_dim
                )
            masked_scores, mask_stats = apply_windowed_token_speaker_mask(
                scores=mask_input,
                lengths=x_lens,
                original_frame_indices=original_frame_indices,
                texts=texts,
                speakers=speakers,
                supervision_start_samples=start_frames,
                supervision_num_samples=num_frames_init,
                compact_token_ids=compact_token_ids,
                sp=graph_compiler.sp,
                token_encoder=graph_compiler.encode,
                frame_duration=params.frame_duration,
                window_duration=params.token_speaker_mask_window_duration,
                boundary_collar=params.token_speaker_mask_boundary_collar,
            )
            if use_viterbi_alignment:
                joint_allowed = torch.isfinite(masked_scores)
                if params.ignore_speaker:
                    # compile_nospk labels are compact token IDs, so retain a
                    # token at a frame when it is locally permitted for any
                    # globally assigned speaker. Speaker scores themselves do
                    # not participate in shuffle-only Viterbi traversal.
                    joint_allowed = torch.cat(
                        [
                            joint_allowed[..., :1],
                            joint_allowed[..., 1:].reshape(
                                *joint_allowed.shape[:-1],
                                out2.size(-1),
                                ctc_output.size(-1) - 1,
                            ).any(dim=-2),
                        ],
                        dim=-1,
                    )
                del masked_scores, mask_input
            else:
                ctc_output = masked_scores
            allowed = mask_stats["allowed_nonblank"]
            possible = mask_stats["possible_nonblank"]
            logging.info(
                "Post-blank-removal token/speaker mask: window=%.3fs "
                "boundary_collar=%.3fs windows=%d blank_only_windows=%d "
                "allowed_nonblank=%d/%d (%.3f%%)",
                params.token_speaker_mask_window_duration,
                params.token_speaker_mask_boundary_collar,
                mask_stats["windows"],
                mask_stats["blank_only_windows"],
                allowed,
                possible,
                100.0 * allowed / max(1, possible),
            )

        segment_windows = None
        viterbi_original_frame_indices = None
        if params.segment_time_window_margin is not None:
            segment_windows = make_segment_time_windows(
                supervision_start_samples=start_frames,
                supervision_num_samples=num_frames_init,
                margin_seconds=params.segment_time_window_margin,
                frame_duration=params.frame_duration,
            ).to(device)
            if original_frame_indices is not None:
                viterbi_original_frame_indices = torch.zeros(
                    (ctc_output.size(0), ctc_output.size(1)),
                    dtype=torch.long,
                    device=device,
                )
                for cut_index, frame_map in enumerate(original_frame_indices):
                    length = int(x_lens[cut_index].item())
                    if frame_map.numel() != length:
                        raise ValueError(
                            f"Original-frame map for cut {cut_index} has "
                            f"{frame_map.numel()} entries, expected {length}"
                        )
                    viterbi_original_frame_indices[cut_index, :length] = (
                        frame_map.to(device=device, dtype=torch.long)
                    )
            logging.info(
                "Hard segment-specific timing windows: margin=%.3fs "
                "segments_per_cut=%s original_frame_map=%s",
                params.segment_time_window_margin,
                [len(x) for x in start_frames],
                original_frame_indices is not None,
            )

        if params.diagnostic_uniform_scores or params.diagnostic_frame_delta:
            if ctc_output.size(0) != 1 or use_factorized_alignment:
                raise ValueError(
                    "Diagnostic score/frame overrides require one cut and a "
                    "non-factorized dense intersection"
                )
            delta = params.diagnostic_frame_delta
            old_length = int(x_lens[0].item())
            new_length = old_length + delta
            if new_length <= 0:
                raise ValueError(
                    f"diagnostic frame delta {delta} gives length {new_length}"
                )
            if delta < 0:
                ctc_output = ctc_output[:, :new_length]
            elif delta > 0:
                padding = ctc_output.new_zeros(
                    1, delta, ctc_output.size(-1)
                )
                ctc_output = torch.cat([ctc_output[:, :old_length], padding], dim=1)
            x_lens = x_lens.new_tensor([new_length])
            supervision_segments[0, 2] = new_length
            if params.diagnostic_uniform_scores:
                ctc_output = torch.zeros_like(ctc_output)
            logging.info(
                "Alignment diagnostic override: uniform_scores=%s "
                "frame_delta=%d frames=%d->%d",
                params.diagnostic_uniform_scores,
                delta,
                old_length,
                new_length,
            )
        
        decoding_graphs = decoding_graphs.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            current_cuda_gib = torch.cuda.memory_allocated(device) / 2**30
        else:
            current_cuda_gib = 0.0
        logging.info(
            "Pre-intersection profile cuts=%s dense_shape=%s "
            "graph_states=%s graph_arcs=%s current_cuda=%.3fGiB",
            [c.id for c in batch["supervisions"]["cut"]],
            tuple(ctc_output.shape),
            decoding_graphs.arcs.shape().tot_size(1),
            decoding_graphs.arcs.num_elements(),
            current_cuda_gib,
        )
        timing_graph = time.perf_counter()

        end_sup = time.time()
        intersection_error = None

        viterbi_result = None
        viterbi_results = None
        if use_viterbi_alignment:
            if use_pruned_viterbi:
                if ctc_output.size(0) != 1:
                    raise ValueError(
                        "Pruned factorized_viterbi currently requires one cut"
                    )
                viterbi_result = factorized_viterbi(
                    decoding_graphs,
                    ctc_output[0, : int(x_lens[0].item())],
                    out2[0, : int(x_lens[0].item())],
                    joint_allowed=(
                        None if joint_allowed is None
                        else joint_allowed[0, : int(x_lens[0].item())]
                    ),
                    segment_windows=(
                        None if segment_windows is None else segment_windows[0]
                    ),
                    original_frame_indices=(
                        None if viterbi_original_frame_indices is None else
                        viterbi_original_frame_indices[
                            0, : int(x_lens[0].item())
                        ]
                    ),
                    score_speakers=(
                        not params.ignore_speaker or use_segment_factor_alignment
                    ),
                    search_beam=params.search_beam,
                    min_active_states=params.pruned_min_active_states,
                    max_active_states=params.pruned_max_active_states,
                )
                viterbi_results = [viterbi_result]
            else:
                viterbi_results = factorized_viterbi_batched(
                    decoding_graphs,
                    ctc_output,
                    out2,
                    x_lens,
                    joint_allowed=joint_allowed,
                    segment_windows=segment_windows,
                    original_frame_indices=viterbi_original_frame_indices,
                    score_speakers=(
                        not params.ignore_speaker or use_segment_factor_alignment
                    ),
                )
                # Retain the historical scalar name for downstream mode tests;
                # actual path tensors are assembled from viterbi_results below.
                viterbi_result = viterbi_results[0]
            if use_pruned_viterbi:
                counts = viterbi_result.active_state_counts
                logging.info(
                    "Pruned Viterbi active states: frames=%d min=%d max=%d "
                    "mean=%.1f final=%d search_beam=%.3f limits=[%d,%d]",
                    counts.numel(), int(counts.min()), int(counts.max()),
                    float(counts.float().mean()), int(counts[-1]),
                    params.search_beam, params.pruned_min_active_states,
                    params.pruned_max_active_states,
                )
            bad_viterbi = [
                index for index, result in enumerate(viterbi_results)
                if not torch.isfinite(result.score)
            ]
            if bad_viterbi:
                cut_ids = [c.id for c in batch["supervisions"]["cut"]]
                intersection_error = (
                    "factorized Viterbi found no complete path for "
                    f"{[cut_ids[index] for index in bad_viterbi]}"
                )
        elif use_factorized_alignment:
            lattice = intersect_dense_factorized(
                a_fsas=decoding_graphs,
                token_log_probs=ctc_output,
                speaker_log_probs=out2,
                supervision_segments=supervision_segments,
                output_beam=beam_size,
                max_states=params.max_states,
                allow_truncate=subsampling_factor - 1,
            )
            totals = lattice.get_tot_scores(
                log_semiring=True, use_double_scores=True
            )
            if not torch.isfinite(totals).all():
                cut_ids = [c.id for c in batch["supervisions"]["cut"]]
                intersection_error = (
                    "factorized intersection produced empty/non-finite lattices "
                    f"for {cut_ids}; output_beam={beam_size}, "
                    f"max_states={params.max_states}"
                )
        elif not use_viterbi_alignment:
            dense_fsa_vec = k2.DenseFsaVec(
                ctc_output.float(),
                supervision_segments.cpu(),
                allow_truncate=subsampling_factor - 1,
            )

        if (
            not use_viterbi_alignment
            and not use_factorized_alignment
            and params.intersection_method == "online_dense_pruned"
        ):
            if ctc_output.size(0) != 1:
                raise ValueError(
                    "online_dense_pruned currently requires a one-cut minibatch"
                )
            block_frames = params.online_intersection_block_frames
            if block_frames <= 0:
                raise ValueError("--online-intersection-block-frames must be positive")

            # OnlineDenseIntersecter requires a single shared decoding graph.
            # DecodeStateInfo deliberately survives between calls.  The k2
            # implementation stores historical frames in it, so logging CUDA
            # allocation after every call lets us test whether this API helps
            # the long-recording FormatOutput memory spike in practice.
            online_intersector = k2.OnlineDenseIntersecter(
                decoding_graph=decoding_graphs,
                num_streams=1,
                search_beam=params.search_beam,
                output_beam=beam_size,
                min_active_states=params.pruned_min_active_states,
                max_active_states=params.pruned_max_active_states,
                allow_partial=True,
            )
            decode_states = [k2.DecodeStateInfo()]
            num_dense_frames = ctc_output.size(1)
            num_blocks = (num_dense_frames + block_frames - 1) // block_frames
            lattice = None
            for block_index, begin in enumerate(
                range(0, num_dense_frames, block_frames)
            ):
                end = min(begin + block_frames, num_dense_frames)
                block_segments = torch.tensor(
                    [[0, 0, end - begin]], dtype=torch.int32
                )
                block_dense = k2.DenseFsaVec(
                    ctc_output[:, begin:end].float(),
                    block_segments,
                    allow_truncate=0,
                )
                block_start = time.perf_counter()
                lattice, decode_states = online_intersector.decode(
                    block_dense, decode_states
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    block_allocated = torch.cuda.memory_allocated(device) / 2**30
                    block_reserved = torch.cuda.memory_reserved(device) / 2**30
                    block_peak = torch.cuda.max_memory_allocated(device) / 2**30
                else:
                    block_allocated = block_reserved = block_peak = 0.0
                logging.info(
                    "Online intersection block=%d/%d retained_frames=[%d,%d) "
                    "elapsed=%.3fs lattice_states=%d lattice_arcs=%d "
                    "cuda_allocated=%.3fGiB cuda_reserved=%.3fGiB "
                    "cuda_peak=%.3fGiB",
                    block_index + 1,
                    num_blocks,
                    begin,
                    end,
                    time.perf_counter() - block_start,
                    lattice.arcs.shape().tot_size(1),
                    lattice.arcs.num_elements(),
                    block_allocated,
                    block_reserved,
                    block_peak,
                )
            assert lattice is not None
            totals = lattice.get_tot_scores(
                log_semiring=True, use_double_scores=True
            )
            if not torch.isfinite(totals).all():
                intersection_error = (
                    "online_dense_pruned produced an empty/non-finite final "
                    "partial lattice; the graph final state may not have "
                    "survived or been reached"
                )
        elif (
            not use_viterbi_alignment
            and not use_factorized_alignment
            and params.intersection_method == "dense_pruned"
        ):
            lattice = k2.intersect_dense_pruned(
                a_fsas=decoding_graphs,
                b_fsas=dense_fsa_vec,
                search_beam=params.search_beam,
                output_beam=beam_size,
                max_active_states=params.pruned_max_active_states,
                min_active_states=params.pruned_min_active_states,
                frame_idx_name="frame",
                allow_partial=False,
            )
            totals = lattice.get_tot_scores(
                log_semiring=True, use_double_scores=True
            )
            if not torch.isfinite(totals).all():
                cut_ids = [c.id for c in batch["supervisions"]["cut"]]
                bad = [
                    cut_ids[i] for i, finite in enumerate(torch.isfinite(totals))
                    if not finite.item()
                ]
                intersection_error = (
                    "intersect_dense_pruned produced empty/non-finite lattices "
                    f"for {bad}; search_beam={params.search_beam}, "
                    f"output_beam={beam_size}, active_states="
                    f"[{params.pruned_min_active_states}, "
                    f"{params.pruned_max_active_states}]"
                )
        elif not use_viterbi_alignment and not use_factorized_alignment:
            lattice = k2.intersect_dense(
                a_fsas=decoding_graphs,
                b_fsas=dense_fsa_vec,
                output_beam=beam_size,
                max_states=params.max_states,
                frame_idx_name="frame",
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_cuda_gib = torch.cuda.max_memory_allocated(device) / 2**30
        else:
            peak_cuda_gib = 0.0
        timing_intersection = time.perf_counter()
        logging.info(
            "Alignment profile cuts=%s model+filter=%.3fs graph=%.3fs "
            "intersection=%.3fs total_to_lattice=%.3fs peak_cuda=%.3fGiB "
            "graph_states=%s graph_arcs=%s",
            [c.id for c in batch["supervisions"]["cut"]],
            timing_model_filter - timing_start,
            timing_graph - timing_model_filter,
            timing_intersection - timing_graph,
            timing_intersection - timing_start,
            peak_cuda_gib,
            decoding_graphs.arcs.shape().tot_size(1),
            decoding_graphs.arcs.num_elements(),
        )
        if intersection_error is not None:
            raise RuntimeError(intersection_error)

        if params.alignment_diagnostics and viterbi_result is None:
            lattice_totals = lattice.get_tot_scores(
                log_semiring=True, use_double_scores=True
            )
            logging.info(
                "Alignment diagnostics lattice: states=%d/%d arcs=%d "
                "totals=%s finite=%s",
                lattice.arcs.shape().tot_size(1),
                params.max_states,
                lattice.arcs.num_elements(),
                lattice_totals.tolist(),
                torch.isfinite(lattice_totals).tolist(),
            )

        if (
            params.intersection_method == "online_dense_pruned"
            and params.online_intersection_profile_only
        ):
            for cut, cut_speakers in zip(
                batch["supervisions"]["cut"], speakers
            ):
                yield cut.id, None, len(set(cut_speakers))
            return

        if viterbi_result is None:
            best_path = k2.shortest_path(lattice, use_double_scores=True)
        else:
            best_path = None
        if params.alignment_diagnostics and best_path is not None:
            best_labels = best_path.labels
            best_aux = best_path.aux_labels
            logging.info(
                "Alignment diagnostics best path: arcs=%d "
                "positive_input=%d blank_input=%d final_input=%d "
                "positive_aux=%d zero_aux=%d final_aux=%d "
                "first_input=%s first_aux=%s",
                best_path.arcs.num_elements(),
                int((best_labels > 0).sum().item()),
                int((best_labels == 0).sum().item()),
                int((best_labels == -1).sum().item()),
                int((best_aux > 0).sum().item()),
                int((best_aux == 0).sum().item()),
                int((best_aux == -1).sum().item()),
                best_labels[:40].tolist(),
                best_aux[:40].tolist(),
            )
        if viterbi_result is None:
            labels_t = best_path.labels.contiguous()
            aux_labels_t = best_path.aux_labels.contiguous()
            frames_t = best_path.frame.contiguous()
            # Collapse the state axis to obtain [FSA][arc].
            arc_shape = best_path.arcs.shape().remove_axis(1)
        else:
            labels_t = torch.cat([
                result.labels for result in viterbi_results
            ]).contiguous()
            aux_labels_t = torch.cat([
                result.aux_labels for result in viterbi_results
            ]).contiguous()
            frames_t = torch.cat([
                result.frames for result in viterbi_results
            ]).contiguous()
            # The Viterbi result already contains only emitting arcs.
            arc_shape = k2.RaggedTensor(
                [result.labels.cpu().tolist() for result in viterbi_results]
            ).to(labels_t.device).shape

        # Build a RaggedTensor for all labels (no filtering yet). Constructing
        # a CUDA RaggedTensor from a separately created shape is unsupported
        # in this k2 build, so create and move the small Viterbi path as one
        # object.
        if viterbi_result is None:
            ragged_all_labels = k2.RaggedTensor(arc_shape, labels_t)
            ragged_all_aux_labels = k2.RaggedTensor(arc_shape, aux_labels_t)
        else:
            ragged_all_labels = k2.RaggedTensor(
                [result.labels.cpu().tolist() for result in viterbi_results]
            ).to(labels_t.device)
            ragged_all_aux_labels = k2.RaggedTensor(
                [result.aux_labels.cpu().tolist() for result in viterbi_results]
            ).to(aux_labels_t.device)
        
        # Remove all non-positive labels (<=0: blanks and possibly -1 sentinels)
        units = ragged_all_labels.remove_values_leq(0)
        if compact_inverse_maps is not None:
            units = restore_compact_units(units, compact_inverse_maps)
        # ragged_clean_labels.shape is now the correct ragged shape for the filtered values

        segs = ragged_all_aux_labels.remove_values_leq(0)
        
        # Build the frames ragged using the SAME shape and the filtered frame values.
        # Compute boolean mask on the original flat tensors to get the same set of kept indices:
        keep_mask = (labels_t > 0)  # True for values we kept; this matches remove_values_leq(0)

        frames_kept = frames_t[keep_mask]

        # Now build ragged frames with the *new* shape.
        if viterbi_result is None:
            ragged_frames = k2.RaggedTensor(units.shape, frames_kept)
        else:
            ragged_frames = k2.RaggedTensor(
                [result.frames.cpu().tolist() for result in viterbi_results]
            ).to(frames_kept.device)
        if original_frame_indices is None:
            ragged_original_frames = ragged_frames
        else:
            mapped_frames = []
            for i in range(ragged_frames.shape.dim0):
                filtered_idx = ragged_frames[i].to(torch.int64)
                frame_map = original_frame_indices[i]
                if filtered_idx.numel() and filtered_idx.max() >= frame_map.numel():
                    raise RuntimeError(
                        f"Filtered alignment frame {filtered_idx.max().item()} "
                        f"is outside frame map of length {frame_map.numel()}"
                    )
                mapped_frames.append(frame_map[filtered_idx])
            mapped_values = torch.cat(mapped_frames) if mapped_frames else frames_kept
            ragged_original_frames = k2.RaggedTensor(
                ragged_frames.shape, mapped_values.to(frames_kept.dtype)
            )
        times = k2.RaggedTensor(
            ragged_original_frames.shape,
            ragged_original_frames.values * params.frame_duration,
        )
       
        # If we don't have the speaker then we use the ground-truth segments
        # alignments, label the segments, and use the speaker labels on the 
        # corresponding time frames
        if use_segment_factor_alignment:
            # Segment identity is part of the scored joint label, while the
            # output speaker remains the known speaker attached to that source
            # supervision. No speaker posterior is predicted by this model.
            spks = segs.clone()
            for i in range(times.shape.dim0):
                speaker_to_index = {
                    name: index
                    for index, name in enumerate(dict.fromkeys(speakers[i]))
                }
                for segment_index, speaker in enumerate(speakers[i], start=1):
                    spks[i][segs[i] == segment_index] = speaker_to_index[speaker]
        elif params.ignore_speaker:
            spks = segs.clone()
            for i in range(times.shape.dim0):
                num_spks = (
                    len(set(speakers[i]))
                    if params.oracle_num_spks == 0
                    else params.oracle_num_spks
                    if params.oracle_num_spks is not None
                    else params.max_num_spks
                )
                for s_i in range(1, len(speakers[i])+1):
                    # out2 was compacted by select_frames_with_map above, so
                    # index it in filtered-frame coordinates. Original-frame
                    # coordinates are used only for reported timestamps.
                    scores_s_i = out2[
                        i, ragged_frames[i][segs[i] == s_i], :
                    ]
                    hyp_lbl = scores_s_i.sum(dim=0)[0:num_spks].argmax()
                    spks[i][segs[i] == s_i] = hyp_lbl.to(torch.int32) 
        else:
            # Joint nonblank labels use
            #   label = token + speaker * (vocab_size - 1),
            # where token is in [1, vocab_size - 1].  Subtract one only when
            # finding the speaker.  Plain quotient/remainder incorrectly maps
            # the final token ID to blank and to the next speaker.
            joint_speakers, _ = split_joint_labels(
                units.values, params.vocab_size
            )
            spks = k2.RaggedTensor(units.shape, joint_speakers)

        if use_segment_factor_alignment:
            joint_segments, token_values = split_joint_labels(
                units.values, params.vocab_size
            )
            expected_segments = segs.values - 1
            if not torch.equal(joint_segments, expected_segments):
                raise RuntimeError(
                    "Segment-factor graph joint labels disagree with aux-label "
                    "segment identities"
                )
        elif params.ignore_speaker:
            token_values = units.values
        else:
            token_values = units.values - joint_speakers * (params.vocab_size - 1)
        syms = k2.RaggedTensor(units.shape, token_values)
  
        cut_ids = [c.id for c in batch["supervisions"]["cut"]]
        for i in range(times.shape.dim0):
            t, s, u, g = times[i], spks[i], syms[i], segs[i]
            if t.numel() == 0:
                yield cut_ids[i], None, len(speakers[0])
                continue

            spk2int = {k: s_i for s_i, k in enumerate(dict.fromkeys(speakers[i]))}
            int2spk = {s_i: k for k, s_i in spk2int.items()}
            # There could be speakers in spks that do not appear in the reference.
            # We need to include them in the dictionary. For now I will assume that
            # the predicted speaker order matches that in the reference but I may
            # need to revisit this. 
            num_spks_observed = 0
            for s_i in range(spks[i].max().item() + 1):
                if s_i not in int2spk:
                    spk_name = f"extra_spk{s_i - num_spks_observed}"
                    int2spk[s_i] = spk_name
                    spk2int[spk_name] = s_i
                num_spks_observed += 1
                        
            max_dur = (feature_lens[i] / 16000 - t[-1]).item()
            avg_token_durs = {}
            for j in range(1, len(speakers[i])+1):
                segment_times = t[g == j]
                segment_units = u[g == j]
                if segment_times.numel() == 0:
                    continue
                if segment_times.numel() == 1:
                    avg_token_durs[j] = fallback_token_duration(
                        graph_compiler.sp.id_to_piece(int(segment_units[0])),
                        params.single_token_duration_per_character,
                        params.frame_duration,
                        params.single_token_fixed_duration,
                    )
                else:
                    avg_token_durs[j] = round(
                        torch.mean(
                            (torch.roll(segment_times, shifts=-1) - segment_times)[:-1]
                        ).item(), 3
                    )
            for j in avg_token_durs:
                if math.isnan(avg_token_durs[j]):
                    first_piece = graph_compiler.sp.id_to_piece(
                        int(u[g == j][0])
                    )
                    avg_token_durs[j] = fallback_token_duration(
                        first_piece,
                        params.single_token_duration_per_character,
                        params.frame_duration,
                        params.single_token_fixed_duration,
                    )
            
            all_durs = (torch.roll(t, shifts=-1) - t).tolist()
            # Let's do this by segment
            for s_idx in range(1, len(speakers[i])+1):
                t_g = t[g == s_idx]
                # Skip the segment if it is not present
                if t_g.numel() == 0:
                    continue
                elif t_g.numel() > 1:
                    durs = (torch.roll(t_g, shifts=-1) - t_g)[:-1].tolist()
                    durs = [round(val, 3) for val in durs]
                else:
                    durs = []

                if g[-1] == s_idx:
                    durs.append(min(avg_token_durs[s_idx], max_dur))
                else:
                    durs.append(avg_token_durs[s_idx])
                
                for d in durs:
                    assert d > 0
                pieces = [
                    graph_compiler.sp.id_to_piece(p)
                    for p in u[g == s_idx].tolist()
                ]
                # spk, segment, piece, start, dur
                result = list(
                    zip(
                        [int2spk[s] for s in s[g == s_idx].tolist()],
                        [s_idx-1] * len(t_g.tolist()),
                        pieces,
                        [round(t_, 3) for t_ in t_g.tolist()],
                        durs,
                    )
                )
                yield cut_ids[i], result, 1
            #for s_idx in range(params.max_num_spks):
            #    t_s = t[s == s_idx]
            #    # Skip the speaker if they aren't present
            #    if t_s.numel() == 0:
            #        continue
            #    elif t_s.numel() > 1:
            #        durs = (torch.roll(t_s, shifts=-1) - t_s)[:-1].tolist()
            #        durs = [round(val, 3) for val in durs]
            #    else:
            #        durs = []

            #    if s[-1] == s_idx:
            #        durs.append(min(avg_token_durs[s_idx], max_dur))
            #    else:
            #        durs.append(avg_token_durs[s_idx])
            #    
            #    for d in durs:
            #        assert d > 0
            #    pieces = [
            #        graph_compiler.sp.id_to_piece(p)
            #        for p in u[s == s_idx].tolist()
            #    ]
            #    result = list(
            #        zip(
            #            [int2spk[s_idx]]*len(t_s.tolist()),
            #            pieces,
            #            [round(t_, 3) for t_ in t_s.tolist()],
            #            durs,
            #        )
            #    )

            #    yield cut_ids[i], result

@torch.no_grad()
def calibrate_blank_threshold(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    graph_compiler: MDCTCGraphCompiler,
) -> float:
    """Find one threshold giving the requested mean per-cut keep fraction."""
    target = params.blank_frame_filter_target_average_fraction
    if not 0.0 < target < 1.0:
        raise ValueError(
            "--blank-frame-filter-target-average-fraction must be in (0, 1)"
        )
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    probabilities = []
    weights = []
    for batch_idx, batch in tqdm(enumerate(dl), desc="blank-threshold calibration"):
        feature = batch["inputs"].to(device)
        feature_lens = batch["num_frames"].to(device)
        texts = batch["texts"]
        speakers = batch["speakers"]
        if params.alignment_chunk_duration > 0:
            if feature.size(0) != 1:
                raise ValueError(
                    "Chunked blank-threshold calibration requires max-cuts=1"
                )
            token_ids, speaker_counts = alignment_candidate_ids(
                texts, speakers, graph_compiler.sp,
                token_encoder=graph_compiler.encode,
            )
            token_piece, _ = chunked_factor_posteriors(
                model=model,
                waveform=feature[0:1],
                waveform_length=int(feature_lens[0].item()),
                token_ids=token_ids[0],
                speaker_count=speaker_counts[0],
                speaker_weight=params.speaker_weight,
                chunk_duration=params.alignment_chunk_duration,
                chunk_overlap=params.alignment_chunk_overlap,
                chunk_context=params.alignment_chunk_context,
                frame_duration=params.frame_duration,
            )
            batch_blank = [token_piece[:, 0].exp().float().cpu()]
        else:
            token_output, _, output_lens = model(
                feature, feature_lens,
                speaker_weight=params.speaker_weight,
                return_factors=True,
            )
            batch_blank = [
                token_output[i, : int(length.item()), 0].exp().float().cpu()
                for i, length in enumerate(output_lens)
            ]
        for blank_prob in batch_blank:
            if blank_prob.numel() == 0:
                continue
            probabilities.append(blank_prob)
            # Giving every frame weight 1/T_i makes every recording contribute
            # total mass one, so the weighted CDF is the arithmetic mean of
            # per-recording retained fractions rather than a duration-weighted
            # pooled fraction.
            weights.append(torch.full_like(
                blank_prob, 1.0 / blank_prob.numel(), dtype=torch.float64
            ))
        logging.info(
            "Blank-threshold calibration batch=%d recordings=%d frames=%d",
            batch_idx, len(probabilities),
            sum(item.numel() for item in probabilities),
        )
    if not probabilities:
        raise ValueError("Blank-threshold calibration set contained no frames")
    values = torch.cat(probabilities)
    frame_weights = torch.cat(weights)
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    cumulative = frame_weights[order].cumsum(0) / len(probabilities)
    position = min(
        int(torch.searchsorted(cumulative, cumulative.new_tensor(target)).item()),
        sorted_values.numel() - 1,
    )
    threshold = float(sorted_values[position].item())
    fractions = torch.tensor([
        float((item <= threshold).float().mean().item())
        for item in probabilities
    ])
    logging.info(
        "Calibrated global blank threshold=%.9g target_mean_keep=%.3f "
        "realized_mean_keep=%.6f pooled_keep=%.6f recordings=%d frames=%d "
        "per_recording_min=%.6f median=%.6f max=%.6f",
        threshold, target, float(fractions.mean()),
        float(sum((item <= threshold).sum().item() for item in probabilities)
              / sum(item.numel() for item in probabilities)),
        len(probabilities), values.numel(), float(fractions.min()),
        float(fractions.median()), float(fractions.max()),
    )
    return threshold


def align_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    graph_compiler: MDCTCGraphCompiler,
):
    """Align dataset
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    alignments = {}
    failed = 0
    total = 0
    for batch_idx, batch in tqdm(enumerate(dl)):
        texts = batch["supervisions"]["text"]
        try:
            batch_results = list(align_one_batch(
                params=params,
                model=model,
                graph_compiler=graph_compiler,
                batch=batch,
            ))
        except RuntimeError as error:
            if not params.skip_empty_alignments or "empty/non-finite" not in str(error):
                raise
            cut_ids = [c.id for c in batch["supervisions"]["cut"]]
            logging.error("Skipping empty alignment for %s: %s", cut_ids, error)
            failed_segments = sum(len(items) for items in batch["speakers"])
            failed += failed_segments
            total += failed_segments
            num_cuts += len(cut_ids)
            continue
        for c_id, ali, num in batch_results:
            total += num
            if ali is None:
                logging.info(f"Alignment {c_id} failed. Skipping ... ")
                failed += num
                continue
            # There are multiple alignments per speaker
            if c_id not in alignments:
                alignments[c_id] = ali
            # The speaker was already seen
            else:
                alignments[c_id] += ali
        num_cuts += len(batch["supervisions"]["cut"])
        batch_str = f"{batch_idx}/{num_batches}"
        logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    for k in alignments:
        alignments[k] = sorted(alignments[k], key=lambda x: x[3])
    logging.info(f"Done. Success for {(total-failed)} / {total}.") 
    alignments = sorted(alignments.items(), key=lambda x: x[0])
    return alignments

 
@torch.no_grad()
def score_speaker_posteriors(dl, params, model):
    """Evaluate argmax and multilabel posterior-only DER systems."""
    blank_thresholds = [
        float(x) for x in params.diarization_blank_thresholds.split(",")
    ]
    speaker_thresholds = [
        float(x) for x in params.diarization_speaker_thresholds.split(",")
    ]
    if not blank_thresholds or not speaker_thresholds:
        raise ValueError("Need at least one blank and one speaker threshold")
    if params.diarization_smoothing_values is None:
        smoothing_values = [params.diarization_smoothing]
    else:
        smoothing_values = [
            float(x) for x in params.diarization_smoothing_values.split(",")
        ]
        if not smoothing_values or any(x < 0 for x in smoothing_values):
            raise ValueError("Diarization smoothing values must be non-negative")
    systems = [
        ("argmax", b, None, smoothing)
        for smoothing in smoothing_values for b in blank_thresholds
    ]
    systems += [
        ("multilabel", b, s, smoothing)
        for smoothing in smoothing_values
        for b in blank_thresholds
        for s in speaker_thresholds
    ]
    totals = {
        system: dict(miss=0.0, false_alarm=0.0, confusion=0.0, reference=0.0)
        for system in systems
    }
    device = next(model.parameters()).device
    for batch in dl:
        features = batch["inputs"].to(device)
        feature_lens = batch["num_frames"].to(device)
        if features.size(0) != 1:
            raise ValueError("Posterior diarization requires --max-cuts 1")
        texts = batch["texts"][0]
        speakers = batch["speakers"][0]
        ordered_speakers = list(dict.fromkeys(speakers))
        seq_idx = batch["supervisions"]["sequence_idx"]
        sort_order = batch["sort_orders"][0]
        starts = [
            int(batch["supervisions"]["start_sample"][seq_idx == 0][j])
            for j in sort_order
        ]
        lengths = [
            int(batch["supervisions"]["num_samples"][seq_idx == 0][j])
            for j in sort_order
        ]

        # Construct the global reference before inference.  The oracle-chunk
        # scorer needs it while resolving each chunk-local output permutation.
        frame_count = round(
            (int(feature_lens[0].item()) / 16000) / params.frame_duration
        )
        times = (
            torch.arange(frame_count, dtype=torch.float64) + 0.5
        ) * params.frame_duration
        reference = torch.zeros(
            frame_count, len(ordered_speakers), dtype=torch.bool
        )
        valid = torch.ones(frame_count, dtype=torch.bool)
        speaker_to_index = {
            name: i for i, name in enumerate(ordered_speakers)
        }
        for name, start, length in zip(speakers, starts, lengths):
            begin = start / 16000
            end = (start + length) / 16000
            reference[:, speaker_to_index[name]] |= (
                (times >= begin) & (times < end)
            )
            valid &= ~(
                ((times >= begin - params.diarization_collar) &
                 (times < begin + params.diarization_collar)) |
                ((times >= end - params.diarization_collar) &
                 (times < end + params.diarization_collar))
            )

        def column_map(input_start, input_end):
            mapping, _, _ = _chunk_speaker_map(
                ordered_speakers, speakers, starts, lengths,
                input_start, input_end,
                strategy=("duration" if params.sort_strategy == "speech_time" else "fifo"),
            )
            return mapping + [
                i for i in range(params.max_num_spks) if i not in set(mapping)
            ]

        oracle_chunks = []

        def remember_oracle_chunk(global_start_frame, local_scores):
            oracle_chunks.append(
                (global_start_frame, global_start_frame + local_scores.size(0))
            )
            # Preserve every raw local slot.  A separate optimal permutation
            # is fitted for each thresholded hypothesis below.
            return list(range(params.max_num_spks))

        if params.diarization_speaker_mapping == "oracle_chunk":
            if params.alignment_chunk_context <= 0:
                raise ValueError(
                    "oracle_chunk posterior scoring currently requires "
                    "center-cropped chunks (--alignment-chunk-context > 0)"
                )
            speaker_count = params.max_num_spks
            time_map = None
            score_map = remember_oracle_chunk
        else:
            speaker_count = len(ordered_speakers)
            time_map = column_map
            score_map = None

        token, speaker = chunked_factor_posteriors(
            model=model,
            waveform=features[0:1],
            waveform_length=int(feature_lens[0]),
            token_ids=[0],
            speaker_count=speaker_count,
            speaker_weight=1.0,
            chunk_duration=params.alignment_chunk_duration,
            chunk_overlap=params.alignment_chunk_overlap,
            chunk_context=params.alignment_chunk_context,
            frame_duration=params.frame_duration,
            speaker_column_map_fn=time_map,
            speaker_column_map_from_scores_fn=score_map,
        )
        blank_probability = token[:, 0].exp().cpu()
        speaker_probability = speaker.exp().cpu()
        frame_count = token.size(0)
        reference = reference[:frame_count]
        valid = valid[:frame_count]
        reference_padded = torch.nn.functional.pad(
            reference,
            (0, params.max_num_spks - reference.size(1)),
        )
        scoring_reference = (
            reference_padded
            if params.diarization_speaker_mapping == "oracle_chunk"
            else reference
        )
        n_ref = reference.sum(-1)
        for system in systems:
            mode, blank_threshold, speaker_threshold, smoothing = system
            smoothing_frames = round(smoothing / params.frame_duration)
            speech = blank_probability < blank_threshold
            if mode == "argmax":
                hypothesis = torch.zeros_like(speaker_probability, dtype=torch.bool)
                winner = speaker_probability.argmax(-1)
                hypothesis[torch.arange(frame_count), winner] = speech
            else:
                hypothesis = (
                    speaker_probability > speaker_threshold
                ) & speech[:, None]

            # A chunk-local FIFO slot has no identity outside its acoustic
            # input chunk.  For an oracle mapping, smooth and permute each
            # retained center independently, then concatenate in global time.
            smoothing_regions = (
                oracle_chunks
                if params.diarization_speaker_mapping == "oracle_chunk"
                else [(0, frame_count)]
            )
            mapped_hypothesis = torch.zeros_like(hypothesis)
            for region_start, region_end in smoothing_regions:
                region_end = min(region_end, frame_count)
                if region_end <= region_start:
                    continue
                region = hypothesis[region_start:region_end].clone()
                region_speech = speech[region_start:region_end]
                region_frames = region_end - region_start
                if mode == "argmax":
                    if smoothing_frames > 0 and region_speech.any():
                        indexes = torch.arange(region_frames)
                        missing_left = torch.full_like(
                            indexes, -region_frames * 2
                        )
                        previous = torch.where(
                            region_speech, indexes, missing_left
                        ).cummax(0).values
                        missing_right = torch.full_like(
                            indexes, region_frames * 2
                        )
                        following = torch.flip(
                            torch.flip(
                                torch.where(
                                    region_speech, indexes, missing_right
                                ),
                                [0],
                            ).cummin(0).values,
                            [0],
                        )
                        use_previous = (
                            indexes - previous <= following - indexes
                        )
                        nearest = torch.where(
                            use_previous, previous, following
                        )
                        distance = torch.minimum(
                            indexes - previous, following - indexes
                        )
                        fill = (~region_speech) & (
                            distance <= smoothing_frames
                        )
                        region[fill] = region[nearest[fill]]
                elif smoothing_frames > 0:
                    # Independent dilation preserves simultaneous speakers.
                    region = torch.nn.functional.max_pool1d(
                        region.T[None].float(),
                        kernel_size=2 * smoothing_frames + 1,
                        stride=1,
                        padding=smoothing_frames,
                    )[0].T.bool()

                if params.diarization_speaker_mapping == "oracle_chunk":
                    from scipy.optimize import linear_sum_assignment

                    region_reference = reference_padded[
                        region_start:region_end
                    ]
                    region_valid = valid[region_start:region_end]
                    overlap = (
                        region[region_valid].float().T
                        @ region_reference[region_valid].float()
                    )
                    local_indexes, global_indexes = linear_sum_assignment(
                        overlap.numpy(), maximize=True
                    )
                    global_to_local = [None] * params.max_num_spks
                    for local_index, global_index in zip(
                        local_indexes.tolist(), global_indexes.tolist()
                    ):
                        global_to_local[global_index] = local_index
                    if any(x is None for x in global_to_local):
                        raise RuntimeError(
                            "Oracle chunk assignment was not a permutation"
                        )
                    region = region[:, global_to_local]
                mapped_hypothesis[region_start:region_end] = region
            hypothesis = mapped_hypothesis
            n_hyp = hypothesis.sum(-1)
            correct = (scoring_reference & hypothesis).sum(-1)
            keep = valid
            scale = params.frame_duration
            value = totals[system]
            value["miss"] += float((n_ref[keep] - n_hyp[keep]).clamp_min(0).sum()) * scale
            value["false_alarm"] += float((n_hyp[keep] - n_ref[keep]).clamp_min(0).sum()) * scale
            value["confusion"] += float((torch.minimum(n_ref[keep], n_hyp[keep]) - correct[keep]).sum()) * scale
            value["reference"] += float(n_ref[keep].sum()) * scale

    results = []
    for mode, blank_threshold, speaker_threshold, smoothing in systems:
        value = totals[(mode, blank_threshold, speaker_threshold, smoothing)]
        denominator = value["reference"]
        result = {
            "mode": mode,
            "speaker_mapping": params.diarization_speaker_mapping,
            "blank_threshold": blank_threshold,
            "speaker_threshold": speaker_threshold,
            "smoothing": smoothing,
            "miss": 100 * value["miss"] / denominator,
            "false_alarm": 100 * value["false_alarm"] / denominator,
            "confusion": 100 * value["confusion"] / denominator,
        }
        result["der"] = result["miss"] + result["false_alarm"] + result["confusion"]
        results.append(result)
        logging.info(
            "POSTERIOR_DER mode=%s blank=%.6g speaker=%s smoothing=%.6g "
            "DER=%.4f%% "
            "miss=%.4f%% false_alarm=%.4f%% confusion=%.4f%%",
            mode, blank_threshold,
            "argmax" if speaker_threshold is None else f"{speaker_threshold:.6g}",
            smoothing,
            result["der"], result["miss"], result["false_alarm"], result["confusion"],
        )
    return results


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)

    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log-argmax-decode/log-decode")
    logging.info("Decoding started")
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    graph_compiler = MDCTCGraphCompiler(
        params.lang_dir,
        device='cpu',
    )
    graph_compiler.set_bpe_max_piece_characters(
        params.alignment_bpe_max_piece_characters
    )
    if graph_compiler.bpe_decomposer is not None:
        logging.info(
            "Using deterministic BPE decomposition with at most %d visible "
            "characters per piece",
            params.alignment_bpe_max_piece_characters,
        )
    
    params.vocab_size = graph_compiler.sp.vocab_size()
    
    logging.info("About to create model")

    model = get_mdctc_model(params)
    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to(device)
    if isinstance(model, FrozenTokenTeacherSpeakerModel):
        # The immutable teacher is deliberately not a registered submodule,
        # so nn.Module.to() cannot move it for us.
        model.move_token_teacher(device)
    model.eval()
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    librispeech = LibriSpeechAsrDataModule(args)

    custom_test_sets = {}
    for item in params.test_cut_manifest:
        if "=" not in item:
            raise ValueError(
                f"Expected NAME=PATH for --test-cut-manifest, got {item!r}"
            )
        name, path = item.split("=", 1)
        if not name or not path or name in custom_test_sets:
            raise ValueError(f"Invalid or duplicate custom test set: {item!r}")
        custom_test_sets[name] = path

    test_sets = params.test_sets.split() if params.test_sets else ["synth2"]
    test_dl = []
    for test_set in test_sets:
        if test_set in custom_test_sets:
            cuts = CutSet.from_file(custom_test_sets[test_set])
        elif test_set == "synth2":
            cuts = librispeech.synth2_cuts(2)
        elif test_set == "synth3":
            cuts = librispeech.synth2_cuts(3)
        elif test_set == "synth":
            cuts = librispeech.synth_cuts()
        else:
            raise ValueError(
                f"Lazy token alignment currently supports synth/synth2/synth3; "
                f"requested {test_set!r}."
            )
        if params.test_cut_id is not None:
            cuts = cuts.filter(
                lambda cut: cut.id == params.test_cut_id
            ).to_eager()
            if len(cuts) != 1:
                raise ValueError(
                    f"Expected exactly one cut with ID {params.test_cut_id!r} "
                    f"in {test_set!r}, found {len(cuts)}"
                )
            logging.info(f"Restricting alignment to cut {params.test_cut_id}")
        test_dl.append(librispeech.valid_dataloaders(cuts))

    if params.blank_frame_filter_target_average_fraction is not None:
        if params.blank_frame_filter_keep_fraction is not None:
            raise ValueError(
                "Dataset-level threshold calibration cannot be combined with "
                "--blank-frame-filter-keep-fraction"
            )
        calibration_set = params.blank_frame_filter_calibration_set
        if calibration_set is None:
            if len(test_sets) != 1:
                raise ValueError(
                    "Specify --blank-frame-filter-calibration-set when "
                    "calibrating with multiple test sets"
                )
            calibration_set = test_sets[0]
        if calibration_set not in test_sets:
            raise ValueError(
                f"Calibration set {calibration_set!r} is not in {test_sets}"
            )
        calibration_index = test_sets.index(calibration_set)
        params.blank_frame_filter_threshold = calibrate_blank_threshold(
            dl=test_dl[calibration_index],
            params=params,
            model=model,
            graph_compiler=graph_compiler,
        )
        logging.info(
            "Applying calibrated blank threshold %.9g to sets %s",
            params.blank_frame_filter_threshold, test_sets,
        )
        if params.blank_frame_filter_threshold_output is not None:
            threshold_path = Path(params.blank_frame_filter_threshold_output)
            threshold_path.parent.mkdir(parents=True, exist_ok=True)
            with open(threshold_path, "w") as threshold_file:
                json.dump(
                    {
                        "blank_probability_threshold": params.blank_frame_filter_threshold,
                        "target_mean_per_recording_keep_fraction": (
                            params.blank_frame_filter_target_average_fraction
                        ),
                        "calibration_set": calibration_set,
                        "applied_to_sets": test_sets,
                    },
                    threshold_file,
                    indent=2,
                )
            logging.info("Stored calibrated blank threshold at %s", threshold_path)

    for test_set, dl in zip(test_sets, test_dl):
        if params.speaker_posterior_diarization:
            results = score_speaker_posteriors(dl, params, model)
            output_dir = params.exp_dir / "align"
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / (
                f"speaker_posterior_der_chkpt{params.iter}_avg{params.avg}_"
                f"{test_set}_{params.suffix}.json"
            )
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            logging.info("Stored posterior diarization sweep at %s", output)
            continue
        alignments = align_dataset(
            dl=dl,
            params=params,
            model=model,
            graph_compiler=graph_compiler,
        )
    
        ali_dir = params.exp_dir / "align"
        ali_dir.mkdir(parents=True, exist_ok=True)
        fname = f"alignments_chkpt{params.iter}_avg{params.avg}_{test_set}_collar{params.collar}_{params.suffix}.json"
        with open(ali_dir / fname, "w") as f:
            json.dump(alignments, f, indent=4)			

    if graph_compiler.bpe_decomposer is not None:
        decomposer = graph_compiler.bpe_decomposer
        logging.info(
            "BPE_DECOMPOSITION canonical_pieces=%d split_pieces=%d "
            "output_pieces=%d unsplittable_pieces=%d max_characters=%d",
            decomposer.num_canonical_pieces,
            decomposer.num_split_pieces,
            decomposer.num_output_pieces,
            decomposer.num_unsplittable_pieces,
            decomposer.max_visible_characters,
        )
    logging.info("Done!")

if __name__ == "__main__":
    main()
