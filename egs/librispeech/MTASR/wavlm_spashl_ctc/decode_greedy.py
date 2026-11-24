#!/usr/bin/env python3
# Copyright 2021 Xiaomi Corporation (Author: Liyong Guo, Fangjun Kuang)
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


import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from mdctc_graph_compiler import MDCTCGraphCompiler
from lhotse import load_manifest_lazy
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from asr_datamodule import LibriSpeechAsrDataModule
from train import (
    get_mdctc_model,
    get_params,
    compute_avg_speaker_density_per_example,
)
from model import split_joint_labels

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import (
    AttributeDict,
    load_averaged_model,
    setup_logger,
    str2bool,
)

import time
from tqdm import tqdm
import json
from text_normalization import normalize_english_transcript


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
        default="zipformer_ctc/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_bpe_500",
        help="The lang dir",
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
        "--max-num-spks",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--factorized-token-speaker-output", "--use-hat",
        dest="factorized_token_speaker_output",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--diarization-loss",
        choices=("none", "bce", "powerset"),
        default="none",
        help=(
            "Architecture selector retained for compatibility with the shared "
            "training model constructor. ASR/SpAShl checkpoints use 'none'."
        ),
    )

    parser.add_argument(
        "--ignore-speaker",
        type=str2bool,
        default=False,
        help=(
            "Decode only the token factor and assign the hypothesis to slot 0. "
            "Use this for token-only/segment-factor checkpoints whose speaker "
            "head was not trained. Requires factorized token/speaker output."
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
        "--frame-duration",
        type=float,
        default=None,
        help=(
            "Seconds per decoder output frame. By default this is inferred "
            "as 20 ms for WavLM output, or 40 ms when --downsample is enabled."
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
        "--downmix-multichannel",
        type=str2bool,
        default=False,
        help="Average multichannel waveform cuts to mono before collation.",
    )
    parser.add_argument(
        "--input-sampling-rate",
        type=int,
        default=0,
        help="Resample waveform cuts to this rate before collation; 0 preserves input rate.",
    )
    parser.add_argument(
        "--word-level-stm",
        type=str2bool,
        default=False,
        help=(
            "Write one timed STM segment per decoded/reference word. This is "
            "recommended for tcpWER on long recordings because a collar cannot "
            "meaningfully constrain a recording-long speaker segment. Reference "
            "word times are read from the CutSet when available."
        ),
    )

    parser.add_argument(
        "--sort-strategy",
        type=str,
        default="start_time"
    )

    parser.add_argument(
        "--speaker-weight",
        type=float,
        default=1.0,
    )

    return parser


def get_nonzero_span_starts(batch, params):
    # batch shape: [B, T]
    B, T = batch.shape

    # Shift the input right by 1 along time dimension, prepend zeros
    shifted = torch.cat(
        [
            torch.zeros(B, 1, dtype=batch.dtype).to(batch.device),
            batch[:, :-1]
        ], dim=1
    )

    # Find where a new span starts: nonzero & (value != previous or previous was zero)
    is_new_span = (batch != 0) & ((shifted != batch) | (shifted == 0))

    # For each batch, get the indices where this is true
    result = [
        torch.nonzero(seq, as_tuple=False).squeeze(1) * params.frame_duration
        for seq in is_new_span
    ]

    return result


def mask_padded_predictions(preds, lengths):
    """Replace predictions beyond each encoder sequence length with blank."""
    if preds.ndim != 2 or lengths.ndim != 1 or preds.size(0) != lengths.numel():
        raise ValueError(
            f"Expected preds [B,T] and lengths [B], got {preds.shape}, "
            f"{lengths.shape}"
        )
    valid = torch.arange(preds.size(1), device=preds.device).unsqueeze(0)
    valid = valid < lengths.to(device=preds.device).unsqueeze(1)
    return preds.masked_fill(~valid, 0)


def factorized_greedy_argmax(token_scores, speaker_scores, blank_weight=0.0):
    """Argmax the implicit blank-plus-(speaker x nonblank-token) output.

    This is exactly equivalent to materializing the joint tensor used by the
    factorized SPaSHL model, but its temporary storage is O(V + S) per frame
    rather than O(V * S).  Joint labels follow the model convention
    ``token + speaker * (vocab_size - 1)``.
    """
    if token_scores.ndim != 3 or speaker_scores.ndim != 3:
        raise ValueError(
            "Expected token and speaker scores with shape [batch, time, classes]"
        )
    if token_scores.shape[:2] != speaker_scores.shape[:2]:
        raise ValueError(
            f"Token/speaker frame shapes differ: {token_scores.shape} vs "
            f"{speaker_scores.shape}"
        )
    blank = token_scores[..., 0] - blank_weight
    token_score, token_index = token_scores[..., 1:].max(dim=-1)
    speaker_score, speaker_index = speaker_scores.max(dim=-1)
    best_joint = token_score + speaker_score
    joint_index = token_index + 1 + speaker_index * (token_scores.size(-1) - 1)
    return torch.where(blank >= best_joint, torch.zeros_like(joint_index), joint_index)


def merge_bpe_with_times(sp, tokens, times, max_time):
    assert len(tokens) == len(times), "Tokens and times must match in length"
    words = []
    current_word = ''
    current_time = None

    for token_, time in zip(tokens, times):
        token = sp.id_to_piece(token_.item())
        if token.startswith('▁') or not current_word:  # new word
            if current_word:
                words.append(
                    (
                        current_word,
                        round(min(current_time, max_time.item()), 2),
                        round(time - current_time, 2)
                    )
                )
            current_word = token.lstrip('▁')
            current_time = time
        else:
            current_word += token
    if current_word:
        words.append(
            (
                current_word,
                round(min(current_time, max_time.item()), 2),
                round(max(0, time - current_time), 2)
            )
        )

    return words


def hyp_to_stm(hyp, cut_idx, word_level=False, min_word_duration=0.02):
    utts = []
    for s in hyp:
        if len(hyp[s]) == 0:
            continue
        if word_level:
            for word, start, duration in hyp[s]:
                utts.append(
                    {
                        "session_id": cut_idx,
                        "words": word,
                        "speaker": str(s),
                        "start_time": round(start, 2),
                        "end_time": round(start + max(duration, min_word_duration), 2),
                    }
                )
            continue
        entry = {
            "session_id": cut_idx,
            "words": " ".join([w[0] for w in hyp[s]]),
            "speaker": str(s),
            "start_time": round(hyp[s][0][1], 2),
            "end_time": round(hyp[s][-1][1] + hyp[s][-1][2], 2),
        }
        utts.append(entry)
    return utts


def cut_ref_to_word_stm(cut, cut_idx):
    """Create a time-accurate tcpWER reference from Lhotse word alignments."""
    utts = []
    for supervision in cut.supervisions:
        alignment = (supervision.alignment or {}).get("word", [])
        if alignment:
            for item in alignment:
                word = normalize_english_transcript(item.symbol)
                if not word:
                    continue
                utts.append(
                    {
                        "session_id": cut_idx,
                        "words": word,
                        "speaker": str(supervision.speaker),
                        "start_time": round(item.start, 2),
                        "end_time": round(item.start + item.duration, 2),
                    }
                )
        else:
            # Preserve compatibility with manifests that only have supervision
            # boundaries, while keeping segments local in time.
            words = normalize_english_transcript(supervision.text)
            if words:
                utts.append(
                    {
                        "session_id": cut_idx,
                        "words": words,
                        "speaker": str(supervision.speaker),
                        "start_time": round(supervision.start, 2),
                        "end_time": round(supervision.end, 2),
                    }
                )
    return utts


def ref_to_stm(ref, cut_idx):
    utts = []
    for s in ref:
        entry = {
            "session_id": cut_idx,
            "words": ref[s][0],
            "speaker": str(s),
            "start_time": round(ref[s][1], 2),
            "end_time": round(ref[s][1] + ref[s][2], 2),
        }
        utts.append(entry)
    return utts


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    graph_compiler: MDCTCGraphCompiler,
    batch: dict,
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if no rescoring is used, the key is the string `no_rescore`.
               If LM rescoring is used, the key is the string `lm_scale_xxx`,
               where `xxx` is the value of `lm_scale`. An example key is
               `lm_scale_0.7`
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.

        - params.method is "1best", it uses 1best decoding without LM rescoring.
        - params.method is "nbest", it uses nbest decoding without LM rescoring.
        - params.method is "nbest-rescoring", it uses nbest LM rescoring.
        - params.method is "whole-lattice-rescoring", it uses whole lattice LM
          rescoring.

      model:
        The neural model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict. Note: If it decodes to nothing, then return None.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"]
    feature_lens = batch["num_frames"].to(device)
    feature = feature.to(device)
    # at entry, feature is (N, T, C)
    # Some real evaluation recordings (including SBCSAE) are stereo and are
    # collated as (N, C, T). The training mixtures are mono, so average real
    # channels before passing waveforms to a waveform encoder.
    if feature.ndim == 3 and feature.size(-1) != 1:
        feature = feature.mean(dim=1)

    supervisions = batch["supervisions"]
    feature = feature.to(device)
    texts = [
        [normalize_english_transcript(t) for t in b]
        for b in batch["texts"]
    ]
    speakers = batch["speakers"]
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
    
    with torch.set_grad_enabled(False):
        start = time.time()
        if params.factorized_token_speaker_output:
            token_output, speaker_output, x_lens = model(
                feature,
                feature_lens,
                speaker_weight=params.speaker_weight,
                return_factors=True,
            )
            ctc_output = None
        else:
            ctc_output, x_lens = model(
                feature,
                feature_lens,
                speaker_weight=params.speaker_weight,
            )
        end_nnet = time.time()
        subsampling_factor = 4 if params.downsample else 2
        use_double_scores = params.use_double_scores


    # Decoding with WFST supervisions
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

    # Works with a BPE model
    densities = compute_avg_speaker_density_per_example(start_frames, num_frames_init)

    if params.factorized_token_speaker_output:
        if params.ignore_speaker:
            token_output = token_output.clone()
            token_output[..., 0] -= params.blank_weight
            preds = token_output.argmax(-1)
        else:
            preds = factorized_greedy_argmax(
                token_output, speaker_output, blank_weight=params.blank_weight
            )
    else:
        ctc_output[..., 0] -= params.blank_weight
        preds = ctc_output.argmax(-1)
    # Fixed-context encoders such as Whisper return states for a fully padded
    # 30-second window.  The graph loss observes x_lens, but greedy decoding
    # must mask the padded states explicitly before timestamps and CTC collapse.
    preds = mask_padded_predictions(preds, x_lens)
    times = get_nonzero_span_starts(preds, params)
    hyps = [
        preds[i].unique_consecutive(dim=-1)[
            preds[i].unique_consecutive(dim=-1) != 0
        ].reshape(-1)
        for i in range(preds.size(0))
    ]
    if params.ignore_speaker:
        spks = [torch.zeros_like(hyp) for hyp in hyps]
        units = hyps
    else:
        # Joint labels are token + speaker * (V - 1), token in [1, V - 1].
        # The previous quotient/remainder split mapped token V-1 to blank in
        # the next speaker block.
        split = [split_joint_labels(hyp, params.vocab_size) for hyp in hyps]
        spks = [item[0] for item in split]
        units = [item[1] for item in split]
    
    cut_ids = [c.id for c in batch["supervisions"]["cut"]]
    tokens = {}
    for i, (cut_id, s, u, t) in enumerate(zip(cut_ids, spks, units, times)):
        spk2int = {k: j for j, k in enumerate(dict.fromkeys(speakers[i]))}
        int2spk = {j: k for k, j in spk2int.items()}
        speaker_hyps = {}
        speaker_refs = {}
        for s_i in range(params.max_num_spks): 
            if s_i in s:
                speaker_hyps[s_i] = merge_bpe_with_times(
                    graph_compiler.sp,
                    u[s == s_i],
                    [t_.item() for t_ in t[s == s_i]],
                    (x_lens[i]+1)*params.frame_duration
                )
            if s_i in int2spk: 
                text_i = []
                found_start = False
                curr_dur = 0
                for j, (text, spk) in enumerate(zip(texts[i], speakers[i])):
                    if spk2int[spk] == s_i:
                        if not found_start:
                            start_i = start_frames[i][j]
                            found_start = True
                        curr_dur = start_frames[i][j] + num_frames_init[i][j]
                        text_i.append(text)
                text_i = " ".join(text_i)
                speaker_refs[s_i] = (
                    graph_compiler.sp.decode(text_i),
                    start_i * (1/16000),
                    (curr_dur - start_i) * (1/16000),
                )
        stm_hyp = hyp_to_stm(
            speaker_hyps,
            cut_id,
            word_level=params.word_level_stm,
            min_word_duration=params.frame_duration,
        )
        if params.word_level_stm:
            stm_ref = cut_ref_to_word_stm(
                batch["supervisions"]["cut"][i], cut_id
            )
        else:
            stm_ref = ref_to_stm(speaker_refs, cut_id)
        yield stm_hyp, stm_ref


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    graph_compiler: MDCTCGraphCompiler,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      graph_compiler:
        The graph compiler to make the shuffle automata
    Returns:
      Return a dict, whose key may be "no-rescore" if no LM rescoring
      is used, or it may be "lm_scale_0.7" if LM rescoring is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    results = defaultdict(list)
    hyps, refs = [], []
    for batch_idx, batch in tqdm(enumerate(dl)):
        texts = batch["supervisions"]["text"]
        for hyps_, refs_ in decode_one_batch(
                params=params,
                model=model,
                graph_compiler=graph_compiler,
                batch=batch,
            ):
            hyps.extend(hyps_)
            refs.extend(refs_)
        batch_str = f"{batch_idx}/{num_batches}"
        logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    hyps = sorted(hyps, key=lambda x: (x["session_id"], x["start_time"]))
    refs = sorted(refs, key=lambda x: (x["session_id"], x["start_time"]))
    return hyps, refs



@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    if args.frame_duration is None:
        args.frame_duration = 0.04 if args.downsample else 0.02
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)

    params = get_params()
    params.update(vars(args))

    if params.ignore_speaker and not params.factorized_token_speaker_output:
        raise ValueError(
            "--ignore-speaker requires --factorized-token-speaker-output=true"
        )

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
    
    params.vocab_size = graph_compiler.sp.vocab_size()

    logging.info("About to create model")
    params.pretrained_dir = None
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
    model.eval()
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    librispeech = LibriSpeechAsrDataModule(args)

    custom_test_sets = {}
    for item in params.test_cut_manifest:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH for --test-cut-manifest, got {item!r}")
        name, path = item.split("=", 1)
        if not name or not path or name in custom_test_sets:
            raise ValueError(f"Invalid or duplicate custom test set: {item!r}")
        custom_test_sets[name] = path

    all_test_sets = [
        "synth", "libri3mix", "l2m_test_clean", "l2m_test_both",
        "lsm_2", "lsm_3", "ls_test_other", "ls_test_clean", "lcss", "ami_dev",
    ] + list(custom_test_sets)
    test_sets = params.test_sets.split() if params.test_sets is not None else all_test_sets
    unknown = set(test_sets) - set(all_test_sets)
    if unknown:
        raise ValueError(f"Unknown test sets: {sorted(unknown)}")

    # Load only requested sets. The portable recipe intentionally does not
    # require LibriMix/AMI/LibriCSS manifests when decoding Synth alone.
    single_speaker = None
    test_dl = []
    for test_set in test_sets:
        if test_set in custom_test_sets:
            cuts = load_manifest_lazy(custom_test_sets[test_set])
        elif test_set == "synth":
            cuts = librispeech.synth_cuts()
        elif test_set == "libri3mix":
            cuts = librispeech.libri3mix_test_clean_cuts()
        elif test_set == "l2m_test_clean":
            cuts = librispeech.libri2mix_test_clean_cuts()
        elif test_set == "l2m_test_both":
            cuts = librispeech.libri2mix_test_both_cuts()
        elif test_set == "lsm_2":
            cuts = librispeech.librispeechmix_2_cuts()
        elif test_set == "lsm_3":
            cuts = librispeech.librispeechmix_3_cuts()
        elif test_set in ("ls_test_other", "ls_test_clean"):
            if single_speaker is None:
                single_speaker = librispeech.single_speaker_cuts()
            cuts = single_speaker[0 if test_set == "ls_test_other" else 1]
        elif test_set == "lcss":
            cuts = librispeech.libricss_cuts()
        else:
            cuts = librispeech.ami_dev()
        test_dl.append(librispeech.valid_dataloaders(cuts))

    for test_set, dl in zip(test_sets, test_dl):
        hyps, refs = decode_dataset(
            dl=dl,
            params=params,
            model=model,
            graph_compiler=graph_compiler,
        )
 
        hyps_sessions = dict.fromkeys([h['session_id'] for h in hyps])
        # For any missing hyps add in an empty line
        for r in refs:
            if r['session_id'] not in hyps_sessions:
                # Do not blank the reference entry in-place when synthesizing an
                # empty hypothesis for a recording with no decoded output.
                h_ = dict(r)
                h_['words'] = ""
                hyps.append(h_)
        hyps = sorted(hyps, key=lambda x: (x["session_id"], x["start_time"]))
        
        # Dump refs and hyps to file
        decode_dir = params.exp_dir / "decode"
        decode_dir.mkdir(parents=True, exist_ok=True)
        with open(decode_dir / f"hyps_chkpt{params.iter}_avg{params.avg}_{test_set}_{params.suffix}.stm", "w") as f:
            for l in hyps:
                print(f"{l['session_id']} 1 {l['speaker']} {l['start_time']} {l['end_time']} {l['words']}", file=f)
        
        with open(decode_dir / f"refs_chkpt{params.iter}_avg{params.avg}_{test_set}_{params.suffix}.stm", "w") as f:
            for l in refs:
                print(f"{l['session_id']} 1 {l['speaker']} {l['start_time']} {l['end_time']} {l['words']}", file=f)

    logging.info("Done!")


if __name__ == "__main__":
    main()
