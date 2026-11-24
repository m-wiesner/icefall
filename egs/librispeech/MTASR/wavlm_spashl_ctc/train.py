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
import itertools
import logging
import math
from contextlib import nullcontext
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import os
import time
import torch
import k2
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from asr_datamodule import LibriSpeechAsrDataModule
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from lhotse import CutSet
from model import (
    FrozenTokenTeacherSpeakerModel,
    MDCTCModel,
    SpeakerDiarizationModel,
    split_joint_labels,
)
from torch.optim import Adam
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from shuffle_ctc import factorized_graph_ctc_loss, graph_ctc_loss

from icefall import diagnostics
from mdctc_graph_compiler import MDCTCGraphCompiler
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    find_checkpoints,
    average_checkpoints,
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
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


LRSchedulerType = Any


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
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )
    parser.add_argument(
        "--profile-cuda-memory",
        type=str2bool,
        default=False,
        help="Log allocated CUDA memory and the per-phase temporary peak.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Stop each epoch after this many training batches; 0 means unlimited.",
    )
    parser.add_argument(
        "--accum-grad",
        type=int,
        default=1,
        help=(
            "Accumulate this many data-loader minibatches before each optimizer "
            "update. Losses are summed, matching one larger minibatch."
        ),
    )
    parser.add_argument(
        "--max-valid-batches",
        type=int,
        default=0,
        help="Stop validation after this many batches; 0 means unlimited.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
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
        "--base-lr", type=float, default=0.0001,
        help="Peak learning rate for the unfrozen OneCycle schedule."
    )

    parser.add_argument(
        "--optimizer-name",
        choices=("adam", "scaled_adam"),
        default="adam",
        help="Optimizer. ScaledAdam is the standard choice for Zipformer.",
    )
    parser.add_argument(
        "--lr-scheduler-name",
        choices=("one_cycle", "eden2"),
        default="one_cycle",
        help="Learning-rate scheduler; Eden2 is update-based and has no epoch term.",
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=5000,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=3.5,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )
    parser.add_argument(
        "--lr-warmup-updates",
        type=float,
        default=500.0,
        help="Number of optimizer updates over which Eden2 warms up.",
    )
    parser.add_argument(
        "--lr-warmup-start",
        type=float,
        default=0.5,
        help="Initial fraction of --base-lr during Eden2 warmup.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=2000,
        help="""Save checkpoint after this number of optimizer updates"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 0.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model after this number of optimizer
        updates. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--collar",
        type=int,
        default=64000,
        help=(
            "Pairwise midpoint collar in input waveform samples used by the "
            "graph compiler (64000 is 4 seconds at 16 kHz)."
        )
    )

    parser.add_argument(
        "--edge-collar",
        type=int,
        default=None,
        help=(
            "Pairwise collar in input waveform samples at utterance boundaries. "
            "Defaults to --collar."
        )
    )

    parser.add_argument(
        "--interval-model",
        choices=("fixed", "brownian_bridge"),
        default="fixed",
        help="Token-onset uncertainty model used to construct graph constraints."
    )

    parser.add_argument(
        "--freeze-lr", type=float, default=0.0001,
        help="The learning rate used to train new unfrozen parameters that are "
        "introduced on top of the base wav2vec2 model. We normally first train "
        "these parameters for some number of iterations/"
    )
    
    parser.add_argument("--freeze-iters", type=int, default=500,
        help="Number of optimizer updates for which the encoder is frozen."
    )

    parser.add_argument(
        "--valid-interval",
        type=int,
        default=3000,
        help="Run validation after this many optimizer updates.",
    )

    parser.add_argument(
        "--validation-batchnorm-train-mode",
        type=str2bool,
        default=False,
        help=(
            "Diagnostic option: use batch statistics for BatchNorm layers "
            "during validation while leaving all other modules in eval mode."
        ),
    )
    parser.add_argument(
        "--validation-only",
        type=str2bool,
        default=False,
        help="Load a checkpoint, evaluate the validation set, and exit.",
    )
    parser.add_argument(
        "--skip-validation",
        type=str2bool,
        default=False,
        help="Skip periodic validation (useful for bounded memory benchmarks).",
    )
    parser.add_argument(
        "--check-batchnorm-finite-interval",
        type=int,
        default=0,
        help=(
            "Check and summarize BatchNorm running statistics every N training "
            "updates; zero disables the check."
        ),
    )

    parser.add_argument(
        "--pct-start", type=float, default=0.08, help="The percent of steps "
        "to use as warmup",
    )

    parser.add_argument(
        "--total-steps", type=int, default=100000,
        help="Total number of optimizer updates, including frozen updates.",
    )    
    
    parser.add_argument(
        "--weight-decay", type=float, default=1e-08, help="Adam weight decay",
    )

    parser.add_argument(
        "--max-num-spks",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-overlap",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--diarization-loss",
        choices=("none", "bce", "powerset"),
        default="none",
        help=(
            "Train a diarization-only WavLM with FIFO frame targets. 'bce' "
            "uses independent binary speaker activities; 'powerset' uses "
            "cross entropy over all speaker subsets containing at most "
            "--max-overlap active speakers. Both modes omit the ASR/token "
            "model, transcript graphs, and CTC intersection entirely."
        ),
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
        help=(
            "Independently normalize the token and speaker heads before "
            "forming their factorized joint distribution. --use-hat is a "
            "deprecated compatibility alias."
        ),
    )
    parser.add_argument(
        "--factorized-ctc-loss",
        type=str2bool,
        default=False,
        help=(
            "Pass token and speaker scores separately to the custom CUDA k2 "
            "intersection instead of materializing token-by-speaker logits."
        ),
    )
    parser.add_argument(
        "--determinize-token-only-graph",
        type=str2bool,
        default=False,
        help=(
            "Determinize the token-projected partial-order transcript FSA "
            "before CTC composition. This removes duplicate token strings "
            "from token-only training graphs; it is ignored when the "
            "speaker branch is enabled."
        ),
    )
    parser.add_argument(
        "--token-only-segment-factor",
        type=str2bool,
        default=False,
        help=(
            "Use unique supervision-segment labels in a token-only graph and "
            "score them with a fixed normalized temporal prior. The prior is "
            "flat inside each collar-expanded supervision interval, decays "
            "exponentially outside it, and has a configurable nonzero floor. "
            "This uses the factorized CUDA loss without adding a learned "
            "segment head."
        ),
    )
    parser.add_argument(
        "--segment-factor-collar",
        type=int,
        default=None,
        help=(
            "Flat supervision-interval expansion in waveform samples for the "
            "fixed segment factor. Defaults to the effective partial-order "
            "collar."
        ),
    )
    parser.add_argument(
        "--segment-factor-decay-scale",
        type=float,
        default=16000.0,
        help=(
            "Exponential-decay scale in waveform samples outside the segment-"
            "factor collar. Must be positive. A value of 16000 corresponds to "
            "one second at 16 kHz."
        ),
    )
    parser.add_argument(
        "--segment-factor-floor-prob",
        type=float,
        default=1.0e-4,
        help=(
            "Total probability mass mixed uniformly over all real segments at "
            "every encoder frame. This keeps every graph path finite even when "
            "the coarse supervision timing is inaccurate."
        ),
    )
    parser.add_argument(
        "--frozen-token-teacher-checkpoint",
        type=str,
        default=None,
        help=(
            "Enable teacher/student SpAShl training. The specified token-only "
            "checkpoint is loaded into a permanently frozen WavLM teacher; a "
            "separate WavLM encoder and speaker head receive all gradients."
        ),
    )
    parser.add_argument(
        "--teacher-local-mask-window-duration",
        type=float,
        default=2.0,
        help=(
            "Width in seconds of each non-overlapping local transcript-mask "
            "window used for frozen-teacher token scores."
        ),
    )
    parser.add_argument(
        "--teacher-local-mask-boundary-collar",
        type=float,
        default=5.0,
        help=(
            "Seconds of coarse-timing context on both sides of each local "
            "token-mask window."
        ),
    )
    parser.add_argument(
        "--speaker-branch",
        type=str2bool,
        default=True,
        help=(
            "Use speaker-attributed output symbols. When false, train a "
            "token-only shuffle graph with no speaker output branch."
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
        type=int,
        default=24,
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=25000000,
    )

    parser.add_argument(
        "--original-topo",
        type=str2bool,
        default=False
    )

    parser.add_argument(
        "--sort-strategy",
        type=str,
        default="start_time",
    )

    parser.add_argument(
        "--pretrained-dir",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--iter",
        type=int,
        default=82000,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=3,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--speaker-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--no-overlap",
        type=bool, default=False,
    )
    
    parser.add_argument(
        "--ami",
        type=str2bool, default=False,
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
            #"subsampling_factor": 4,  # not passed in, this is fixed.
            "warm_step": 2000,
            "env_info": get_env_info(),
            # parameters for loss
            #"beam_size": 24,
            "reduction": "sum",
            "use_double_scores": True,
            # parameters for decoding
            "search_beam": 20,
            "output_beam": 24,
            "min_active_states": 30,
            "max_active_states": 10000,
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
    
    if params.pretrained_dir is not None and params.pretrained_dir.strip() != "":
        # Get the averaged model from the last few saved checkpoints
        filenames = find_checkpoints(
            params.pretrained_dir,
            iteration=-params.iter
        )[:params.avg]
        state_dict = average_checkpoints(filenames, device='cpu')
        
        # Loading pretrained model
        for name, p in model.named_parameters():
            p.data.copy_(state_dict[f'encoder.{name}'].data)

    return model, odim


def powerset_class_bitmasks(num_speakers: int, max_overlap: int) -> Tuple[int, ...]:
    """Return deterministic bitmasks for the admissible powerset classes."""
    if num_speakers < 1:
        raise ValueError(f"num_speakers must be positive: {num_speakers}")
    if not 0 <= max_overlap <= num_speakers:
        raise ValueError(
            f"max_overlap must be in [0, {num_speakers}]: {max_overlap}"
        )
    masks = []
    for cardinality in range(max_overlap + 1):
        for active in itertools.combinations(range(num_speakers), cardinality):
            mask = 0
            for speaker in active:
                mask |= 1 << speaker
            masks.append(mask)
    return tuple(masks)


def get_mdctc_model(
    params: AttributeDict,
) -> nn.Module:
    encoder, encoder_dim = get_encoder_model(params)

    if params.diarization_loss != "none":
        if params.diarization_loss == "bce":
            output_dim = params.max_num_spks
        else:
            output_dim = len(
                powerset_class_bitmasks(params.max_num_spks, params.max_overlap)
            )
        logging.info(
            "Constructing diarization-only WavLM: loss=%s output_dim=%d",
            params.diarization_loss,
            output_dim,
        )
        return SpeakerDiarizationModel(
            encoder=encoder,
            encoder_dim=encoder_dim,
            output_dim=output_dim,
            layer_norm=params.use_layer_norm,
            downsample=params.downsample,
        )

    if not getattr(params, "frozen_token_teacher_checkpoint", None):
        return MDCTCModel(
            encoder=encoder,
            encoder_dim=encoder_dim,
            vocab_size1=params.vocab_size,
            vocab_size2=params.max_num_spks,
            factorized_token_speaker_output=params.factorized_token_speaker_output,
            layer_norm=params.use_layer_norm,
            downsample=params.downsample,
        )

    teacher_checkpoint = Path(params.frozen_token_teacher_checkpoint)
    if not teacher_checkpoint.is_file():
        raise FileNotFoundError(
            f"Frozen token-teacher checkpoint not found: {teacher_checkpoint}"
        )

    # ``encoder`` above becomes the independently pretrained speaker student.
    # Build the legacy token-only architecture separately so its checkpoint is
    # interpreted exactly as it was during the original run.
    teacher_encoder, teacher_dim = get_encoder_model(params)
    if teacher_dim != encoder_dim:
        raise RuntimeError(
            f"Teacher/student encoder dimensions differ: {teacher_dim} != {encoder_dim}"
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

    model = FrozenTokenTeacherSpeakerModel(
        token_teacher=token_teacher,
        speaker_encoder=encoder,
        encoder_dim=encoder_dim,
        vocab_size=params.vocab_size,
        num_speakers=params.max_num_spks,
        layer_norm=params.use_layer_norm,
        downsample=params.downsample,
    )
    logging.info(
        "Teacher/student parameter counts: frozen token teacher=%d; "
        "trainable speaker model=%d",
        sum(parameter.numel() for parameter in token_teacher.parameters()),
        sum(parameter.numel() for parameter in model.parameters()),
    )
    return model


def get_legacy_mdctc_model(
    params: AttributeDict,
) -> nn.Module:
    """Compatibility constructor retained for external recipe imports."""
    encoder, encoder_dim = get_encoder_model(params)
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


def make_uniform_segment_log_probs(
    start_samples_list,
    num_samples_list,
    input_sample_lengths: torch.Tensor,
    output_lengths: torch.Tensor,
    max_output_frames: int,
    collar: int,
    device: torch.device,
    decay_scale: float = 16000.0,
    floor_probability: float = 1.0e-4,
) -> torch.Tensor:
    """Create a normalized, nonlearned soft per-frame segment distribution.

    A segment receives a constant unnormalized score inside its supervision
    interval expanded by ``collar`` waveform samples.  Outside that interval,
    its score decays exponentially with distance using ``decay_scale``.  After
    normalization, ``floor_probability`` total mass is mixed uniformly over
    all real segments.  Consequently every segment has finite log probability
    at every valid encoder frame, so inaccurate coarse timing cannot eliminate
    every accepting graph path.

    Padded segment columns remain ``-inf`` because they are not labels in the
    corresponding graph. Padded encoder frames are normalized on segment zero;
    they are ignored by the dense intersection.
    """
    if collar < 0:
        raise ValueError(f"segment-factor collar must be non-negative: {collar}")
    if decay_scale <= 0:
        raise ValueError(
            "segment-factor decay scale must be positive: "
            f"{decay_scale}"
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
        left_edges = starts_tensor - collar
        right_edges = ends_tensor + collar
        distance = torch.maximum(
            left_edges[None, :] - frame_centers[:, None],
            frame_centers[:, None] - right_edges[None, :],
        ).clamp_min_(0)
        local_log_probs = torch.log_softmax(-distance / decay_scale, dim=1)

        if floor_probability > 0.0:
            num_segments = len(starts)
            log_local_weight = math.log1p(-floor_probability)
            log_uniform_floor = math.log(floor_probability / num_segments)
            local_log_probs = torch.logaddexp(
                local_log_probs + log_local_weight,
                torch.full_like(local_log_probs, log_uniform_floor),
            )

        ans[i, :output_length, :len(starts)] = local_log_probs

        # Padded encoder frames are not selected by DenseFsaVec.  Give them a
        # benign normalized value so diagnostics never encounter an all--inf
        # row accidentally.
        if output_length < max_output_frames:
            ans[i, output_length:, 0] = 0.0
    return ans


def apply_teacher_local_token_mask(
    token_log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    input_sample_lengths: torch.Tensor,
    texts,
    supervision_start_samples,
    supervision_num_samples,
    sp,
    window_duration: float,
    boundary_collar: float,
    sample_rate: int = 16000,
):
    """Apply the long-form local transcript mask to teacher token scores.

    Windows are non-overlapping on the original waveform time axis.  For a
    window ``[a,b)``, a supervision contributes its BPE types when its coarse
    interval intersects ``[a-c,b+c]``.  Blank is always legal.  The teacher's
    original log-softmax denominator is deliberately preserved: masking sets
    inadmissible types to ``-inf`` but does not renormalize the remaining
    scores, matching the long-form alignment implementation.
    """
    if token_log_probs.ndim != 3:
        raise ValueError(
            f"Expected teacher scores [N,T,V], got {token_log_probs.shape}"
        )
    if window_duration <= 0:
        raise ValueError(
            f"Teacher local-mask window must be positive: {window_duration}"
        )
    if boundary_collar < 0:
        raise ValueError(
            f"Teacher local-mask collar must be non-negative: {boundary_collar}"
        )
    batch_size, max_frames, vocab_size = token_log_probs.shape
    if not (
        batch_size
        == len(texts)
        == len(supervision_start_samples)
        == len(supervision_num_samples)
        == int(output_lengths.numel())
        == int(input_sample_lengths.numel())
    ):
        raise ValueError("Teacher local-mask batch dimensions do not agree")

    window_samples = window_duration * sample_rate
    collar_samples = boundary_collar * sample_rate
    allowed = torch.zeros(
        (batch_size, max_frames, vocab_size),
        dtype=torch.bool,
        device=token_log_probs.device,
    )
    allowed[..., 0] = True
    total_windows = 0
    blank_only_windows = 0
    allowed_nonblank = 0
    possible_nonblank = 0

    for cut_index in range(batch_size):
        length = int(output_lengths[cut_index].item())
        input_length = int(input_sample_lengths[cut_index].item())
        if length <= 0 or input_length <= 0:
            raise ValueError(
                f"Cut {cut_index} has input_length={input_length}, "
                f"output_length={length}"
            )
        starts = supervision_start_samples[cut_index]
        durations = supervision_num_samples[cut_index]
        cut_texts = texts[cut_index]
        if not (len(starts) == len(durations) == len(cut_texts)):
            raise ValueError(
                f"Cut {cut_index} has {len(cut_texts)} texts, {len(starts)} "
                f"starts, and {len(durations)} durations"
            )

        supervision_tokens = []
        for text, start, duration in zip(cut_texts, starts, durations):
            token_ids = set(sp.encode(text))
            if any(token < 1 or token >= vocab_size for token in token_ids):
                raise ValueError(
                    f"Cut {cut_index} contains a token outside [1,{vocab_size}): "
                    f"{sorted(token_ids)}"
                )
            supervision_tokens.append(
                (float(start), float(start + duration), token_ids)
            )

        frame_centers = (
            torch.arange(
                length, device=token_log_probs.device, dtype=torch.float64
            )
            + 0.5
        ) * (input_length / length)
        window_ids = torch.floor(frame_centers / window_samples).to(torch.long)
        for window_id in torch.unique(window_ids).tolist():
            frame_mask = window_ids == window_id
            window_start = window_id * window_samples
            window_end = window_start + window_samples
            active_tokens = set()
            for sup_start, sup_end, token_ids in supervision_tokens:
                if (
                    sup_start < window_end + collar_samples
                    and sup_end > window_start - collar_samples
                ):
                    active_tokens.update(token_ids)
            if active_tokens:
                frame_index = torch.nonzero(frame_mask, as_tuple=False).squeeze(1)
                token_index = torch.tensor(
                    sorted(active_tokens),
                    device=token_log_probs.device,
                    dtype=torch.long,
                )
                allowed[
                    cut_index, frame_index[:, None], token_index[None, :]
                ] = True
            else:
                blank_only_windows += 1
            total_windows += 1

        allowed_nonblank += int(allowed[cut_index, :length, 1:].sum().item())
        possible_nonblank += length * max(vocab_size - 1, 0)

    masked = token_log_probs.masked_fill(~allowed, -torch.inf)
    stats = {
        "windows": total_windows,
        "blank_only_windows": blank_only_windows,
        "allowed_nonblank": allowed_nonblank,
        "possible_nonblank": possible_nonblank,
    }
    return masked, stats


def make_fifo_speaker_targets(
    speakers,
    supervision_start_samples,
    supervision_num_samples,
    input_sample_lengths: torch.Tensor,
    output_lengths: torch.Tensor,
    max_output_frames: int,
    max_num_speakers: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """Rasterize supervision intervals into FIFO speaker activities.

    The first distinct speaker to begin in a cut is slot 0, the next unseen
    speaker is slot 1, and so on.  ``speakers`` and the interval lists are
    expected to be in the dataset's ``start_time`` order, exactly matching the
    FIFO convention used by the teacher-token SpAShl graph.
    """
    batch_size = len(speakers)
    if not (
        batch_size
        == len(supervision_start_samples)
        == len(supervision_num_samples)
        == int(input_sample_lengths.numel())
        == int(output_lengths.numel())
    ):
        raise ValueError("FIFO target batch dimensions differ")

    targets = torch.zeros(
        batch_size,
        max_output_frames,
        max_num_speakers,
        dtype=torch.float32,
        device=device,
    )
    valid_frames = (
        torch.arange(max_output_frames, device=device)[None, :]
        < output_lengths.to(device=device, dtype=torch.long)[:, None]
    )
    num_speakers_per_cut = []

    for cut_index, (cut_speakers, starts, durations) in enumerate(
        zip(speakers, supervision_start_samples, supervision_num_samples)
    ):
        if not (len(cut_speakers) == len(starts) == len(durations)):
            raise ValueError(
                f"Cut {cut_index} has mismatched speakers/starts/durations: "
                f"{len(cut_speakers)}/{len(starts)}/{len(durations)}"
            )
        fifo_slots = {}
        segment_slots = []
        for speaker in cut_speakers:
            if speaker not in fifo_slots:
                fifo_slots[speaker] = len(fifo_slots)
            segment_slots.append(fifo_slots[speaker])
        if len(fifo_slots) > max_num_speakers:
            raise ValueError(
                f"Cut {cut_index} has {len(fifo_slots)} speakers, exceeding "
                f"--max-num-spks={max_num_speakers}"
            )
        num_speakers_per_cut.append(len(fifo_slots))

        output_length = int(output_lengths[cut_index].item())
        input_length = int(input_sample_lengths[cut_index].item())
        if output_length <= 0 or input_length <= 0:
            raise ValueError(
                f"Cut {cut_index} has input_length={input_length}, "
                f"output_length={output_length}"
            )
        frame_centers = (
            torch.arange(output_length, device=device, dtype=torch.float32) + 0.5
        ) * (input_length / output_length)
        for start, duration, slot in zip(starts, durations, segment_slots):
            end = start + duration
            active = (frame_centers >= start) & (frame_centers < end)
            targets[cut_index, :output_length, slot] = torch.maximum(
                targets[cut_index, :output_length, slot], active.to(torch.float32)
            )

    return targets, valid_frames, tuple(num_speakers_per_cut)


def compute_diarization_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_frames: torch.Tensor,
    loss_type: str,
    max_overlap: int,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Compute masked FIFO BCE or powerset cross entropy."""
    if logits.shape[:2] != targets.shape[:2] or valid_frames.shape != logits.shape[:2]:
        raise ValueError(
            f"Diarization tensor shape mismatch: logits={tuple(logits.shape)}, "
            f"targets={tuple(targets.shape)}, valid={tuple(valid_frames.shape)}"
        )
    num_valid_frames = int(valid_frames.sum().item())
    if num_valid_frames == 0:
        raise ValueError("Diarization batch contains no valid output frames")

    if loss_type == "bce":
        if logits.size(-1) != targets.size(-1):
            raise ValueError(
                f"BCE output dimension {logits.size(-1)} != "
                f"speaker slots {targets.size(-1)}"
            )
        element_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        speaker_mask = valid_frames.unsqueeze(-1).expand_as(element_loss)
        loss = element_loss.masked_select(speaker_mask).mean()
        return loss, {"powerset_classes": 0, "overlap_violations": 0}

    if loss_type != "powerset":
        raise ValueError(f"Unsupported diarization loss: {loss_type}")

    num_speakers = targets.size(-1)
    class_masks = powerset_class_bitmasks(num_speakers, max_overlap)
    if logits.size(-1) != len(class_masks):
        raise ValueError(
            f"Powerset output dimension {logits.size(-1)} != {len(class_masks)}"
        )
    bit_weights = 1 << torch.arange(num_speakers, device=targets.device)
    target_masks = (targets.to(torch.long) * bit_weights).sum(dim=-1)
    lookup = torch.full(
        (1 << num_speakers,), -1, dtype=torch.long, device=targets.device
    )
    lookup[torch.tensor(class_masks, device=targets.device)] = torch.arange(
        len(class_masks), device=targets.device
    )
    target_classes = lookup[target_masks]
    violations = valid_frames & target_classes.lt(0)
    num_violations = int(violations.sum().item())
    if num_violations:
        observed_overlap = int(
            targets[violations].sum(dim=-1).max().item()
        )
        raise ValueError(
            f"Found {num_violations} valid frames with more than "
            f"--max-overlap={max_overlap} active speakers "
            f"(observed maximum {observed_overlap})"
        )
    loss = F.cross_entropy(logits[valid_frames], target_classes[valid_frames])
    return loss, {
        "powerset_classes": len(class_masks),
        "overlap_violations": num_violations,
    }


def _configure_cuda_memory_reporter(params, model, device):
    """Return a phase profiler whose peak is reset after every checkpoint."""
    bare_model = model.module if isinstance(model, DDP) else model
    if not params.profile_cuda_memory or device.type != "cuda":
        reporter = None
    else:
        gib = 1024 ** 3

        def reporter(label):
            torch.cuda.synchronize(device)
            logging.info(
                "CUDA_MEMORY phase=%s allocated=%.3fGiB phase_peak=%.3fGiB "
                "reserved=%.3fGiB",
                label,
                torch.cuda.memory_allocated(device) / gib,
                torch.cuda.max_memory_allocated(device) / gib,
                torch.cuda.memory_reserved(device) / gib,
            )
            torch.cuda.reset_peak_memory_stats(device)

    bare_model._memory_reporter = reporter
    if hasattr(bare_model, "encoder"):
        bare_model.encoder.memory_reporter = reporter
    return reporter


def compute_loss(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    graph_compiler: MDCTCGraphCompiler,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
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
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    memory_reporter = _configure_cuda_memory_reporter(params, model, device)
    if memory_reporter is not None:
        torch.cuda.reset_peak_memory_stats(device)
        memory_reporter("batch/start")
    feature = batch["inputs"]
    max_density = batch["max_density"]
    feature_lens = batch["num_frames"].to(device)
    # at entry, feature is (N, T, C)
    feature = feature.to(device)
    feature_lens = feature_lens.to(device)
    if memory_reporter is not None:
        memory_reporter("batch/inputs_on_gpu")
    texts = batch["texts"]
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

    if params.diarization_loss != "none":
        with torch.set_grad_enabled(is_training):
            start = time.time()
            logits, x_lens = model(feature, feature_lens)
            end_nnet = time.time()
            targets, valid_frames, num_speakers_per_cut = make_fifo_speaker_targets(
                speakers=speakers,
                supervision_start_samples=start_frames,
                supervision_num_samples=num_frames_init,
                input_sample_lengths=feature_lens,
                output_lengths=x_lens,
                max_output_frames=logits.size(1),
                max_num_speakers=params.max_num_spks,
                device=device,
            )
            loss, objective_stats = compute_diarization_objective(
                logits=logits,
                targets=targets,
                valid_frames=valid_frames,
                loss_type=params.diarization_loss,
                max_overlap=params.max_overlap,
            )
            end_loss = time.time()

        if memory_reporter is not None:
            memory_reporter("diarization/loss")
        valid_count = int(valid_frames.sum().item())
        weighted_loss = float(loss.detach().cpu().item()) * valid_count
        info = MetricsTracker()
        info["frames"] = valid_count
        info["loss"] = weighted_loss
        info["diarization_loss"] = weighted_loss
        info["speaker_activity"] = float(
            targets.masked_select(
                valid_frames.unsqueeze(-1).expand_as(targets)
            ).sum().item()
        )
        info["num_speakers"] = (
            sum(num_speakers_per_cut) / len(num_speakers_per_cut) * valid_count
        )
        info["pct_nnet"] = (
            (end_nnet - start) / max(end_loss - start, 1.0e-12) * valid_count
        )
        info["powerset_classes"] = objective_stats["powerset_classes"] * valid_count
        info["utterances"] = feature.size(0)
        info["utt_duration"] = feature_lens.sum().item()
        info["utt_pad_proportion"] = (
            ((feature.size(1) - feature_lens) / feature.size(1)).sum().item()
        )
        assert loss.requires_grad == is_training, (
            f"{loss.requires_grad} != {is_training}"
        )
        return loss, info

    beam_factor = 1
    teacher_mask_stats = None
    with torch.set_grad_enabled(is_training):
        start = time.time()
        use_speaker_factor = params.factorized_ctc_loss and params.speaker_branch
        use_segment_factor = params.token_only_segment_factor
        use_factorized_ctc = use_speaker_factor or use_segment_factor
        model_output = model(
            feature, feature_lens,
            speaker_weight=params.speaker_weight,
            # Both the speaker-factor and segment-factor losses consume the
            # token branch before speaker x token expansion.  In segment mode
            # the model's speaker factor is deliberately discarded and
            # replaced below by the fixed temporal segment prior.
            return_factors=use_factorized_ctc,
        )
        if use_factorized_ctc:
            token_output, model_factor_output, x_lens = model_output
            ctc_output = None
            speaker_output = (
                model_factor_output if use_speaker_factor else None
            )
            if use_speaker_factor and params.frozen_token_teacher_checkpoint:
                token_output, teacher_mask_stats = apply_teacher_local_token_mask(
                    token_log_probs=token_output,
                    output_lengths=x_lens,
                    input_sample_lengths=feature_lens,
                    texts=texts,
                    supervision_start_samples=start_frames,
                    supervision_num_samples=num_frames_init,
                    sp=graph_compiler.sp,
                    window_duration=params.teacher_local_mask_window_duration,
                    boundary_collar=params.teacher_local_mask_boundary_collar,
                )
        else:
            ctc_output, x_lens = model_output
        end_nnet = time.time()
        subsampling_factor = params.subsampling_factor
        beam_size = params.beam_size * beam_factor
        reduction = params.reduction
        use_double_scores = params.use_double_scores


        # With gradient accumulation, batch_idx_train advances only after an
        # optimizer update. Emit this expensive diagnostic on the first
        # microbatch of the selected update, not on every accumulated batch.
        if (
            params.batch_idx_train % params.decode_hyp_interval == 0
            and getattr(params, "microbatch_idx_train", 0) % params.accum_grad == 0
        ):
            if use_speaker_factor:
                token_score, token_index = token_output[..., 1:].max(-1)
                speaker_score, speaker_index = speaker_output.max(-1)
                best_joint = token_score + speaker_score
                joint_index = token_index + 1 + speaker_index * (
                    token_output.size(-1) - 1
                )
                preds = torch.where(
                    token_output[..., 0] >= best_joint,
                    torch.zeros_like(joint_index), joint_index,
                )
            elif use_segment_factor:
                # Segment-factor training has no predicted speaker/factor
                # branch.  These are genuine SentencePiece token IDs.
                preds = token_output.argmax(-1)
            else:
                preds = ctc_output.argmax(-1)
            hyps = []
            spk_hyps = []
            for i in range(preds.size(0)):
                hyps_ = {}
                preds_no_repeats = preds[i].unique_consecutive()
                preds_no_repeats_or_blanks = preds_no_repeats[
                    preds_no_repeats != 0
                ].reshape(-1)
                if not use_segment_factor:
                    # Joint labels are token + speaker * (V - 1), with token
                    # in [1, V - 1].  Subtracting one before division is
                    # essential for token V-1 at every block boundary.
                    spks, units = split_joint_labels(
                        preds_no_repeats_or_blanks, params.vocab_size
                    )
                    for s in range(params.max_num_spks):
                        s_units = units[spks == s].tolist()
                        hyps_[s] = s_units
                else:
                    spks = torch.zeros_like(preds_no_repeats_or_blanks)
                    units = preds_no_repeats_or_blanks
                    hyps_[0] = units.tolist()
                hyps.append(hyps_)
                spk_hyps.append(spks.view(-1).tolist())

            # Construct the speaker ref
            text_lens = []
            num_frames_ref, num_tokens_ref = 0, 0
            for t_idx in range(len(texts[0])): 
                text_lens.append(start_frames[0][t_idx] + num_frames_init[0][t_idx])
                num_frames_ref += num_frames_init[0][t_idx]
                num_tokens_ref += len(graph_compiler.sp.encode(texts[0][t_idx]))
            max_len = max(text_lens)
            frames_per_token = num_frames_ref // num_tokens_ref + 1
            start_tokens = [start_frames[0][t_idx] // frames_per_token for t_idx in range(len(texts[0]))]
            spk2int = {k: i for i, k in enumerate(dict.fromkeys(speakers[0]))}

            logging.info(f" ===================================== " )
            logging.info(f"spk_ref: ")
            for t_idx, st in enumerate(start_tokens):
                blanks = " "*st
                tokens = str(spk2int[speakers[0][t_idx]])*len(graph_compiler.sp.encode(texts[0][t_idx]))
                logging.info(f"{blanks}{tokens}")
            logging.info(f" ---------------")
            logging.info(f"spk_hyp: {''.join(map(str, spk_hyps[0]))}")
            for t_idx in range(len(hyps[0])):
                logging.info(f"hyp_{t_idx}: {graph_compiler.sp.decode(hyps[0][t_idx])}")
                logging.info(f"")
            logging.info(f" --------")
            for t_idx, text in enumerate(texts[0]):
                logging.info(f"ref_{t_idx}: {text}")
                logging.info(f"")
            logging.info("--------------------------------------")
        
        # Train with WFST supervisions
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
        
        collar = params.collar
        # Besides the graph compiler's historical ambiguity-count guard, keep
        # every individual graph within the exact signed-int32 address space
        # used by k2's dense intersection.  Arc count is not the relevant
        # quantity: k2 allocates/indexes metadata bounded by
        # (num_frames + 1) * num_graph_states.
        while True:
            effective_edge_collar = (
                None if params.edge_collar is None
                else min(params.edge_collar, collar)
            )
            if use_segment_factor:
                segment_ids = [list(range(len(cut_texts))) for cut_texts in texts]
                decoding_graphs = graph_compiler.compile(
                    texts,
                    start_frames,
                    num_frames_init,
                    segment_ids,
                    collar=collar,
                    edge_collar=effective_edge_collar,
                    interval_model=params.interval_model,
                    max_overlaps=params.total_max_overlaps,
                    original_topo=params.original_topo,
                    # Every supervision receives a unique factor label; retain
                    # the same cross-segment partial order as token-only mode.
                    allow_self_overlap=True,
                )
            elif params.speaker_branch:
                decoding_graphs = graph_compiler.compile(
                    texts, start_frames, num_frames_init, speakers,
                    collar=collar,
                    edge_collar=effective_edge_collar,
                    interval_model=params.interval_model,
                    max_overlaps=params.total_max_overlaps,
                    original_topo=params.original_topo,
                    allow_self_overlap=params.allow_self_overlap,
                )
            else:
                decoding_graphs = graph_compiler.compile_nospk(
                    texts,
                    start_frames,
                    num_frames_init,
                    collar=collar,
                    edge_collar=effective_edge_collar,
                    interval_model=params.interval_model,
                    max_overlaps=params.total_max_overlaps,
                    original_topo=params.original_topo,
                    determinize=params.determinize_token_only_graph,
                )
            graph_state_splits_cpu = (
                decoding_graphs.arcs.shape().row_splits(1).cpu()
            )
            graph_states_per_cut = (
                graph_state_splits_cpu[1:] - graph_state_splits_cpu[:-1]
            ).to(torch.int64)
            intersection_products = (
                graph_states_per_cut
                * (x_lens.detach().cpu().to(torch.int64) + 1)
            )
            int32_product_limit = torch.iinfo(torch.int32).max - 1
            worst_product, worst_index = intersection_products.max(dim=0)
            if int(worst_product.item()) <= int32_product_limit:
                break
            if collar == 0:
                raise RuntimeError(
                    "A single training graph exceeds k2's int32 dense-"
                    "intersection address space even at zero collar: "
                    f"cut_index={int(worst_index.item())}, "
                    f"frames={int(x_lens[worst_index].item())}, "
                    f"graph_states={int(graph_states_per_cut[worst_index].item())}, "
                    f"product={int(worst_product.item())}, "
                    f"limit={int32_product_limit}"
                )
            old_collar = collar
            collar //= 2
            logging.warning(
                "DYNAMIC_COLLAR_ADDRESS_GUARD cut_index=%d frames=%d "
                "graph_states=%d address_product=%d int32_limit=%d "
                "collar=%d->%d edge_collar=%s; recompiling batch",
                int(worst_index.item()),
                int(x_lens[worst_index].item()),
                int(graph_states_per_cut[worst_index].item()),
                int(worst_product.item()),
                int32_product_limit,
                old_collar,
                collar,
                effective_edge_collar,
            )
        
        decoding_graphs = decoding_graphs.to(device)
        if memory_reporter is not None:
            memory_reporter("graph/compiled_and_on_gpu")
        
        end_sup = time.time()

        factorized_diagnostics = None
        zero_ctc_examples = 0
        if use_factorized_ctc:
            if use_segment_factor:
                segment_collar = (
                    collar
                    if params.segment_factor_collar is None
                    else params.segment_factor_collar
                )
                speaker_output = make_uniform_segment_log_probs(
                    start_frames,
                    num_frames_init,
                    feature_lens,
                    x_lens,
                    token_output.size(1),
                    segment_collar,
                    device,
                    decay_scale=params.segment_factor_decay_scale,
                    floor_probability=params.segment_factor_floor_prob,
                )
            ctc_loss, factorized_diagnostics = factorized_graph_ctc_loss(
                token_output,
                speaker_output,
                decoding_graphs,
                x_lens,
                supervision_segments=supervision_segments,
                reduction=reduction,
                output_beam=beam_size,
                max_states=params.max_states,
                allow_truncate=subsampling_factor - 1,
                use_double_scores=use_double_scores,
                # Preserve the historical training behavior, but retain the
                # pre-zeroing losses below so every skipped example is visible.
                zero_infinity=is_training,
                clamp_negative=True,
                memory_reporter=memory_reporter,
                return_diagnostics=True,
            )
        else:
            ctc_loss = graph_ctc_loss(
                ctc_output,
                decoding_graphs,
                x_lens,
                supervision_segments=supervision_segments,
                reduction=reduction,
                output_beam=beam_size,
                max_states=params.max_states,
                allow_truncate=subsampling_factor - 1,
                use_double_scores=use_double_scores,
                zero_infinity=is_training,
                clamp_negative=True,
                memory_reporter=memory_reporter,
            )

        if factorized_diagnostics is not None:
            raw_losses = factorized_diagnostics["raw_losses"]
            bad_examples = (~torch.isfinite(raw_losses)).nonzero().flatten()
            zero_ctc_examples = int(bad_examples.numel()) if is_training else 0
            if bad_examples.numel() and is_training:
                cuts = batch["supervisions"].get("cut", [])
                graph_state_splits = decoding_graphs.arcs.shape().row_splits(1)
                graph_arc_splits = decoding_graphs.arcs.shape().row_splits(2)
                for bad_index_tensor in bad_examples:
                    bad_index = int(bad_index_tensor.item())
                    num_input_frames = int(x_lens[bad_index].item())
                    token_count = sum(
                        len(graph_compiler.sp.encode(text))
                        for text in texts[bad_index]
                    )
                    token_scores_i = token_output[
                        bad_index, :num_input_frames
                    ]
                    token_scores_all_finite = bool(
                        torch.isfinite(token_scores_i).all().item()
                    )
                    token_scores_valid = bool(
                        (
                            ~torch.isnan(token_scores_i)
                            & ~torch.isposinf(token_scores_i)
                        ).all().item()
                        and torch.isfinite(token_scores_i).any(dim=-1).all().item()
                    )

                    factor_scores_i = speaker_output[
                        bad_index, :num_input_frames
                    ]
                    if use_segment_factor:
                        # Padded segment columns and hard-mask configurations
                        # may legitimately contain -inf. Validate the real
                        # segment distribution instead of requiring every
                        # tensor element to be finite.
                        num_factors = len(texts[bad_index])
                        factor_scores_i = factor_scores_i[:, :num_factors]
                        factor_rows_supported = bool(
                            torch.isfinite(factor_scores_i)
                            .any(dim=-1)
                            .all()
                            .item()
                        )
                        factor_rows_normalized = bool(
                            torch.allclose(
                                torch.logsumexp(factor_scores_i, dim=-1),
                                torch.zeros(
                                    num_input_frames,
                                    dtype=factor_scores_i.dtype,
                                    device=factor_scores_i.device,
                                ),
                                atol=1.0e-4,
                                rtol=1.0e-4,
                            )
                        )
                        factor_scores_valid = bool(
                            (
                                ~torch.isnan(factor_scores_i)
                                & ~torch.isposinf(factor_scores_i)
                            ).all().item()
                            and factor_rows_supported
                            and factor_rows_normalized
                        )
                    else:
                        factor_rows_supported = bool(
                            torch.isfinite(factor_scores_i)
                            .any(dim=-1)
                            .all()
                            .item()
                        )
                        factor_rows_normalized = True
                        factor_scores_valid = bool(
                            torch.isfinite(factor_scores_i).all().item()
                        )

                    # Independently retry this one example with the explicit
                    # token x speaker logits and standard k2 intersection.  A
                    # finite reference loss isolates a factorized-CUDA bug;
                    # non-finite in both paths indicates a graph/frame issue.
                    reference_loss = float("nan")
                    reference_error = None
                    try:
                        with torch.no_grad():
                            token_i = token_output[
                                bad_index : bad_index + 1, :num_input_frames
                            ].detach()
                            speaker_i = speaker_output[
                                bad_index : bad_index + 1, :num_input_frames
                            ].detach()
                            blank_i = token_i[..., :1]
                            joint_i = (
                                token_i[..., 1:].unsqueeze(-1)
                                + speaker_i.unsqueeze(-2)
                            )
                            joint_i = joint_i.permute(0, 1, 3, 2).reshape(
                                1, num_input_frames, -1
                            )
                            expanded_i = torch.cat([blank_i, joint_i], dim=-1)
                            graph_i = k2.create_fsa_vec(
                                [decoding_graphs[bad_index]]
                            )
                            reference_loss = float(
                                graph_ctc_loss(
                                    expanded_i,
                                    graph_i,
                                    x_lens[bad_index : bad_index + 1],
                                    reduction="none",
                                    output_beam=beam_size,
                                    max_states=params.max_states,
                                    allow_truncate=subsampling_factor - 1,
                                    use_double_scores=use_double_scores,
                                    zero_infinity=False,
                                    clamp_negative=False,
                                )[0].item()
                            )
                    except Exception as error:
                        reference_error = repr(error)

                    if not token_scores_valid:
                        cause = "invalid_token_model_scores"
                    elif not factor_scores_valid:
                        cause = (
                            "invalid_segment_factor_scores"
                            if use_segment_factor
                            else "invalid_speaker_model_scores"
                        )
                    elif np.isfinite(reference_loss):
                        cause = "factorized_cuda_mismatch"
                    elif token_count > num_input_frames:
                        cause = "token_lower_bound_exceeds_input_frames"
                    elif reference_error is not None:
                        cause = "reference_diagnostic_failed"
                    else:
                        cause = "no_accepting_path_in_factorized_or_reference_k2"

                    graph_state_begin = int(graph_state_splits[bad_index].item())
                    graph_state_end = int(graph_state_splits[bad_index + 1].item())
                    graph_num_arcs = int(
                        (
                            graph_arc_splits[graph_state_end]
                            - graph_arc_splits[graph_state_begin]
                        ).item()
                    )
                    cut = cuts[bad_index] if bad_index < len(cuts) else None
                    logging.error(
                        "ZERO_CTC_DIAGNOSTIC cause=%s cut_id=%s recording_id=%s "
                        "raw_loss=%s reference_expanded_loss=%s "
                        "reference_error=%s input_frames=%d bpe_tokens=%d "
                        "token_frame_ratio=%.4f supervisions=%d speakers=%d "
                        "graph_states=%d graph_arcs=%d lattice_states=%d "
                        "lattice_arcs=%d token_scores_all_finite=%s "
                        "token_scores_valid=%s factor_scores_valid=%s "
                        "factor_rows_supported=%s factor_rows_normalized=%s "
                        "output_beam=%s "
                        "max_states=%d start_frames=%s supervision_frames=%s",
                        cause,
                        getattr(cut, "id", f"batch-index-{bad_index}"),
                        getattr(cut, "recording_id", "unknown"),
                        float(raw_losses[bad_index].item()),
                        reference_loss,
                        reference_error,
                        num_input_frames,
                        token_count,
                        token_count / max(num_input_frames, 1),
                        len(texts[bad_index]),
                        len(set(speakers[bad_index])),
                        graph_state_end - graph_state_begin,
                        graph_num_arcs,
                        int(factorized_diagnostics["lattice_num_states"][bad_index].item()),
                        int(factorized_diagnostics["lattice_num_arcs"][bad_index].item()),
                        token_scores_all_finite,
                        token_scores_valid,
                        factor_scores_valid,
                        factor_rows_supported,
                        factor_rows_normalized,
                        beam_size,
                        params.max_states,
                        start_frames[bad_index],
                        num_frames_init[bad_index],
                    )
        end_ctc = time.time()
        
    nnet_time = end_nnet - start
    sup_time = end_sup - end_nnet
    ctc_time = end_ctc - end_sup
    total_time = end_ctc - start
    tot_frames = num_frames.sum().item()
    info = MetricsTracker()
    info["frames"] = tot_frames
    info["ctc_loss"] = ctc_loss.detach().cpu().item()
    info["zero_ctc_examples"] = zero_ctc_examples * tot_frames
    info["beam_size"] = beam_size * tot_frames 
    info["max_length"] = num_frames.max() * tot_frames
    info["max_duration"] = num_frames.max() * 0.01 * params.subsampling_factor * tot_frames
    info["max_spk_density"] = max_density * tot_frames
    avg_num_texts = sum([len(t) for t in texts])/len(texts)
    info["num_texts"] = avg_num_texts * tot_frames
    info["avg_spk_density"] = (sum(densities) / len(densities)) * tot_frames
    info["arc_density"] = decoding_graphs.arcs.values().size(0)
    if teacher_mask_stats is not None:
        possible = teacher_mask_stats["possible_nonblank"]
        allowed_fraction = (
            teacher_mask_stats["allowed_nonblank"] / possible
            if possible > 0
            else 0.0
        )
        info["teacher_mask_allowed_fraction"] = allowed_fraction * tot_frames
        info["teacher_mask_blank_only_windows"] = (
            teacher_mask_stats["blank_only_windows"] * tot_frames
        )
    info["pct_ctc"] = ctc_time / total_time * tot_frames
    info["pct_nnet"] = nnet_time / total_time * tot_frames
    info["pct_sup"] = sup_time / total_time * tot_frames
    loss = ctc_loss
    assert loss.requires_grad == is_training, f"{loss.requires_grad} != {is_training}"
    info["loss"] = loss.detach().cpu().item()

    # `utt_duration` and `utt_pad_proportion` would be normalized by `utterances`  # noqa
    info["utterances"] = feature.size(0)
    # averaged input duration in frames over utterances
    info["utt_duration"] = feature_lens.sum().item()
    # averaged padding proportion over utterances
    info["utt_pad_proportion"] = (
        ((feature.size(1) - feature_lens) / feature.size(1)).sum().item()
    )
    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: nn.Module,
    graph_compiler: MDCTCGraphCompiler,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()
    if params.validation_batchnorm_train_mode:
        num_batchnorm = 0
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.train()
                num_batchnorm += 1
        logging.info(
            "Validation diagnostic: using batch statistics for %d BatchNorm layers",
            num_batchnorm,
        )

    tot_loss = MetricsTracker()
    nonfinite_batches = 0
    nonfinite_utterances = 0

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            batch=batch,
            graph_compiler=graph_compiler,
            is_training=False,
        )
        assert loss.requires_grad is False
        if not torch.isfinite(loss):
            nonfinite_batches += 1
            nonfinite_utterances += int(batch["num_frames"].numel())
            # MetricsTracker skips non-finite entries.  If only some DDP
            # ranks skip these keys, reduce() constructs different collective
            # payloads and deadlocks.  Preserve identical keys here and
            # restore the global failure marker after reduction below.
            if params.diarization_loss == "none":
                loss_info["ctc_loss"] = 0.0
            else:
                loss_info["diarization_loss"] = 0.0
            loss_info["loss"] = 0.0
        tot_loss = tot_loss + loss_info
        if params.max_valid_batches > 0 and batch_idx + 1 >= params.max_valid_batches:
            break

    if world_size > 1:
        tot_loss.reduce(loss.device)
        nonfinite_counts = torch.tensor(
            [nonfinite_batches, nonfinite_utterances],
            dtype=torch.long,
            device=loss.device,
        )
        torch.distributed.all_reduce(
            nonfinite_counts, op=torch.distributed.ReduceOp.SUM
        )
        nonfinite_batches, nonfinite_utterances = nonfinite_counts.tolist()

    if nonfinite_batches:
        if world_size == 1 or torch.distributed.get_rank() == 0:
            logging.error(
                "Validation produced non-finite loss in %d batches containing "
                "%d utterances",
                nonfinite_batches,
                nonfinite_utterances,
            )
        # MetricsTracker deliberately skips non-finite values while adding.
        # Restore failure semantics: any failed validation batch makes the
        # aggregate loss non-finite rather than silently reporting zero or a
        # partial average.
        if params.diarization_loss == "none":
            tot_loss["ctc_loss"] = float("inf")
        else:
            tot_loss["diarization_loss"] = float("inf")
        tot_loss["loss"] = float("inf")

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    graph_compiler: MDCTCGraphCompiler,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    init_lr: float, 
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      graph_compiler:
        It is used to convert transcripts to FSAs.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
    """
    model.train()

    if params.accum_grad < 1:
        raise ValueError(f"--accum-grad must be at least 1, got {params.accum_grad}")

    tot_loss = MetricsTracker()
    
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    autocast_args = {'device_type': 'cuda'} if device.type == 'cuda' else {'device_type': 'cpu'}

    start = time.time()
    num_egs = 0
    num_frames = 0 
    stopped_at_total_steps = False
    optimizer.zero_grad()
    try:
        num_train_batches = len(train_dl)
    except TypeError:
        num_train_batches = None
    
    for batch_idx, batch in enumerate(train_dl):
        if params.batch_idx_train >= params.total_steps:
            stopped_at_total_steps = True
            break
        batch_size = batch["num_frames"].size(0)
        params.microbatch_idx_train = batch_idx
        is_frozen = model.module.frozen if isinstance(model, DDP) else model.frozen
        unfreeze = model.module.unfreeze_encoder if isinstance(model, DDP) else model.unfreeze_encoder
        next_update = params.batch_idx_train + 1
        if next_update > params.freeze_iters and is_frozen:
            logging.info("Unfreezing ...")
            unfreeze()
            is_frozen = False
            for param_group in optimizer.param_groups:
                param_group['lr'] = init_lr

        accumulation_boundary = (batch_idx + 1) % params.accum_grad == 0
        loader_boundary = (
            num_train_batches is not None and batch_idx + 1 == num_train_batches
        )
        configured_boundary = (
            params.max_train_batches > 0
            and batch_idx + 1 >= params.max_train_batches
        )
        should_step = accumulation_boundary or loader_boundary or configured_boundary

        try:
            sync_context = (
                model.no_sync()
                if isinstance(model, DDP) and not should_step
                else nullcontext()
            )
            with sync_context:
                with torch.amp.autocast(**autocast_args):
                    loss, loss_info = compute_loss(
                        params=params,
                        model=model,
                        batch=batch,
                        graph_compiler=graph_compiler,
                        is_training=True,
                    )
                scaler.scale(loss).backward()
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            #if loss.isinf():
            #    logging.warning(f"Infinite Loss {params.batch_idx_train}")
            #    continue;
            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            memory_reporter = getattr(
                model.module if isinstance(model, DDP) else model,
                "_memory_reporter",
                None,
            )
            if memory_reporter is not None:
                memory_reporter("train/backward")
            if not should_step:
                continue

            params.batch_idx_train += 1
            if not is_frozen:
                if hasattr(scheduler, "step_batch"):
                    scheduler.step_batch(params.batch_idx_train)
                else:
                    scheduler.step()
            
            scaler.step(optimizer)
            scaler.update()
            if memory_reporter is not None:
                memory_reporter("train/optimizer_step")
            optimizer.zero_grad()
            if memory_reporter is not None:
                memory_reporter("train/zero_grad")

            check_interval = params.check_batchnorm_finite_interval
            if check_interval > 0 and params.batch_idx_train % check_interval == 0:
                model_to_check = model.module if isinstance(model, DDP) else model
                batchnorm_rows = []
                for name, module in model_to_check.named_modules():
                    if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        continue
                    for buffer_name in ("running_mean", "running_var"):
                        value = getattr(module, buffer_name, None)
                        if value is not None and not torch.isfinite(value).all():
                            bad = int((~torch.isfinite(value)).sum().item())
                            raise FloatingPointError(
                                f"Non-finite BatchNorm {buffer_name} at update "
                                f"{params.batch_idx_train}: {name} ({bad}/{value.numel()} bad)"
                            )
                    if module.running_mean is not None:
                        batchnorm_rows.append(
                            (
                                module.running_mean.detach().float().abs().max().item(),
                                module.running_var.detach().float().min().item(),
                                module.running_var.detach().float().max().item(),
                            )
                        )
                if batchnorm_rows and rank == 0:
                    logging.info(
                        "BatchNorm statistics at update %d: layers=%d, "
                        "max_abs_mean=%.6g, min_var=%.6g, max_var=%.6g",
                        params.batch_idx_train,
                        len(batchnorm_rows),
                        max(row[0] for row in batchnorm_rows),
                        min(row[1] for row in batchnorm_rows),
                        max(row[2] for row in batchnorm_rows),
                    )
        except:  # noqa
            #display_and_save_batch(batch, params=params, graph_compiler=graph_compiler)
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            params.cur_batch_idx = batch_idx
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            del params.cur_batch_idx
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if params.batch_idx_train % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler.get_scale()
            if cur_grad_scale < 1.0 or (cur_grad_scale < 8.0 and batch_idx % 400 == 0):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                raise_grad_scale_is_too_small_error(cur_grad_scale)

        if (
            params.batch_idx_train == 1
            or params.batch_idx_train % params.log_interval == 0
        ):
            cur_lr = max(scheduler.get_last_lr()) if not is_frozen else params.freeze_lr
            cur_grad_scale = scaler.get_scale() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"microbatch {batch_idx}, update {params.batch_idx_train}, "
                f"loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"accum_grad: {params.accum_grad}, "
                f"lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler.get_scale()}" if params.use_fp16 else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale",
                        cur_grad_scale,
                        params.batch_idx_train,
                    )

        if (
            (params.batch_idx_train == 1
             or params.batch_idx_train % params.valid_interval == 0)
            and not params.print_diagnostics
            and not params.skip_validation
        ):
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                graph_compiler=graph_compiler,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

        if params.batch_idx_train >= params.total_steps:
            logging.info("Reached --total-steps=%d", params.total_steps)
            stopped_at_total_steps = True
            break

        if params.max_train_batches > 0 and batch_idx + 1 >= params.max_train_batches:
            logging.info("Reached --max-train-batches=%d", params.max_train_batches)
            break

    if device.type == "cuda":
        logging.info(
            "Maximum CUDA memory allocated in training: %.2f GiB",
            torch.cuda.max_memory_allocated(device) / (1024**3),
        )

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss

    if stopped_at_total_steps:
        params.cur_batch_idx = batch_idx
        logging.info(
            "Saving final resumable checkpoint at update %d", params.batch_idx_train
        )
        save_checkpoint_with_global_batch_idx(
            out_dir=params.exp_dir,
            global_batch_idx=params.batch_idx_train,
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )
        del params.cur_batch_idx


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    #local_rank = int(os.environ.get("LOCAL_RANK", 0))
    #torch.cuda.set_device(local_rank)
    params = get_params()
    params.update(vars(args))
    if params.diarization_loss != "none":
        if params.sort_strategy != "start_time":
            raise ValueError(
                "Diarization-only FIFO targets require --sort-strategy start_time"
            )
        if params.frozen_token_teacher_checkpoint:
            raise ValueError(
                "Diarization-only training has no ASR teacher; omit "
                "--frozen-token-teacher-checkpoint"
            )
        if params.token_only_segment_factor:
            raise ValueError(
                "Diarization-only training cannot use --token-only-segment-factor"
            )
        powerset_class_bitmasks(params.max_num_spks, params.max_overlap)
    if params.token_only_segment_factor:
        if params.speaker_branch:
            raise ValueError(
                "--token-only-segment-factor requires --speaker-branch false"
            )
        if not params.factorized_ctc_loss:
            raise ValueError(
                "--token-only-segment-factor requires --factorized-ctc-loss true"
            )
        if not params.factorized_token_speaker_output:
            raise ValueError(
                "--token-only-segment-factor requires "
                "--factorized-token-speaker-output true so the loss receives "
                "the token branch rather than expanded speaker x token labels"
            )
        if params.determinize_token_only_graph:
            raise ValueError(
                "Segment-labelled graphs are already unambiguous; do not combine "
                "--token-only-segment-factor with "
                "--determinize-token-only-graph"
            )
        if (
            params.segment_factor_collar is not None
            and params.segment_factor_collar < 0
        ):
            raise ValueError("--segment-factor-collar must be non-negative")
        if params.segment_factor_decay_scale <= 0:
            raise ValueError("--segment-factor-decay-scale must be positive")
        if not 0.0 <= params.segment_factor_floor_prob < 1.0:
            raise ValueError(
                "--segment-factor-floor-prob must be in [0, 1)"
            )
    if params.frozen_token_teacher_checkpoint:
        if not params.speaker_branch:
            raise ValueError(
                "--frozen-token-teacher-checkpoint requires --speaker-branch true"
            )
        if not params.factorized_token_speaker_output:
            raise ValueError(
                "Frozen-token-teacher training requires "
                "--factorized-token-speaker-output true"
            )
        if not params.factorized_ctc_loss:
            raise ValueError(
                "Frozen-token-teacher training requires --factorized-ctc-loss true"
            )
        if params.token_only_segment_factor:
            raise ValueError(
                "Frozen-token-teacher training cannot use the segment factor"
            )
        if params.sort_strategy != "start_time":
            raise ValueError(
                "Frozen-token-teacher speaker labels are FIFO; use "
                "--sort-strategy start_time"
            )
        if params.teacher_local_mask_window_duration <= 0:
            raise ValueError("Teacher local-mask window must be positive")
        if params.teacher_local_mask_boundary_collar < 0:
            raise ValueError("Teacher local-mask collar must be non-negative")
    #params.master_port = int(os.environ.get("MASTER_PORT", 12355))
    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")
    logging.info(params)

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None


    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)

    if params.diarization_loss == "none":
        graph_compiler = MDCTCGraphCompiler(
            params.lang_dir,
            device='cpu',
        )
        params.vocab_size = graph_compiler.sp.vocab_size()
    else:
        graph_compiler = None
        params.vocab_size = 0
    params.subsampling_factor = 4 if params.downsample else 2
    logging.info("About to create model")

    model = get_mdctc_model(params)
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        if isinstance(model, FrozenTokenTeacherSpeakerModel):
            # The immutable 4.8GB teacher is neither averaged nor checkpointed.
            # Detach it before deepcopy so rank zero does not create a useless
            # second CPU copy for model_avg.
            token_teacher = model.release_token_teacher()
            try:
                model_avg = copy.deepcopy(model).to(torch.float64)
            finally:
                model.attach_token_teacher(token_teacher)
        else:
            model_avg = copy.deepcopy(model).to(torch.float64)
 
    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if isinstance(model, FrozenTokenTeacherSpeakerModel):
        model.move_token_teacher(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
     
    parameters_names = []
    parameters_names.append(
        [name_param_pair[0] for name_param_pair in model.named_parameters()]
    )
    
    if params.optimizer_name == "scaled_adam":
        if params.lr_scheduler_name != "eden2":
            raise ValueError("ScaledAdam currently requires --lr-scheduler-name eden2")
        from optim import Eden2, ScaledAdam

        optimizer = ScaledAdam(
            get_parameter_groups_with_lrs(
                model, lr=params.base_lr, include_names=True
            ),
            lr=params.base_lr,
            clipping_scale=2.0,
        )
        scheduler = Eden2(
            optimizer,
            lr_batches=params.lr_batches,
            warmup_batches=params.lr_warmup_updates,
            warmup_start=params.lr_warmup_start,
        )
        scheduler.step_batch(params.batch_idx_train)
        init_lr = scheduler.get_last_lr()[0]
    else:
        if params.lr_scheduler_name != "one_cycle":
            raise ValueError("Adam currently requires --lr-scheduler-name one_cycle")
        optimizer = Adam(
            list(filter(lambda p: p.requires_grad, model.parameters())),
            lr=params.base_lr,
            betas=(0.9, 0.98),
            eps=1e-08,
            weight_decay=params.weight_decay,
        )

        if params.freeze_lr <= 0:
            raise ValueError(
                f"Expected --freeze-lr > 0, got {params.freeze_lr}"
            )
        unfrozen_steps = params.total_steps - params.freeze_iters
        if unfrozen_steps < 1:
            raise ValueError(
                "--total-steps must be larger than --freeze-iters; got "
                f"{params.total_steps} and {params.freeze_iters}"
            )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=params.base_lr,
            total_steps=unfrozen_steps,
            pct_start=params.pct_start,
            anneal_strategy="cos",
            # Newer pretrained-encoder recipes use a low freeze LR and expect
            # a continuous transition.  The original WavLM recipe instead
            # trains its newly initialized heads at a much larger freeze LR;
            # retain its historical OneCycle start factor in that case.
            div_factor=(
                params.base_lr / params.freeze_lr
                if params.freeze_lr <= params.base_lr
                else 200.0
            ),
        )
        init_lr = scheduler.get_lr()[0]
    if params.batch_idx_train < params.freeze_iters:
        if isinstance(model, DDP):
            model.module.freeze_encoder()
        else:
            model.freeze_encoder()
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.freeze_lr 

    #optimizer = ScaledAdam(
    #    model.parameters(),
    #    lr=params.base_lr,
    #    clipping_scale=2.0,
    #    parameters_names=parameters_names,
    #)

    #scheduler = Eden(
    #    optimizer,
    #    params.lr_batches,
    #    params.lr_epochs,
    #    warmup_batches=1000
    #)

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    args.return_cuts = False
    librispeech = LibriSpeechAsrDataModule(args)

    if params.validation_only:
        if params.ami:
            valid_cuts = librispeech.valid_ami_cuts()
        else:
            valid_cuts = librispeech.valid_cuts(params.no_overlap)

        def keep_validation_cut(c: Cut):
            overlaps = [len(c.supervisions) for c in c.cut_into_windows(0.1)]
            max_overlap = max(overlaps)
            num_ge_max = len([o for o in overlaps if o >= params.max_overlap])
            return (
                1.0 <= c.duration <= 60.0
                and len(CutSet([c]).speakers) <= params.max_num_spks
                and num_ge_max <= 10
                and max_overlap <= params.max_overlap
            )

        valid_dl = librispeech.valid_dataloaders(valid_cuts.filter(keep_validation_cut))
        valid_info = compute_validation_loss(
            params=params,
            model=model,
            graph_compiler=graph_compiler,
            valid_dl=valid_dl,
            world_size=world_size,
        )
        logging.info("Validation-only result: %s", valid_info)
        return

    if params.ami:
        train_cuts = librispeech.train_ami_cuts()
    else:
        train_cuts = librispeech.train_cuts()

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = copy.deepcopy(checkpoints["sampler"])
        # Resource-related sampler settings are intentionally controlled by
        # the current command line. This permits a memory-safe resume after an
        # OOM instead of silently restoring the failing batch-size settings.
        sampler_state_dict["max_duration"] = params.max_duration
        sampler_state_dict["num_cuts_for_bins_estimate"] = (
            params.num_cuts_for_bins_estimate
        )
        sampler_state_dict["buffer_size"] = params.sampler_buffer_size
    else:
        sampler_state_dict = None


    def remove_short_and_long_utt(c: Cut):
        if not (1.0 <= c.duration <= params.max_train_cut_duration):
            return False
        if len(CutSet([c]).speakers) > params.max_num_spks:
            return False
        overlaps = [len(c.supervisions) for c in c.cut_into_windows(0.1)]
        max_overlap = max(overlaps)
        # Do not reject cuts merely because they sustain the configured
        # maximum concurrency. FastMSS intentionally contains regions with
        # four simultaneous speakers; --max-overlap is the upper bound.
        return max_overlap <= params.max_overlap

    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    train_dl = librispeech.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )

    if params.ami:
        valid_cuts = librispeech.valid_ami_cuts()
    else:
        valid_cuts = librispeech.valid_cuts(params.no_overlap)
    valid_cuts = valid_cuts.filter(remove_short_and_long_utt)
    valid_dl = librispeech.valid_dataloaders(valid_cuts)

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    fix_random_seed(params.seed)
    train_dl.sampler.set_epoch(0)

    params.cur_epoch = 0

    train_one_epoch(
        params=params,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
        graph_compiler=graph_compiler,
        train_dl=train_dl,
        valid_dl=valid_dl,
        scaler=scaler,
        tb_writer=tb_writer,
        world_size=world_size,
        rank=rank,
        init_lr=init_lr,
    )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()

 
#def display_and_save_batch(
#    batch: dict,
#    params: AttributeDict,
#    graph_compiler: MDCTCGraphCompiler,
#) -> None:
#    """Display the batch statistics and save the batch into disk.
#
#    Args:
#      batch:
#        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
#        for the content in it.
#      params:
#        Parameters for training. See :func:`get_params`.
#      graph_compiler:
#        It is used to build a decoding graph from a ctc topo and training
#        transcript. The training transcript is contained in the given `batch`,
#        while the ctc topo is built when this compiler is instantiated.
#    """
#    from lhotse.utils import uuid4
#
#    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
#    logging.info(f"Saving batch to {filename}")
#    torch.save(batch, filename)
#
#    supervisions = batch["supervisions"]
#    features = batch["inputs"]
#
#    logging.info(f"features shape: {features.shape}")
#
#    y = graph_compiler.texts_to_ids(supervisions["text"])
#    num_tokens = sum(len(i) for i in y)
#    logging.info(f"num tokens: {num_tokens}")


def scan_pessimistic_batches_for_oom(
    model: nn.Module,
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    graph_compiler: MDCTCGraphCompiler,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 0 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            optimizer.zero_grad()
            loss, _ = compute_loss(
                params=params,
                model=model,
                batch=batch,
                graph_compiler=graph_compiler,
                is_training=True,
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0, 2.0)
            optimizer.step()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            raise


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)
#    rank = int(os.environ.get("RANK", 0))
#    world_size = int(os.environ.get("WORLD_SIZE", 1))
#    args.master_port = int(os.environ.get("MASTER_PORT", 12355))
#    run(rank=rank, world_size=world_size, args=args)


if __name__ == "__main__":
    main()
