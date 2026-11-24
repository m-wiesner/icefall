#!/usr/bin/env python3
"""Train a Whisper encoder with factorized Shuffle-CTC heads."""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path

import torch.nn as nn
from lhotse import CutSet, fastcopy, load_manifest_lazy

_recipe_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_recipe_root))
sys.path.insert(0, str(_recipe_root / "wavlm_spashl_ctc"))

import wavlm_spashl_ctc.train as base
import wavlm_spashl_ctc.asr_datamodule as base_data
from wavlm_spashl_ctc.train import *  # noqa: F401,F403
from icefall.utils import AttributeDict
from local.rechunk_supervision_groups import overlap_components, pack_components
from whisper_spashl_ctc.encoder import WhisperAudioEncoder
from whisper_spashl_ctc.model import MDCTCModel, TokenOnlyMDCTCModel


_base_get_parser = base.get_parser
_encoder_overrides = {}


def configure_whisper_encoder(**kwargs) -> None:
    """Supply encoder identity when reusing the generic decode programs."""
    _encoder_overrides.update(
        {key: value for key, value in kwargs.items() if value is not None}
    )


def get_parser():
    parser = _base_get_parser()
    group = parser.add_argument_group("Whisper encoder")
    group.add_argument(
        "--whisper-model-id", default="openai/whisper-large-v3-turbo"
    )
    group.add_argument("--whisper-revision", default=None)
    group.add_argument(
        "--whisper-valid-cut-manifest",
        default="data/manifests/cuts_librispeech_dev_synth_max30.jsonl.gz",
    )
    return parser


class WhisperAsrDataModule(base.LibriSpeechAsrDataModule):
    @lru_cache()
    def valid_cuts(self, no_overlap: bool = False) -> CutSet:
        if no_overlap:
            return super().valid_cuts(no_overlap=True)
        path = Path(self.args.whisper_valid_cut_manifest)
        if not path.is_file():
            raise FileNotFoundError(f"Missing <=30-second Whisper dev CutSet: {path}")
        logging.info("Using Whisper dev cuts from %s", path)
        return load_manifest_lazy(path)


class WhisperYodasSharGroups(base_data._YodasSharGroups):
    """Create <=30 s YODAS cuts without cutting through an utterance.

    The generic YODAS reader uses fixed-duration windows.  That is acceptable
    for long-context encoders, but can discard a supervision that crosses a
    window boundary.  Whisper instead packs complete overlap-connected
    components and drops only a component that is itself longer than 30 s.
    """

    def _split(self, cut):
        components = overlap_components(cut.supervisions)
        for group_idx, group in enumerate(
            pack_components(
                components,
                max_duration=self.max_duration,
                max_pause=self.max_pause,
            )
        ):
            start = min(supervision.start for supervision in group)
            end = max(supervision.end for supervision in group)
            if end - start > self.max_duration + 1.0e-6:
                logging.warning(
                    "Dropping YODAS overlap component %s/%d: %.3f s > %.3f s",
                    cut.id,
                    group_idx,
                    end - start,
                    self.max_duration,
                )
                continue
            piece = cut.truncate(
                offset=start,
                duration=end - start,
                keep_excessive_supervisions=False,
                preserve_id=True,
            )
            expected_ids = {supervision.id for supervision in group}
            actual_ids = {supervision.id for supervision in piece.supervisions}
            if actual_ids != expected_ids:
                raise RuntimeError(
                    f"YODAS supervision mismatch after truncating {cut.id}"
                )
            num_speakers = len(
                {supervision.speaker for supervision in piece.supervisions}
            )
            if self.min_speakers <= num_speakers <= self.max_speakers:
                yield fastcopy(piece, id=f"{cut.id}-g{group_idx}")


def get_encoder_model(params: AttributeDict) -> tuple[nn.Module, int]:
    encoder = WhisperAudioEncoder.from_pretrained(
        model_id=_encoder_overrides.get(
            "model_id",
            getattr(params, "whisper_model_id", "openai/whisper-large-v3-turbo"),
        ),
        revision=_encoder_overrides.get(
            "revision", getattr(params, "whisper_revision", None)
        ),
    )
    return encoder, encoder.output_dim


def get_mdctc_model(params: AttributeDict) -> nn.Module:
    if params.downsample:
        raise ValueError("Use --downsample false to retain Whisper's 50-Hz output")
    # speaker_branch is a training-only parser option, so alignment programs
    # do not normally define it.  Factorized heads identify the historical
    # SPaSHL checkpoints; a non-factorized checkpoint is the token-only
    # Shuffle model and must instantiate TokenOnlyMDCTCModel when decoding.
    speaker_branch = getattr(
        params, "speaker_branch", params.factorized_token_speaker_output
    )
    if speaker_branch and not params.factorized_token_speaker_output:
        raise ValueError("Whisper Shuffle training requires factorized heads")
    encoder, encoder_dim = get_encoder_model(params)
    if speaker_branch:
        model = MDCTCModel(
            encoder=encoder,
            encoder_dim=encoder_dim,
            vocab_size1=params.vocab_size,
            vocab_size2=params.max_num_spks,
            factorized_token_speaker_output=True,
            layer_norm=params.use_layer_norm,
            downsample=False,
        )
        head_names = ("ctc_output1.", "ctc_output2.", "layer_norm.")
        head_label = "SPaSHL heads"
    else:
        model = TokenOnlyMDCTCModel(
            encoder=encoder,
            encoder_dim=encoder_dim,
            vocab_size=params.vocab_size,
            layer_norm=params.use_layer_norm,
            max_num_spks=params.max_num_spks,
        )
        head_names = ("ctc_output1.", "layer_norm.")
        head_label = "Shuffle token head"
    encoder_parameters = sum(p.numel() for p in encoder.encoder.parameters())
    head_parameters = sum(
        p.numel()
        for name, p in model.named_parameters()
        if name.startswith(head_names)
    )
    logging.info(
        "Parameter breakdown: pretrained Whisper encoder=%s; %s=%s; total=%s",
        f"{encoder_parameters:,}",
        head_label,
        f"{head_parameters:,}",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )
    return model


base.get_parser = get_parser
base.get_encoder_model = get_encoder_model
base.get_mdctc_model = get_mdctc_model
base.MDCTCModel = MDCTCModel
base.LibriSpeechAsrDataModule = WhisperAsrDataModule
base_data._YodasSharGroups = WhisperYodasSharGroups


if __name__ == "__main__":
    base.main()
