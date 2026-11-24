#!/usr/bin/env python3
"""Train token-only shuffle CTC with a Parakeet TDT encoder."""

from __future__ import annotations

import sys
from pathlib import Path

_recipe_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_recipe_root))
sys.path.insert(0, str(_recipe_root / "wavlm_spashl_ctc"))

import wavlm_spashl_ctc.train as base
import parakeet_spashl_ctc.train as speaker_recipe
from parakeet_spashl_ctc.train import *  # noqa: F401,F403


_speaker_get_parser = speaker_recipe.get_parser


def get_parser():
    parser = _speaker_get_parser()
    parser.set_defaults(speaker_branch=False)
    return parser


def get_mdctc_model(params):
    params.speaker_branch = False
    return speaker_recipe.get_mdctc_model(params)


base.get_parser = get_parser
base.get_encoder_model = speaker_recipe.get_encoder_model
base.get_mdctc_model = get_mdctc_model


if __name__ == "__main__":
    base.main()
