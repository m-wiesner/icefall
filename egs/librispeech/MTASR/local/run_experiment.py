#!/usr/bin/env python3
"""Run or print a portable MTASR training preset."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

COMMON = [
    "--bpe-model", "data/lang_bpe_5000/bpe.model",
    "--lang-dir", "data/lang_bpe_5000",
    "--use-large", "true", "--downsample", "false",
    "--use-layer-norm", "false", "--max-num-spks", "4",
    "--max-overlap", "3", "--collar", "64000", "--beam-size", "30",
    "--total-max-overlaps", "6000", "--base-lr", "0.00005",
    "--pct-start", "0.15", "--freeze-iters", "2000",
    "--freeze-lr", "0.005", "--original-topo", "false",
    "--sort-strategy", "speech_time", "--max-splices", "10",
    "--max-splice-duration", "50", "--max-unique", "4",
    "--overlap", "1.0", "--min-overlap", "0.5", "--drift", "0.5",
    "--reverb", "true", "--normalize-loudness", "true",
    "--max-snr", "30", "--allow-self-overlap", "false",
    "--duration-increment", "0.00001", "--num-buckets", "100",
    "--num-cuts-for-bins-estimate", "10000",
    "--sampler-buffer-size", "20000", "--num-workers", "3",
    "--use-fp16", "true", "--save-every-n", "2000",
    "--average-period", "200", "--tensorboard", "true", "--seed", "42",
]

PRESETS = {
    "paper": {
        "trainer": "wavlm_spashl_ctc/train.py",
        "exp_dir": "exp/shuffle_large_length_paper",
        "world_size": 1, "total_steps": 200_000,
        "args": ["--factorized-token-speaker-output", "true",
                 "--speaker-weight", "1.0", "--interval-model", "fixed"],
    },
    "brownian": {
        "trainer": "wavlm_spashl_ctc/train.py",
        "exp_dir": "exp/shuffle_large_length_brownian",
        "world_size": 1, "total_steps": 200_000,
        "args": ["--factorized-token-speaker-output", "true",
                 "--speaker-weight", "1.0",
                 "--interval-model", "brownian_bridge", "--edge-collar", "0"],
    },
    "four-gpu-400k": {
        "trainer": "wavlm_spashl_ctc/train.py",
        "exp_dir": "exp/shuffle_large_length_4gpu_400k",
        "world_size": 4, "total_steps": 400_000,
        "args": ["--factorized-token-speaker-output", "true",
                 "--speaker-weight", "1.0", "--interval-model", "fixed"],
    },
    "token-only": {
        "trainer": "wavlm_shuffle_ctc/train.py",
        "exp_dir": "exp/shuffle_only_large_length",
        "world_size": 1, "total_steps": 200_000,
        "args": ["--interval-model", "fixed"],
    },
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=PRESETS)
    parser.add_argument("--exp-dir", type=Path)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--max-duration", type=float, default=300)
    parser.add_argument("--print-only", action="store_true")
    args, extra_args = parser.parse_known_args()
    args.extra_args = extra_args[1:] if extra_args[:1] == ["--"] else extra_args
    return args


def main() -> None:
    args = get_args()
    preset = PRESETS[args.preset]
    exp_dir = args.exp_dir or Path(preset["exp_dir"])
    world_size = args.world_size or preset["world_size"]
    total_steps = args.total_steps or preset["total_steps"]
    command = [
        sys.executable, preset["trainer"],
        "--world-size", str(world_size), "--exp-dir", str(exp_dir),
        "--total-steps", str(total_steps),
        "--max-duration", str(args.max_duration),
        *COMMON, *preset["args"], *args.extra_args,
    ]
    print(shlex.join(command))
    if not args.print_only:
        exp_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
