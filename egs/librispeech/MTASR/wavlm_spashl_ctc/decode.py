#!/usr/bin/env python3
"""Decode with a speaker-attributed shuffle-CTC model."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

METHODS = {
    "greedy": "decode_greedy.py",
    "hlg": "decode_hlg_algorithm.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="greedy")
    parser.add_argument("--method-help", action="store_true", help="Show the selected algorithm's arguments and exit.")
    args, method_args = parser.parse_known_args()
    target = Path(__file__).with_name(METHODS[args.method])
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{target.parent}{os.pathsep}{env.get('PYTHONPATH', '')}"
    if method_args[:1] == ["--"]:
        method_args = method_args[1:]
    if args.method_help:
        method_args = ["--help"]
    raise SystemExit(subprocess.call([sys.executable, str(target), *method_args], env=env))


if __name__ == "__main__":
    main()
