#!/usr/bin/env python3
"""Reproducibly prepare a LibriSpeech n-gram HLG decoding graph."""

import argparse
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from compile_hlg import compile_HLG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lm-dir", type=Path, required=True,
                        help="OpenSLR SLR11 directory containing the ARPA LM.")
    parser.add_argument("--lang-dir", type=Path, default=Path("data/lang_bpe_5000"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/lm"))
    parser.add_argument("--arpa", default="3-gram.pruned.1e-7.arpa.gz")
    parser.add_argument("--lm-name", default="G_3_gram_1e_7")
    parser.add_argument("--output", type=Path,
                        default=Path("data/lang_bpe_5000/HLG_3_gram_1e_7.pt"))
    parser.add_argument("--modified-topology", action="store_true",
                        help="Use k2's modified topology (not identical to training).")
    parser.add_argument(
        "--training-topology",
        action="store_true",
        help="Use the exact sparse CTC topology used by shuffle-CTC training.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.lm_dir / args.arpa
    if not source.is_file():
        raise FileNotFoundError(source)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    arpa_input = source
    if source.suffix == ".gz":
        arpa_input = args.work_dir / source.stem
        if args.force or not arpa_input.is_file():
            with gzip.open(source, "rb") as src, arpa_input.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    fst_txt = args.work_dir / f"{args.lm_name}.fst.txt"
    if args.force or not fst_txt.is_file() or fst_txt.stat().st_size == 0:
        command = [
            sys.executable, "-m", "kaldilm",
            f"--read-symbol-table={args.lang_dir / 'words.txt'}",
            "--disambig-symbol=#0", "--max-order=3",
            str(arpa_input),
        ]
        with fst_txt.open("w") as out:
            subprocess.run(command, check=True, stdout=out)

    # compile_HLG uses the conventional data/lm location and caches G as .pt.
    if args.modified_topology and args.training_topology:
        raise ValueError("Select only one of --modified-topology/--training-topology")
    hlg = compile_HLG(
        str(args.lang_dir),
        lm=args.lm_name,
        modified=args.modified_topology,
        training_topology=args.training_topology,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hlg.as_dict(), args.output)
    metadata = {
        "source_arpa": str(source.resolve()),
        "lm_name": args.lm_name,
        "lang_dir": str(args.lang_dir.resolve()),
        "fst_text": str(fst_txt.resolve()),
        "output": str(args.output.resolve()),
        "modified_topology": args.modified_topology,
        "training_topology": args.training_topology,
        "kaldilm_command": command if 'command' in locals() else "reused existing FST",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
