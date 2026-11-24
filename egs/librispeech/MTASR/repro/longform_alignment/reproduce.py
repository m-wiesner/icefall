#!/usr/bin/env python3
"""Check artifacts, inspect results, and submit the paper experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
RECIPE = HERE.parents[1]
REPO = RECIPE.parents[2]
ARTIFACTS = json.loads((HERE / "artifacts.json").read_text())
EXPERIMENTS = json.loads((HERE / "experiments.json").read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_files(checkpoint: Path) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = [("paper checkpoint", checkpoint)]
    lang_dir = RECIPE / ARTIFACTS["language_dir"]
    paths.extend(
        (f"language asset {name}", lang_dir / name)
        for name in ARTIFACTS["language_files"]
    )
    relative_paths = [
        "data/manifests/cuts_librispeech_dev_synth_3spk_8splices.jsonl.gz",
        "data/alignments/alignments_dev_synth_3spk_8splices_speech_length.json",
        "data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz",
        "data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_aligned.jsonl.gz",
        "data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_max30_sc_plaza_0_normalized_word_alignments.json",
        "data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_max30_sc_plaza_0_normalized_word_alignments.json",
        "data/manifests/real_meetings/notsofar/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz",
        "data/manifests/real_meetings/notsofar/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_aligned.jsonl.gz",
        "data/alignments/notsofar_sdm_dev_240415.2_dev_with_GT_sc_plaza_0_full_normalized_word_alignments.json",
        "data/alignments/notsofar_sdm_test_240629.1_eval_small_with_GT_sc_plaza_0_full_normalized_word_alignments.json",
        "data/manifests/real_meetings/ami/ami_sdm_dev_full_normalized_word_aligned.jsonl.gz",
        "data/manifests/real_meetings/ami/ami_sdm_test_full_normalized_word_aligned.jsonl.gz",
        "data/alignments/ami_sdm_dev_full_normalized_word_alignments.json",
        "data/alignments/ami_sdm_test_full_normalized_word_alignments.json",
    ]
    paths.extend(("evaluation input", RECIPE / name) for name in relative_paths)
    return paths


def cmd_check(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    missing = 0
    for description, path in required_files(checkpoint):
        if path.is_file():
            print(f"[OK]      {description}: {path}")
        else:
            print(f"[MISSING] {description}: {path}")
            missing += 1

    if checkpoint.is_file():
        model = ARTIFACTS["paper_model"]
        variants = (
            model["release_artifact"],
            model["source_training_checkpoint"],
        )
        size = checkpoint.stat().st_size
        matching = [variant for variant in variants if size == variant["size_bytes"]]
        if not matching:
            print(
                f"[MISMATCH] checkpoint size: {size}; "
                f"expected one of {[item['size_bytes'] for item in variants]}"
            )
            missing += 1
        elif args.hash_checkpoint:
            actual = sha256(checkpoint)
            if actual != matching[0]["sha256"]:
                print(f"[MISMATCH] checkpoint SHA-256: {actual}")
                missing += 1
            else:
                print(f"[OK]      checkpoint SHA-256: {actual}")

    if missing:
        print(f"\n{missing} required item(s) are missing or mismatched.")
        return 1
    print("\nAll paper artifacts and evaluation inputs are present.")
    return 0


def print_csv(path: Path) -> None:
    with path.open(newline="") as source:
        rows = list(csv.reader(source))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    print(f"\n{path.name}")
    for row_index, row in enumerate(rows):
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))
        if row_index == 0:
            print("  ".join("-" * width for width in widths))


def cmd_list(_: argparse.Namespace) -> int:
    print(json.dumps(EXPERIMENTS, indent=2))
    for table in sorted((HERE / "expected").glob("*.csv")):
        print_csv(table)
    return 0


def command_text(command: Iterable[str], exports: dict[str, str] | None = None) -> str:
    prefix = ""
    if exports:
        prefix = " ".join(f"{key}={value}" for key, value in exports.items()) + " "
    return prefix + " ".join(command)


def submit(
    script: str,
    args: argparse.Namespace,
    *,
    array: str | None = None,
    exports: dict[str, str] | None = None,
    dependency: str | None = None,
) -> str:
    command = ["sbatch", "--parsable"]
    if args.gres:
        command.append(f"--gres={args.gres}")
    if array:
        command.append(f"--array={array}")
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    all_exports = {
        "ALL": "",
        "RECIPE_DIR": str(RECIPE),
        "EXP_DIR": str(Path(args.exp_dir).expanduser()),
    }
    if exports:
        all_exports.update(exports)
    export_arg = ",".join(
        key if value == "" else f"{key}={value}" for key, value in all_exports.items()
    )
    command.extend([f"--export={export_arg}", str(RECIPE / script)])
    if args.dry_run:
        print(command_text(command))
        return f"dry-{Path(script).stem}"
    result = subprocess.run(
        command, cwd=RECIPE, check=True, text=True, capture_output=True
    )
    job_id = result.stdout.strip().split(";", maxsplit=1)[0]
    print(f"Submitted {script}: {job_id}")
    return job_id


def submit_longform(args: argparse.Namespace) -> None:
    submit(
        EXPERIMENTS["longform_ablation"]["script"],
        args,
        array=EXPERIMENTS["longform_ablation"]["array"],
    )

    full_script = EXPERIMENTS["longform_full_recordings"]["script"]
    for corpus in ("notsofar", "ami"):
        attributed_dev = submit(
            full_script,
            args,
            exports={"CORPUS": corpus, "MODE": "attributed", "SPLIT": "dev"},
        )
        token_dev = submit(
            full_script,
            args,
            exports={"CORPUS": corpus, "MODE": "token", "SPLIT": "dev"},
            dependency=attributed_dev,
        )
        submit(
            full_script,
            args,
            exports={"CORPUS": corpus, "MODE": "attributed", "SPLIT": "test"},
            dependency=attributed_dev,
        )
        submit(
            full_script,
            args,
            exports={"CORPUS": corpus, "MODE": "token", "SPLIT": "test"},
            dependency=token_dev,
        )

    submit(
        EXPERIMENTS["bpe_trim"]["script"],
        args,
        array=EXPERIMENTS["bpe_trim"]["array"],
        exports={"KEEP_FRACTION": "0.60"},
    )


def cmd_submit(args: argparse.Namespace) -> int:
    if not args.dry_run and shutil.which("sbatch") is None:
        print("sbatch is not available; use --dry-run to inspect commands.", file=sys.stderr)
        return 2
    if args.exp_dir is None:
        defaults = {
            "train-speaker": "exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4",
            "train-token": "exp/fastmss_yodas_en_wavlm_large_shuffle_v100x4_accum4",
            "shortform": ARTIFACTS["paper_model"]["destination"].rsplit("/", 1)[0],
            "longform": ARTIFACTS["paper_model"]["destination"].rsplit("/", 1)[0],
        }
        args.exp_dir = defaults[args.target]

    if args.target == "train-speaker":
        submit(
            "slurm/train_fastmss_yodas_wavlm_large_spashl_factorized_v100x4_accum4.sbatch",
            args,
            exports={"YODAS_SHAR_DIR": args.yodas_shar_dir},
        )
    elif args.target == "train-token":
        submit(
            "slurm/train_fastmss_yodas_wavlm_large_shuffle_v100x4_accum4.sbatch",
            args,
            exports={"YODAS_SHAR_DIR": args.yodas_shar_dir},
        )
    elif args.target == "shortform":
        submit(
            EXPERIMENTS["shortform"]["script"],
            args,
            array=EXPERIMENTS["shortform"]["array"],
        )
    elif args.target == "longform":
        submit_longform(args)
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    model = ARTIFACTS["paper_model"]
    release = model["release_artifact"]
    url = args.url or model["download_url"]
    if not url:
        print(
            "No public model URL is recorded yet. Pass --url after uploading the "
            "checkpoint to the project artifact host.",
            file=sys.stderr,
        )
        return 2
    destination = Path(args.destination or RECIPE / model["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    print(f"Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, temporary)
    actual = sha256(temporary)
    if actual != release["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded checkpoint SHA-256 {actual} does not match {release['sha256']}"
        )
    temporary.replace(destination)
    print(f"Verified SHA-256 and installed {destination}")
    return 0


def cmd_paper(_: argparse.Namespace) -> int:
    subprocess.run([REPO / "paper/longform_alignment/build.sh"], check=True)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="check required artifacts")
    check_parser.add_argument(
        "--checkpoint",
        default=RECIPE / ARTIFACTS["paper_model"]["destination"],
        type=Path,
    )
    check_parser.add_argument("--hash-checkpoint", action="store_true")
    check_parser.set_defaults(func=cmd_check)

    list_parser = subparsers.add_parser("list", help="show experiments and results")
    list_parser.set_defaults(func=cmd_list)

    submit_parser = subparsers.add_parser("submit", help="submit paper Slurm jobs")
    submit_parser.add_argument(
        "target", choices=("train-speaker", "train-token", "shortform", "longform")
    )
    submit_parser.add_argument("--dry-run", action="store_true")
    submit_parser.add_argument("--gres", help="override Slurm GRES, e.g. gpu:a100:1")
    submit_parser.add_argument("--exp-dir")
    submit_parser.add_argument(
        "--yodas-shar-dir",
        default=os.environ.get("YODAS_SHAR_DIR", "/path/to/yodas_filtered_en"),
    )
    submit_parser.set_defaults(func=cmd_submit)

    fetch_parser = subparsers.add_parser("fetch-model", help="download and verify model")
    fetch_parser.add_argument("--url")
    fetch_parser.add_argument("--destination", type=Path)
    fetch_parser.set_defaults(func=cmd_fetch)

    paper_parser = subparsers.add_parser("paper", help="build the manuscript")
    paper_parser.set_defaults(func=cmd_paper)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(arguments.func(arguments))
