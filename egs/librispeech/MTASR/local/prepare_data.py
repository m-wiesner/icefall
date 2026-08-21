#!/usr/bin/env python3
"""Prepare LibriSpeech data for the MTASR shuffle-CTC recipe."""

import argparse
import gzip
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from lhotse import CutSet, RecordingSet, SupervisionSet, load_manifest_lazy
from lhotse.recipes.librispeech import download_librispeech, prepare_librispeech


PARTS = (
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
)
TRAIN_PARTS = PARTS[:3]
MFA_URL = (
    "https://zenodo.org/records/2619474/files/"
    "librispeech_alignments.zip?download=1"
)
MFA_MD5 = "2bab567d0ace651a4ba254e813629f46"
LM_FILES = {
    "librispeech-lexicon.txt": (
        "https://www.openslr.org/resources/11/librispeech-lexicon.txt"
    ),
    "3-gram.pruned.3e-7.arpa.gz": (
        "https://www.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz"
    ),
    "4-gram.arpa.gz": "https://www.openslr.org/resources/11/4-gram.arpa.gz",
}


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("download", "manifests", "speakers", "synth", "bpe", "validate"),
        default=("manifests", "speakers", "bpe", "validate"),
        help="Preparation stages to run, in the supplied order.",
    )
    parser.add_argument(
        "--librispeech-dir",
        type=Path,
        help="LibriSpeech directory, or its SLR12 parent directory.",
    )
    parser.add_argument(
        "--lm-dir",
        type=Path,
        help="SLR11 language-model resource directory.",
    )
    parser.add_argument(
        "--alignments-dir",
        type=Path,
        help="Existing extracted MFA alignment directory.",
    )
    parser.add_argument(
        "--synth-manifests",
        nargs="+",
        type=Path,
        help=(
            "Existing synthetic cut manifests to install. Their embedded "
            "LibriSpeech recording paths are relocated to --librispeech-dir."
        ),
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/downloads"),
        help="Destination used by the optional download stage.",
    )
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("data/manifests")
    )
    parser.add_argument(
        "--speaker-dir",
        type=Path,
        default=Path("data/manifests/librispeech_speakers"),
    )
    parser.add_argument("--lang-dir", type=Path, default=Path("data/lang_bpe_5000"))
    parser.add_argument("--num-jobs", type=int, default=8)
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Seed for the deterministic ordering of per-speaker training cuts.",
    )
    parser.add_argument("--vocab-size", type=int, default=5000)
    parser.add_argument(
        "--bpe-mode",
        choices=("preserved", "train", "skip"),
        default="preserved",
        help="Install preserved paper assets, train bpe.model, or skip BPE setup.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate existing stage outputs."
    )
    return parser.parse_args()


def resolve_corpus_dir(path):
    if path is None:
        raise ValueError(
            "Provide --librispeech-dir, or run the download stage first."
        )
    path = path.expanduser().resolve()
    candidates = (path, path / "LibriSpeech")
    for candidate in candidates:
        if all((candidate / part).is_dir() for part in PARTS):
            return candidate
    raise FileNotFoundError(
        f"Could not find all LibriSpeech partitions under {path} or {path / 'LibriSpeech'}"
    )


def download_file(url, destination, expected_md5=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and (
        expected_md5 is None or md5(destination) == expected_md5
    ):
        logging.info("Using existing download %s", destination)
        return
    logging.info("Downloading %s", url)
    urllib.request.urlretrieve(url, destination)
    if expected_md5 is not None and md5(destination) != expected_md5:
        raise RuntimeError(f"Checksum mismatch for {destination}")


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract_zip(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
        stream.extractall(destination)


def run_download(args):
    args.download_dir.mkdir(parents=True, exist_ok=True)
    corpus_parent = args.download_dir / "SLR12"
    download_librispeech(corpus_parent, dataset_parts="all")
    args.librispeech_dir = corpus_parent / "LibriSpeech"

    lm_dir = args.download_dir / "SLR11"
    for name, url in LM_FILES.items():
        download_file(url, lm_dir / name)
    args.lm_dir = lm_dir

    alignment_root = args.download_dir / "mfa_alignments"
    archive = args.download_dir / "librispeech_alignments.zip"
    download_file(MFA_URL, archive, MFA_MD5)
    marker = alignment_root / ".complete"
    if args.force or not marker.is_file():
        safe_extract_zip(archive, alignment_root)
        marker.touch()
    args.alignments_dir = alignment_root


def manifest_paths(manifest_dir, part):
    return (
        manifest_dir / f"librispeech_recordings_{part}.jsonl.gz",
        manifest_dir / f"librispeech_supervisions_{part}.jsonl.gz",
    )


def run_manifests(args):
    corpus_dir = resolve_corpus_dir(args.librispeech_dir)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        part
        for part in PARTS
        if not all(path.is_file() for path in manifest_paths(args.manifest_dir, part))
    ]
    if args.force or missing:
        prepare_librispeech(
            corpus_dir=corpus_dir,
            dataset_parts=PARTS,
            output_dir=args.manifest_dir,
            num_jobs=args.num_jobs,
        )

    for part in PARTS:
        recordings_path, supervisions_path = manifest_paths(args.manifest_dir, part)
        recordings = RecordingSet.from_file(recordings_path)
        supervisions = SupervisionSet.from_file(supervisions_path)
        cuts_path = args.manifest_dir / f"cuts_librispeech_{part}.jsonl.gz"
        if args.force or not cuts_path.is_file():
            CutSet.from_manifests(
                recordings=recordings, supervisions=supervisions
            ).trim_to_supervisions(
                keep_overlapping=False, keep_all_channels=False
            ).to_file(cuts_path)


def run_speakers(args):
    args.speaker_dir.mkdir(parents=True, exist_ok=True)
    marker = args.speaker_dir / ".complete"
    if marker.is_file() and not args.force:
        logging.info("Using existing speaker manifests in %s", args.speaker_dir)
        return
    script = Path(__file__).with_name("split_cuts_by_speaker.py")
    command = [
        sys.executable,
        str(script),
        "--recos",
        *[str(manifest_paths(args.manifest_dir, p)[0]) for p in TRAIN_PARTS],
        "--sups",
        *[str(manifest_paths(args.manifest_dir, p)[1]) for p in TRAIN_PARTS],
        "--trim-to-supervisions",
        "--oname",
        "librispeech",
        "--shuffle-seed",
        str(args.shuffle_seed),
        str(args.speaker_dir),
    ]
    subprocess.run(command, check=True)
    marker.touch()


def relocate_synthetic_manifest(source, destination, corpus_dir):
    """Copy a JSONL manifest while relocating embedded LibriSpeech audio paths."""
    corpus_dir = corpus_dir.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    opener_in = gzip.open if source.suffix == ".gz" else open
    opener_out = gzip.open if destination.suffix == ".gz" else open
    recording_count = 0

    def relocate(value):
        nonlocal recording_count
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str) and "LibriSpeech/" in value:
            relative = value.split("LibriSpeech/", maxsplit=1)[1]
            target = (corpus_dir / relative).resolve()
            try:
                target.relative_to(corpus_dir)
            except ValueError as error:
                raise RuntimeError(f"Unsafe recording path in {source}: {value}") from error
            if not target.is_file():
                raise FileNotFoundError(
                    f"Recording referenced by {source} does not exist: {target}"
                )
            recording_count += 1
            return str(target)
        return value

    try:
        with opener_in(source, "rt", encoding="utf-8") as input_stream, opener_out(
            temporary, "wt", encoding="utf-8"
        ) as output_stream:
            for line in input_stream:
                output_stream.write(json.dumps(relocate(json.loads(line))) + "\n")
        if recording_count == 0:
            raise RuntimeError(f"No embedded LibriSpeech paths found in {source}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    logging.info(
        "Installed %s with %d relocated recording references",
        destination,
        recording_count,
    )


def run_synth(args):
    if not args.synth_manifests:
        raise ValueError("The synth stage requires --synth-manifests.")
    corpus_dir = resolve_corpus_dir(args.librispeech_dir)
    for source in args.synth_manifests:
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Missing synthetic cut manifest: {source}")
        relocate_synthetic_manifest(
            source=source,
            destination=args.manifest_dir / source.name,
            corpus_dir=corpus_dir,
        )


def write_training_text(args):
    transcript = args.lang_dir / "transcript_words.txt"
    if transcript.is_file() and not args.force:
        return transcript
    args.lang_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    with transcript.open("w", encoding="utf-8") as stream:
        for part in TRAIN_PARTS:
            _, supervision_path = manifest_paths(args.manifest_dir, part)
            for supervision in SupervisionSet.from_file(supervision_path):
                if supervision.id not in seen:
                    print(supervision.text, file=stream)
                    seen.add(supervision.id)
    return transcript


def run_bpe(args):
    if args.bpe_mode == "skip":
        return
    if args.bpe_mode == "preserved":
        source = Path(__file__).resolve().parents[1] / "data-lang-bpe-5000"
        if not source.is_dir():
            raise FileNotFoundError(f"Missing preserved language assets: {source}")
        args.lang_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, args.lang_dir, dirs_exist_ok=True)
        return
    transcript = write_training_text(args)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("train_bpe_model.py")),
            "--lang-dir",
            str(args.lang_dir),
            "--transcript",
            str(transcript),
            "--vocab-size",
            str(args.vocab_size),
        ],
        check=True,
    )


def run_validate(args):
    errors = []
    for part in PARTS:
        for path in (*manifest_paths(args.manifest_dir, part),):
            if not path.is_file():
                errors.append(f"missing {path}")
        cuts_path = args.manifest_dir / f"cuts_librispeech_{part}.jsonl.gz"
        if not cuts_path.is_file():
            errors.append(f"missing {cuts_path}")
    speaker_manifests = list(args.speaker_dir.glob("*.jsonl.gz"))
    if not speaker_manifests:
        errors.append(f"no speaker manifests in {args.speaker_dir}")
    if args.bpe_mode != "skip" and not (args.lang_dir / "bpe.model").is_file():
        errors.append(f"missing {args.lang_dir / 'bpe.model'}")
    if args.lm_dir is not None:
        for name in LM_FILES:
            if not (args.lm_dir / name).is_file():
                errors.append(f"missing LM resource {args.lm_dir / name}")
    if args.alignments_dir is not None and not args.alignments_dir.is_dir():
        errors.append(f"missing alignment directory {args.alignments_dir}")
    synthetic_manifests = sorted(
        args.manifest_dir.glob("cuts_librispeech_*_synth*.jsonl.gz")
    )
    for path in synthetic_manifests:
        try:
            for cut in load_manifest_lazy(path):
                cut.load_audio()
        except Exception as error:
            errors.append(f"invalid synthetic manifest {path}: {error}")
    if errors:
        raise RuntimeError("Preparation validation failed:\n  " + "\n  ".join(errors))

    metadata = {
        "librispeech_dir": str(resolve_corpus_dir(args.librispeech_dir)),
        "lm_dir": str(args.lm_dir.resolve()) if args.lm_dir else None,
        "alignments_dir": (
            str(args.alignments_dir.resolve()) if args.alignments_dir else None
        ),
        "parts": list(PARTS),
        "speaker_manifest_count": len(speaker_manifests),
        "speaker_shuffle_seed": args.shuffle_seed,
        "synthetic_manifests": [path.name for path in synthetic_manifests],
        "bpe_mode": args.bpe_mode,
        "vocab_size": args.vocab_size,
        "mfa_source": "https://doi.org/10.5281/zenodo.2619474",
        "mfa_archive_md5": MFA_MD5,
    }
    output = args.manifest_dir / "mtasr_preparation.json"
    output.write_text(json.dumps(metadata, indent=2) + "\n")
    logging.info("Preparation validated; wrote %s", output)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    handlers = {
        "download": run_download,
        "manifests": run_manifests,
        "speakers": run_speakers,
        "synth": run_synth,
        "bpe": run_bpe,
        "validate": run_validate,
    }
    for stage in args.stages:
        logging.info("Running stage: %s", stage)
        handlers[stage](args)


if __name__ == "__main__":
    main()
    "3-gram.pruned.1e-7.arpa.gz": (
        "https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz"
    ),
