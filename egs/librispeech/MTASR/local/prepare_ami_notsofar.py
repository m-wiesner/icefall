#!/usr/bin/env python3
"""Prepare DiCoW-style AMI/NOTSOFAR SDM CutSets without copying audio."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

from lhotse import CutSet, Recording, SupervisionSegment, fastcopy
from lhotse.recipes.ami import prepare_ami
from lhotse.supervision import AlignmentItem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text_normalization import normalize_english_transcript  # noqa: E402


def _activity_groups(supervisions, max_pause: float):
    groups = []
    end = None
    for supervision in sorted(supervisions, key=lambda item: (item.start, item.end)):
        if end is None or supervision.start - end > max_pause:
            groups.append([supervision])
            end = supervision.end
        else:
            groups[-1].append(supervision)
            end = max(end, supervision.end)
    return groups


def _alignment_fragment(supervision, begin: float, end: float, is_last: bool):
    alignments = (supervision.alignment or {}).get("word", [])
    selected = [
        item
        for item in alignments
        if begin <= item.start + item.duration / 2 < end
        or (is_last and item.start + item.duration / 2 == end)
    ]
    if selected:
        text = normalize_english_transcript(" ".join(item.symbol for item in selected))
        if not text:
            return None
        start = max(begin, min(item.start for item in selected))
        stop = min(end, max(item.end for item in selected))
        clipped = [
            AlignmentItem(
                symbol=item.symbol,
                start=max(begin, item.start) - begin,
                duration=max(
                    1.0e-3,
                    min(end, item.end) - max(begin, item.start),
                ),
            )
            for item in selected
        ]
        return text, start, stop, {"word": clipped}

    # NOTSOFAR and AMI normally have word timings. Retain an unaligned segment
    # in exactly one window as a conservative fallback.
    midpoint = supervision.start + supervision.duration / 2
    if not (begin <= midpoint < end or (is_last and midpoint == end)):
        return None
    text = normalize_english_transcript(supervision.text)
    if not text:
        return None
    return (
        text,
        max(begin, supervision.start),
        min(end, supervision.end),
        None,
    )


def segment_cut(cut, max_duration: float, max_pause: float, stats: Counter):
    """Yield active, normalized windows with each aligned word exactly once."""
    for group_index, group in enumerate(
        _activity_groups(cut.supervisions, max_pause=max_pause)
    ):
        group_start = min(item.start for item in group)
        group_end = max(item.end for item in group)
        window_start = group_start
        window_index = 0
        while window_start < group_end:
            window_end = min(window_start + max_duration, group_end)
            is_last = window_end >= group_end - 1.0e-7
            fragments = []
            for supervision in group:
                if supervision.end <= window_start or supervision.start >= window_end:
                    continue
                fragment = _alignment_fragment(
                    supervision, window_start, window_end, is_last
                )
                if fragment is None:
                    continue
                text, start, stop, alignment = fragment
                if stop <= start:
                    continue
                fragments.append(
                    fastcopy(
                        supervision,
                        id=(
                            f"{supervision.id}-w{group_index:04d}"
                            f"-{window_index:04d}"
                        ),
                        start=start - window_start,
                        duration=stop - start,
                        text=text,
                        alignment=alignment,
                        channel=0,
                    )
                )
            if fragments:
                piece = cut.truncate(
                    offset=window_start,
                    duration=window_end - window_start,
                    keep_excessive_supervisions=False,
                    preserve_id=True,
                )
                # Truncation rounds the requested boundary to an audio sample.
                # Clamp supervision ends to that exact realized duration (most
                # visible for SBCSAE's 44.1-kHz recordings).
                bounded_fragments = []
                for fragment in fragments:
                    start = max(0.0, min(fragment.start, piece.duration - 1.0e-6))
                    stop = min(fragment.end, piece.duration)
                    if stop > start:
                        bounded_fragments.append(
                            fastcopy(fragment, start=start, duration=stop - start)
                        )
                if not bounded_fragments:
                    window_start = window_end
                    window_index += 1
                    continue
                piece = fastcopy(
                    piece,
                    id=f"{cut.id}-g{group_index:04d}-w{window_index:04d}",
                    supervisions=bounded_fragments,
                )
                stats["cuts"] += 1
                stats["supervisions"] += len(bounded_fragments)
                stats["words"] += sum(
                    len(item.text.split()) for item in bounded_fragments
                )
                stats["max_speakers"] = max(
                    stats["max_speakers"],
                    len({item.speaker for item in bounded_fragments}),
                )
                yield piece
            window_start = window_end
            window_index += 1


def write_segmented(cuts: CutSet, output: Path, max_duration: float, max_pause: float):
    stats = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    segmented = CutSet.from_cuts(
        piece
        for cut in cuts
        for piece in segment_cut(cut, max_duration, max_pause, stats)
    )
    segmented.to_file(output)
    summary = dict(stats)
    summary["max_duration_seconds"] = max_duration
    summary["max_pause_seconds"] = max_pause
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    logging.info("Wrote %s: %s", output, summary)


def prepare_ami_sdm(root: Path, output_dir: Path):
    raw_dir = output_dir / "ami"
    manifests = prepare_ami(
        data_dir=root,
        annotations_dir=root / "annotations",
        output_dir=raw_dir,
        mic="sdm",
        partition="full-corpus-asr",
        normalize_text="none",
    )
    for split in ("train", "dev", "test"):
        cutset = CutSet.from_manifests(**manifests[split])
        raw = raw_dir / f"ami_sdm_{split}.jsonl.gz"
        cutset.to_file(raw)
        write_segmented(
            cutset,
            raw_dir / f"ami_sdm_{split}_max30_normalized.jsonl.gz",
            max_duration=30.0,
            max_pause=2.0,
        )


def _notsofar_versions(root: Path):
    mapping = {"train_set": "train", "dev_set": "dev", "eval_set": "test"}
    for part_dir, split in mapping.items():
        parent = root / part_dir
        for version in sorted(path for path in parent.iterdir() if path.is_dir()):
            yield split, version.name, version / "MTG"


def _notsofar_cut(meeting_dir: Path, device_dir: Path, split: str, version: str):
    meeting = meeting_dir.name
    device = device_dir.name
    recording_id = f"nsf-{split}-{meeting}-{device}"
    recording = Recording.from_file(device_dir / "ch0.wav", recording_id=recording_id)
    transcription = json.loads((meeting_dir / "gt_transcription.json").read_text())
    supervisions = []
    for index, segment in enumerate(transcription):
        begin = float(segment["start_time"])
        end = float(segment["end_time"])
        alignments = []
        for symbol, word_begin, word_end in segment.get("word_timing", []):
            if "<" in symbol or ">" in symbol:
                continue
            word_begin = float(word_begin)
            word_end = float(word_end)
            if word_end > word_begin:
                alignments.append(
                    AlignmentItem(
                        symbol=symbol,
                        start=word_begin,
                        duration=word_end - word_begin,
                    )
                )
        text = normalize_english_transcript(segment["text"])
        if end <= begin or not text:
            continue
        supervisions.append(
            SupervisionSegment(
                id=f"{recording_id}-sup{index:05d}",
                recording_id=recording_id,
                start=begin,
                duration=end - begin,
                channel=0,
                text=text,
                language="English",
                speaker=segment["speaker_id"],
                alignment={"word": alignments},
                custom={"corpus": "NOTSOFAR", "split": split, "version": version},
            )
        )
    return fastcopy(recording.to_cut(), supervisions=supervisions)


def prepare_notsofar_sdm(root: Path, output_dir: Path):
    raw_dir = output_dir / "notsofar"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for split, version, meetings_dir in _notsofar_versions(root):
        cuts = []
        for meeting_dir in sorted(path for path in meetings_dir.iterdir() if path.is_dir()):
            for device_dir in sorted(meeting_dir.glob("sc_*")):
                audio = device_dir / "ch0.wav"
                if audio.is_file():
                    cuts.append(
                        _notsofar_cut(meeting_dir, device_dir, split, version)
                    )
        cutset = CutSet.from_cuts(cuts)
        tag = f"notsofar_sdm_{split}_{version}"
        raw = raw_dir / f"{tag}.jsonl.gz"
        cutset.to_file(raw)
        write_segmented(
            cutset,
            raw_dir / f"{tag}_max30_normalized.jsonl.gz",
            max_duration=30.0,
            max_pause=2.0,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ami-root",
        type=Path,
        default=Path(os.environ["AMI_ROOT"]) if "AMI_ROOT" in os.environ else None,
    )
    parser.add_argument(
        "--notsofar-root",
        type=Path,
        default=(
            Path(os.environ["NOTSOFAR_ROOT"])
            if "NOTSOFAR_ROOT" in os.environ
            else None
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/manifests/real_meetings")
    )
    parser.add_argument("--corpus", choices=("all", "ami", "notsofar"), default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.corpus in ("all", "ami"):
        if args.ami_root is None:
            parser.error("--ami-root or AMI_ROOT is required for AMI preparation")
        prepare_ami_sdm(args.ami_root, args.output_dir)
    if args.corpus in ("all", "notsofar"):
        if args.notsofar_root is None:
            parser.error(
                "--notsofar-root or NOTSOFAR_ROOT is required for NOTSOFAR preparation"
            )
        prepare_notsofar_sdm(args.notsofar_root, args.output_dir)


if __name__ == "__main__":
    main()
