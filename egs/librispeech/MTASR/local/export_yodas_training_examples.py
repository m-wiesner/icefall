#!/usr/bin/env python3
"""Export exact lazily grouped YODAS training examples as ordinary WAVs."""

import argparse
import csv
from pathlib import Path

import soundfile as sf

from wavlm_spashl_ctc.asr_datamodule import _YodasSharGroups


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-cuts", type=int, default=12)
    parser.add_argument("--min-duration", type=float, default=20.0)
    parser.add_argument("--max-duration", type=float, default=60.0)
    parser.add_argument(
        "--unique-recordings",
        action="store_true",
        help="Export at most one grouped cut from each parent recording.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = _YodasSharGroups(
        scheme_dir=args.scheme_dir,
        hypothesis="doverlap",
        max_pause=1.0,
        max_duration=60.0,
        min_speakers=2,
        max_speakers=8,
    )

    rows = []
    seen_recordings = set()
    for cut in groups:
        if not args.min_duration <= cut.duration <= args.max_duration:
            continue
        source_recording_id = cut.recording_id.split("__chunk", maxsplit=1)[0]
        if args.unique_recordings and source_recording_id in seen_recordings:
            continue
        speakers = sorted({str(s.speaker) for s in cut.supervisions})
        wav_path = args.output_dir / f"{len(rows):02d}_{cut.id}.wav"
        audio = cut.load_audio()
        sf.write(wav_path, audio[0], cut.sampling_rate, subtype="PCM_16")
        rows.append(
            {
                "file": wav_path.name,
                "cut_id": cut.id,
                "recording_id": cut.recording_id,
                "start": f"{cut.start:.3f}",
                "duration": f"{cut.duration:.3f}",
                "num_speakers": len(speakers),
                "speakers": " ".join(speakers),
                "num_supervisions": len(cut.supervisions),
                "transcript": " | ".join(s.text for s in cut.supervisions),
            }
        )
        seen_recordings.add(source_recording_id)
        if len(rows) >= args.num_cuts:
            break

    if len(rows) != args.num_cuts:
        raise RuntimeError(f"Found only {len(rows)} matching cuts")
    with (args.output_dir / "metadata.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} cuts to {args.output_dir}")


if __name__ == "__main__":
    main()
