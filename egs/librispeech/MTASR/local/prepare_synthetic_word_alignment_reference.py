#!/usr/bin/env python3
"""Export word-timing references from synthetic Lhotse cuts."""

import argparse
import json
from pathlib import Path

from lhotse import CutSet

from text_normalization import normalize_english_transcript


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize alignment symbols instead of preserving the canonical words.",
    )
    args = parser.parse_args()

    references = []
    for cut in CutSet.from_file(args.cuts):
        durations = {}
        for supervision in cut.supervisions:
            durations.setdefault(supervision.speaker, 0.0)
            durations[supervision.speaker] += supervision.duration

        # This is the recipe's historical speech_time ordering: speakers are
        # globally numbered by decreasing total speech duration, while each
        # speaker's supervision order is retained.
        speakers = sorted(durations, key=durations.get, reverse=True)
        ordered = [
            supervision
            for speaker in speakers
            for supervision in cut.supervisions
            if supervision.speaker == speaker
        ]

        rows = []
        for segment_index, supervision in enumerate(ordered):
            for item in (supervision.alignment or {}).get("word", []):
                if args.normalize:
                    words = normalize_english_transcript(item.symbol).split()
                else:
                    words = item.symbol.split()
                for word_index, word in enumerate(words):
                    duration = item.duration / len(words)
                    rows.append([
                        supervision.speaker,
                        segment_index,
                        word,
                        supervision.start + item.start + word_index * duration,
                        duration,
                    ])
        references.append([cut.id, rows])

    references.sort(key=lambda item: item[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(references, indent=2) + "\n")
    print(f"cuts={len(references)} output={args.output}")


if __name__ == "__main__":
    main()
