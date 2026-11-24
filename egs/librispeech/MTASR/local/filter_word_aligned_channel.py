#!/usr/bin/env python3
"""Filter a word-aligned CutSet to one recording channel and write its reference."""

import argparse
import json

from lhotse import CutSet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-output", required=True)
    parser.add_argument("--recording-suffix", required=True)
    args = parser.parse_args()

    cuts = CutSet.from_file(args.input).filter(
        lambda cut: cut.recording_id.endswith(args.recording_suffix)
    )
    # The source manifests occasionally contain fragments such as partial-word
    # annotations without word timing.  They were excluded from the established
    # Dev evaluation subset and must likewise be excluded from Eval.
    cuts = cuts.map(
        lambda cut: cut.filter_supervisions(
            lambda supervision: bool(supervision.alignment)
            and "word" in supervision.alignment
        )
    ).filter(lambda cut: len(cut.supervisions) > 0).to_eager()
    cuts.to_file(args.output)

    reference = []
    for cut in cuts:
        speaker_durations = {}
        for supervision in cut.supervisions:
            speaker_durations.setdefault(supervision.speaker, 0.0)
            speaker_durations[supervision.speaker] += supervision.duration
        speaker_order = sorted(
            speaker_durations, key=speaker_durations.get, reverse=True
        )
        ordered_supervisions = [
            supervision
            for speaker in speaker_order
            for supervision in cut.supervisions
            if supervision.speaker == speaker
        ]
        rows = []
        for segment, supervision in enumerate(ordered_supervisions):
            for item in supervision.alignment["word"]:
                rows.append(
                    [
                        supervision.speaker,
                        segment,
                        item.symbol.upper(),
                        item.start,
                        item.duration,
                    ]
                )
        reference.append([cut.id, rows])

    with open(args.reference_output, "w") as f:
        json.dump(reference, f)
        f.write("\n")

    print(f"Wrote {len(cuts)} cuts to {args.output}")
    print(f"Wrote {len(reference)} reference alignments to {args.reference_output}")


if __name__ == "__main__":
    main()
