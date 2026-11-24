#!/usr/bin/env python3
"""Keep aligned supervisions and export score_alignments.py references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lhotse import CutSet, fastcopy

from text_normalization import normalize_english_transcript


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cuts-output", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument(
        "--cut-id-suffix",
        type=str,
        default=None,
        help="Keep only cuts whose IDs end with this suffix.",
    )
    parser.add_argument(
        "--recording-id-contains",
        type=str,
        default=None,
        help="Keep only cuts whose recording ID contains this string.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Keep only cuts with at most this many distinct supervision speakers.",
    )
    parser.add_argument(
        "--replace-text-with-alignment",
        action="store_true",
        help=(
            "Replace supervision text with the normalized sequence of words "
            "that actually have reference timings. This is needed for raw "
            "NOTSOFAR manifests where some transcript words are unaligned."
        ),
    )
    args = parser.parse_args()

    source = CutSet.from_file(args.input)
    cuts = []
    references = []
    dropped_supervisions = 0
    words = 0
    for cut in source:
        if args.cut_id_suffix is not None and not cut.id.endswith(args.cut_id_suffix):
            continue
        if (
            args.recording_id_contains is not None
            and args.recording_id_contains not in cut.recording_id
        ):
            continue
        if (
            args.max_speakers is not None
            and len({s.speaker for s in cut.supervisions}) > args.max_speakers
        ):
            continue
        aligned = []
        rows_by_supervision = []
        for supervision in cut.supervisions:
            alignment = (supervision.alignment or {}).get("word", [])
            normalized_items = []
            for item in alignment:
                pieces = normalize_english_transcript(item.symbol).split()
                for piece_index, piece in enumerate(pieces):
                    # Hyphen removal can turn one timed annotation into multiple
                    # normalized words. Divide its interval evenly so the
                    # normalized transcript and reference remain identical.
                    normalized_items.append(
                        (
                            piece,
                            item.start + item.duration * piece_index / len(pieces),
                            item.duration / len(pieces),
                        )
                    )
            if not normalized_items:
                dropped_supervisions += 1
                continue
            normalized = [word for word, _, _ in normalized_items]
            normalized_text = " ".join(normalized)
            if normalized_text != supervision.text and not args.replace_text_with_alignment:
                raise ValueError(
                    f"Alignment text mismatch in {supervision.id}: "
                    f"{supervision.text!r} != {normalized_text!r}"
                )
            if args.replace_text_with_alignment:
                supervision = fastcopy(supervision, text=normalized_text)
            aligned.append(supervision)
            supervision_rows = []
            for word, start, duration in normalized_items:
                supervision_rows.append(
                    [
                        supervision.speaker,
                        None,
                        word,
                        start,
                        duration,
                    ]
                )
                words += 1
            rows_by_supervision.append(supervision_rows)
        if aligned:
            speaker_durations = {}
            for supervision in aligned:
                speaker_durations.setdefault(supervision.speaker, 0.0)
                speaker_durations[supervision.speaker] += supervision.duration
            speakers = sorted(
                speaker_durations,
                key=speaker_durations.get,
                reverse=True,
            )
            sort_order = [
                index
                for speaker in speakers
                for index, supervision in enumerate(aligned)
                if supervision.speaker == speaker
            ]
            rows = []
            for segment_index, supervision_index in enumerate(sort_order):
                for row in rows_by_supervision[supervision_index]:
                    row[1] = segment_index
                    rows.append(row)
            cuts.append(fastcopy(cut, supervisions=aligned))
            references.append([cut.id, rows])

    args.cuts_output.parent.mkdir(parents=True, exist_ok=True)
    args.reference_output.parent.mkdir(parents=True, exist_ok=True)
    CutSet.from_cuts(cuts).to_file(args.cuts_output)
    references.sort(key=lambda item: item[0])
    args.reference_output.write_text(json.dumps(references, indent=2) + "\n")
    print(
        f"cuts={len(cuts)} words={words} "
        f"dropped_unaligned_supervisions={dropped_supervisions}"
    )


if __name__ == "__main__":
    main()
