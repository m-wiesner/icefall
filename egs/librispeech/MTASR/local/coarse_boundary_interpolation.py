#!/usr/bin/env python3
"""Create a non-acoustic word-alignment baseline from supervision boundaries.

For every supervision, this script assigns equal duration to each transcript
word and tiles those word intervals across the complete coarse supervision
interval.  The output uses the same JSON representation as ``ali_to_word_ali``
and can therefore be scored directly with ``local/score_alignments.py``.

Reference word timestamps are used only to recover the evaluation's segment
identifiers and word instances.  They are never used to place a word in time.
The coarse start and end times always come from the Lhotse supervision.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Sequence, Tuple

from lhotse import CutSet


TokenRow = List[Any]  # [speaker, segment, word, start, duration]
Signature = Tuple[str, Tuple[str, ...]]


def segment_rows(rows: Sequence[TokenRow]) -> List[Tuple[Any, List[TokenRow]]]:
    """Group rows by segment while preserving first-appearance order."""
    grouped: Dict[Any, List[TokenRow]] = {}
    for row in rows:
        grouped.setdefault(row[1], []).append(row)
    return list(grouped.items())


def signature(speaker: Any, words: Iterable[str]) -> Signature:
    return str(speaker), tuple(words)


def interpolate_cut(cut, ref_rows: Sequence[TokenRow]) -> List[TokenRow]:
    """Interpolate one cut using only its supervision start/end boundaries."""
    available: Dict[Signature, Deque[Any]] = defaultdict(deque)
    for supervision in cut.supervisions:
        words = tuple((supervision.text or "").split())
        available[signature(supervision.speaker, words)].append(supervision)

    output: List[TokenRow] = []
    for segment, rows in segment_rows(ref_rows):
        speakers = {row[0] for row in rows}
        if len(speakers) != 1:
            raise ValueError(
                f"{cut.id}: segment {segment!r} has multiple speakers: {speakers}"
            )
        speaker = next(iter(speakers))
        words = [row[2] for row in rows]
        key = signature(speaker, words)
        if not available[key]:
            same_speaker = [
                (sup.id, sup.start, sup.duration, sup.text)
                for sup in cut.supervisions
                if str(sup.speaker) == str(speaker)
            ]
            raise ValueError(
                f"{cut.id}: could not match reference segment {segment!r} "
                f"with speaker={speaker!r}, text={' '.join(words)!r}. "
                f"Same-speaker supervisions: {same_speaker}"
            )

        supervision = available[key].popleft()
        word_duration = float(supervision.duration) / len(words)
        for index, word in enumerate(words):
            start = float(supervision.start) + index * word_duration
            # Calculate the last duration from the endpoint to avoid cumulative
            # floating-point drift beyond the supervision boundary.
            duration = (
                float(supervision.end) - start
                if index == len(words) - 1
                else word_duration
            )
            output.append([speaker, segment, word, start, duration])

    leftovers = [sup.id for queue in available.values() for sup in queue]
    if leftovers:
        raise ValueError(
            f"{cut.id}: {len(leftovers)} supervisions were not represented in "
            f"the reference: {leftovers}"
        )
    return output


def interpolate_manifest(
    manifest: Path, reference: Path
) -> Tuple[List[List[Any]], Dict[str, int]]:
    references = json.load(reference.open())
    ref_by_id = {cut_id: rows for cut_id, rows in references}
    output: List[List[Any]] = []
    seen = set()
    num_segments = 0
    num_words = 0

    for cut in CutSet.from_file(manifest):
        rows = ref_by_id.get(cut.id)
        if rows is None:
            continue
        interpolated = interpolate_cut(cut, rows)
        output.append([cut.id, interpolated])
        seen.add(cut.id)
        num_segments += len(segment_rows(rows))
        num_words += len(rows)

    missing = set(ref_by_id) - seen
    if missing:
        preview = sorted(missing)[:10]
        raise ValueError(
            f"Manifest is missing {len(missing)} reference cuts; first IDs: {preview}"
        )
    return output, {
        "cuts": len(output),
        "segments": num_segments,
        "words": num_words,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    alignments, counts = interpolate_manifest(args.manifest, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(alignments, f)
        f.write("\n")
    print(
        f"Wrote {counts['cuts']} cuts, {counts['segments']} supervisions, and "
        f"{counts['words']} uniformly interpolated words to {args.output}"
    )


if __name__ == "__main__":
    main()
