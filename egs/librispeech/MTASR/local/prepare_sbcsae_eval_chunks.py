#!/usr/bin/env python3
"""Create normalized, fixed-context SBCSAE evaluation chunks."""

import argparse
import sys
from collections import Counter
from pathlib import Path

from lhotse import CutSet, fastcopy, load_manifest_lazy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text_normalization import normalize_english_transcript  # noqa: E402
from local.prepare_ami_notsofar import segment_cut  # noqa: E402


def supervision_components(supervisions):
    """Group temporally overlapping supervisions so a boundary never splits one."""
    components = []
    for supervision in sorted(supervisions, key=lambda item: (item.start, item.end)):
        if not components or supervision.start >= max(item.end for item in components[-1]):
            components.append([supervision])
        else:
            components[-1].append(supervision)
    return components


def pack_components(components, max_duration):
    group = []
    for component in components:
        candidate = group + component
        span = max(item.end for item in candidate) - min(item.start for item in candidate)
        if group and span > max_duration:
            yield group
            group = list(component)
        else:
            group = candidate
    if group:
        yield group


def chunk_cut(cut, max_duration):
    components = supervision_components(cut.supervisions)
    for index, group in enumerate(pack_components(components, max_duration)):
        start = min(item.start for item in group)
        end = max(item.end for item in group)
        normalized = [
            fastcopy(
                supervision,
                text=normalize_english_transcript(supervision.text),
            )
            for supervision in group
            if normalize_english_transcript(supervision.text)
        ]
        if not normalized:
            continue
        piece = cut.truncate(
            offset=start,
            duration=end - start,
            keep_excessive_supervisions=False,
            preserve_id=True,
        )
        # Use the explicitly selected group to guarantee every supervision is
        # represented exactly once, including utterances touching a boundary.
        yield fastcopy(
            piece,
            id=f"{cut.id}-eval-{index:04d}",
            supervisions=[
                fastcopy(s, start=s.start - start) for s in normalized
            ],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-duration", type=float, default=60.0)
    parser.add_argument("--max-pause", type=float, default=2.0)
    args = parser.parse_args()

    source = load_manifest_lazy(args.input)
    stats = Counter()
    chunks = CutSet.from_cuts(
        piece
        for cut in source
        for piece in segment_cut(
            cut,
            max_duration=args.max_duration,
            max_pause=args.max_pause,
            stats=stats,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_file(args.output)
    print(
        f"Wrote {len(chunks)} normalized evaluation chunks to {args.output}; "
        f"supervisions={stats['supervisions']}, words={stats['words']}"
    )


if __name__ == "__main__":
    main()
