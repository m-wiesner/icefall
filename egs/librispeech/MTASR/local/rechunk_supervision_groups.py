#!/usr/bin/env python3
"""Create bounded Lhotse cuts without copying audio or splitting utterances."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from lhotse import CutSet, fastcopy, load_manifest_lazy


def overlap_components(supervisions):
    """Return components whose utterances are connected by temporal overlap."""
    components = []
    component_end = None
    for supervision in sorted(supervisions, key=lambda item: (item.start, item.end)):
        if component_end is None or supervision.start >= component_end:
            components.append([supervision])
            component_end = supervision.end
        else:
            components[-1].append(supervision)
            component_end = max(component_end, supervision.end)
    return components


def pack_components(components, max_duration: float, max_pause: float):
    """Pack whole overlap components while respecting duration and pause."""
    group = []
    for component in components:
        if not group:
            group = list(component)
            continue
        group_start = min(item.start for item in group)
        group_end = max(item.end for item in group)
        component_end = max(item.end for item in component)
        span = component_end - group_start
        gap = min(item.start for item in component) - group_end
        if span > max_duration or gap > max_pause:
            yield group
            group = list(component)
        else:
            group.extend(component)
    if group:
        yield group


def rechunk_cut(cut, max_duration: float, max_pause: float, oversize_policy: str):
    components = overlap_components(cut.supervisions)
    for index, group in enumerate(
        pack_components(components, max_duration=max_duration, max_pause=max_pause)
    ):
        start = min(item.start for item in group)
        end = max(item.end for item in group)
        duration = end - start
        if duration > max_duration + 1.0e-6:
            if oversize_policy == "error":
                raise ValueError(
                    f"Cut {cut.id} has an indivisible overlap component of "
                    f"{duration:.3f}s (> {max_duration:.3f}s)"
                )
            continue
        piece = cut.truncate(
            offset=start,
            duration=duration,
            keep_excessive_supervisions=False,
            preserve_id=True,
        )
        # MixedCut derives its supervisions from its tracks, so replacing the
        # top-level supervision list is invalid.  Truncation adjusts the track
        # supervisions correctly; verify that it retained exactly this group.
        expected_ids = {supervision.id for supervision in group}
        actual_ids = {supervision.id for supervision in piece.supervisions}
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            unexpected = sorted(actual_ids - expected_ids)
            raise RuntimeError(
                f"Cut {cut.id} supervision mismatch after truncation: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        yield fastcopy(piece, id=f"{cut.id}-chunk-{index:04d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--max-pause", type=float, default=1.0)
    parser.add_argument(
        "--oversize-policy", choices=("error", "drop"), default="error"
    )
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    totals = Counter()

    def pieces():
        for cut in load_manifest_lazy(args.input):
            totals["source_cuts"] += 1
            totals["source_supervisions"] += len(cut.supervisions)
            for piece in rechunk_cut(
                cut,
                max_duration=args.max_duration,
                max_pause=args.max_pause,
                oversize_policy=args.oversize_policy,
            ):
                totals["cuts_written"] += 1
                totals["supervisions_written"] += len(piece.supervisions)
                yield piece
            components = overlap_components(cut.supervisions)
            oversize = [
                component
                for component in components
                if max(s.end for s in component) - min(s.start for s in component)
                > args.max_duration + 1.0e-6
            ]
            totals["oversize_components"] += len(oversize)
            totals["oversize_supervisions"] += sum(map(len, oversize))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    CutSet.from_cuts(pieces()).to_file(args.output)
    summary_path = args.summary or args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(dict(totals), indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(totals), sort_keys=True))


if __name__ == "__main__":
    main()
