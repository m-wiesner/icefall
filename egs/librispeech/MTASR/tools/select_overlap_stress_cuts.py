#!/usr/bin/env python3
"""Select FastMSS cuts with the greatest concurrent-speech burden."""

from __future__ import annotations

import argparse
import heapq
import logging
from pathlib import Path

from lhotse import CutSet, load_manifest_lazy


def overlap_stats(cut):
    events = []
    for supervision in cut.supervisions:
        events.append((supervision.start, 1))
        events.append((supervision.end, -1))
    # Process ends before starts when boundaries coincide.
    events.sort(key=lambda event: (event[0], event[1]))
    active = 0
    peak = 0
    overlap_seconds = 0.0
    previous_time = events[0][0] if events else 0.0
    for time, delta in events:
        overlap_seconds += max(active - 1, 0) * (time - previous_time)
        active += delta
        peak = max(peak, active)
        previous_time = time
    return peak, overlap_seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-cuts", type=int, default=160)
    parser.add_argument("--log-interval", type=int, default=1000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    best = []
    seen = 0
    for manifest in args.manifests:
        logging.info("Scanning %s", manifest)
        for cut in load_manifest_lazy(manifest):
            peak, overlap_seconds = overlap_stats(cut)
            # Heap entries must remain totally ordered even for equal scores.
            score = (
                peak,
                overlap_seconds,
                len(cut.supervisions),
                cut.duration,
                cut.id,
            )
            entry = (score, cut)
            if len(best) < args.num_cuts:
                heapq.heappush(best, entry)
            elif score > best[0][0]:
                heapq.heapreplace(best, entry)
            seen += 1
            if seen % args.log_interval == 0:
                logging.info("Scanned %d cuts", seen)

    selected = []
    for score, cut in sorted(best, reverse=True):
        peak, overlap_seconds, num_supervisions, _, _ = score
        cut = cut.with_custom(
            "stress_peak_concurrency", peak
        ).with_custom(
            "stress_overlap_speaker_seconds", overlap_seconds
        ).with_custom(
            "stress_num_supervisions", num_supervisions
        )
        selected.append(cut)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    CutSet.from_cuts(selected).to_file(args.output)
    logging.info(
        "Wrote %d/%d cuts to %s; weakest retained score=%s",
        len(selected),
        seen,
        args.output,
        min(score for score, _ in best),
    )


if __name__ == "__main__":
    main()
