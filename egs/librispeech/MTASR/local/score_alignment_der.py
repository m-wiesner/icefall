#!/usr/bin/env python3
"""Score diarization error rate from MTASR word-alignment JSON files."""

import argparse
import json
from collections import Counter, defaultdict

from lhotse import CutSet

from score_alignments import (
    apply_mapping_at_segment_level,
    estimate_timing_bias,
    hungarian_mapping_token_cost,
)


def segment_regions(words):
    grouped = defaultdict(list)
    for word in words:
        grouped[word[1]].append(word)
    regions = []
    for segment, items in grouped.items():
        speakers = {item[0] for item in items}
        if len(speakers) != 1:
            raise ValueError(f"Segment {segment} has multiple speakers: {speakers}")
        start = min(item[3] for item in items)
        end = max(item[3] + item[4] for item in items)
        regions.append((start, end, next(iter(speakers))))
    return regions


def majority_segment_regions(words):
    """Collapse word-level labels to one duration-majority label per segment."""
    grouped = defaultdict(list)
    for word in words:
        grouped[word[1]].append(word)
    regions = []
    for items in grouped.values():
        start = min(item[3] for item in items)
        end = max(item[3] + item[4] for item in items)
        duration_by_speaker = Counter()
        for item in items:
            duration_by_speaker[item[0]] += item[4]
        speaker = duration_by_speaker.most_common(1)[0][0]
        regions.append((start, end, speaker))
    return regions


def word_regions(words):
    """Use every aligned word interval as a diarization hypothesis region."""
    return [(word[3], word[3] + word[4], word[0]) for word in words]


def supervision_regions(cut):
    """Return reference regions with the segment numbering used by alignment JSON."""
    supervisions = list(cut.supervisions)
    speaker_durations = defaultdict(float)
    for supervision in supervisions:
        speaker_durations[supervision.speaker] += supervision.duration
    speakers = sorted(
        speaker_durations,
        key=speaker_durations.get,
        reverse=True,
    )
    sort_order = [
        index
        for speaker in speakers
        for index, supervision in enumerate(supervisions)
        if supervision.speaker == speaker
    ]
    return [
        (
            supervisions[index].start,
            supervisions[index].start + supervisions[index].duration,
            supervisions[index].speaker,
        )
        for index in sort_order
    ]


def active(regions, time):
    return {speaker for start, end, speaker in regions if start <= time < end}


def excluded(collars, time):
    return any(start <= time < end for start, end in collars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument(
        "--ref-cut-manifest",
        help=(
            "Use supervision start/end boundaries from this CutSet as the DER "
            "reference, instead of spans from the first through last aligned word."
        ),
    )
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument(
        "--hyp-region-unit",
        choices=("segment", "segment_majority", "word"),
        default="segment",
        help=(
            "Construct hypothesis regions from original segments, collapse "
            "word labels to a duration-majority segment label, or retain one "
            "region per aligned word."
        ),
    )
    parser.add_argument(
        "--relabel", action="store_true",
        help="Use the same token-cost Hungarian relabeling as score_alignments.py",
    )
    parser.add_argument("--calibration-hyp")
    parser.add_argument("--calibration-ref")
    parser.add_argument("--calibration-offsets")
    parser.add_argument(
        "--calibration-statistic",
        choices=("median", "mean"),
        default="median",
    )
    args = parser.parse_args()

    hyp = dict(json.load(open(args.hyp)))
    ref = dict(json.load(open(args.ref)))
    ref_cuts = None
    if args.ref_cut_manifest is not None:
        ref_cuts = {
            cut.id: cut for cut in CutSet.from_file(args.ref_cut_manifest)
        }
    if (args.calibration_hyp is None) != (args.calibration_ref is None):
        raise ValueError("--calibration-hyp and --calibration-ref must be used together")
    if args.calibration_offsets is not None and args.calibration_hyp is not None:
        raise ValueError("Use saved offsets or calibration inputs, not both")
    bias = duration_bias = 0.0
    if args.calibration_offsets is not None:
        saved = json.load(open(args.calibration_offsets))
        bias = float(saved["start_bias_seconds"])
        duration_bias = float(saved["duration_bias_seconds"])
        print(
            f"Loaded calibration offsets from {args.calibration_offsets}: "
            f"start={1000 * bias:.3f} ms duration={1000 * duration_bias:.3f} ms"
        )
    elif args.calibration_hyp is not None:
        bias, duration_bias, calibration_tokens = estimate_timing_bias(
            json.load(open(args.calibration_hyp)),
            json.load(open(args.calibration_ref)),
            relabel=args.relabel,
            statistic=args.calibration_statistic,
        )
        print(
            f"Calibration offsets ({args.calibration_statistic}) from "
            f"{calibration_tokens} tokens: start={1000 * bias:.3f} ms "
            f"duration={1000 * duration_bias:.3f} ms"
        )
    miss = false_alarm = confusion = reference_speaker_time = 0.0
    scored_cuts = 0

    for cut_id, ref_words in ref.items():
        if cut_id not in hyp:
            continue
        hyp_words = hyp[cut_id]
        if args.relabel:
            h = sorted(hyp_words, key=lambda x: (x[1], x[3]))
            r = sorted(ref_words, key=lambda x: (x[1], x[3]))
            hyp_words = apply_mapping_at_segment_level(
                h, hungarian_mapping_token_cost(h, r)
            )
        if bias != 0.0 or duration_bias != 0.0:
            hyp_words = [
                [word[0], word[1], word[2], word[3] - bias, word[4] - duration_bias]
                for word in hyp_words
            ]

        if ref_cuts is not None:
            if cut_id not in ref_cuts:
                raise KeyError(f"Cut {cut_id!r} is absent from --ref-cut-manifest")
            ref_regions = supervision_regions(ref_cuts[cut_id])
        else:
            ref_regions = segment_regions(ref_words)
        if args.hyp_region_unit == "word":
            hyp_regions = word_regions(hyp_words)
        elif args.hyp_region_unit == "segment_majority":
            hyp_regions = majority_segment_regions(hyp_words)
        else:
            hyp_regions = segment_regions(hyp_words)
        collars = [
            (max(0.0, boundary - args.collar), boundary + args.collar)
            for start, end, _ in ref_regions for boundary in (start, end)
        ]
        boundaries = {0.0}
        for start, end, _ in ref_regions + hyp_regions:
            boundaries.update((start, end))
        for start, end in collars:
            boundaries.update((start, end))
        boundaries = sorted(boundaries)

        for left, right in zip(boundaries, boundaries[1:]):
            if right <= left:
                continue
            midpoint = (left + right) / 2
            if excluded(collars, midpoint):
                continue
            ref_active = active(ref_regions, midpoint)
            hyp_active = active(hyp_regions, midpoint)
            duration = right - left
            n_ref, n_hyp = len(ref_active), len(hyp_active)
            correct = len(ref_active & hyp_active)
            miss += duration * max(0, n_ref - n_hyp)
            false_alarm += duration * max(0, n_hyp - n_ref)
            confusion += duration * (min(n_ref, n_hyp) - correct)
            reference_speaker_time += duration * n_ref
        scored_cuts += 1

    errors = miss + false_alarm + confusion
    print(f"Scored cuts: {scored_cuts}")
    print(f"Reference speaker time: {reference_speaker_time:.3f} s")
    print(f"Miss: {100 * miss / reference_speaker_time:.4f}%")
    print(f"False alarm: {100 * false_alarm / reference_speaker_time:.4f}%")
    print(f"Speaker confusion: {100 * confusion / reference_speaker_time:.4f}%")
    print(f"DER: {100 * errors / reference_speaker_time:.4f}%")


if __name__ == "__main__":
    main()
