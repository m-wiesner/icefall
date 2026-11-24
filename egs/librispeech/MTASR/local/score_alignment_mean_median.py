#!/usr/bin/env python3
"""Report mean, median, and corpus-level forced-alignment metrics.

This complements ``score_alignments.py`` while making each aggregation unit
explicit.  It supports the two calibration protocols used in the alignment
experiments:

* ``--calibrate-half``: estimate mean timing offsets on the first half and
  score only the second half (the legacy dev_synth protocol).
* external calibration inputs/offsets: estimate median offsets on a complete
  development set and apply them to a target set.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from score_alignments import (
    apply_mapping_at_segment_level,
    estimate_timing_bias,
    hungarian_mapping_token_cost,
    kendall_distance_fenwick,
)


def load_pairs(hyp_path: Path, ref_path: Path):
    hyp = json.load(hyp_path.open())
    ref = json.load(ref_path.open())
    hyp_by_id = {item[0]: item for item in hyp}
    pairs = [(hyp_by_id[item[0]], item) for item in ref if item[0] in hyp_by_id]
    return pairs, len(ref) - len(pairs), len(ref)


def estimate_half_mean(pairs):
    split = len(pairs) // 2
    start_errors = []
    duration_errors = []
    for hyp_item, ref_item in pairs[:split]:
        hyp = sorted(hyp_item[1], key=lambda x: (x[1], x[3]))
        ref = sorted(ref_item[1], key=lambda x: (x[1], x[3]))
        if len(hyp) != len(ref):
            raise ValueError(f"Calibration length mismatch for {ref_item[0]}")
        start_errors.extend(h[3] - r[3] for h, r in zip(hyp, ref))
        duration_errors.extend(h[4] - r[4] for h, r in zip(hyp, ref))
    return (
        float(np.mean(start_errors)),
        float(np.mean(duration_errors)),
        split,
        len(start_errors),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--relabel", action="store_true")
    parser.add_argument("--calibrate-half", action="store_true")
    parser.add_argument("--calibration-hyp", type=Path)
    parser.add_argument("--calibration-ref", type=Path)
    parser.add_argument("--calibration-offsets", type=Path)
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    external = args.calibration_hyp is not None or args.calibration_ref is not None
    if external and (args.calibration_hyp is None or args.calibration_ref is None):
        raise ValueError("--calibration-hyp and --calibration-ref are a pair")
    if sum((args.calibrate_half, external, args.calibration_offsets is not None)) > 1:
        raise ValueError("Choose only one calibration mode")

    pairs, skipped, total_refs = load_pairs(args.hyp, args.ref)
    start_bias = 0.0
    duration_bias = 0.0
    calibration = {"mode": "none"}
    if args.calibrate_half:
        start_bias, duration_bias, split, num_tokens = estimate_half_mean(pairs)
        pairs = pairs[split:]
        calibration = {
            "mode": "first_half_mean_to_second_half",
            "num_calibration_cuts": split,
            "num_calibration_tokens": num_tokens,
        }
    elif external:
        calibration_hyp = json.load(args.calibration_hyp.open())
        calibration_ref = json.load(args.calibration_ref.open())
        start_bias, duration_bias, num_tokens = estimate_timing_bias(
            calibration_hyp,
            calibration_ref,
            relabel=args.relabel,
            statistic="median",
        )
        calibration = {
            "mode": "external_median",
            "num_calibration_tokens": num_tokens,
            "calibration_hyp": str(args.calibration_hyp),
            "calibration_ref": str(args.calibration_ref),
        }
        if args.calibration_output is not None:
            args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
            with args.calibration_output.open("w") as f:
                json.dump(
                    {
                        "start_bias_seconds": start_bias,
                        "duration_bias_seconds": duration_bias,
                        **calibration,
                    },
                    f,
                    indent=2,
                )
                f.write("\n")
    elif args.calibration_offsets is not None:
        saved = json.load(args.calibration_offsets.open())
        start_bias = float(saved["start_bias_seconds"])
        duration_bias = float(saved["duration_bias_seconds"])
        calibration = {"mode": "saved_offsets", "path": str(args.calibration_offsets)}

    token_ious = []
    token_boundary_errors = []
    token_start_errors = []
    token_duration_errors = []
    cut_boundary_errors = []
    cut_kendall = []
    cut_inversions = []
    total_intersection = 0.0
    total_union = 0.0
    speaker_errors = 0
    segment_errors = 0
    segment_any_word_errors = 0
    num_segments = 0
    cut_segment_error_rates = []
    partition_errors = 0
    string_errors = 0

    for hyp_item, ref_item in pairs:
        hyp = sorted(hyp_item[1], key=lambda x: (x[1], x[3]))
        ref = sorted(ref_item[1], key=lambda x: (x[1], x[3]))
        if len(hyp) != len(ref):
            raise ValueError(f"Scoring length mismatch for {ref_item[0]}")
        mapped_hyp = hyp
        if args.relabel:
            mapped_hyp = apply_mapping_at_segment_level(
                hyp, hungarian_mapping_token_cost(hyp, ref)
            )

        this_boundary = []
        this_speaker_error = False
        for h, r, mapped in zip(hyp, ref, mapped_hyp):
            hyp_start = h[3] - start_bias
            hyp_end = hyp_start + h[4] - duration_bias
            ref_start = r[3]
            ref_end = ref_start + r[4]
            intersection = max(0.0, min(hyp_end, ref_end) - max(hyp_start, ref_start))
            union = max(hyp_end, ref_end) - min(hyp_start, ref_start)
            boundary_error = 0.5 * (
                abs(hyp_start - ref_start) + abs(hyp_end - ref_end)
            )
            token_ious.append(intersection / union)
            token_boundary_errors.append(boundary_error)
            this_boundary.append(boundary_error)
            token_start_errors.append(hyp_start - ref_start)
            token_duration_errors.append(
                (hyp_end - hyp_start) - (ref_end - ref_start)
            )
            total_intersection += intersection
            total_union += union
            mismatch = mapped[0] != r[0]
            speaker_errors += int(mismatch)
            this_speaker_error = this_speaker_error or mismatch
        cut_boundary_errors.append(float(np.mean(this_boundary)))
        partition_errors += int(this_speaker_error)

        hyp_by_segment = defaultdict(list)
        ref_by_segment = defaultdict(list)
        for mapped, reference in zip(mapped_hyp, ref):
            hyp_by_segment[mapped[1]].append(mapped[0])
            ref_by_segment[reference[1]].append(reference[0])
        this_segment_errors = 0
        for segment, ref_speakers in ref_by_segment.items():
            if len(set(ref_speakers)) != 1:
                raise ValueError(
                    f"Reference segment {segment} has multiple speakers"
                )
            predicted = hyp_by_segment[segment]
            majority_speaker = Counter(predicted).most_common(1)[0][0]
            reference_speaker = ref_speakers[0]
            mismatch = majority_speaker != reference_speaker
            any_word_mismatch = any(
                speaker != reference_speaker for speaker in predicted
            )
            segment_errors += int(mismatch)
            segment_any_word_errors += int(any_word_mismatch)
            this_segment_errors += int(mismatch)
            num_segments += 1
        cut_segment_error_rates.append(
            this_segment_errors / max(1, len(ref_by_segment))
        )

        hyp_tokens = [x[2] for x in sorted(hyp_item[1], key=lambda x: x[3])]
        ref_tokens = [x[2] for x in sorted(ref_item[1], key=lambda x: x[3])]
        inversions = kendall_distance_fenwick(ref_tokens, hyp_tokens)
        cut_inversions.append(inversions)
        cut_kendall.append(inversions / max(1, len(ref_tokens)))
        string_errors += int(hyp_tokens != ref_tokens)

    num_tokens = len(token_ious)
    result = {
        "hyp": str(args.hyp),
        "ref": str(args.ref),
        "num_scored_cuts": len(pairs),
        "num_scored_tokens": num_tokens,
        "failed_percent": 100.0 * skipped / max(1, total_refs),
        "calibration": {
            **calibration,
            "start_bias_seconds": start_bias,
            "duration_bias_seconds": duration_bias,
        },
        "iou_token_mean_percent": 100.0 * float(np.mean(token_ious)),
        "iou_token_median_percent": 100.0 * float(np.median(token_ious)),
        "iou_pooled_percent": 100.0 * total_intersection / total_union,
        "boundary_cut_mean_ms": 1000.0 * float(np.mean(cut_boundary_errors)),
        "boundary_cut_median_ms": 1000.0 * float(np.median(cut_boundary_errors)),
        "boundary_token_mean_ms": 1000.0 * float(np.mean(token_boundary_errors)),
        "boundary_token_median_ms": 1000.0 * float(np.median(token_boundary_errors)),
        "start_error_token_mean_ms": 1000.0 * float(np.mean(token_start_errors)),
        "start_error_token_median_ms": 1000.0 * float(np.median(token_start_errors)),
        "duration_error_token_mean_ms": 1000.0 * float(np.mean(token_duration_errors)),
        "duration_error_token_median_ms": 1000.0 * float(np.median(token_duration_errors)),
        "kendall_cut_mean": float(np.mean(cut_kendall)),
        "kendall_cut_median": float(np.median(cut_kendall)),
        "kendall_corpus_micro": float(sum(cut_inversions) / max(1, num_tokens)),
        "inversions_cut_mean": float(np.mean(cut_inversions)),
        "inversions_cut_median": float(np.median(cut_inversions)),
        "wder_percent": 100.0 * speaker_errors / max(1, num_tokens),
        "segment_error_percent": 100.0 * segment_errors / max(1, num_segments),
        "segment_error_cut_mean_percent": 100.0 * float(
            np.mean(cut_segment_error_rates)
        ),
        "segment_any_word_error_percent": (
            100.0 * segment_any_word_errors / max(1, num_segments)
        ),
        "num_scored_segments": num_segments,
        "partition_error_percent": 100.0 * partition_errors / max(1, len(pairs)),
        "string_error_rate": string_errors / max(1, len(pairs)),
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
