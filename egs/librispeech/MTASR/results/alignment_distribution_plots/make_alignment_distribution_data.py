#!/usr/bin/env python3
"""Create ECDF data for the long-form token-only alignment figures.

The timing metrics exactly follow local/score_alignment_mean_median.py:
word boundary error is the mean absolute onset/offset error, and a recording
score is the unweighted mean of its word scores.  The word-level Kendall
quantity is an additive decomposition of the existing recording metric.  A
word receives half of the number of inversions in which it participates, so
the mean contribution in a recording is exactly inversions / number of words.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "alignment_distribution_data"
HIST_OUT = HERE / "alignment_histogram_data"

METRICS = {
    "notsofar_eval": ROOT
    / "exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4"
    / "pinned/align/word_alignments_chkpt72000_avg1_notsofar_eval_sc_plaza_0_full_"
    "wavlm_spashl72k_bpechars4_tokoracle_collar32000_longform_center20_graph20_"
    "remove40_1state_tokenonly_oraclemap_bpechars4_20260923_dev_calibrated_"
    "mean_median.json",
    "ami_eval": ROOT
    / "exp/fastmss_yodas_en_wavlm_large_spashl_factorized_v100x4_accum4"
    / "pinned/align/word_alignments_chkpt72000_avg1_ami_sdm_test_full_wavlm_"
    "spashl72k_bpechars4_tokoracle_collar32000_longform_center20_graph20_"
    "remove40_1state_tokenonly_oraclemap_bpechars4_20260923_dev_calibrated_"
    "mean_median.json",
}


class Fenwick:
    def __init__(self, size: int):
        self.bit = [0] * (size + 1)

    def add(self, index: int) -> None:
        index += 1
        while index < len(self.bit):
            self.bit[index] += 1
            index += index & -index

    def prefix(self, index: int) -> int:
        index += 1
        total = 0
        while index > 0:
            total += self.bit[index]
            index -= index & -index
        return total


def occurrence_ranks(reference: list[str], hypothesis: list[str]) -> list[int]:
    """Map repeated hypothesis words to reference positions as in the scorer."""
    available: dict[str, deque[int]] = defaultdict(deque)
    for index, token in enumerate(reference):
        available[token].append(index)
    ranks = []
    for token in hypothesis:
        if not available[token]:
            raise ValueError(f"Hypothesis has an unmatched occurrence of {token!r}")
        ranks.append(available[token].popleft())
    return ranks


def inversion_contributions(reference: list[str], hypothesis: list[str]) -> np.ndarray:
    """Return half of each word's inversion degree.

    Every inversion contributes 0.5 to each of its two words.  Consequently,
    ``contributions.sum()`` is the raw inversion count.
    """
    ranks = occurrence_ranks(reference, hypothesis)
    n = len(ranks)
    left_greater = np.zeros(n, dtype=np.int64)
    right_smaller = np.zeros(n, dtype=np.int64)

    bit = Fenwick(n)
    for i, rank in enumerate(ranks):
        left_greater[i] = i - bit.prefix(rank)
        bit.add(rank)

    bit = Fenwick(n)
    for i in range(n - 1, -1, -1):
        rank = ranks[i]
        right_smaller[i] = bit.prefix(rank - 1)
        bit.add(rank)

    return 0.5 * (left_greater + right_smaller)


def write_ecdf(
    path: Path,
    values,
    floor: float | None = None,
    collapse_identical: bool = False,
    round_decimals: int | None = None,
) -> None:
    selected = np.sort(np.asarray(values, dtype=np.float64))
    population = len(selected)
    if floor is not None:
        selected = np.maximum(selected, floor)
    if round_decimals is not None:
        # Quantize only below the visible resolution of the corresponding
        # axis. All observations still contribute to the exact ECDF count.
        selected = np.round(selected, round_decimals)
    if collapse_identical:
        selected, counts = np.unique(selected, return_counts=True)
        probabilities = np.cumsum(counts, dtype=np.float64) / population
    else:
        probabilities = np.arange(1, population + 1, dtype=np.float64) / population
    with path.open("w") as stream:
        stream.write("value ecdf\n")
        for value, probability in zip(selected, probabilities):
            stream.write(f"{value:.10g} {probability:.10g}\n")


def write_histogram(path: Path, values, edges: np.ndarray) -> None:
    """Write a normalized histogram in PGFPlots ``ybar interval`` format."""
    values = np.asarray(values, dtype=np.float64)
    counts, _ = np.histogram(values, bins=edges)
    if int(counts.sum()) != len(values):
        raise ValueError(
            f"Histogram range for {path.name} contains {counts.sum()} of "
            f"{len(values)} observations"
        )
    probabilities = counts.astype(np.float64) / len(values)
    with path.open("w") as stream:
        stream.write("edge probability\n")
        for edge, probability in zip(edges[:-1], probabilities):
            stream.write(f"{edge:.10g} {probability:.10g}\n")
        # ybar interval uses the final x coordinate as the last right edge.
        stream.write(f"{edges[-1]:.10g} 0\n")


def load_dataset(metrics_path: Path):
    metrics = json.load(metrics_path.open())
    hypothesis = json.load((ROOT / metrics["hyp"]).open())
    reference = json.load((ROOT / metrics["ref"]).open())
    hyp_by_id = {item[0]: item for item in hypothesis}
    start_bias = float(metrics["calibration"]["start_bias_seconds"])
    duration_bias = float(metrics["calibration"]["duration_bias_seconds"])

    word_iou, word_boundary, word_kendall = [], [], []
    recording_iou, recording_boundary, recording_kendall = [], [], []

    for ref_item in reference:
        hyp_item = hyp_by_id.get(ref_item[0])
        if hyp_item is None:
            continue

        hyp_timing = sorted(hyp_item[1], key=lambda row: (row[1], row[3]))
        ref_timing = sorted(ref_item[1], key=lambda row: (row[1], row[3]))
        if len(hyp_timing) != len(ref_timing):
            raise ValueError(f"Length mismatch for {ref_item[0]}")

        this_iou, this_boundary = [], []
        for hyp_word, ref_word in zip(hyp_timing, ref_timing):
            hyp_start = hyp_word[3] - start_bias
            hyp_end = hyp_start + hyp_word[4] - duration_bias
            ref_start = ref_word[3]
            ref_end = ref_start + ref_word[4]
            intersection = max(0.0, min(hyp_end, ref_end) - max(hyp_start, ref_start))
            union = max(hyp_end, ref_end) - min(hyp_start, ref_start)
            this_iou.append(100.0 * intersection / union)
            this_boundary.append(
                500.0 * (abs(hyp_start - ref_start) + abs(hyp_end - ref_end))
            )

        hyp_tokens = [row[2] for row in sorted(hyp_item[1], key=lambda row: row[3])]
        ref_tokens = [row[2] for row in sorted(ref_item[1], key=lambda row: row[3])]
        this_kendall = inversion_contributions(ref_tokens, hyp_tokens)

        word_iou.extend(this_iou)
        word_boundary.extend(this_boundary)
        word_kendall.extend(this_kendall.tolist())
        recording_iou.append(float(np.mean(this_iou)))
        recording_boundary.append(float(np.mean(this_boundary)))
        recording_kendall.append(float(np.mean(this_kendall)))

    return {
        "word_iou": word_iou,
        "recording_iou": recording_iou,
        "word_boundary": word_boundary,
        "recording_boundary": recording_boundary,
        "word_kendall": word_kendall,
        "recording_kendall": recording_kendall,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    HIST_OUT.mkdir(parents=True, exist_ok=True)
    histogram_edges = {
        "word_iou": np.linspace(0.0, 100.0, 31),
        "recording_iou": np.arange(50.0, 76.0 + 1e-9, 2.0),
        "word_boundary": np.geomspace(0.1, 20000.0, 37),
        "recording_boundary": np.arange(40.0, 190.0 + 1e-9, 10.0),
        "word_kendall": np.arange(0.0, 23.0 + 1e-9, 0.5),
        "recording_kendall": np.arange(0.0, 0.5 + 1e-9, 0.05),
    }
    summary = {}
    for dataset, metrics_path in METRICS.items():
        values = load_dataset(metrics_path)
        summary[dataset] = {}
        for metric, observations in values.items():
            # Log-scaled plots need a small display floor for exact zeros.
            floor = {
                "word_boundary": 0.1,
                "word_kendall": 0.01,
            }.get(metric)
            write_ecdf(
                OUT / f"{dataset}_{metric}.dat",
                observations,
                floor=floor,
                # This is count-preserving ECDF compression, not observation
                # downsampling: every observation contributes to a jump.
                collapse_identical=metric.startswith("word_"),
                round_decimals={
                    "word_iou": 1,       # 0.1 percentage point
                    "word_boundary": 0,  # 1 ms
                    "word_kendall": 1,
                }.get(metric),
            )
            histogram_values = observations
            if metric == "word_boundary":
                # Exact zeros share the visible 0.1-ms bin on the log axis.
                histogram_values = np.maximum(observations, 0.1)
            write_histogram(
                HIST_OUT / f"{dataset}_{metric}.dat",
                histogram_values,
                histogram_edges[metric],
            )
            summary[dataset][metric] = {
                "count": len(observations),
                "mean": float(np.mean(observations)),
                "p50": float(np.percentile(observations, 50)),
                "p90": float(np.percentile(observations, 90)),
                "p95": float(np.percentile(observations, 95)),
                "max": float(np.max(observations)),
            }
    with (HERE / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
