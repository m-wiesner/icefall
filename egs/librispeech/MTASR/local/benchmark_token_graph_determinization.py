#!/usr/bin/env python3
"""Benchmark token-only transcript-FSA determinization on a CutSet."""

import argparse
import statistics
import time
from itertools import islice

import k2
from lhotse import CutSet

from wavlm_spashl_ctc.mdctc_graph_compiler import MDCTCGraphCompiler


def fsa_size(fsa):
    return int(fsa.shape[0]), int(fsa.arcs.num_elements())


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return float("nan")
    return values[round(fraction * (len(values) - 1))]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lang-dir", required=True)
    parser.add_argument("--num-cuts", type=int, default=100)
    parser.add_argument("--collar", type=int, default=32000)
    parser.add_argument("--edge-collar", type=int, default=0)
    parser.add_argument(
        "--interval-model",
        choices=("fixed", "brownian_bridge"),
        default="brownian_bridge",
    )
    return parser.parse_args()


def main():
    args = get_args()
    compiler = MDCTCGraphCompiler(args.lang_dir, device="cpu")
    rows = []

    for cut in islice(CutSet.from_file(args.manifest), args.num_cuts):
        supervisions = sorted(cut.supervisions, key=lambda s: (s.start, s.id))
        texts = [s.text for s in supervisions]
        offsets = [round(s.start * cut.sampling_rate) for s in supervisions]
        lengths = [round(s.duration * cut.sampling_rate) for s in supervisions]

        start = time.perf_counter()
        seqs, constraints, num_overlaps, _ = (
            compiler.get_seqs_and_constraints_no_spk(
                texts,
                offsets,
                lengths,
                collar=args.collar,
                edge_collar=args.edge_collar,
                interval_model=args.interval_model,
            )
        )
        transcript = compiler.build_topo_sort_fsa(
            seqs, constraints, use_speaker=False, aux_labels=True
        )
        built = time.perf_counter()

        acceptor = transcript.clone()
        del acceptor.aux_labels
        acceptor = k2.arc_sort(k2.connect(acceptor))
        was_deterministic = bool(
            acceptor.properties
            & k2.fsa_properties.ARC_SORTED_AND_DETERMINISTIC
        )
        determinized = k2.determinize(acceptor)
        determinized = k2.arc_sort(k2.connect(determinized))
        finished = time.perf_counter()

        before_states, before_arcs = fsa_size(transcript)
        after_states, after_arcs = fsa_size(determinized)
        rows.append(
            {
                "cut": cut.id,
                "duration": cut.duration,
                "supervisions": len(supervisions),
                "tokens": sum(map(len, seqs)),
                "unordered_pairs": num_overlaps,
                "build_seconds": built - start,
                "determinize_seconds": finished - built,
                "before_states": before_states,
                "before_arcs": before_arcs,
                "after_states": after_states,
                "after_arcs": after_arcs,
                "was_deterministic": was_deterministic,
            }
        )

    if not rows:
        raise RuntimeError("The manifest contained no cuts")

    det_times = [row["determinize_seconds"] for row in rows]
    build_times = [row["build_seconds"] for row in rows]
    changed = [
        row
        for row in rows
        if (row["before_states"], row["before_arcs"])
        != (row["after_states"], row["after_arcs"])
    ]
    state_ratios = [row["after_states"] / row["before_states"] for row in rows]
    arc_ratios = [row["after_arcs"] / row["before_arcs"] for row in rows]

    nondeterministic = sum(not row["was_deterministic"] for row in rows)
    print(
        f"cuts={len(rows)} nondeterministic={nondeterministic} "
        f"changed_shape={len(changed)}"
    )
    print(
        "build_seconds "
        f"total={sum(build_times):.6f} mean={statistics.mean(build_times):.6f} "
        f"median={statistics.median(build_times):.6f} "
        f"p95={percentile(build_times, 0.95):.6f} max={max(build_times):.6f}"
    )
    print(
        "determinize_seconds "
        f"total={sum(det_times):.6f} mean={statistics.mean(det_times):.6f} "
        f"median={statistics.median(det_times):.6f} "
        f"p95={percentile(det_times, 0.95):.6f} max={max(det_times):.6f}"
    )
    print(
        "size_ratio_after_over_before "
        f"states_mean={statistics.mean(state_ratios):.6f} "
        f"states_max={max(state_ratios):.6f} "
        f"arcs_mean={statistics.mean(arc_ratios):.6f} "
        f"arcs_max={max(arc_ratios):.6f}"
    )
    for deterministic in (True, False):
        group = [
            row for row in rows
            if row["was_deterministic"] == deterministic
        ]
        times = [row["determinize_seconds"] for row in group]
        label = "already_deterministic" if deterministic else "nondeterministic"
        print(
            f"{label} cuts={len(group)} "
            f"mean={statistics.mean(times):.6f} "
            f"median={statistics.median(times):.6f} "
            f"p95={percentile(times, 0.95):.6f} max={max(times):.6f}"
        )
    print("slowest_determinization:")
    for row in sorted(
        rows, key=lambda item: item["determinize_seconds"], reverse=True
    )[:10]:
        print(
            f"  {row['cut']} det={row['determinize_seconds']:.6f}s "
            f"build={row['build_seconds']:.6f}s "
            f"states={row['before_states']}->{row['after_states']} "
            f"arcs={row['before_arcs']}->{row['after_arcs']} "
            f"tokens={row['tokens']} supervisions={row['supervisions']} "
            f"was_deterministic={row['was_deterministic']}"
        )


if __name__ == "__main__":
    main()
