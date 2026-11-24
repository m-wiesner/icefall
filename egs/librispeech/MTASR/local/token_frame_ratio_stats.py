#!/usr/bin/env python3
"""Compute per-cut BPE-token/output-frame load statistics for training data."""

import argparse
import gzip
import json
from pathlib import Path

import sentencepiece as spm


def update(stats, duration, texts, sp, frame_rate):
    if duration <= 0:
        return
    tokens = sum(len(sp.encode(text)) for text in texts if text)
    ratio = tokens / (duration * frame_rate)
    stats["cuts"] += 1
    stats["tokens"] += tokens
    stats["frames"] += duration * frame_rate
    stats["ratio_sum"] += ratio
    stats["above_one"] += ratio > 1.0


def manifest_stats(paths, sp, frame_rate, max_duration):
    stats = dict(cuts=0, tokens=0, frames=0.0, ratio_sum=0.0, above_one=0)
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                cut = json.loads(line)
                duration = cut["duration"]
                if max_duration is not None and duration > max_duration:
                    continue
                update(
                    stats, duration,
                    [sup.get("text", "") for sup in cut.get("supervisions", [])],
                    sp, frame_rate,
                )
    return stats


def yodas_stats(scheme_dir, sp, frame_rate, max_duration):
    # Import the recipe's exact streaming supervision swap/group implementation.
    from lhotse import CutSet
    from wavlm_spashl_ctc.asr_datamodule import _YodasSharGroups

    stats = dict(cuts=0, tokens=0, frames=0.0, ratio_sum=0.0, above_one=0)
    grouper = _YodasSharGroups(
        scheme_dir=scheme_dir,
        hypothesis="doverlap",
        max_pause=1.0,
        max_duration=60.0,
        min_speakers=2,
        max_speakers=8,
    )
    meta_dir = scheme_dir.parent / f"{scheme_dir.name}.meta"
    keep_ids = set((meta_dir / "keep_ids.txt").read_text().splitlines())
    scheme = json.loads((meta_dir / "scheme.json").read_text())
    shar_dir = Path(scheme["shar_dir"])
    candidates = [shar_dir / name for name in scheme["kept_source_shards"]]
    # Reading cuts.*.jsonl.gz directly avoids walking the recording tar files;
    # grouping still uses the exact training-time implementation above.
    for cuts_path in candidates:
        for parent in CutSet.from_file(cuts_path):
            if parent.id not in keep_ids:
                continue
            parent = grouper._swap_supervisions(parent, grouper.hypothesis)
            for cut in grouper._split(parent):
                if max_duration is not None and cut.duration > max_duration:
                    continue
                update(
                    stats, cut.duration, [s.text for s in cut.supervisions],
                    sp, frame_rate,
                )
    return stats


def report(name, stats):
    print(json.dumps({
        "dataset": name,
        "cuts": stats["cuts"],
        "mean_cut_ratio": stats["ratio_sum"] / stats["cuts"],
        "aggregate_token_frame_ratio": stats["tokens"] / stats["frames"],
        "cuts_above_1": stats["above_one"],
        "percent_cuts_above_1": 100 * stats["above_one"] / stats["cuts"],
        "total_tokens": stats["tokens"],
        "total_frames": stats["frames"],
    }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpe-model", required=True)
    parser.add_argument("--frame-rate", type=float, default=12.5)
    parser.add_argument("--max-duration", type=float, default=100.0)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--yodas-scheme-dir")
    args = parser.parse_args()
    sp = spm.SentencePieceProcessor(model_file=args.bpe_model)
    if args.yodas_scheme_dir:
        stats = yodas_stats(
            Path(args.yodas_scheme_dir), sp, args.frame_rate, args.max_duration
        )
    else:
        stats = manifest_stats(
            [Path(p) for p in args.manifest], sp, args.frame_rate, args.max_duration
        )
    report(args.dataset, stats)


if __name__ == "__main__":
    main()
