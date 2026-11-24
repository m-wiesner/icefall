#!/usr/bin/env python3
"""Prepare oracle-supervision audio crops for external forced aligners.

The reference alignment is used only to recover the canonical segment IDs and
ground-truth word strings expected by the common scorer.  Its word times are
never used to make the audio crop or to constrain either aligner.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from lhotse import CutSet


def normalized_words(words):
    return tuple(
        x for x in (re.sub(r"[^A-Z']", "", w.upper()) for w in words) if x
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--max-cuts", type=int)
    args = parser.parse_args()

    if args.margin < 0:
        raise ValueError("--margin must be non-negative")
    corpus_dir = args.output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    reference_items = json.load(args.reference.open())
    if args.max_cuts is not None:
        reference_items = reference_items[: args.max_cuts]
    cuts = {cut.id: cut for cut in CutSet.from_file(args.manifest)}

    metadata = []
    for cut_index, (cut_id, reference_words) in enumerate(reference_items):
        if cut_id not in cuts:
            raise KeyError(f"Reference cut {cut_id!r} is absent from {args.manifest}")
        cut = cuts[cut_id]
        audio = cut.load_audio()
        if audio.ndim != 2:
            raise ValueError(f"Unexpected audio shape for {cut_id}: {audio.shape}")
        if audio.shape[0] != 1:
            audio = audio.mean(axis=0, keepdims=True)
        sample_rate = cut.sampling_rate

        grouped = defaultdict(list)
        for word in reference_words:
            grouped[int(word[1])].append(word)

        # The established reference writer numbers segments by total-duration
        # speaker order and then preserves manifest order within each speaker
        # (see filter_word_aligned_channel.py).  Reconstruct that numbering
        # directly.  Matching only on transcript text is ambiguous for common
        # one-word segments such as "OK" and for repeated synthetic prompts.
        speaker_durations = defaultdict(float)
        for supervision in cut.supervisions:
            speaker_durations[str(supervision.speaker)] += supervision.duration
        speaker_order = sorted(
            speaker_durations, key=speaker_durations.get, reverse=True
        )
        ordered_supervisions = [
            supervision
            for speaker in speaker_order
            for supervision in cut.supervisions
            if str(supervision.speaker) == speaker
        ]
        if len(grouped) != len(ordered_supervisions):
            raise ValueError(
                f"Segment/supervision count mismatch for {cut_id}: "
                f"reference={len(grouped)} manifest={len(ordered_supervisions)}"
            )
        for segment_id in sorted(grouped):
            entries = grouped[segment_id]
            words = [str(x[2]) for x in entries]
            speaker = str(entries[0][0])
            expected = normalized_words(words)
            if segment_id < 0 or segment_id >= len(ordered_supervisions):
                raise ValueError(
                    f"Out-of-range segment ID for {cut_id}: {segment_id}"
                )
            supervision = ordered_supervisions[segment_id]
            observed = normalized_words(supervision.text.split())
            if str(supervision.speaker) != speaker or observed != expected:
                raise ValueError(
                    f"Reference/manifest mismatch for cut={cut_id} "
                    f"segment={segment_id}: reference speaker/text="
                    f"{speaker}/{expected}, manifest speaker/text="
                    f"{supervision.speaker}/{observed}"
                )

            crop_start = max(0.0, float(supervision.start) - args.margin)
            crop_end = min(
                float(cut.duration),
                float(supervision.start + supervision.duration) + args.margin,
            )
            begin_sample = max(0, round(crop_start * sample_rate))
            end_sample = min(audio.shape[1], round(crop_end * sample_rate))
            if end_sample <= begin_sample:
                raise ValueError(f"Empty crop for {cut_id} segment {segment_id}")

            item_id = f"c{cut_index:04d}_s{segment_id:04d}"
            wav_path = corpus_dir / f"{item_id}.wav"
            lab_path = corpus_dir / f"{item_id}.lab"
            sf.write(
                wav_path,
                np.asarray(audio[0, begin_sample:end_sample]),
                sample_rate,
                subtype="PCM_16",
            )
            lab_path.write_text(" ".join(words) + "\n")
            metadata.append(
                {
                    "item_id": item_id,
                    "cut_index": cut_index,
                    "cut_id": cut_id,
                    "segment_id": segment_id,
                    "speaker": speaker,
                    "words": words,
                    "crop_start": begin_sample / sample_rate,
                    "crop_duration": (end_sample - begin_sample) / sample_rate,
                    "sample_rate": sample_rate,
                    "wav": str(wav_path.resolve()),
                    "supervision_start": float(supervision.start),
                    "supervision_duration": float(supervision.duration),
                }
            )

    metadata_path = args.output_dir / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    summary = {
        "manifest": str(args.manifest),
        "reference": str(args.reference),
        "margin_seconds_each_side": args.margin,
        "num_cuts": len({x["cut_id"] for x in metadata}),
        "num_segments": len(metadata),
        "audio_hours": sum(x["crop_duration"] for x in metadata) / 3600,
        "metadata": str(metadata_path),
        "corpus": str(corpus_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
