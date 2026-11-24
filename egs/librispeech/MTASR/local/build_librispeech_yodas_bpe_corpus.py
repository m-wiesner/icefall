#!/usr/bin/env python3
"""Create normalized BPE training text from LibriSpeech-960 and filtered YODAS."""

import argparse
import gzip
import json
import sys
from pathlib import Path

from lhotse import SupervisionSet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text_normalization import normalize_english_transcript  # noqa: E402


LIBRISPEECH_PARTS = ("train-clean-100", "train-clean-360", "train-other-500")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--yodas-scheme-dir", type=Path, required=True)
    parser.add_argument("--yodas-hypothesis", default="doverlap")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalized(text):
    text = normalize_english_transcript(text)
    if text:
        yield text


def iter_librispeech(manifest_dir):
    for part in LIBRISPEECH_PARTS:
        path = manifest_dir / f"librispeech_supervisions_{part}.jsonl.gz"
        for supervision in SupervisionSet.from_file(path):
            yield from normalized(supervision.text)


def iter_yodas(scheme_dir, hypothesis):
    meta_dir = scheme_dir.parent / f"{scheme_dir.name}.meta"
    keep_ids = set((meta_dir / "keep_ids.txt").read_text().splitlines())
    for path in sorted(scheme_dir.glob("cuts.*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                cut = json.loads(line)
                if cut["id"] not in keep_ids:
                    continue
                hyp = cut["custom"]["hypotheses"][hypothesis]
                for segment in hyp["segments"]:
                    yield from normalized(segment.get("text", ""))


def main():
    args = get_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    libri_count = yodas_count = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for text in iter_librispeech(args.manifest_dir):
            print(text, file=stream)
            libri_count += 1
        for text in iter_yodas(args.yodas_scheme_dir, args.yodas_hypothesis):
            print(text, file=stream)
            yodas_count += 1
    print(f"Wrote {libri_count} LibriSpeech and {yodas_count} YODAS transcripts")


if __name__ == "__main__":
    main()
