#!/usr/bin/env python3
"""Validate the trained SentencePiece model against its normalized corpus."""

import argparse
from pathlib import Path

import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--expected-vocab-size", type=int, default=5000)
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=str(args.model))
    if sp.vocab_size() != args.expected_vocab_size:
        raise ValueError(f"Expected {args.expected_vocab_size} pieces, got {sp.vocab_size()}")
    lines = tokens = unknown = 0
    with args.corpus.open(encoding="utf-8") as stream:
        for text in stream:
            ids = sp.encode(text.rstrip("\n"))
            lines += 1
            tokens += len(ids)
            unknown += sum(token == sp.unk_id() for token in ids)
    print(f"lines={lines} tokens={tokens} unknown={unknown} vocab={sp.vocab_size()}")
    if unknown:
        raise ValueError(f"Found {unknown} unknown tokens in the BPE training corpus")


if __name__ == "__main__":
    main()
