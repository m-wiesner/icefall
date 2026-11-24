#!/usr/bin/env python3
"""Convert per-supervision MFA JSON output to the common alignment format."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def normalize(word: str) -> str:
    return re.sub(r"[^A-Z']", "", word.upper())


def reconcile_words(words, entries):
    """Map MFA word entries back to the original transcript tokenization.

    MFA sometimes exports a contraction as two entries (e.g. ``THERE`` and
    ``'S``) even though the input transcript contained ``THERE'S``.  It also
    writes an out-of-vocabulary word as ``<unk>``.  Neither case is an
    alignment failure: merge split clitics and retain the original spelling
    for a one-to-one OOV entry.
    """
    observed = [normalize(str(x[2])) for x in entries]
    reconciled = []
    j = 0
    for original_word in words:
        expected = normalize(original_word)
        if j >= len(entries):
            raise ValueError(
                f"MFA words differ: expected={[normalize(x) for x in words]} "
                f"observed={observed}"
            )

        # The usual one-entry case, including an MFA OOV placeholder.
        if observed[j] == expected or observed[j] == "UNK":
            end_j = j + 1
        else:
            # MFA can split clitics/possessives in its JSON output.  Find the
            # shortest consecutive group that reconstructs the input word.
            combined = ""
            end_j = None
            for k in range(j, min(len(entries), j + 4)):
                combined += observed[k]
                if combined == expected:
                    end_j = k + 1
                    break
                if not expected.startswith(combined):
                    break
            if end_j is None:
                raise ValueError(
                    f"MFA words differ: expected={[normalize(x) for x in words]} "
                    f"observed={observed}"
                )

        reconciled.append(
            (original_word, float(entries[j][0]), float(entries[end_j - 1][1]))
        )
        j = end_j

    if j != len(entries):
        raise ValueError(
            f"MFA words differ: expected={[normalize(x) for x in words]} "
            f"observed={observed}"
        )
    return reconciled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--mfa-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    items = json.load(args.metadata.open())
    aligned_by_cut = defaultdict(list)
    failed_by_cut = defaultdict(list)
    for item in items:
        path = args.mfa_output_dir / f"{item['item_id']}.json"
        try:
            data = json.load(path.open())
            word_tiers = [
                tier for name, tier in data["tiers"].items()
                if name == "words" or name.endswith(" - words")
            ]
            if len(word_tiers) != 1:
                raise ValueError(f"Expected one words tier, found {len(word_tiers)}")
            entries = [x for x in word_tiers[0]["entries"] if str(x[2]).strip()]
            for original_word, entry_begin, entry_end in reconcile_words(
                item["words"], entries
            ):
                begin = item["crop_start"] + entry_begin
                end = item["crop_start"] + entry_end
                aligned_by_cut[item["cut_id"]].append(
                    [
                        item["speaker"],
                        int(item["segment_id"]),
                        original_word,
                        round(begin, 6),
                        round(max(0.001, end - begin), 6),
                    ]
                )
        except Exception as error:
            failed_by_cut[item["cut_id"]].append(
                {
                    "item_id": item["item_id"],
                    "segment_id": item["segment_id"],
                    "error": repr(error),
                }
            )

    cut_order = []
    seen = set()
    for item in items:
        if item["cut_id"] not in seen:
            cut_order.append(item["cut_id"])
            seen.add(item["cut_id"])
    result = [
        [cut_id, sorted(aligned_by_cut[cut_id], key=lambda x: (x[1], x[3]))]
        for cut_id in cut_order
        if cut_id not in failed_by_cut
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    with args.output.with_suffix(".failures.json").open("w") as f:
        json.dump(failed_by_cut, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "num_input_cuts": len(cut_order),
                "num_aligned_cuts": len(result),
                "num_failed_cuts": len(failed_by_cut),
                "num_failed_segments": sum(len(x) for x in failed_by_cut.values()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
