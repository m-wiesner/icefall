#!/usr/bin/env python3
"""Force-align ground-truth words with WAV2VEC2_ASR_BASE_960H emissions."""

from __future__ import annotations

import argparse
import json
import re
import wave
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio


def load_pcm16_wav(path: str) -> tuple[torch.Tensor, int]:
    """Read the PCM16 WAVs produced by prepare_external_forced_alignment.py."""
    with wave.open(path, "rb") as stream:
        sample_rate = stream.getframerate()
        num_channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
        frames = stream.readframes(stream.getnframes())
    if sample_width != 2:
        raise ValueError(f"Expected PCM16 WAV, got {sample_width * 8}-bit: {path}")
    wave_tensor = torch.frombuffer(bytearray(frames), dtype=torch.int16)
    wave_tensor = wave_tensor.reshape(-1, num_channels).T.float().div_(32768.0)
    return wave_tensor, sample_rate


def alignment_word(word: str) -> str:
    return re.sub(r"[^A-Z']", "", word.upper())


def make_batches(items, max_items: int, max_audio_seconds: float):
    items = sorted(items, key=lambda x: x["crop_duration"])
    batch = []
    longest = 0.0
    for item in items:
        proposed_longest = max(longest, item["crop_duration"])
        if batch and (
            len(batch) >= max_items
            or proposed_longest * (len(batch) + 1) > max_audio_seconds
        ):
            yield batch
            batch = []
            longest = 0.0
        batch.append(item)
        longest = max(longest, item["crop_duration"])
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batch-items", type=int, default=16)
    parser.add_argument("--max-batch-audio-seconds", type=float, default=120.0)
    args = parser.parse_args()

    items = json.load(args.metadata.open())
    device = torch.device(args.device)
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model(dl_kwargs={"model_dir": str(args.model_dir)}).to(device)
    model.eval()
    labels = bundle.get_labels()
    label_to_id = {label: index for index, label in enumerate(labels)}
    blank = label_to_id["-"]
    delimiter = label_to_id["|"]

    aligned_by_cut = defaultdict(list)
    failed_by_cut = defaultdict(list)
    num_batches = 0
    for batch in make_batches(
        items, args.max_batch_items, args.max_batch_audio_seconds
    ):
        waves = []
        lengths = []
        for item in batch:
            wave, sample_rate = load_pcm16_wav(item["wav"])
            if sample_rate != bundle.sample_rate:
                wave = torchaudio.functional.resample(
                    wave, sample_rate, bundle.sample_rate
                )
            if wave.size(0) != 1:
                wave = wave.mean(0, keepdim=True)
            waves.append(wave[0])
            lengths.append(wave.size(1))
        padded = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True).to(device)
        input_lengths = torch.tensor(lengths, device=device, dtype=torch.long)
        with torch.inference_mode():
            emissions, output_lengths = model(padded, input_lengths)

        for index, item in enumerate(batch):
            try:
                normalized = [alignment_word(word) for word in item["words"]]
                if any(not word for word in normalized):
                    raise ValueError(f"Empty normalized word in {item['words']}")
                target_text = "|".join(normalized)
                unsupported = sorted(set(target_text) - set(label_to_id))
                if unsupported:
                    raise ValueError(f"Unsupported transcript characters: {unsupported}")
                target_ids = torch.tensor(
                    [label_to_id[c] for c in target_text],
                    device=device,
                    dtype=torch.int32,
                ).unsqueeze(0)
                output_length = int(output_lengths[index].item())
                log_probs = emissions[index, :output_length].float().log_softmax(-1)
                path, scores = torchaudio.functional.forced_align(
                    log_probs.unsqueeze(0), target_ids, blank=blank
                )
                spans = torchaudio.functional.merge_tokens(
                    path[0], scores[0], blank=blank
                )
                if [int(span.token) for span in spans] != target_ids[0].tolist():
                    raise RuntimeError("Forced-alignment target/path mismatch")

                seconds_per_frame = (
                    float(lengths[index]) / bundle.sample_rate / output_length
                )
                span_index = 0
                for word_index, (original_word, normalized_word) in enumerate(
                    zip(item["words"], normalized)
                ):
                    word_spans = spans[
                        span_index : span_index + len(normalized_word)
                    ]
                    if len(word_spans) != len(normalized_word):
                        raise RuntimeError("Missing character spans")
                    begin = (
                        item["crop_start"]
                        + int(word_spans[0].start) * seconds_per_frame
                    )
                    end = (
                        item["crop_start"]
                        + int(word_spans[-1].end) * seconds_per_frame
                    )
                    aligned_by_cut[item["cut_id"]].append(
                        [
                            item["speaker"],
                            int(item["segment_id"]),
                            original_word,
                            round(begin, 6),
                            round(max(0.001, end - begin), 6),
                        ]
                    )
                    span_index += len(normalized_word)
                    if word_index + 1 < len(normalized):
                        if int(spans[span_index].token) != delimiter:
                            raise RuntimeError("Missing word delimiter span")
                        span_index += 1
                if span_index != len(spans):
                    raise RuntimeError("Unused forced-alignment spans")
            except Exception as error:
                failed_by_cut[item["cut_id"]].append(
                    {
                        "item_id": item["item_id"],
                        "segment_id": item["segment_id"],
                        "error": repr(error),
                    }
                )
        num_batches += 1
        if num_batches % 10 == 0:
            print(
                f"batches={num_batches} segments_done="
                f"{sum(len(x) for x in aligned_by_cut.values())} "
                f"failed_segments={sum(len(x) for x in failed_by_cut.values())}",
                flush=True,
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
