#!/usr/bin/env python3
"""Extract factorized WavLM token/speaker posteriors for one validation cut."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lhotse import CutSet

from icefall.checkpoint import average_checkpoints
from icefall.utils import AttributeDict
from mdctc_graph_compiler import MDCTCGraphCompiler
from wavlm_spashl_ctc.decode_greedy import factorized_greedy_argmax
from wavlm_spashl_ctc.model import split_joint_labels
from wavlm_spashl_ctc.train import get_mdctc_model, get_params


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cut-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lang-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-wav", type=Path, required=True)
    parser.add_argument("--max-num-spks", type=int, default=8)
    parser.add_argument("--speaker-weight", type=float, default=1.0)
    parser.add_argument("--frame-duration", type=float, default=0.02)
    return parser.parse_args()


def find_cut(manifest: Path, cut_id: str):
    for cut in CutSet.from_file(manifest):
        if cut.id == cut_id:
            return cut
    raise KeyError(f"Cut not found: {cut_id}")


def speaker_order(cut) -> list[str]:
    durations: dict[str, float] = {}
    first_onsets: dict[str, float] = {}
    for supervision in cut.supervisions:
        speaker = supervision.speaker or "noise"
        durations[speaker] = durations.get(speaker, 0.0) + supervision.duration
        first_onsets[speaker] = min(
            first_onsets.get(speaker, float("inf")), supervision.start
        )
    return sorted(
        durations,
        key=lambda speaker: (
            -durations[speaker],
            first_onsets[speaker],
            speaker,
        ),
    )


def overlap_statistics(cut) -> tuple[float, int]:
    events = []
    for supervision in cut.supervisions:
        events.extend(
            [
                (max(0.0, supervision.start), 1),
                (min(cut.duration, supervision.end), -1),
            ]
        )
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    last = 0.0
    overlap = 0.0
    maximum = 0
    for time, delta in events:
        if active >= 2:
            overlap += time - last
        active += delta
        maximum = max(maximum, active)
        last = time
    return overlap / cut.duration, maximum


def collapse_emissions(
    joint_labels: torch.Tensor,
    token_scores: torch.Tensor,
    speaker_scores: torch.Tensor,
    sentencepiece,
    frame_duration: float,
) -> list[dict]:
    previous = torch.cat(
        [torch.zeros(1, dtype=joint_labels.dtype, device=joint_labels.device), joint_labels[:-1]]
    )
    indices = torch.nonzero(
        (joint_labels != 0) & ((joint_labels != previous) | (previous == 0)),
        as_tuple=False,
    ).flatten()
    labels = joint_labels[indices]
    speaker_ids, token_ids = split_joint_labels(labels, token_scores.size(-1))
    emissions = []
    for frame, speaker_id, token_id in zip(indices, speaker_ids, token_ids):
        frame_index = int(frame.item())
        speaker = int(speaker_id.item())
        token = int(token_id.item())
        token_probability = float(token_scores[frame_index, token].exp().item())
        speaker_probability = float(speaker_scores[frame_index, speaker].exp().item())
        blank_probability = float(token_scores[frame_index, 0].exp().item())
        emissions.append(
            {
                "frame": frame_index,
                "time": frame_index * frame_duration,
                "speaker_slot": speaker,
                "token_id": token,
                "piece": sentencepiece.id_to_piece(token),
                "token_probability": token_probability,
                "speaker_probability": speaker_probability,
                "blank_probability": blank_probability,
                "joint_log_score": float(
                    token_scores[frame_index, token].item()
                    + speaker_scores[frame_index, speaker].item()
                ),
            }
        )
    return emissions


def merge_words(emissions: list[dict], duration: float) -> list[dict]:
    """Merge BPE pieces per speaker while retaining global interleaving."""
    current: dict[int, dict] = {}
    words = []
    for emission in emissions:
        speaker = emission["speaker_slot"]
        piece = emission["piece"]
        starts_word = piece.startswith("▁")
        if starts_word and speaker in current:
            words.append(current.pop(speaker))
        if speaker not in current:
            current[speaker] = {
                "speaker_slot": speaker,
                "word": piece.lstrip("▁"),
                "start": emission["time"],
                "end": emission["time"] + 0.02,
                "pieces": [piece],
                "mean_speaker_probability": emission["speaker_probability"],
                "mean_token_probability": emission["token_probability"],
                "num_pieces": 1,
            }
        else:
            word = current[speaker]
            word["word"] += piece.lstrip("▁")
            word["end"] = emission["time"] + 0.02
            word["pieces"].append(piece)
            word["mean_speaker_probability"] += emission["speaker_probability"]
            word["mean_token_probability"] += emission["token_probability"]
            word["num_pieces"] += 1
    for word in current.values():
        words.append(word)
    for word in words:
        count = word.pop("num_pieces")
        word["mean_speaker_probability"] /= count
        word["mean_token_probability"] /= count
        word["end"] = min(duration, word["end"])
    return sorted(words, key=lambda item: (item["start"], item["speaker_slot"]))


def main() -> None:
    args = get_args()
    cut = find_cut(args.manifest, args.cut_id)
    audio = cut.load_audio()
    if audio.shape[0] != 1:
        audio = audio.mean(axis=0, keepdims=True)
    waveform = torch.from_numpy(audio[0]).unsqueeze(0).to(torch.float32)
    lengths = torch.tensor([waveform.size(1)], dtype=torch.int64)

    graph_compiler = MDCTCGraphCompiler(args.lang_dir, device="cpu")
    params = get_params()
    params.update(
        AttributeDict(
            {
                "use_large": True,
                "use_layer_norm": False,
                "downsample": False,
                "factorized_token_speaker_output": True,
                "max_num_spks": args.max_num_spks,
                "vocab_size": graph_compiler.sp.vocab_size(),
                "pretrained_dir": None,
                "diarization_loss": "none",
            }
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_mdctc_model(params).to(device)
    model.load_state_dict(average_checkpoints([args.checkpoint], device=device))
    model.eval()
    with torch.inference_mode():
        token_scores, speaker_scores, output_lengths = model(
            waveform.to(device),
            lengths.to(device),
            speaker_weight=args.speaker_weight,
            return_factors=True,
        )
    frames = int(output_lengths[0].item())
    token_scores = token_scores[0, :frames].float().cpu()
    speaker_scores = speaker_scores[0, :frames].float().cpu()
    joint_labels = factorized_greedy_argmax(
        token_scores.unsqueeze(0), speaker_scores.unsqueeze(0)
    )[0]
    emissions = collapse_emissions(
        joint_labels,
        token_scores,
        speaker_scores,
        graph_compiler.sp,
        args.frame_duration,
    )
    words = merge_words(emissions, cut.duration)
    overlap_fraction, max_concurrent = overlap_statistics(cut)
    order = speaker_order(cut)

    output = {
        "model": str(args.checkpoint),
        "manifest": str(args.manifest),
        "cut_id": cut.id,
        "duration": cut.duration,
        "sampling_rate": cut.sampling_rate,
        "frame_duration": args.frame_duration,
        "num_frames": frames,
        "overlap_fraction": overlap_fraction,
        "max_concurrent_speakers": max_concurrent,
        "speaker_ordering": "descending total supervision duration",
        "slot_to_reference_speaker": {
            str(index): speaker for index, speaker in enumerate(order)
        },
        "reference_supervisions": [
            {
                "speaker": supervision.speaker,
                "slot": order.index(supervision.speaker),
                "start": supervision.start,
                "duration": supervision.duration,
                "text": supervision.text,
            }
            for supervision in cut.supervisions
        ],
        "times": (np.arange(frames) * args.frame_duration).tolist(),
        "blank_probability": token_scores[:, 0].exp().numpy().tolist(),
        "speaker_probability": speaker_scores.exp().numpy().tolist(),
        "greedy_emissions": emissions,
        "greedy_words": words,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    import soundfile as sf

    sf.write(args.output_wav, audio[0], cut.sampling_rate, subtype="PCM_16")
    print(json.dumps({
        "output_json": str(args.output_json),
        "output_wav": str(args.output_wav),
        "frames": frames,
        "emissions": len(emissions),
        "words": len(words),
        "overlap_fraction": overlap_fraction,
    }, indent=2))


if __name__ == "__main__":
    main()
