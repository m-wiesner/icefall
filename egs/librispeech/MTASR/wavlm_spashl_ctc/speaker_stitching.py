"""Embedding-based stitching of chunk-local speaker output slots."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist


DEFAULT_ECAPA = Path(
    "/home/hltcoe/mwiesner/.cache/huggingface/hub/"
    "models--speechbrain--spkrec-ecapa-voxceleb/snapshots/"
    "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
)
DEFAULT_VBX = Path(__file__).resolve().parent.parent / ".tools/VBx"


class EcapaSpeakerEmbedder:
    """Thin adapter around the locally cached SpeechBrain ECAPA model."""

    def __init__(self, model_dir: str | Path, device: torch.device):
        package_dir = Path(__file__).resolve().parent.parent / ".tools/speaker-stitch"
        if str(package_dir) not in sys.path:
            sys.path.insert(0, str(package_dir))
        from speechbrain.inference.speaker import EncoderClassifier

        model_dir = Path(model_dir)
        if not (model_dir / "hyperparams.yaml").is_file():
            raise FileNotFoundError(f"Missing cached ECAPA model at {model_dir}")
        self.classifier = EncoderClassifier.from_hparams(
            source=str(model_dir),
            # Array tasks can share a compute node.  A process-unique staging
            # directory avoids SpeechBrain's non-atomic symlink creation race.
            savedir=str(Path("/tmp") / f"mtasr-ecapa-voxceleb-{os.getpid()}"),
            run_opts={"device": str(device)},
            overrides={"pretrained_path": str(model_dir)},
        )
        self.device = device

    @property
    def min_num_samples(self) -> int:
        """Shortest input accepted by ECAPA (pyannote's runtime convention)."""
        if not hasattr(self, "_min_num_samples"):
            lower, upper = 2, round(0.5 * 16000)
            with torch.inference_mode():
                while lower + 1 < upper:
                    middle = (lower + upper) // 2
                    try:
                        self.classifier.encode_batch(
                            torch.randn(1, middle, device=self.device)
                        )
                        upper = middle
                    except RuntimeError:
                        lower = middle
            self._min_num_samples = upper
            logging.info("ECAPA minimum valid input is %d samples (%.4fs)",
                         upper, upper / 16000)
        return self._min_num_samples

    @torch.no_grad()
    def __call__(self, clips: list[torch.Tensor], batch_size: int = 32) -> np.ndarray:
        output = []
        for begin in range(0, len(clips), batch_size):
            batch_clips = clips[begin : begin + batch_size]
            lengths = torch.tensor(
                [clip.numel() for clip in batch_clips],
                device=self.device, dtype=torch.float32,
            )
            padded = torch.zeros(
                len(batch_clips), int(lengths.max().item()),
                device=self.device, dtype=torch.float32,
            )
            for index, clip in enumerate(batch_clips):
                padded[index, : clip.numel()] = clip.to(self.device).float()
            embeddings = self.classifier.encode_batch(
                padded, wav_lens=lengths / lengths.max(), normalize=True
            ).squeeze(1)
            output.append(embeddings.float().cpu().numpy())
        return np.concatenate(output, axis=0)


def _speaker_activity(
    block, blank_threshold: float, speaker_threshold: float,
    smoothing_seconds: float, frame_duration: float,
) -> torch.Tensor:
    if "embedding_activity" in block:
        return block["embedding_activity"]
    speech = block["diarization_blank_probability"] < blank_threshold
    active = (
        block["diarization_speaker_probability"] > speaker_threshold
    ) & speech.unsqueeze(-1)
    smoothing_frames = round(smoothing_seconds / frame_duration)
    if smoothing_frames > 0:
        active = F.max_pool1d(
            active.T.float().unsqueeze(0),
            kernel_size=2 * smoothing_frames + 1,
            stride=1,
            padding=smoothing_frames,
        ).squeeze(0).T.bool()
    return active


@torch.no_grad()
def diarize_and_map_chunk_speaker_slots_vbx(
    blocks,
    waveform: torch.Tensor,
    frame_duration: float,
    num_clusters: int | None = None,
    vbx_dir: str | Path = DEFAULT_VBX,
    blank_threshold: float = 0.4,
    speaker_threshold: float = 0.45,
    smoothing_seconds: float = 0.5,
    min_clean_seconds: float = 1.44,
    sample_rate: int = 16000,
):
    """Run the official 16-kHz VBx recipe and map local slots to VBx speakers.

    The union of the posterior- or token-derived local activity is used only as
    VAD.  The chronological x-vector sequence is extracted with VBx's bundled
    ResNet101 ONNX model, then clustered with its bundled transform and PLDA
    using the AMI AHC+VB parameters.
    """
    vbx_dir = Path(vbx_dir).resolve()
    predict = vbx_dir / "VBx/predict_batched.py"
    vbhmm = vbx_dir / "VBx/vbhmm.py"
    backend = vbx_dir / "VBx/models/ResNet101_16kHz"
    required = [predict, vbhmm, backend / "nnet/raw_81.pth",
                backend / "transform.h5", backend / "plda"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete VBx checkout; missing {missing}")

    activities = []
    vad_activities = []
    min_clean_frames = max(1, round(min_clean_seconds / frame_duration))
    for block in blocks:
        if "embedding_activity" in block:
            active = block["embedding_activity"].cpu()
            clean = block.get("embedding_clean_activity")
            clean = clean.cpu() if clean is not None else None
        else:
            active = _speaker_activity(
                block, blank_threshold, speaker_threshold,
                smoothing_seconds, frame_duration,
            ).cpu()
            clean = None
        activities.append(active)

        # Build VBx's VAD preferentially from confident, neighbor-consistent
        # token regions.  For each local slot, add ordinary non-overlapped
        # token speech and finally overlapped speech only when the cleaner
        # masks do not contain one full x-vector window.
        if clean is None:
            vad_activities.append(active)
            continue
        multiplicity = active.sum(-1)
        selected = torch.zeros_like(active)
        for slot in range(active.size(-1)):
            preferred = clean[:, slot] & (clean.sum(-1) == 1)
            selected[:, slot] |= preferred
            if int(selected[:, slot].sum()) < min_clean_frames:
                selected[:, slot] |= active[:, slot] & (multiplicity == 1)
            if int(selected[:, slot].sum()) < min_clean_frames:
                selected[:, slot] |= active[:, slot]
        vad_activities.append(selected)
    speech_frames = set()
    for block, active in zip(blocks, vad_activities):
        absolute = block["diarization_absolute_frames"].cpu().tolist()
        for local_frame in torch.nonzero(active.any(-1), as_tuple=False).flatten().tolist():
            speech_frames.add(int(absolute[local_frame]))
    if not speech_frames:
        raise RuntimeError("VBx received an empty posterior-derived VAD")
    ordered = sorted(speech_frames)
    runs, begin, previous = [], ordered[0], ordered[0]
    for frame in ordered[1:]:
        if frame != previous + 1:
            runs.append((begin, previous + 1))
            begin = frame
        previous = frame
    runs.append((begin, previous + 1))

    import soundfile as sf
    with tempfile.TemporaryDirectory(prefix="mtasr-vbx-") as directory:
        directory = Path(directory)
        wav_dir, lab_dir, output_dir = directory / "wav", directory / "lab", directory / "out"
        wav_dir.mkdir(); lab_dir.mkdir(); output_dir.mkdir()
        recording = "recording"
        sf.write(wav_dir / f"{recording}.wav", waveform.float().cpu().numpy(), sample_rate)
        with open(lab_dir / f"{recording}.lab", "w") as stream:
            for start, end in runs:
                stream.write(f"{start * frame_duration:.3f} {end * frame_duration:.3f}\n")
        (directory / "list.txt").write_text(f"{recording}\n")
        ark, segments = directory / "xvectors.ark", directory / "segments"
        extraction = subprocess.run([
            sys.executable, str(predict), "--in-file-list", str(directory / "list.txt"),
            "--in-lab-dir", str(lab_dir), "--in-wav-dir", str(wav_dir),
            "--out-ark-fn", str(ark), "--out-seg-fn", str(segments),
            "--weights", str(backend / "nnet/raw_81.pth"), "--batch-size", "16",
        ], cwd=vbx_dir, capture_output=True, text=True)
        if extraction.returncode:
            raise RuntimeError(
                "Batched VBx x-vector extraction failed:\n"
                f"stdout:\n{extraction.stdout}\nstderr:\n{extraction.stderr}"
            )
        vbhmm_command = [
            sys.executable, str(vbhmm), "--init", "AHC+VB",
            "--out-rttm-dir", str(output_dir), "--xvec-ark-file", str(ark),
            "--segments-file", str(segments), "--xvec-transform", str(backend / "transform.h5"),
            "--plda-file", str(backend / "plda"), "--threshold", "-0.015",
            "--init-smoothing", "7.0", "--lda-dim", "128", "--Fa", "0.4",
            "--Fb", "64", "--loopP", "0.65",
        ]
        if num_clusters is not None:
            vbhmm_command += ["--num-speakers", str(num_clusters)]
        subprocess.run(
            vbhmm_command, cwd=vbx_dir, check=True,
            capture_output=True, text=True,
        )
        intervals = []
        with open(output_dir / f"{recording}.rttm") as stream:
            for line in stream:
                fields = line.split()
                start, duration = float(fields[3]), float(fields[4])
                intervals.append((start, start + duration, int(fields[7]) - 1))

    if not intervals:
        raise RuntimeError("VBx produced no diarization intervals")
    # RTTM labels can be sparse after VB-HMM resegmentation.  Normalize them
    # before allocating score vectors and report the number actually occupied.
    occupied_labels = sorted({label for _, _, label in intervals})
    label_remap = {label: index for index, label in enumerate(occupied_labels)}
    intervals = [
        (start, end, label_remap[label]) for start, end, label in intervals
    ]
    num_clusters = len(occupied_labels)
    duration_by_cluster = [0.0] * num_clusters
    intervals_by_cluster = [0] * num_clusters
    for start, end, label in intervals:
        duration_by_cluster[label] += end - start
        intervals_by_cluster[label] += 1
    logging.info(
        "VBx RTTM cluster durations=%s interval counts=%s",
        [round(value, 2) for value in duration_by_cluster],
        intervals_by_cluster,
    )
    num_slots = activities[0].size(-1)
    maps = []
    for block, active in zip(blocks, activities):
        votes = torch.zeros(num_slots, num_clusters, dtype=torch.long)
        times = block["diarization_absolute_frames"].cpu().float() * frame_duration
        for start, end, label in intervals:
            inside = (times >= start) & (times < end)
            if inside.any():
                votes[:, label] += active[inside].sum(0).long()
        mapping = torch.full((num_slots,), -1, dtype=torch.long)
        has_votes = votes.sum(-1) > 0
        mapping[has_votes] = votes[has_votes].argmax(-1)
        maps.append(mapping)
    logging.info(
        "VBx stitching produced %d speakers from %d VAD regions; "
        "active slots per block=%s token_derived=%s min_clean_seconds=%.2f",
        num_clusters, len(runs), [int((mapping >= 0).sum()) for mapping in maps],
        any("embedding_activity" in block for block in blocks), min_clean_seconds,
    )
    return maps, num_clusters, intervals


@torch.no_grad()
def cluster_chunk_speaker_slots(
    blocks,
    waveform: torch.Tensor,
    num_clusters: int,
    frame_duration: float,
    embedder: EcapaSpeakerEmbedder,
    min_seconds: float = 0.5,
    max_seconds: float = 5.0,
    blank_threshold: float = 0.4,
    speaker_threshold: float = 0.45,
    smoothing_seconds: float = 0.5,
    clustering_threshold: float = 0.7153814381597874,
    min_cluster_size: int = 15,
    sample_rate: int = 16000,
):
    """Cluster one ECAPA embedding per active (graph block, local slot).

    Full-rate local posteriors are thresholded and smoothed to diarize each
    acoustic core.  Embeddings use single-speaker regions whenever possible;
    overlapped regions are added only when needed to reach ``min_seconds``.
    No reference identity, timing, or local-to-global permutation is used.
    """
    if num_clusters is not None and num_clusters < 1:
        raise ValueError("Need at least one transcript speaker")
    minimum_samples = max(embedder.min_num_samples, round(min_seconds * sample_rate))
    min_frames = max(1, int(np.ceil(minimum_samples / sample_rate / frame_duration)))
    max_frames = (
        max(min_frames, round(max_seconds / frame_duration))
        if max_seconds > 0 else None
    )
    clips, keys = [], []
    extraction_stats = []
    num_slots = blocks[0]["diarization_speaker_probability"].size(-1)

    def contiguous_runs(mask: torch.Tensor):
        indexes = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        if not indexes:
            return []
        runs, begin, previous = [], indexes[0], indexes[0]
        for index in indexes[1:]:
            if index != previous + 1:
                runs.append((begin, previous + 1))
                begin = index
            previous = index
        runs.append((begin, previous + 1))
        return runs

    def select_runs(mask: torch.Tensor, budget: int | None):
        # Prefer long coherent regions for robust embeddings, then restore
        # chronological order before concatenating their waveform samples.
        selected = []
        remaining = int(mask.sum()) if budget is None else budget
        for begin, end in sorted(
            contiguous_runs(mask), key=lambda item: item[1] - item[0], reverse=True
        ):
            if remaining <= 0:
                break
            use = min(end - begin, remaining)
            selected.append((begin, begin + use))
            remaining -= use
        used = int(mask.sum()) if budget is None else budget - remaining
        return sorted(selected), used

    for block_index, block in enumerate(blocks):
        posterior = block["diarization_speaker_probability"]
        if "embedding_activity" in block:
            active = block["embedding_activity"]
        else:
            active = _speaker_activity(
                block, blank_threshold, speaker_threshold,
                smoothing_seconds, frame_duration,
            )
        multiplicity = active.sum(-1)
        clean_activity = block.get("embedding_clean_activity")
        clean_multiplicity = (
            clean_activity.sum(-1) if clean_activity is not None else None
        )
        for slot in range(num_slots):
            exclusive = active[:, slot] & (multiplicity == 1)
            preferred = (
                clean_activity[:, slot] & (clean_multiplicity == 1)
                if clean_activity is not None else exclusive
            )
            overlap = active[:, slot] & (multiplicity > 1)
            exclusive_runs, exclusive_count = select_runs(preferred, max_frames)
            # If stable token neighborhoods do not provide enough audio, first
            # fall back to all token-derived single-speaker regions, and only
            # then to regions that may contain overlap.
            if exclusive_count <= min_frames and clean_activity is not None:
                exclusive_runs, exclusive_count = select_runs(exclusive, max_frames)
            # pyannote uses clean speech only when it is long enough; otherwise
            # it falls back to the complete speaker mask (clean + overlap).
            selected_runs = list(exclusive_runs)
            overlap_count = 0
            if exclusive_count <= min_frames:
                total_budget = max_frames
                selected_runs, total_count = select_runs(active[:, slot], total_budget)
                overlap_count = max(0, total_count - int(exclusive.sum()))
                exclusive_count = min(int(exclusive.sum()), total_count)
            total_count = exclusive_count + overlap_count
            if total_count < min_frames:
                extraction_stats.append((block_index, slot, exclusive_count, overlap_count, False))
                continue
            pieces = []
            absolute = block["diarization_absolute_frames"]
            for begin, end in sorted(selected_runs):
                start_frame = int(absolute[begin])
                end_frame = int(absolute[end - 1]) + 1
                start = max(0, round(start_frame * frame_duration * sample_rate))
                end = min(waveform.numel(), round(end_frame * frame_duration * sample_rate))
                if end > start:
                    pieces.append(waveform[start:end])
            if pieces:
                clip = torch.cat(pieces)
                if clip.numel() >= round(min_seconds * sample_rate):
                    clips.append(clip)
                    keys.append((block_index, slot))
                    extraction_stats.append((block_index, slot, exclusive_count, overlap_count, True))

    if not clips:
        raise RuntimeError("No chunk-local speaker slot had enough speech to embed")
    embeddings = embedder(clips, batch_size=32)
    requested_clusters = (
        None if num_clusters is None else min(num_clusters, len(embeddings))
    )
    if requested_clusters == 1:
        labels = np.zeros(len(embeddings), dtype=np.int64)
    else:
        labels = _pyannote_ecapa_ahc(
            embeddings, num_clusters=requested_clusters,
            threshold=clustering_threshold,
            min_cluster_size=min_cluster_size,
        )
    actual_clusters = int(labels.max()) + 1

    maps = [torch.full((num_slots,), -1, dtype=torch.long) for _ in blocks]
    for (block_index, slot), label in zip(keys, labels):
        maps[block_index][slot] = int(label)
    logging.info(
        "Speaker stitching embedded %d chunk-local slots into %d clusters; "
        "active slots per block=%s; exclusive=%.2fs overlap_fallback=%.2fs "
        "skipped_slots=%d thresholds=(blank<%.3f,speaker>%.3f) smoothing=%.3fs "
        "requested_clusters=%s",
        len(keys), actual_clusters,
        [int((mapping >= 0).sum()) for mapping in maps],
        sum(item[2] for item in extraction_stats if item[4]) * frame_duration,
        sum(item[3] for item in extraction_stats if item[4]) * frame_duration,
        sum(not item[4] for item in extraction_stats),
        blank_threshold, speaker_threshold, smoothing_seconds,
        requested_clusters,
    )
    return maps, actual_clusters


def _pyannote_ecapa_ahc(
    embeddings: np.ndarray,
    num_clusters: int | None,
    threshold: float = 0.7153814381597874,
    min_cluster_size: int = 15,
) -> np.ndarray:
    """Pyannote 2.1-style ECAPA clustering with a known speaker count."""
    embeddings = embeddings.copy()
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-12)
    dendrogram = linkage(embeddings, method="centroid", metric="euclidean")
    if num_clusters is not None:
        # Oracle-K means exactly K, not merely using K as a soft target.  The
        # previous threshold search frequently returned fewer speakers.
        # Do not use scipy.cut_tree here: centroid linkage may have distance
        # inversions, for which its height-based cut can also return fewer
        # than K groups.  Applying exactly N-K chronological merges is an
        # unambiguous cut of the already constructed hierarchy.
        num_items = len(embeddings)
        components = {index: [index] for index in range(num_items)}
        for merge_index, row in enumerate(dendrogram[: num_items - num_clusters]):
            left, right = int(row[0]), int(row[1])
            components[num_items + merge_index] = (
                components.pop(left) + components.pop(right)
            )
        labels = np.empty(num_items, dtype=np.int64)
        for label, members in enumerate(components.values()):
            labels[members] = label
        assert len(np.unique(labels)) == num_clusters
        return labels
    clusters = fcluster(dendrogram, threshold, criterion="distance") - 1
    effective_min = min(min_cluster_size, max(1, round(0.1 * len(embeddings))))

    def large(labels):
        unique, counts = np.unique(labels, return_counts=True)
        return unique[counts >= effective_min], unique, counts

    large_clusters, unique, counts = large(clusters)
    if num_clusters is not None and len(large_clusters) != num_clusters:
        indexed = dendrogram.copy()
        indexed[:, 2] = np.arange(len(embeddings) - 1)
        best_labels, best_delta = clusters, abs(len(large_clusters) - num_clusters)
        for iteration in np.argsort(np.abs(dendrogram[:, 2] - threshold)):
            if indexed[iteration, 3] < effective_min:
                continue
            candidate = fcluster(indexed, iteration, criterion="distance") - 1
            candidate_large, _, _ = large(candidate)
            delta = abs(len(candidate_large) - num_clusters)
            if delta < best_delta:
                best_labels, best_delta = candidate, delta
            if delta == 0:
                break
        clusters = best_labels
        large_clusters, unique, counts = large(clusters)

    if not len(large_clusters):
        clusters[:] = 0
    else:
        small = unique[counts < effective_min]
        if len(small):
            large_centroids = np.vstack([
                embeddings[clusters == label].mean(0) for label in large_clusters
            ])
            small_centroids = np.vstack([
                embeddings[clusters == label].mean(0) for label in small
            ])
            nearest = cdist(large_centroids, small_centroids, metric="cosine").argmin(0)
            for small_label, large_index in zip(small, nearest):
                clusters[clusters == small_label] = large_clusters[large_index]
    _, clusters = np.unique(clusters, return_inverse=True)

    # Match pyannote's final assignment pass against cluster centroids.
    centroids = np.vstack([
        embeddings[clusters == label].mean(0) for label in range(clusters.max() + 1)
    ])
    return cdist(embeddings, centroids, metric="cosine").argmin(1)
