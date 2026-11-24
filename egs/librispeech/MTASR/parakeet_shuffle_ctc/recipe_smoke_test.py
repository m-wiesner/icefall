#!/usr/bin/env python3
"""Run one real Lhotse mixture through token-only Parakeet shuffle CTC."""

import torch

import train
from asr_datamodule import LibriSpeechAsrDataModule
from mdctc_graph_compiler import MDCTCGraphCompiler


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires CUDA")
    parser = train.get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args([])
    args.max_duration = 20
    args.max_splice_duration = 20
    args.max_splices = 2
    args.min_splices = 2
    args.max_unique = 2
    args.max_num_spks = 4
    args.max_overlap = 3
    args.num_buckets = 4
    args.num_cuts_for_bins_estimate = 40
    args.sampler_buffer_size = 40
    args.num_workers = 0
    args.reverb = False
    args.normalize_loudness = False
    args.speaker_branch = False
    args.downsample = False
    args.encoder_upsample_factor = 4
    args.high_resolution_fusion = True
    args.speaker_layer = 12

    data = LibriSpeechAsrDataModule(args)
    loader = data.train_dataloaders(data.train_cuts())
    batch = next(iter(loader))

    params = train.get_params()
    params.update(vars(args))
    graph_compiler = MDCTCGraphCompiler(args.lang_dir, device="cpu")
    params.vocab_size = graph_compiler.sp.vocab_size()
    params.subsampling_factor = 2
    params.batch_idx_train = 1

    model = train.get_mdctc_model(params).cuda()
    model.freeze_encoder()
    with torch.amp.autocast("cuda"):
        loss, info = train.compute_loss(
            params=params,
            model=model,
            batch=batch,
            graph_compiler=graph_compiler,
            is_training=True,
        )
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite token-only shuffle loss: {loss}")
    if model.speaker_branch or model.ctc_output2 is not None:
        raise RuntimeError("Token-only smoke test unexpectedly constructed a speaker head")
    print(
        {
            "objective": "token_only",
            "batch_size": int(batch["inputs"].size(0)),
            "loss": float(loss.detach()),
            "frames": float(info["frames"]),
            "arc_density": float(info["arc_density"] / info["frames"]),
            "peak_cuda_mb": torch.cuda.max_memory_allocated() // 1_000_000,
        }
    )


if __name__ == "__main__":
    main()
