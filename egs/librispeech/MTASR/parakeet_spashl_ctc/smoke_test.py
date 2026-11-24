#!/usr/bin/env python3
"""Exercise the real pretrained encoder, upsampler, heads, and backward pass."""

import os

import torch

from model import MDCTCModel
from parakeet_encoder import ParakeetWaveformEncoder


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a CUDA GPU")
    device = torch.device("cuda")
    encoder = ParakeetWaveformEncoder.from_pretrained(
        os.environ.get("PARAKEET_MODEL_ID", "nvidia/parakeet-tdt-0.6b-v3"),
        upsample_factor=4,
    )
    model = MDCTCModel(
        encoder=encoder,
        encoder_dim=encoder.output_dim,
        vocab_size1=5000,
        vocab_size2=4,
        factorized_token_speaker_output=True,
        high_resolution_fusion=True,
    ).to(device)
    model.freeze_encoder()
    waveforms = torch.randn(2, 64000, device=device) * 0.01
    lengths = torch.tensor([64000, 48000], device=device)
    with torch.amp.autocast("cuda"):
        output, output_lengths = model(waveforms, lengths)
        loss = -output[0, 0, 1] - output[1, 0, 1]
    loss.backward()
    assert torch.isfinite(output).all()
    assert output_lengths[0] > output_lengths[1] > 0
    assert abs(output_lengths[0].item() - 200) <= 1  # 4 s at 50 Hz
    assert model.ctc_output1[1].weight.grad is not None
    assert model.recovery_adapter.residual_output.weight.grad is not None
    print(
        {
            "output_shape": tuple(output.shape),
            "output_lengths": output_lengths.tolist(),
            "frame_shift_seconds": encoder.timing.output_stride_seconds,
            "loss": loss.item(),
            "peak_cuda_mb": torch.cuda.max_memory_allocated() // 1_000_000,
        }
    )


if __name__ == "__main__":
    main()
