#!/usr/bin/env python3
"""Probe Parakeet BatchNorm buffers with synthetic training-mode batches."""

from __future__ import annotations

import argparse
import math

import torch

from parakeet_spashl_ctc.parakeet_encoder import ParakeetWaveformEncoder


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-id", default="nvidia/parakeet-tdt-0.6b-v3")
    return parser.parse_args()


def bn_modules(model):
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]


def save_buffers(modules):
    return {
        name: (
            module.running_mean.detach().clone(),
            module.running_var.detach().clone(),
            module.num_batches_tracked.detach().clone(),
        )
        for name, module in modules
    }


def restore_buffers(modules, saved):
    for name, module in modules:
        mean, var, count = saved[name]
        module.running_mean.copy_(mean)
        module.running_var.copy_(var)
        module.num_batches_tracked.copy_(count)


def make_batch(kind, device):
    generator = torch.Generator().manual_seed(0)
    if kind == "b1_1s_random":
        x = torch.randn(1, 16000, generator=generator) * 0.05
        lengths = torch.tensor([16000])
    elif kind == "b1_20s_random":
        x = torch.randn(1, 320000, generator=generator) * 0.05
        lengths = torch.tensor([320000])
    elif kind == "b1_10s_sine":
        t = torch.arange(160000) / 16000
        x = (0.1 * torch.sin(2 * math.pi * 440 * t)).unsqueeze(0)
        lengths = torch.tensor([160000])
    elif kind == "b1_10s_zero":
        x = torch.zeros(1, 160000)
        lengths = torch.tensor([160000])
    elif kind == "b2_padded":
        x = torch.zeros(2, 320000)
        x[0, :16000] = torch.randn(16000, generator=generator) * 0.05
        x[1] = torch.randn(320000, generator=generator) * 0.05
        lengths = torch.tensor([16000, 320000])
    else:
        raise ValueError(kind)
    return x.to(device), lengths.to(device)


def run_case(model, modules, original_buffers, kind, use_fp16):
    restore_buffers(modules, original_buffers)
    observations = {}
    handles = []

    def pre_hook(name):
        def hook(_module, inputs):
            value = inputs[0]
            observations[name] = {
                "input_dtype": str(value.dtype),
                "input_nonfinite": int((~torch.isfinite(value)).sum()),
                "input_absmax": float(value.float().abs().max()),
            }
        return hook

    for name, module in modules:
        handles.append(module.register_forward_pre_hook(pre_hook(name)))

    x, lengths = make_batch(kind, next(model.parameters()).device)
    model.train()
    error = None
    output_finite = None
    try:
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_fp16
        ):
            states, _ = model(x, lengths)
        output_finite = bool(torch.isfinite(states).all())
    except Exception as exception:  # diagnostic: report rather than abort suite
        error = repr(exception)
    finally:
        for handle in handles:
            handle.remove()

    bad_buffers = []
    for name, module in modules:
        mean_bad = int((~torch.isfinite(module.running_mean)).sum())
        var_bad = int((~torch.isfinite(module.running_var)).sum())
        if mean_bad or var_bad:
            bad_buffers.append((name, mean_bad, var_bad))
    bad_inputs = [
        (name, value)
        for name, value in observations.items()
        if value["input_nonfinite"]
    ]
    dtypes = sorted({value["input_dtype"] for value in observations.values()})
    max_input = max(
        (value["input_absmax"] for value in observations.values()), default=float("nan")
    )
    print(
        f"CASE kind={kind} fp16={use_fp16} error={error} "
        f"output_finite={output_finite} bn_input_dtypes={dtypes} "
        f"max_bn_input_abs={max_input:.8g} bad_input_layers={len(bad_inputs)} "
        f"bad_buffer_layers={len(bad_buffers)}"
    )
    for name, value in bad_inputs[:3]:
        print(f"BAD_INPUT layer={name} {value}")
    for name, mean_bad, var_bad in bad_buffers[:3]:
        print(f"BAD_BUFFER layer={name} mean_bad={mean_bad} var_bad={var_bad}")


def main():
    options = args()
    device = torch.device("cuda")
    model = ParakeetWaveformEncoder.from_pretrained(options.model_id).to(device)
    modules = bn_modules(model)
    pretrained_buffers = save_buffers(modules)
    print(f"MODEL pretrained bn_layers={len(modules)}")
    for fp16 in (False, True):
        for kind in (
            "b1_1s_random",
            "b1_20s_random",
            "b1_10s_sine",
            "b1_10s_zero",
            "b2_padded",
        ):
            run_case(model, modules, pretrained_buffers, kind, fp16)

    if options.checkpoint:
        checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=False)
        encoder_state = {
            key[len("encoder.encoder.") :]: value
            for key, value in checkpoint["model"].items()
            if key.startswith("encoder.encoder.")
        }
        model.encoder.load_state_dict(encoder_state, strict=True)
        # Test the fine-tuned weights with known-finite pretrained BN buffers.
        restore_buffers(modules, pretrained_buffers)
        print(f"MODEL checkpoint={options.checkpoint} bn_buffers=pretrained")
        for fp16 in (False, True):
            for kind in ("b1_1s_random", "b1_20s_random", "b2_padded"):
                run_case(model, modules, pretrained_buffers, kind, fp16)


if __name__ == "__main__":
    main()
