#!/usr/bin/env python3
"""Compare pretrained and checkpointed Parakeet BatchNorm buffers."""

from __future__ import annotations

import argparse
from collections import OrderedDict

import torch
from transformers import AutoConfig, ParakeetEncoder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--model-id", default="nvidia/parakeet-tdt-0.6b-v3")
    return parser.parse_args()


def summarize(name, state, reference):
    prefix = "encoder.encoder."
    rows = []
    for key, value in state.items():
        if not key.startswith(prefix) or not key.endswith("running_var"):
            continue
        short = key[len(prefix) : -len("running_var")]
        mean_key = key[: -len("running_var")] + "running_mean"
        count_key = key[: -len("running_var")] + "num_batches_tracked"
        ref_var = reference[short + "running_var"].float()
        ref_mean = reference[short + "running_mean"].float()
        var = value.float()
        mean = state[mean_key].float()
        count = int(state[count_key])
        rows.append(
            OrderedDict(
                layer=short.rstrip("."),
                batches=count,
                finite=bool(torch.isfinite(var).all() and torch.isfinite(mean).all()),
                mean_nonfinite=int((~torch.isfinite(mean)).sum()),
                mean_elements=mean.numel(),
                var_nonfinite=int((~torch.isfinite(var)).sum()),
                var_elements=var.numel(),
                mean_min=float(mean.min()),
                mean_max=float(mean.max()),
                mean_rms=float(mean.square().mean().sqrt()),
                var_min=float(var.min()),
                var_max=float(var.max()),
                var_median=float(var.median()),
                var_mean=float(var.mean()),
                var_ratio_median=float((var / ref_var.clamp_min(1e-30)).median()),
                mean_delta_rms=float((mean - ref_mean).square().mean().sqrt()),
            )
        )

    print(f"MODEL_STATE={name} BATCHNORM_LAYERS={len(rows)}")
    for row in rows:
        print(" ".join(f"{key}={value}" for key, value in row.items()))
    if rows:
        print(
            "SUMMARY "
            f"state={name} "
            f"finite_layers={sum(row['finite'] for row in rows)}/{len(rows)} "
            f"batches_min={min(row['batches'] for row in rows)} "
            f"batches_max={max(row['batches'] for row in rows)} "
            f"global_var_min={min(row['var_min'] for row in rows):.8g} "
            f"global_var_max={max(row['var_max'] for row in rows):.8g} "
            f"median_layer_var_ratio={torch.tensor([row['var_ratio_median'] for row in rows]).median().item():.8g}"
        )


def main():
    args = get_args()
    config = AutoConfig.from_pretrained(args.model_id, local_files_only=True)
    pretrained = ParakeetEncoder.from_pretrained(
        args.model_id, config=config.encoder_config, local_files_only=True
    ).state_dict()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    for state_name in ("model", "model_avg"):
        state = checkpoint.get(state_name)
        if state is None:
            continue
        bad = []
        for key, value in state.items():
            if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
                bad.append((key, int((~torch.isfinite(value)).sum()), value.numel()))
        print(f"NONFINITE_TENSORS state={state_name} count={len(bad)}")
        for key, count, total in bad:
            print(f"NONFINITE state={state_name} key={key} count={count} total={total}")
    summarize("model", checkpoint["model"], pretrained)
    if checkpoint.get("model_avg") is not None:
        summarize("model_avg", checkpoint["model_avg"], pretrained)


if __name__ == "__main__":
    main()
