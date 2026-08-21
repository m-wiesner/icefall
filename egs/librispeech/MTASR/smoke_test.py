#!/usr/bin/env python3
"""Import and API smoke test for the complete recipe environment."""

import importlib
import os
import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ICEFALL_ROOT = ROOT.parents[2]
sys.path[:0] = [str(ROOT), str(ICEFALL_ROOT)]


def main() -> None:
    modules = [
        "torch", "torchaudio", "k2", "lhotse", "sentencepiece", "scipy",
        "pyloudnorm", "meeteval", "tensorboard", "icefall",
        "vendor.lhotse.cut_splice3",
    ]
    for name in modules:
        importlib.import_module(name)
        print(f"OK import {name}")

    implementations = {
        "speaker-attributed shuffle": ("wavlm_spashl_ctc", ("greedy", "hlg"), ("graph", "token", "oracle-mono")),
        "speaker-distinguished CTC": ("wavlm_sdctc", ("greedy", "hlg"), ("graph",)),
        "token-only shuffle": ("wavlm_shuffle_ctc", ("greedy", "hlg", "no-speaker"), ("graph",)),
    }
    for label, (directory, decode_methods, align_methods) in implementations.items():
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT / directory), str(ROOT), str(ICEFALL_ROOT), env.get("PYTHONPATH", "")]
        )
        subprocess.run(
            [sys.executable, "-c", "import dataset, asr_datamodule, model, mdctc_graph_compiler, train"],
            cwd=ROOT,
            env=env,
            check=True,
        )
        for source in (ROOT / directory).glob("*.py"):
            py_compile.compile(str(source), doraise=True)
        for command in ("decode.py", "align.py"):
            subprocess.run(
                [sys.executable, str(ROOT / directory / command), "--help"],
                cwd=ROOT,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        print(f"OK implementation {label} ({directory})")

    import torch
    import k2
    import lhotse

    fsa = k2.linear_fsa([1, 2])
    assert fsa.arcs.num_elements() == 3, fsa
    print("OK k2 FSA construction")

    print(f"Python {sys.version.split()[0]}")
    print(f"PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}; available={torch.cuda.is_available()}")
    k2_version = getattr(k2, "__version__", getattr(k2, "__dev_version__", "unknown"))
    print(f"k2 {k2_version}; Lhotse {lhotse.__version__}")


if __name__ == "__main__":
    main()
