#!/usr/bin/env python3
"""Build the narrow factorized k2 CUDA extension with PyTorch/Ninja."""

from pathlib import Path
import os

import k2
from torch.utils.cpp_extension import load


recipe = Path(__file__).resolve().parents[1]
workspace = recipe.parents[3]
k2_source = Path(
    os.environ.get("K2_SOURCE_DIR", workspace / "third_party" / "k2")
)
moderngpu_source = Path(
    os.environ.get("MODERNGPU_SOURCE_DIR", workspace / "third_party" / "moderngpu")
)
if not (k2_source / "k2/csrc/intersect_dense.cu").is_file():
    raise RuntimeError(
        "Set K2_SOURCE_DIR to k2 commit "
        "0a09f67e1ae9c4f90ea61a161633baacdd5b5af8 with "
        "shuffle_ctc/csrc/k2-factorized-intersect.patch applied"
    )
if not (moderngpu_source / "src/moderngpu").is_dir():
    raise RuntimeError("Set MODERNGPU_SOURCE_DIR to a moderngpu source checkout")
k2_package = Path(k2.__file__).resolve().parent
build_dir = recipe / ".extensions" / "factorized_k2"
build_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;8.0")

module = load(
    name="_k2_factorized",
    sources=[
        str(recipe / "shuffle_ctc/csrc/factorized_intersect_bindings.cu"),
        str(k2_source / "k2/csrc/intersect_dense.cu"),
    ],
    extra_include_paths=[
        str(k2_source), str(moderngpu_source / "src")
    ],
    extra_cflags=["-O3", "-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_cuda_cflags=[
        "-O3", "--use_fast_math", "--expt-extended-lambda",
        "-ccbin=/usr/bin/g++",
        "-DK2_WITH_CUDA", "-DK2_FACTOR_EXT_ONLY",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    extra_ldflags=[
        f"-L{k2_package / 'lib64'}", "-lk2context", "-lk2fsa", "-lk2_log",
        f"-Wl,-rpath,{k2_package / 'lib64'}",
    ],
    build_directory=str(build_dir),
    verbose=True,
)
print(module.__file__)
