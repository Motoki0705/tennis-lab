"""Extension loading logic for MSDA CUDA kernels."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def try_load_msda_extension(*, already_attempted: bool, cached_ext: object | None) -> tuple[object | None, bool]:
    """Load CUDA extension once and return updated `(ext, attempted)` state."""
    if already_attempted:
        return cached_ext, True

    attempted = True
    if os.environ.get("MSDA_FORCE_FALLBACK", "0") == "1":
        return None, attempted
    if not torch.cuda.is_available():
        return None, attempted

    csrc_dir = Path(__file__).resolve().parent.parent / "csrc"
    sources = [
        str(csrc_dir / "binding.cpp"),
        str(csrc_dir / "deformable_cuda.cpp"),
        str(csrc_dir / "deformable_cuda_kernel.cu"),
    ]

    try:
        ext = load(
            name="msda_ops",
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception:
        ext = None

    return ext, attempted

