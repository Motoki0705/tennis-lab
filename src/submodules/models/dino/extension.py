"""CPU-only compatibility preflight for the explicitly installed DINO op."""

from __future__ import annotations

import importlib
from pathlib import Path

import torch


def validate_dino_extension() -> Path:
    """Check both C++ dispatch entry points without allocating CUDA tensors.

    The pinned upstream op rejects CPU tensors with a specific error. Reaching
    that error proves that tensor/backend dispatch works with this PyTorch;
    importing an old binary alone does not. This is NOT a CUDA kernel test.
    No extension is built, searched for elsewhere, or substituted on failure.
    """
    rebuild = (
        "Rebuild DINO with this checkout's tests/benchmarks/build_dino_extension.sh "
        "and select its lib directory explicitly via PYTHONPATH."
    )
    try:
        extension = importlib.import_module("MultiScaleDeformableAttention")
    except (ImportError, OSError) as error:
        raise RuntimeError(f"Cannot import the DINO CUDA extension. {rebuild}") from error
    if extension.__file__ is None:
        raise RuntimeError("DINO extension must have a concrete binary path")
    path = Path(extension.__file__).resolve(strict=True)
    # Valid minimal shapes ensure that only the deliberate CPU rejection is
    # accepted; unrelated errors and unexpected CPU implementations fail closed.
    value = torch.ones((1, 1, 1, 1), device="cpu", dtype=torch.float32)
    shapes = torch.tensor([[1, 1]], device="cpu", dtype=torch.int64)
    starts = torch.tensor([0], device="cpu", dtype=torch.int64)
    locations = torch.full((1, 1, 1, 1, 1, 2), .5, device="cpu", dtype=torch.float32)
    weights = torch.ones((1, 1, 1, 1, 1), device="cpu", dtype=torch.float32)
    gradient = torch.ones((1, 1, 1), device="cpu", dtype=torch.float32)
    for entry, tail in (("forward", ()), ("backward", (gradient,))):
        function = getattr(extension, f"ms_deform_attn_{entry}", None)
        if not callable(function):
            raise RuntimeError(f"DINO extension {path} has no {entry} entry point. {rebuild}")
        try:
            function(value, shapes, starts, locations, weights, *tail, 1)
        except RuntimeError as error:
            if str(error).split("\n", 1)[0] == "Not implemented on the CPU":
                continue
            raise RuntimeError(
                f"DINO extension {path} failed CPU {entry} dispatch with PyTorch "
                f"{torch.__version__}: {error}. {rebuild}"
            ) from error
        raise RuntimeError(f"DINO extension {path} unexpectedly accepted CPU {entry}. {rebuild}")
    return path
