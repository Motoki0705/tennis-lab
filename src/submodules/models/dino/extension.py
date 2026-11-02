"""CPU-only compatibility preflight for the explicitly installed DINO op."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path
from threading import RLock
from types import ModuleType

import torch

_MODULE = "MultiScaleDeformableAttention"
_SELECTION = "TENNIS_LAB_DINO_EXTENSION"
_LOAD_LOCK = RLock()
_REBUILD = (
    "Follow docs/installation/README.md#dino-cuda-extension to rebuild with "
    "scripts/build_dino_extension.sh and set TENNIS_LAB_DINO_EXTENSION to its binary."
)


def load_dino_extension() -> ModuleType:
    """Load the explicitly selected binary under the name used by upstream DINO.

    Loading from a file bypasses cwd/PYTHONPATH shadowing. Never replace a cached
    module: upstream code could already hold a reference to it.
    """
    selected = os.environ.get(_SELECTION)
    if not selected:
        raise RuntimeError(f"Set {_SELECTION} to the absolute DINO binary path. {_REBUILD}")
    path = Path(selected)
    if not path.is_absolute():
        raise RuntimeError(f"{_SELECTION} must be an absolute binary path: {path}")
    try:
        path = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"DINO extension selected by {_SELECTION} does not exist: {path}. {_REBUILD}") from error
    if not path.is_file() or not path.name.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
        raise RuntimeError(f"DINO extension selected by {_SELECTION} is not a binary file: {path}. {_REBUILD}")

    with _LOAD_LOCK:
        if _MODULE in sys.modules:
            cached = sys.modules[_MODULE]
            cached_file = getattr(cached, "__file__", None)
            if cached_file is None or Path(cached_file).resolve() != path:
                raise RuntimeError(
                    f"DINO extension already loaded from {cached_file!r}; "
                    f"{_SELECTION} selects {path}. Start a fresh process."
                )
            return cached

        try:
            spec = importlib.util.spec_from_file_location(_MODULE, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"No extension loader for {path}")
            extension = importlib.util.module_from_spec(spec)
            sys.modules[_MODULE] = extension
            try:
                spec.loader.exec_module(extension)
            except BaseException:
                sys.modules.pop(_MODULE, None)
                raise
        except (ImportError, OSError) as error:
            raise RuntimeError(f"Cannot import DINO extension {path}. {_REBUILD}") from error
        loaded_file = getattr(extension, "__file__", None)
        if loaded_file is None or Path(loaded_file).resolve() != path:
            sys.modules.pop(_MODULE, None)
            raise RuntimeError(f"DINO extension resolved to {loaded_file!r}, expected {path}")
        return extension


def validate_dino_extension() -> Path:
    """Check both C++ dispatch entry points without allocating CUDA tensors.

    The pinned upstream op rejects CPU tensors with a specific error. Reaching
    that error proves that tensor/backend dispatch works with this PyTorch;
    importing an old binary alone does not. This is NOT a CUDA kernel test.
    No extension is built, searched for elsewhere, or substituted on failure.
    """
    extension = load_dino_extension()
    assert extension.__file__ is not None  # Checked by load_dino_extension.
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
            raise RuntimeError(f"DINO extension {path} has no {entry} entry point. {_REBUILD}")
        try:
            function(value, shapes, starts, locations, weights, *tail, 1)
        except RuntimeError as error:
            if str(error).split("\n", 1)[0] == "Not implemented on the CPU":
                continue
            raise RuntimeError(
                f"DINO extension {path} failed CPU {entry} dispatch with PyTorch "
                f"{torch.__version__}: {error}. {_REBUILD}"
            ) from error
        raise RuntimeError(f"DINO extension {path} unexpectedly accepted CPU {entry}. {_REBUILD}")
    return path
