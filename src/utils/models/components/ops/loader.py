from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType

TIME_LOCAL_EXTENSION_NAME = "src.utils.models.components.ops.time_local._C"
COMPRESSED_TIME_LOCAL_EXTENSION_NAME = (
    "src.utils.models.components.ops.compressed_time_local._C"
)


@lru_cache(maxsize=1)
def get_time_local_cuda_extension() -> ModuleType | None:
    try:
        return importlib.import_module(TIME_LOCAL_EXTENSION_NAME)
    except (ImportError, OSError):
        return None


@lru_cache(maxsize=1)
def get_compressed_time_local_cuda_extension() -> ModuleType | None:
    try:
        return importlib.import_module(COMPRESSED_TIME_LOCAL_EXTENSION_NAME)
    except (ImportError, OSError):
        return None


def is_time_local_cuda_available() -> bool:
    return get_time_local_cuda_extension() is not None


def is_compressed_time_local_cuda_available() -> bool:
    return get_compressed_time_local_cuda_extension() is not None


def require_time_local_cuda_extension() -> ModuleType:
    extension = get_time_local_cuda_extension()
    if extension is None:
        raise RuntimeError(
            "Time-local CUDA extension is not available. Build it with "
            "`TENNIS_LAB_BUILD_CUDA_OPS=1 .venv/bin/python -m pip install -e . "
            "--no-build-isolation`, or call the local-attention API with use_cuda=False."
        )
    return extension


def require_compressed_time_local_cuda_extension() -> ModuleType:
    extension = get_compressed_time_local_cuda_extension()
    if extension is None:
        raise RuntimeError(
            "Compressed time-local CUDA extension is not available. Build it with "
            "`TENNIS_LAB_BUILD_CUDA_OPS=1 .venv/bin/python -m pip install -e . "
            "--no-build-isolation`, or select backend='reference'."
        )
    return extension
