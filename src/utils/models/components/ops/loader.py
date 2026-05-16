from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType

MOE_EXTENSION_NAME = "src.utils.models.components.ops.moe._C"
TIME_LOCAL_EXTENSION_NAME = "src.utils.models.components.ops.time_local._C"


@lru_cache(maxsize=1)
def get_moe_cuda_extension() -> ModuleType | None:
    try:
        return importlib.import_module(MOE_EXTENSION_NAME)
    except (ImportError, OSError):
        return None


@lru_cache(maxsize=1)
def get_time_local_cuda_extension() -> ModuleType | None:
    try:
        return importlib.import_module(TIME_LOCAL_EXTENSION_NAME)
    except (ImportError, OSError):
        return None


def is_moe_cuda_available() -> bool:
    return get_moe_cuda_extension() is not None


def is_time_local_cuda_available() -> bool:
    return get_time_local_cuda_extension() is not None


def require_moe_cuda_extension() -> ModuleType:
    extension = get_moe_cuda_extension()
    if extension is None:
        raise RuntimeError(
            "MoE CUDA extension is not available. Build it with "
            "`TENNIS_LAB_BUILD_CUDA_OPS=1 uv pip install -e . "
            "--no-build-isolation --python .venv/bin/python`, or call the API "
            "with use_cuda=False."
        )
    return extension


def require_time_local_cuda_extension() -> ModuleType:
    extension = get_time_local_cuda_extension()
    if extension is None:
        raise RuntimeError(
            "Time-local CUDA extension is not available. Build it with "
            "`TENNIS_LAB_BUILD_CUDA_OPS=1 uv pip install -e . "
            "--no-build-isolation --python .venv/bin/python`, or call the "
            "local-attention API with use_cuda=False."
        )
    return extension
