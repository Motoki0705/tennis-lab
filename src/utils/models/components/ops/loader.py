from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType

COMPRESSED_TIME_LOCAL_EXTENSION_NAME = "tennis_lab_compressed_time_local_cuda"


@lru_cache(maxsize=1)
def get_compressed_time_local_cuda_extension() -> ModuleType | None:
    try:
        return importlib.import_module(COMPRESSED_TIME_LOCAL_EXTENSION_NAME)
    except (ImportError, OSError):
        return None


def is_compressed_time_local_cuda_available() -> bool:
    return get_compressed_time_local_cuda_extension() is not None


def require_compressed_time_local_cuda_extension() -> ModuleType:
    extension = get_compressed_time_local_cuda_extension()
    if extension is None:
        raise RuntimeError(
            "Compressed time-local CUDA extension is not available. From the "
            "repository root, initialize third_party/DINO and set CUDA_HOME and "
            "TORCH_CUDA_ARCH_LIST as in docs/installation/README.md#dino-cuda-extension; "
            'then run `bash scripts/build_dino_extension.sh "$PWD" '
            '"$PWD/outputs/dino_extension"` and '
            '`export PYTHONPATH="$PWD/outputs/dino_extension/lib"`; restart Python and retry, '
            "or select backend='reference'."
        )
    return extension
