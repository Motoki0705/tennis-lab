"""Construction-time backend dispatch for compressed time-local attention."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from importlib import import_module
from typing import Literal, cast

from torch import Tensor

from src.utils.models.components.ops.compressed_time_local.reference import (
    reference_compressed_time_local_attention,
)
from src.utils.models.components.ops.loader import (
    require_compressed_time_local_cuda_extension,
)

CompressedTimeLocalAttentionExecutor = Callable[..., Tensor]
_MAX_CUDA_WINDOW_RADIUS = 64


def _require_cuda_executor() -> CompressedTimeLocalAttentionExecutor:
    """Load the optional CUDA executor or fail without a reference fallback."""
    require_compressed_time_local_cuda_extension()
    try:
        module = import_module(
            "src.utils.models.components.ops.compressed_time_local._autograd"
        )
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "compressed time-local CUDA backend was requested, but its "
            "extension is unavailable"
        ) from error
    executor = getattr(module, "cuda_compressed_time_local_attention", None)
    if not callable(executor):
        raise RuntimeError(
            "compressed time-local CUDA backend was requested, but its executor "
            "is unavailable"
        )
    return cast(CompressedTimeLocalAttentionExecutor, executor)


def resolve_compressed_time_local_attention(
    backend: Literal["reference", "cuda"],
    *,
    compression_ratio: int,
    window_radius: int,
) -> CompressedTimeLocalAttentionExecutor:
    """Resolve and bind one executor at module construction time."""
    if type(compression_ratio) is not int or compression_ratio < 2:
        raise ValueError(
            f"compression_ratio must be an int of at least 2, got {compression_ratio!r}"
        )
    if type(window_radius) is not int or window_radius < 0:
        raise ValueError(
            f"window_radius must be a non-negative int, got {window_radius!r}"
        )
    if backend == "reference":
        executor = reference_compressed_time_local_attention
    elif backend == "cuda":
        if window_radius > _MAX_CUDA_WINDOW_RADIUS:
            raise ValueError(
                "compressed time-local CUDA supports window_radius <= "
                f"{_MAX_CUDA_WINDOW_RADIUS}, got {window_radius}"
            )
        executor = _require_cuda_executor()
    else:
        raise ValueError(f"Unsupported compressed time-local backend: {backend!r}")
    return partial(
        executor,
        compression_ratio=compression_ratio,
        window_radius=window_radius,
    )
