"""Construction-time backend resolution for token-compressor pooling."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from importlib import import_module
from typing import Literal, Protocol, cast

from torch import Tensor

from src.utils.models.components.ops.token_compressor.reference import (
    reference_token_compressor_pool,
)

TokenCompressorPool = Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]
_CUDA_COMPRESSION_RATIO = 4
_CUDA_HEAD_DIM = 64


class _TokenCompressorImplementation(Protocol):
    def __call__(
        self,
        raw_kv: Tensor,
        raw_gate: Tensor,
        state_valid: Tensor,
        *,
        compression_ratio: int,
    ) -> tuple[Tensor, Tensor]: ...


def _require_cuda_pool() -> _TokenCompressorImplementation:
    """Lazily load the Triton executor or fail without a reference fallback."""
    try:
        module = import_module(
            "src.utils.models.components.ops.token_compressor._triton"
        )
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "token-compressor CUDA backend was requested, but Triton is unavailable"
        ) from error
    executor = getattr(module, "cuda_token_compressor_pool", None)
    if not callable(executor):
        raise RuntimeError(
            "token-compressor CUDA backend was requested, but its executor "
            "is unavailable"
        )
    return cast(_TokenCompressorImplementation, executor)


def resolve_token_compressor_pool(
    backend: Literal["reference", "cuda"],
    *,
    compression_ratio: int,
    head_dim: int,
) -> TokenCompressorPool:
    """Resolve and bind one pooling implementation at module construction."""
    if type(compression_ratio) is not int or compression_ratio < 2:
        raise ValueError(
            f"compression_ratio must be an int of at least 2, got {compression_ratio!r}"
        )
    if type(head_dim) is not int or head_dim <= 0:
        raise ValueError(f"head_dim must be a positive int, got {head_dim!r}")
    if backend == "reference":
        return partial(
            reference_token_compressor_pool,
            compression_ratio=compression_ratio,
        )
    if backend != "cuda":
        raise ValueError(f"Unsupported token-compressor backend: {backend!r}")
    if compression_ratio != _CUDA_COMPRESSION_RATIO:
        raise ValueError(
            "token-compressor CUDA supports compression_ratio=4, "
            f"got {compression_ratio}"
        )
    if head_dim != _CUDA_HEAD_DIM:
        raise ValueError(f"token-compressor CUDA supports head_dim=64, got {head_dim}")
    return partial(
        _require_cuda_pool(),
        compression_ratio=compression_ratio,
    )


__all__ = ["TokenCompressorPool", "resolve_token_compressor_pool"]
