"""Backend-aware loader for operator implementations."""

from __future__ import annotations

import torch

from src.utils.models.components.ops.core.backend import OpBackend
from src.utils.models.components.ops.core.errors import BackendNotAvailableError, OperatorNotFoundError
from src.utils.models.components.ops.core.registry import list_backends, resolve_operator
from src.utils.models.components.ops.core.types import OperatorRequest


def _backend_available(backend: OpBackend) -> bool:
    if backend == OpBackend.CUDA:
        return torch.cuda.is_available()
    return True


def load_operator(request: OperatorRequest):
    """Resolve an operator implementation with optional backend fallback."""
    preferred = request.prefer_backend
    ordered = [preferred]
    if request.allow_fallback:
        ordered.extend(b for b in (OpBackend.CUDA, OpBackend.CPU, OpBackend.PYTORCH) if b != preferred)

    available = set(list_backends(request.key))
    for backend in ordered:
        if backend not in available:
            continue
        if not _backend_available(backend):
            continue
        return resolve_operator(request.key, backend)

    if preferred in available and not _backend_available(preferred):
        raise BackendNotAvailableError(f"Requested backend unavailable at runtime: {preferred}")
    raise OperatorNotFoundError(f"No implementation found for {request.key} across backends={tuple(ordered)}")
