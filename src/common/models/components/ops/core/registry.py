"""In-process registry for operator implementations."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

from src.common.models.components.ops.core.backend import OpBackend
from src.common.models.components.ops.core.errors import OperatorImplementationError, OperatorNotFoundError
from src.common.models.components.ops.core.types import OperatorHandle, OperatorKey

_REGISTRY: dict[OperatorKey, dict[OpBackend, Callable]] = defaultdict(dict)


def register_operator(key: OperatorKey, backend: OpBackend, fn: Callable) -> None:
    """Register an implementation for an operator key/backend pair."""
    if not callable(fn):
        raise OperatorImplementationError(f"Registered implementation is not callable: {key} ({backend})")
    _REGISTRY[key][backend] = fn


def resolve_operator(key: OperatorKey, backend: OpBackend) -> OperatorHandle:
    """Resolve an operator handle for the requested backend."""
    backends = _REGISTRY.get(key)
    if backends is None or backend not in backends:
        raise OperatorNotFoundError(f"Operator not found: family={key.family} name={key.name} backend={backend}")
    return OperatorHandle(key=key, backend=backend, fn=backends[backend])


def list_backends(key: OperatorKey) -> tuple[OpBackend, ...]:
    """Return available backends for a key."""
    backends = _REGISTRY.get(key, {})
    return tuple(backends.keys())
