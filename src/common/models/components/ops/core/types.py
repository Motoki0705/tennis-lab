"""Datatypes used by operator registry/loader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.common.models.components.ops.core.backend import OpBackend


@dataclass(frozen=True)
class OperatorKey:
    """Fully-qualified operator key."""

    family: str
    name: str


@dataclass(frozen=True)
class OperatorRequest:
    """Runtime request for resolving an operator implementation."""

    key: OperatorKey
    prefer_backend: OpBackend
    allow_fallback: bool = True


@dataclass(frozen=True)
class OperatorHandle:
    """Resolved operator implementation."""

    key: OperatorKey
    backend: OpBackend
    fn: Callable
