"""Core registry/dispatch utilities for custom ops."""

from src.common.models.components.ops.core.backend import OpBackend
from src.common.models.components.ops.core.loader import load_operator
from src.common.models.components.ops.core.registry import list_backends, register_operator, resolve_operator
from src.common.models.components.ops.core.types import OperatorHandle, OperatorKey, OperatorRequest

__all__ = [
    "OpBackend",
    "OperatorHandle",
    "OperatorKey",
    "OperatorRequest",
    "load_operator",
    "register_operator",
    "resolve_operator",
    "list_backends",
]
