"""Operator layer for custom and extension-backed kernels."""

from src.utils.models.components.ops.core.backend import OpBackend
from src.utils.models.components.ops.core.loader import load_operator
from src.utils.models.components.ops.core.registry import register_operator

__all__ = [
    "OpBackend",
    "load_operator",
    "register_operator",
]
