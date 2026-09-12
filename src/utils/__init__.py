"""Shared utilities for tennis-lab.

This package provides common utilities used across multiple modules:
- geometry: Court dimensions, keypoints, camera models
- rendering: Visualization components for courts, skeletons, and balls
"""

from importlib import import_module
from types import ModuleType


def __getattr__(name: str) -> ModuleType:
    """Load visualization dependencies only when rendering is requested."""
    if name != "rendering":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("src.utils.rendering")
    globals()[name] = module
    return module


__all__ = ["rendering"]
