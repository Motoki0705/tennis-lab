"""Typed inference decorators for PyTorch gradient-mode boundaries."""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar

import torch

_P = ParamSpec("_P")
_R = TypeVar("_R")


def no_grad(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Apply :func:`torch.no_grad` without erasing the callable signature."""
    decorator: Callable[[Callable[_P, _R]], Callable[_P, _R]] = torch.no_grad()
    return decorator(function)
