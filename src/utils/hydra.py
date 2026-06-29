"""Typed wrapper around :func:`hydra.main`.

``hydra.main`` is an untyped decorator factory, so every CLI entry point used to
re-declare a private ``hydra_main`` shim purely to keep mypy/pyright happy. This
module is the single shared implementation; import it instead of copying the
wrapper into each script.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

import hydra

F = TypeVar("F", bound=Callable[..., Any])

__all__ = ["hydra_main"]


def hydra_main(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper for :func:`hydra.main`."""
    return cast(Callable[[F], F], hydra.main(*args, **kwargs))
