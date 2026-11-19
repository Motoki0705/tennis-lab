"""Frozen publication ownership, rechecked without holding locks during Git I/O."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .journal import require


def publication_owner(
    store: Any, repo: str, parent: int, identity: str | None, generation: int | None
) -> Callable[[], bool]:
    """Keep the admitting actor fixed; never adopt a successor's generation."""
    key = f"{repo.casefold()}#{parent}"

    def current() -> bool:
        with store.operation(key), store.locked():
            row = store.parents.get(key, {})
            return bool(
                identity is not None
                and generation is not None
                and row.get("session") == identity
                and row.get("generation") == generation
            )

    return current


def require_owner(check: Callable[[], bool] | None) -> None:
    """Admit one publication before acquiring its turn lock."""
    require(check is None or check(), "log_owner_mismatch")
