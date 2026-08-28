"""Resolve the optional training-queue reproduction artifact boundary."""

from __future__ import annotations

import os
from pathlib import Path

QUEUE_REPRO_DIR_ENV = "TENNIS_REPRO_DIR"


class QueueReproDirError(ValueError):
    """Raised when the queue reproduction directory is present but unsafe."""


def resolve_queue_repro_dir() -> Path | None:
    """Return the resolved queue-owned repro directory when explicitly set.

    Queue jobs export an absolute per-job directory. The value is an external
    write authority, so malformed, traversal-bearing, root, and non-directory
    values fail closed instead of falling back to the configured artifact root.
    """
    raw = os.environ.get(QUEUE_REPRO_DIR_ENV)
    if raw is None:
        return None
    if not raw or raw != raw.strip() or any(ord(character) < 32 for character in raw):
        raise QueueReproDirError(
            f"{QUEUE_REPRO_DIR_ENV} must be a non-empty absolute directory path."
        )

    candidate = Path(raw)
    if not candidate.is_absolute():
        raise QueueReproDirError(
            f"{QUEUE_REPRO_DIR_ENV} must be absolute; got {raw!r}."
        )
    if ".." in candidate.parts:
        raise QueueReproDirError(
            f"{QUEUE_REPRO_DIR_ENV} must not contain parent traversal; got {raw!r}."
        )
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise QueueReproDirError(
            f"{QUEUE_REPRO_DIR_ENV} cannot be resolved safely: {raw!r}."
        ) from exc
    if resolved == Path(resolved.anchor):
        raise QueueReproDirError(
            f"{QUEUE_REPRO_DIR_ENV} must not grant the filesystem root."
        )
    if resolved.exists() and not resolved.is_dir():
        raise QueueReproDirError(
            f"{QUEUE_REPRO_DIR_ENV} must identify a directory; got {resolved}."
        )
    return resolved
