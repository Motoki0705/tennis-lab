"""Validated paths for queue-owned reproducibility bundles."""

from __future__ import annotations

import os
from pathlib import Path


def active_queue_repro_dir() -> Path | None:
    """Return the explicit queue repro directory, rejecting unsafe values."""
    raw = os.environ.get("TENNIS_REPRO_DIR")
    if raw is None:
        return None
    if not raw or raw != raw.strip():
        raise RuntimeError("TENNIS_REPRO_DIR must be non-empty and trimmed.")
    path = Path(raw)
    if not path.is_absolute():
        raise RuntimeError("TENNIS_REPRO_DIR must be an absolute path.")
    resolved = path.resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise RuntimeError("TENNIS_REPRO_DIR must not be the filesystem root.")
    return resolved


__all__ = ["active_queue_repro_dir"]
