"""Filesystem helpers: directory creation and JSON persistence.

These wrap the ubiquitous ``Path.mkdir(parents=True, exist_ok=True)`` and
``mkdir(...) + json.dump(...)`` idioms that were copy-pasted across dataset
writers, pipeline ``Result.save()`` methods and scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and any missing parents) and return it as a ``Path``.

    Idempotent: an existing directory is left untouched.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Serialize ``data`` to ``path`` as UTF-8 JSON, creating parent dirs.

    Args:
        data: JSON-serializable object.
        path: Destination file path.
        indent: ``json.dump`` indentation (the project default is 2).

    Returns:
        The written path as a ``Path``.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent)
    return destination


def load_json(path: str | Path) -> Any:
    """Load and return the JSON content of ``path`` (UTF-8)."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = ["ensure_dir", "save_json", "load_json"]
