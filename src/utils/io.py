"""Filesystem helpers: directory creation and JSON persistence.

These wrap the ubiquitous ``Path.mkdir(parents=True, exist_ok=True)`` and
``mkdir(...) + json.dump(...)`` idioms that were copy-pasted across dataset
writers, pipeline ``Result.save()`` methods and scripts.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

JSONDict = dict[str, Any]


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and any missing parents) and return it as a ``Path``.

    Idempotent: an existing directory is left untouched.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_dirs(paths: Iterable[str | Path]) -> list[Path]:
    """Create multiple directories and return them as ``Path`` objects."""
    return [ensure_dir(path) for path in paths]


def find_existing_file(
    directory: str | Path,
    stem: str,
    extensions: Iterable[str],
) -> Path | None:
    """Return the first existing ``<directory>/<stem><extension>``, else ``None``.

    Wraps the "image id with .png/.jpg fallback" lookup duplicated across
    dataset loaders and annotation tooling.
    """
    base = Path(directory)
    for extension in extensions:
        candidate = base / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def save_json(
    data: Any,
    path: str | Path,
    *,
    indent: int = 2,
    default: Callable[[Any], Any] | None = None,
) -> Path:
    """Serialize ``data`` to ``path`` as UTF-8 JSON, creating parent dirs.

    Args:
        data: JSON-serializable object.
        path: Destination file path.
        indent: ``json.dump`` indentation (the project default is 2).
        default: Optional ``json.dump`` fallback serializer (e.g. ``str``).

    Returns:
        The written path as a ``Path``.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, default=default)
    return destination


def save_json_atomic(
    data: Any,
    path: str | Path,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> Path:
    """Atomically serialize ``data`` to ``path`` as UTF-8 JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    text = json.dumps(data, ensure_ascii=ensure_ascii, indent=indent) + "\n"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_json(path: str | Path) -> Any:
    """Load and return the JSON content of ``path`` (UTF-8)."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_if_exists(path: str | Path, default: Any = None) -> Any:
    """Load JSON from ``path`` or return ``default`` when it is missing."""
    source = Path(path)
    return default if not source.exists() else load_json(source)


def read_jsonl(path: str | Path) -> list[JSONDict]:
    """Read a UTF-8 JSONL file into a list of dictionaries."""
    source = Path(path)
    if not source.exists():
        return []
    return [
        cast(JSONDict, json.loads(line))
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(
    path: str | Path,
    records: Iterable[JSONDict],
    *,
    ensure_ascii: bool = False,
) -> Path:
    """Write dictionaries to ``path`` as UTF-8 JSONL."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(record, ensure_ascii=ensure_ascii) + "\n" for record in records
    )
    destination.write_text(text, encoding="utf-8")
    return destination


def relative_path(path: str | Path, root: str | Path) -> str:
    """Return ``path`` relative to ``root`` after resolving both."""
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


__all__ = [
    "JSONDict",
    "ensure_dir",
    "ensure_dirs",
    "find_existing_file",
    "load_json",
    "load_json_if_exists",
    "read_jsonl",
    "relative_path",
    "save_json",
    "save_json_atomic",
    "utc_now_iso",
    "write_jsonl",
]
