"""Opt-in, private command timings; no request bodies or identities are logged."""

from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_current: dict[str, Any] | None = None


def count(field: str, value: float | int = 1) -> None:
    if _current is not None:
        _current[field] = _current.get(field, 0) + value


def flock(file: Any, operation: int) -> None:
    started = time.monotonic()
    fcntl.flock(file, operation)
    if operation & fcntl.LOCK_EX:
        count("lock_wait_seconds", time.monotonic() - started)


@contextmanager
def command(name: str, path: Path | None) -> Iterator[dict[str, Any]]:
    global _current
    row: dict[str, Any] = {
        "schema_version": 1,
        "command": name,
        "started_at": time.time(),
        "lock_wait_seconds": 0.0,
        "github_seconds": 0.0,
        "github_requests": 0,
        "github_pages": 0,
        "cache_hits": 0,
        "conflicts": 0,
        "exit_code": None,
        "attempt": int(os.environ.get("PARENT_METRICS_ATTEMPT", "1")),
    }
    if row["attempt"] < 1:
        raise ValueError("PARENT_METRICS_ATTEMPT must be positive")
    row["retry_count"] = row["attempt"] - 1
    previous, _current = _current, row
    started = time.monotonic()
    try:
        yield row
    finally:
        row["elapsed_seconds"] = time.monotonic() - started
        row["processing_seconds"] = max(
            0.0,
            row["elapsed_seconds"] - row["github_seconds"] - row["lock_wait_seconds"],
        )
        _current = previous
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(
                path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(fd, "a", encoding="utf-8") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
