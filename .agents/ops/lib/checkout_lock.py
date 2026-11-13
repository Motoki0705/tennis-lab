"""Coordinate checkout replacement with job lifetimes and dispatcher controls."""

from __future__ import annotations

import fcntl
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def state_root() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    if not base.is_absolute():
        raise ValueError("XDG_STATE_HOME must be absolute")
    return base / "tennis-lab-agents"


@contextmanager
def checkout_lock(
    root: Path,
    *,
    name: str = "ops-code",
    exclusive: bool = False,
    blocking: bool = True,
) -> Iterator[int | None]:
    """Never unlink these files: all checkouts must refer to the same inode."""
    common = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        text=True,
    ).strip()
    with (Path(common) / f"{name}.lock").open("a", encoding="utf-8") as stream:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(stream, mode | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield None
        else:
            yield stream.fileno()
