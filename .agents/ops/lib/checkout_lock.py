"""Coordinate checkout replacement with job lifetimes and dispatcher controls."""

from __future__ import annotations

import fcntl
import os
import stat
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

CHECKOUT_ROOT_ENV = "TENNIS_LAB_CHECKOUT_ROOT"
CHECKOUT_FD_ENV = "TENNIS_LAB_CHECKOUT_FD"


def require_checkout_job() -> None:
    """Reject CLI jobs without the launcher's inherited, real checkout lock FD."""
    try:
        root = Path(os.environ[CHECKOUT_ROOT_ENV])
        descriptor = int(os.environ[CHECKOUT_FD_ENV])
        if not root.is_absolute():
            raise ValueError("checkout root must be absolute")
        actual = os.fstat(descriptor)
        expected = (common_directory(root) / "ops-code.lock").stat()
        if not stat.S_ISREG(actual.st_mode) or (actual.st_dev, actual.st_ino) != (
            expected.st_dev,
            expected.st_ino,
        ):
            raise ValueError("not the checkout lock descriptor")
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        if checkout_barrier(root).exists():
            raise ValueError("checkout recovery pending")
    except (KeyError, ValueError, OSError, OverflowError) as exc:
        print(
            f"checkout job refused: {exc}; use checkout_job.py --repo <main> -- <command...>",
            file=sys.stderr,
        )
        raise SystemExit(75) from exc


def state_root() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    if not base.is_absolute():
        raise ValueError("XDG_STATE_HOME must be absolute")
    return base / "tennis-lab-agents"


def common_directory(root: Path) -> Path:
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
    return Path(common)


def checkout_barrier(root: Path) -> Path:
    return common_directory(root) / "ops-code.blocked"


@contextmanager
def checkout_lock(
    root: Path,
    *,
    name: str = "ops-code",
    exclusive: bool = False,
    blocking: bool = True,
) -> Iterator[int | None]:
    """Never unlink these files: all checkouts must refer to the same inode."""
    with (common_directory(root) / f"{name}.lock").open(
        "a", encoding="utf-8"
    ) as stream:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(stream, mode | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield None
        else:
            yield stream.fileno()
