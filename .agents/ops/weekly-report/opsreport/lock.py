"""Machine-wide lock serializing every issue-writing step of report and triage.

``report`` (same-week check -> create) and ``triage`` (child lookup -> create ->
body update) are check-then-act sequences against GitHub. Two runs at once (timer
plus a manual run, or two checkouts) could both decide that nothing exists yet and
both create it; the lock makes those sequences mutually exclusive.
"""

from __future__ import annotations

import fcntl
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .shell import CommandError

LOG = logging.getLogger("weekly_report.lock")
POLL_SECONDS = 0.2


class LockTimeout(CommandError):
    """Another run held the lock for longer than the timeout."""


@contextmanager
def exclusive(path: Path, *, timeout: float) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        deadline = time.monotonic() + timeout
        waited = False
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise LockTimeout(
                        f"lock {path} still held by another run after {timeout}s"
                    ) from None
                if not waited:
                    LOG.info("waiting for lock %s", path)
                    waited = True
                time.sleep(POLL_SECONDS)
        LOG.info("acquired lock %s", path)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
