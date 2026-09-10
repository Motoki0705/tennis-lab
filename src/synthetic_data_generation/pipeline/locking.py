"""One advisory scene writer lock shared by the pipeline and manual editor."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def scene_write_lock(scene_root: Path) -> Iterator[None]:
    """Fail explicitly if another process is mutating this scene."""
    scene_root.mkdir(parents=True, exist_ok=True)
    with (scene_root / ".scene-writer.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Scene is busy in another pipeline or editor process."
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
