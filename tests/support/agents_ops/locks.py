"""Observe real flock ownership from an independent process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROBE = """
import fcntl, pathlib, sys
with pathlib.Path(sys.argv[1]).open('a') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('held')
    else:
        print('free')
"""


def available(path: Path) -> bool:
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.stdout.strip() in {"held", "free"}
    return result.stdout.strip() == "free"
