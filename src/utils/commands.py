"""Subprocess command helpers."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence


def run_command(cmd: Sequence[str], *, echo: bool = True) -> None:
    """Run ``cmd`` with ``check=True`` and optional shell-like echoing."""
    if echo:
        print("  $ " + " ".join(cmd))
    subprocess.run(list(cmd), check=True)


__all__ = ["run_command"]
