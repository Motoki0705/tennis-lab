"""Shell-free subprocess boundary shared by reconstruction and rendering."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path


def run_process(
    command: Sequence[str],
    *,
    working_directory: Path | None,
    environment: Mapping[str, str],
) -> None:
    resolved_environment = os.environ.copy()
    resolved_environment.update(environment)
    completed = subprocess.run(
        list(command),
        cwd=working_directory,
        env=resolved_environment,
        check=False,
    )
    if completed.returncode < 0:
        raise RuntimeError(
            f"External process terminated by signal {-completed.returncode}: "
            f"{list(command)}"
        )
    if completed.returncode:
        raise RuntimeError(
            f"External process exited with code {completed.returncode}: {list(command)}"
        )
