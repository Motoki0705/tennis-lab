"""Run real hooks with /proc ancestry bounded at the invoking test process.

Only that process's parent PID is replaced with 1. Native process names,
PID/start-time stamps and every descendant relationship below it stay real.
The patch exists only in this disposable hook process, never in production.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch


def hook_command(hook: Path) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), str(os.getpid()), str(hook)]


def main() -> None:
    boundary_pid = int(sys.argv[1])
    hook = sys.argv[2]
    boundary = Path(f"/proc/{boundary_pid}/stat")
    pid = os.getpid()
    while pid > 1:
        if pid == boundary_pid:
            break
        record = Path(f"/proc/{pid}/stat").read_text()
        pid = int(record.rsplit(")", 1)[1].split()[1])
    else:
        raise ValueError("hook isolation boundary must be an ancestor")

    read_text = Path.read_text

    def bounded_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        record = read_text(path, *args, **kwargs)
        if path == boundary:
            name, fields_text = record.rsplit(")", 1)
            fields = fields_text.split()
            fields[1] = "1"
            return name + ") " + " ".join(fields) + "\n"
        return record

    sys.argv = [hook, *sys.argv[3:]]
    with patch.object(Path, "read_text", bounded_read_text):
        runpy.run_path(hook, run_name="__main__")


if __name__ == "__main__":
    main()
