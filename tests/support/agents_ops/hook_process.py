"""Run real hooks with /proc ancestry bounded at the invoking test process.

The boundary process's parent PID is replaced with 1. agy fixtures can also
supply the argv for disposable TUI processes. Native process names, PID/start
time, controlling terminals, locks and descendant relationships stay real.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch


def hook_command(hook: Path | str) -> list[str]:
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
    read_bytes = Path.read_bytes
    cmdlines = json.loads(os.environ.pop("TEST_AGY_CMDLINES", "{}"))

    def bounded_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        record = read_text(path, *args, **kwargs)
        if path == boundary:
            name, fields_text = record.rsplit(")", 1)
            fields = fields_text.split()
            fields[1] = "1"
            return name + ") " + " ".join(fields) + "\n"
        return record

    def fixture_read_bytes(path: Path) -> bytes:
        if path.name == "cmdline" and path.parent.parent == Path("/proc"):
            command = cmdlines.get(path.parent.name)
            if command is not None:
                return b"\0".join(os.fsencode(arg) for arg in command) + b"\0"
        return read_bytes(path)

    sys.argv = [hook, *sys.argv[3:]]
    with (
        patch.object(Path, "read_text", bounded_read_text),
        patch.object(Path, "read_bytes", fixture_read_bytes),
    ):
        if hook.startswith("-m:"):
            runpy.run_module(
                hook.removeprefix("-m:"), run_name="__main__", alter_sys=True
            )
        else:
            runpy.run_path(hook, run_name="__main__")


if __name__ == "__main__":
    main()
