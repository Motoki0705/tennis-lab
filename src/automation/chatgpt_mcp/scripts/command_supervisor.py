"""Container command supervisor; diagnostic outcomes never authorize teardown."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def supervise(command: str, timeout: float, destination: Path) -> int:
    """Record observed completion separately from the application's exit code."""
    process = subprocess.Popen(["/bin/bash", "-lc", command], start_new_session=True)
    reason = "unknown"
    try:
        code = process.wait(timeout=timeout)
        reason = "succeeded" if code == 0 else "failed"
    except subprocess.TimeoutExpired:
        reason = "timed_out"
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        code = 124
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps({"outcome": reason, "exit_code": code}) + "\n")
    temporary.replace(destination)
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(supervise(sys.argv[1], float(sys.argv[2]), Path(sys.argv[3])))
