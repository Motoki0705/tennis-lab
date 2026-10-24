"""Observe subprocess costs without logging response contents or configuration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    arguments = sys.argv[1:]
    if arguments[:1] != ["--config"] or len(arguments) < 4:
        raise ValueError("Expected the Drive skill's explicit configuration argument.")
    command = arguments[2:]
    if command[0] not in {"lsjson", "copyto"}:
        raise ValueError("This read probe only observes lsjson and download copyto.")
    if command[0] == "copyto" and not command[1].startswith("gdrive:"):
        raise ValueError("Uploads are outside this read-only probe.")
    started = time.perf_counter()
    result = subprocess.run(
        [os.environ["COURT_COST_REAL_RCLONE"], *arguments],
        check=False,
        capture_output=True,
    )
    elapsed = time.perf_counter() - started
    remote = command[1].startswith("gdrive:")
    listing = command[0] == "lsjson" and "--stat" not in command
    entry_count = None
    if listing and result.returncode == 0:
        entries = json.loads(result.stdout)
        entry_count = len(entries)
    event = {
        "category": "resolve_listing" if listing else command[0],
        "remote": remote,
        "remote_target": command[1] if remote else None,
        "elapsed_seconds": elapsed,
        "returncode": result.returncode,
        "entry_count": entry_count,
        "stdout_bytes": len(result.stdout),
    }
    with Path(os.environ["COURT_COST_TRACE"]).open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
