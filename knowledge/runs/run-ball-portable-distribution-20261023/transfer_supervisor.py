"""Preserve the lifetime and exit receipt of a detached local Drive transfer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def save(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def supervise(path: Path, argv: list[str]) -> int:
    if path.exists() or not argv:
        raise ValueError("A supervised transfer needs a new receipt and an explicit command")
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state: dict[str, Any] = dict(
        schema_version=1, status="starting", argv=argv, cwd=str(Path.cwd()),
        started_at=datetime.now(UTC).isoformat(), pid=os.getpid(),
        start_token=Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        session_id=os.getsid(0), process_group_id=os.getpgrp(),
    )
    save(path, state)
    try:
        # The launcher supplies a new OS session and file-backed standard streams.
        # Keep the command in this supervisor's session, independent of the agent.
        with subprocess.Popen(argv, stdin=subprocess.DEVNULL) as process:
            state.update(status="running", command_pid=process.pid)
            save(path, state)
            returncode = process.wait()
        state.update(status="completed" if returncode == 0 else "failed", returncode=returncode)
    except Exception as error:
        state.update(status="failed", returncode=1, error=str(error))
        returncode = 1
    state.update(finished_at=datetime.now(UTC).isoformat(), elapsed_seconds=time.monotonic() - started)
    save(path, state)
    return returncode


def launch(path: Path, stdout: Path, stderr: Path, argv: list[str]) -> dict[str, Any]:
    if path.exists() or not argv:
        raise ValueError("A detached transfer needs a new receipt and an explicit command")
    for destination in (path, stdout, stderr):
        destination.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("xb") as out, stderr.open("xb") as err:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--state", str(path.resolve()), "--", *argv],
            stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            start_new_session=True, close_fds=True,
        )
    return dict(schema_version=1, supervisor_pid=process.pid, state=str(path.resolve()),
                stdout=str(stdout.resolve()), stderr=str(stderr.resolve()), start_new_session=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--stdout", type=Path)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.argv[1:] if arguments.argv[:1] == ["--"] else arguments.argv
    if arguments.detach:
        if arguments.stdout is None or arguments.stderr is None:
            parser.error("Detached transfer requires new --stdout and --stderr files")
        print(json.dumps(launch(arguments.state, arguments.stdout, arguments.stderr, command)))
    else:
        if arguments.stdout is not None or arguments.stderr is not None:
            parser.error("Output paths are supplied only to the detached launcher")
        raise SystemExit(supervise(arguments.state, command))
