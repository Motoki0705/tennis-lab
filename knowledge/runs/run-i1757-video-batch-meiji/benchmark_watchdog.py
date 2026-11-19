"""Supervise the benchmark worker using its heartbeat and a wall time limit."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path

from health_guard import terminate_variant, utc, write, write_anomaly


def capture_thread_stack(process: subprocess.Popen, stack_path: Path) -> bool:
    process.send_signal(signal.SIGUSR1)
    deadline = time.monotonic() + 5
    while process.poll() is None and time.monotonic() < deadline:
        if stack_path.is_file() and stack_path.stat().st_size:
            return True
        time.sleep(0.02)
    return False


def supervise_worker(
    phase: Path,
    command: list[str],
    cwd: Path,
    *,
    heartbeat_timeout: float,
    time_limit: float,
):
    health = phase / "health"
    health.mkdir()
    stack_path = health / "worker-threads.log"
    log_path = phase / "worker.log"
    heartbeat_path = phase / "worker" / "heartbeat.json"
    result = {
        "variant": "worker",
        "monitor": "heartbeat",
        "started_at_utc": utc(),
        "heartbeat_timeout_seconds": heartbeat_timeout,
        "time_limit_seconds": time_limit,
    }
    last_heartbeat = None

    def record_failure(kind, detail):
        result.update(error_kind=kind, error=detail)
        write_anomaly(
            phase,
            {
                "kind": kind,
                "variant": "worker",
                "detail": detail,
                "log": str(log_path),
                "pid": result["pid"],
                "last_heartbeat": last_heartbeat,
            },
        )

    with log_path.open("x") as log:
        started = time.monotonic()
        deadline = started + time_limit
        environment = {**os.environ, "TENNIS_STACK_FILE": str(stack_path)}
        process = subprocess.Popen(
            command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=environment
        )
        result["pid"] = process.pid
        try:
            try:
                while process.poll() is None:
                    last_update = started
                    if heartbeat_path.is_file():
                        last_heartbeat = json.loads(heartbeat_path.read_text())
                        last_update = last_heartbeat["monotonic"]
                        if not math.isfinite(last_update):
                            raise ValueError(
                                "Heartbeat needs a finite monotonic timestamp"
                            )
                    heartbeat_deadline = last_update + heartbeat_timeout
                    remaining = min(deadline, heartbeat_deadline) - time.monotonic()
                    if remaining <= 0:
                        if deadline <= heartbeat_deadline:
                            record_failure(
                                "time_limit",
                                f"External time limit of {time_limit:g} s elapsed",
                            )
                        else:
                            record_failure(
                                "worker_heartbeat_timeout",
                                f"No worker heartbeat for {heartbeat_timeout:g} s",
                            )
                        break
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=min(1, remaining))
            except (KeyError, OSError, TypeError, ValueError) as error:
                record_failure("monitor_error", repr(error))

            if "error_kind" in result and last_heartbeat is not None:
                try:
                    result["stack_obtained_before_sigterm"] = capture_thread_stack(
                        process, stack_path
                    )
                except OSError as error:
                    result["diagnostic_error"] = repr(error)
        finally:
            result["termination_pending"] = terminate_variant(process)

    if "error_kind" not in result and process.returncode != 0:
        record_failure("worker_failed", f"Worker exited with code {process.returncode}")
    result.update(
        status="failed" if "error_kind" in result else "complete",
        returncode=process.returncode,
        finished_at_utc=utc(),
    )
    write(health / "worker.json", result)
    return result
