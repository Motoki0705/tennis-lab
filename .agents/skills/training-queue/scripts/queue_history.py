#!/usr/bin/env python3
"""Queue filename ordering and bounded history (stdlib only).

The shell owns process teardown. Only its terminal publications, or legacy
files already in terminal directories, authorize history collection here.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import stat
import sys
import tempfile
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# The entry point is also called from arbitrary workload CWDs.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from src.utils.training_queue import (  # noqa: E402
    CEILING,
    RETENTION_NS,
    TERMINAL,
    completion_path,
    read_completion,
    registration_order,
    regular,
)


def now_ns() -> int:
    value = os.environ.get("TRAINING_QUEUE_TEST_NOW_NS")
    result = time.time_ns() if value is None else int(value)
    if not 0 < result <= CEILING:
        raise ValueError("queue clock must be a positive epoch nanosecond value")
    return result


def job_name(pid: str, name: str) -> str:
    timestamp = now_ns()
    seconds, nanos = divmod(timestamp, 10**9)
    readable = datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H-%M-%S")
    return f"0-{CEILING - timestamp:019d}_{readable}-{nanos:09d}Z_{pid}_{name}"


def ordered_jobs(directory: Path, newest: bool = False) -> list[str]:
    jobs = [p.name for p in directory.glob("*.job") if not p.name.startswith(".")]
    return sorted(jobs, key=registration_order, reverse=newest)


def values(path: Path) -> dict[str, str]:
    if not regular(path):
        return {}
    result = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            raise ValueError(f"invalid queue state/owner record: {path}")
        result[key] = value
    if not result:
        raise ValueError(f"empty queue state/owner record: {path}")
    return result


def write_completion(queue: Path, record: dict[str, Any]) -> None:
    path = completion_path(queue, record["job"])
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=".tmp.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(record, stream, sort_keys=True)
            stream.write("\n")
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def history_lock(queue: Path) -> Iterator[None]:
    for name in (
        "jobs",
        "running",
        *TERMINAL,
        "logs",
        "state",
        "cancel-requests",
        "control",
        "control/completions",
        "control/external-acks",
        "control/registrations",
    ):
        directory = queue / name
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"queue history directory is unsafe: {directory}")
    descriptor = os.open(
        queue / "control/history.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("queue history lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def complete(queue: Path, job: str, state: str) -> None:
    if state not in TERMINAL:
        raise ValueError(f"invalid terminal state: {state}")
    with history_lock(queue):
        if not regular(queue / state / job):
            raise ValueError(f"cannot complete job without terminal history: {job}")
        previous = read_completion(queue, job)
        if previous is not None:
            if previous["state"] != state or "expired_at_ns" in previous:
                raise ValueError(f"conflicting completion for {job}")
            if previous["time_source"] == "recorded":
                return
        timestamp = now_ns()
        write_completion(
            queue,
            dict(
                version=1,
                job=job,
                state=state,
                finished_at_ns=timestamp,
                retention_started_at_ns=timestamp,
                time_source="recorded",
            ),
        )


def protected(queue: Path, job: str) -> bool:
    for directory in ("jobs", "running"):
        path = queue / directory / job
        if path.exists() or path.is_symlink():
            return True
    state = values(queue / "state" / f"{job[:-4]}.state").get("state")
    if state is not None and state not in TERMINAL:
        return True
    owner = values(queue / "state" / f"{job[:-4]}.owner")
    if owner:
        if (
            owner.get("job") != job
            or not owner.get("pid", "").isdigit()
            or not owner.get("start_time", "").isdigit()
        ):
            raise ValueError(f"invalid queue owner record: {job}")
        try:
            # comm may contain spaces or parentheses; starttime is field 22.
            process = (
                Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(")", 1)[1].split()
            )
        except FileNotFoundError:
            return False
        return process[19] == owner["start_time"]
    return False


def maintain(queue: Path) -> None:
    migrated = orphaned = 0
    timestamp = now_ns()
    with history_lock(queue):
        terminal: dict[str, str] = {}
        for state in TERMINAL:
            for path in (queue / state).glob("*.job"):
                if path.name.startswith("."):
                    continue
                if path.name in terminal:
                    raise ValueError(
                        f"job appears in multiple terminal directories: {path.name}"
                    )
                terminal[path.name] = state
        jobs = set(terminal)
        jobs.update(
            f"{p.stem}.job" for p in (queue / "control/completions").glob("*.json")
        )
        for job in sorted(jobs):
            if protected(queue, job):
                continue
            record = read_completion(queue, job)
            state_value = values(queue / "state" / f"{job[:-4]}.state").get("state")
            expected = terminal.get(job) if record is None else record["state"]
            if state_value is not None and state_value != expected:
                raise ValueError(f"terminal state disagrees with history: {job}")
            if record is None:
                # No reliable end timestamp exists for old data. Start a full
                # grace period on first observation, never use registration/mtime.
                regular(queue / terminal[job] / job)
                record = dict(
                    version=1,
                    job=job,
                    state=terminal[job],
                    finished_at_ns=None,
                    retention_started_at_ns=timestamp,
                    time_source="legacy-observed",
                )
                write_completion(queue, record)
                migrated += 1
            if job in terminal and record["state"] != terminal[job]:
                raise ValueError(f"terminal history disagrees with completion: {job}")
            if timestamp - record["retention_started_at_ns"] <= RETENTION_NS:
                continue
            if job not in terminal and "expired_at_ns" not in record:
                # An unexplained disappearance is not an intentional expiry.
                raise ValueError(f"terminal history missing before expiry: {job}")
            stem = job[:-4]
            paths = [
                queue / record["state"] / job,
                queue / "logs" / f"{stem}.log",
                queue / "state" / f"{stem}.state",
                queue / "state" / f"{stem}.owner",
                queue / "cancel-requests" / f"{stem}.cancel",
                queue / "control/external-acks" / f"{stem}.ack",
            ]
            for path in paths:
                regular(path)  # Validate the complete deletion set before acting.
            if "expired_at_ns" not in record:
                record["expired_at_ns"] = timestamp
                write_completion(queue, record)
            # Publish the small tombstone first. Retrying after interruption is
            # safe both for collection and for idempotent registration.
            for path in paths:
                path.unlink(missing_ok=True)
        for path in (queue / "logs").glob("*.log"):
            job = f"{path.stem}.job"
            if (
                job not in jobs
                and not (queue / "jobs" / job).exists()
                and not (queue / "running" / job).exists()
            ):
                orphaned += 1
    if migrated:
        print(
            f"[history] {migrated} legacy terminal jobs: retaining for 7 days from first observation (end time unknown)",
            file=sys.stderr,
        )
    if orphaned:
        print(
            f"[history] retaining {orphaned} logs without terminal evidence; inspect manually",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("name", "order", "maintain", "complete", "expired", "list-expired"),
    )
    parser.add_argument("path")
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    queue = Path(args.path)
    try:
        if args.action == "name":
            print(job_name(args.path, args.arguments[0]))
        elif args.action == "order":
            jobs = ordered_jobs(queue, newest="newest" in args.arguments)
            if "first" in args.arguments:
                jobs = jobs[:1]
            elif "ten" in args.arguments:
                jobs = jobs[:10]
            for job in jobs:
                print(job)
        elif args.action == "maintain":
            maintain(queue)
        elif args.action == "complete":
            complete(queue, args.arguments[0], args.arguments[1])
        elif args.action == "expired":
            record = read_completion(queue, args.arguments[0])
            if record is None or "expired_at_ns" not in record:
                return 1
            print(record["state"])
        else:
            for path in sorted((queue / "control/completions").glob("*.json")):
                record = read_completion(queue, f"{path.stem}.job")
                if record is not None and "expired_at_ns" in record:
                    print(
                        f"{record['job']} resource=- slot=- pid=- pgid=- state={record['state']} wait=history-expired history=expired"
                    )
    except (OSError, ValueError, KeyError, IndexError) as exc:
        print(f"error: queue history: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
