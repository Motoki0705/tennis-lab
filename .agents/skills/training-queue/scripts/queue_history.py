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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Checkouts and installed bundles use the same shared source module. Resolve
# only these explicit layouts, never an unrelated parent checkout or the CWD.
script_directory = Path(__file__).resolve().parent
module_root = (
    script_directory.parents[3]
    if script_directory.parts[-4:] == (".agents", "skills", "training-queue", "scripts")
    else script_directory
)
if not (module_root / "src/utils/training_queue.py").is_file():
    raise ImportError(f"training queue shared module is missing under {module_root}")
sys.path.insert(0, str(module_root))
from src.utils.training_queue import (  # noqa: E402
    CEILING,
    RETENTION_NS,
    TERMINAL,
    InvalidQueueRecord,
    completion_path,
    parse_completion,
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


def valid_pid(value: str) -> bool:
    return (
        value.isascii()
        and value.isdigit()
        and len(value) <= 10
        and not value.startswith("0")
        and int(value) <= 2**31 - 1
    )


def values(entry: os.DirEntry[str] | None) -> dict[str, str]:
    """Read an optional state/owner file already type-checked by the scan."""
    if entry is None:
        return {}
    path = Path(entry.path)
    result = {}
    try:
        lines = path.read_text().splitlines()
    except UnicodeError as exc:
        raise InvalidQueueRecord(f"invalid queue state/owner record: {path}") from exc
    for line in lines:
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            raise InvalidQueueRecord(f"invalid queue state/owner record: {path}")
        result[key] = value
    if not result:
        raise InvalidQueueRecord(f"empty queue state/owner record: {path}")
    if path.suffix == ".state" and result.get("state") not in (
        "queued",
        "waiting",
        "running",
        "terminating",
        *TERMINAL,
    ):
        raise InvalidQueueRecord(f"invalid queue lifecycle state: {path}")
    if path.suffix == ".state":
        for key, allowed in (
            ("resource", ("half", "all", "invalid")),
            ("slot", ("-", "0", "1", "all")),
        ):
            if key in result and result[key] not in allowed:
                raise InvalidQueueRecord(f"invalid queue state {key}: {path}")
        for key in ("pid", "pgid"):
            if key in result and result[key] != "-" and not valid_pid(result[key]):
                raise InvalidQueueRecord(f"invalid queue state {key}: {path}")
        if "wait" in result and not result["wait"]:
            raise InvalidQueueRecord(f"empty queue state wait reason: {path}")
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


def protected(
    job: str, state: str | None, owner: dict[str, str], active: bool
) -> bool:
    owned = False
    if owner:
        if (
            owner.get("job") != job
            or not valid_pid(owner.get("pid", ""))
            or not owner.get("start_time", "").isascii()
            or not owner.get("start_time", "").isdigit()
            or len(owner["start_time"]) > 20
        ):
            raise InvalidQueueRecord(f"invalid queue owner record: {job}")
        try:
            # comm may contain spaces or parentheses; starttime is field 22.
            process = (
                Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(")", 1)[1].split()
            )
        except FileNotFoundError:
            pass
        else:
            owned = process[19] == owner["start_time"]
    return owned or (state is not None and state not in TERMINAL) or active


def collection_paths(queue: Path, job: str, state: str) -> list[Path]:
    stem = job[:-4]
    return [
        queue / state / job,
        queue / "logs" / f"{stem}.log",
        queue / "state" / f"{stem}.state",
        queue / "state" / f"{stem}.owner",
        queue / "cancel-requests" / f"{stem}.cancel",
        queue / "control/external-acks" / f"{stem}.ack",
    ]


@dataclass
class HistoryEntry:
    job: str
    directories: list[str]
    record: dict[str, Any] | None = None
    protected: bool = False
    problem: str | None = None
    orphaned: bool = False


@dataclass
class HistoryFiles:
    files: dict[str, dict[str, os.DirEntry[str]]]
    terminal: dict[str, list[str]]
    related: set[str]


def scan_history(queue: Path) -> HistoryFiles:
    """Index each directory once, using entry types rather than candidate stats."""
    files: dict[str, dict[str, os.DirEntry[str]]] = {}
    terminal: dict[str, list[str]] = {}
    related: set[str] = set()
    for directory, suffixes in (
        ("jobs", (".job",)),
        ("running", (".job",)),
        *((state, (".job",)) for state in TERMINAL),
        ("logs", (".log",)),
        ("state", (".state", ".owner")),
        ("cancel-requests", (".cancel",)),
        ("control/external-acks", (".ack",)),
        ("control/completions", (".json",)),
    ):
        files[directory] = {}
        with os.scandir(queue / directory) as entries:
            for entry in entries:
                if entry.name.startswith(".") or not entry.name.endswith(suffixes):
                    continue
                # Validate before inspecting any records. Corrupt job data must
                # not hide a symlink, directory or special file for that job.
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise ValueError(
                        f"queue history path must be a regular file: {entry.path}"
                    )
                files[directory][entry.name] = entry
                if directory != "control/completions":
                    job = entry.name.rsplit(".", 1)[0] + ".job"
                    related.add(job)
                    if directory in TERMINAL:
                        terminal.setdefault(job, []).append(directory)
    return HistoryFiles(files, terminal, related)


def history_entries(
    queue: Path, *, include_expired: bool = True
) -> Iterator[HistoryEntry]:
    """Inspect under history_lock; only job-local record errors are recoverable."""
    snapshot = scan_history(queue)
    files = snapshot.files
    jobs = set(snapshot.terminal)
    jobs.update(name[:-5] + ".job" for name in files["control/completions"])
    jobs.update(name[:-4] + ".job" for name in files["logs"])
    for job in sorted(jobs):
        entry = HistoryEntry(job, snapshot.terminal.get(job, []))
        try:
            completion = files["control/completions"].get(f"{job[:-4]}.json")
            if completion is not None:
                entry.record = parse_completion(Path(completion.path).read_bytes(), job)
            # Retain record validation (including missing unexpired history),
            # but do no per-job path probes or deletion retries for an already
            # collected tombstone. Any reappearing related file disables this.
            if (
                job not in snapshot.related
                and entry.record is not None
                and "expired_at_ns" in entry.record
            ):
                if include_expired:
                    yield entry
                continue
            if len(entry.directories) > 1:
                raise InvalidQueueRecord("job appears in multiple terminal directories")
            state_value = values(files["state"].get(f"{job[:-4]}.state")).get("state")
            owner = values(files["state"].get(f"{job[:-4]}.owner"))
            entry.protected = protected(
                job, state_value, owner, job in files["jobs"] or job in files["running"]
            )
            terminal_directory = entry.directories[0] if entry.directories else None
            expected = (
                terminal_directory if entry.record is None else entry.record["state"]
            )
            if (
                expected is not None
                and state_value is not None
                and state_value != expected
            ):
                raise InvalidQueueRecord(
                    f"terminal state {state_value} disagrees with history {expected}"
                )
            if entry.record is not None:
                if (
                    terminal_directory is not None
                    and entry.record["state"] != terminal_directory
                ):
                    raise InvalidQueueRecord(
                        "terminal directory disagrees with completion"
                    )
                if (
                    terminal_directory is None
                    and "expired_at_ns" not in entry.record
                    and not entry.protected
                ):
                    raise InvalidQueueRecord("terminal history missing before expiry")
            elif terminal_directory is None and not entry.protected:
                entry.orphaned = True
        except InvalidQueueRecord as exc:
            entry.problem = str(exc)
        yield entry


def maintain(queue: Path) -> None:
    migrated = orphaned = inconsistent = 0
    timestamp = now_ns()
    with history_lock(queue):
        for entry in history_entries(queue, include_expired=False):
            if entry.problem:
                inconsistent += 1
                continue
            if entry.orphaned:
                orphaned += 1
                continue
            if entry.protected:
                continue
            job, record = entry.job, entry.record
            if record is None:
                # No reliable end timestamp exists for old data. Start a full
                # grace period on first observation, never use registration/mtime.
                record = dict(
                    version=1,
                    job=job,
                    state=entry.directories[0],
                    finished_at_ns=None,
                    retention_started_at_ns=timestamp,
                    time_source="legacy-observed",
                )
                write_completion(queue, record)
                migrated += 1
            if timestamp - record["retention_started_at_ns"] <= RETENTION_NS:
                continue
            paths = collection_paths(queue, job, record["state"])
            for path in paths:
                regular(path)  # Validate the complete deletion set before acting.
            if "expired_at_ns" not in record:
                record["expired_at_ns"] = timestamp
                write_completion(queue, record)
            # Publish the small tombstone first. Retrying after interruption is
            # safe both for collection and for idempotent registration.
            for path in paths:
                path.unlink(missing_ok=True)
    if migrated:
        print(
            f"[history] {migrated} legacy terminal jobs: retaining for 7 days from first observation (end time unknown)",
            file=sys.stderr,
        )
    if orphaned:
        print(
            f"[history] retaining {orphaned} logs without terminal evidence; inspect with list --include-expired (history=orphaned)",
            file=sys.stderr,
        )
    if inconsistent:
        print(
            f"[history] retaining {inconsistent} inconsistent jobs; inspect with list --include-expired (history=inconsistent)",
            file=sys.stderr,
        )


def list_history(queue: Path) -> None:
    with history_lock(queue):
        for entry in history_entries(queue):
            if entry.problem or entry.orphaned:
                kind = "inconsistent" if entry.problem else "orphaned"
                reason = entry.problem or "log without terminal evidence"
                print(
                    f"{entry.job} resource=- slot=- pid=- pgid=- state=unknown "
                    f"wait=history-{kind} history={kind} "
                    f"directories={','.join(entry.directories) or '-'} reason={json.dumps(reason)}"
                )
            elif (
                not entry.protected
                and entry.record is not None
                and "expired_at_ns" in entry.record
            ):
                print(
                    f"{entry.job} resource=- slot=- pid=- pgid=- state={entry.record['state']} wait=history-expired history=expired"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("name", "order", "maintain", "complete", "expired", "list-history"),
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
            list_history(queue)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        print(f"error: queue history: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
