"""Private, durable creation records and a host-wide cleanup lock."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HOUR_NS = 3600 * 10**9
LIFETIME_NS = 336 * HOUR_NS


class CleanupError(RuntimeError):
    pass


class Busy(CleanupError):
    pass


def state_root() -> Path:
    root = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    if not root.is_absolute():
        raise CleanupError("XDG_STATE_HOME must be absolute")
    return root / "tennis-lab-agents"


def utc(ns: int) -> str:
    seconds, remainder = divmod(ns, 10**9)
    return (
        datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%S")
        + f".{remainder:09d}Z"
    )


def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len(pairs) != len({key for key, _ in pairs}):
        raise CleanupError("duplicate keys in worktree state")
    return dict(pairs)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise CleanupError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=".worktree-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        sync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_DIRECTORY | os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def locked(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(
        directory / "cleanup.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(fd, "w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Busy("another worktree scan/cleanup holds cleanup.lock") from exc
        yield


class State:
    def __init__(self, directory: Path, repo: Path) -> None:
        self.directory = directory
        self.path = directory / "worktrees.json"
        self.data: dict[str, Any] = (
            read_json(self.path)
            if self.path.exists()
            else {
                "schema_version": 1,
                "repo": str(repo),
                "records": {},
            }
        )
        if (
            set(self.data) != {"schema_version", "repo", "records"}
            or type(self.data["schema_version"]) is not int
            or self.data["schema_version"] != 1
            or self.data["repo"] != str(repo)
            or not isinstance(self.data["records"], dict)
        ):
            raise CleanupError("unknown, malformed or foreign worktree state")
        for key, row in self.records.items():
            if (
                not isinstance(row, dict)
                or re.fullmatch(r"[0-9a-f]{64}", key) is None
                or row.get("id") != key
                or not isinstance(row.get("path"), str)
                or not isinstance(row.get("admin"), str)
                or not {"admin_identity", "directory_identity", "head"} <= row.keys()
                or not isinstance(row.get("admin_identity"), dict)
                or set(row["admin_identity"])
                not in (
                    {"device", "inode", "birth_time_ns"},
                    {"device", "inode", "birth_time_ns", "token"},
                )
                or (
                    row["admin_identity"].get("token") is not None
                    and re.fullmatch(
                        r"[0-9a-f]{32}", str(row["admin_identity"]["token"])
                    )
                    is None
                )
                or any(
                    type(row["admin_identity"].get(k)) is not int
                    for k in ("device", "inode")
                )
                or type(row.get("created_at_ns")) is not int
                or row["created_at_ns"] < 0
                or row.get("source") not in {"birth_time", "first_observed"}
                or row.get("created_at") != utc(row["created_at_ns"])
                or row.get("phase")
                not in {
                    "observed",
                    "stopping",
                    "quarantined",
                    "unregistered",
                    "deleted",
                    "absent",
                    "superseded",
                    "quarantine_conflict",
                }
            ):
                raise CleanupError(f"malformed worktree record: {key}")

    @property
    def records(self) -> dict[str, Any]:
        records: dict[str, Any] = self.data["records"]
        return records

    def save(self) -> None:
        write_json(self.path, self.data)

    def log(self, row: dict[str, Any], **fields: Any) -> None:
        fd = os.open(
            self.directory / "results.jsonl",
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, "a") as stream:
            stream.write(
                json.dumps(
                    {
                        "path": row["path"],
                        "head": row.get("head"),
                        "id": row["id"],
                        **fields,
                    }
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
