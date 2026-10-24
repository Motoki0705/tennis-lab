"""Machine-local ownership: one locked registry, atomic durable writes."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from lib.identity import session_identity as session_identity

from .execution import validate_authorization
from .policy import digest, object_value


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len(pairs) != len({key for key, _ in pairs}):
        raise ValueError("duplicate keys in parent-session state")
    return dict(pairs)


def state_directory() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    if not base.is_absolute():
        raise ValueError("XDG_STATE_HOME must be absolute")
    return base / "tennis-lab-agents/parent-sessions"


class Store:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "registry.json"
        self.data: dict[str, Any] = {}
        self._lock: TextIO | None = None

    @contextmanager
    def operation(self, key: str) -> Iterator[None]:
        """Serialize a parent's mutations while its registry lock is released."""
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.directory / f"operation-{digest(key)}.lock").open(
            "a", encoding="utf-8"
        ) as lock:
            os.chmod(lock.name, 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    @contextmanager
    def locked(self) -> Iterator[Store]:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.directory / "registry.lock").open("a", encoding="utf-8") as lock:
            os.chmod(lock.name, 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.data = (
                object_value(
                    json.loads(
                        self.path.read_text(encoding="utf-8"),
                        object_pairs_hook=unique_object,
                    )
                )
                if self.path.exists() or self.path.is_symlink()
                else {"version": 1, "parents": {}}
            )
            if (
                set(self.data) != {"version", "parents"}
                or type(self.data["version"]) is not int
                or self.data["version"] != 1
            ):
                raise ValueError("unknown or malformed parent-session state")
            parents: dict[str, Any] = {}
            for key, row in object_value(self.data["parents"]).items():
                canonical = key.casefold()
                if canonical in parents:
                    raise ValueError(
                        f"registry contains a repository alias collision: {canonical}"
                    )
                parents[canonical] = row
                if (
                    not isinstance(row, dict)
                    or not {"session", "status", "generation", "tasks", "reviews"}
                    <= row.keys()
                    or row["status"]
                    not in {"active", "released", "ended", "handoff_pending"}
                ):
                    raise ValueError(f"invalid parent-session record: {key}")
                if (
                    not isinstance(row["session"], str)
                    or not row["session"]
                    or not isinstance(row["tasks"], dict)
                    or not isinstance(row["reviews"], list)
                    or type(row["generation"]) is not int
                    or row["generation"] < 1
                ):
                    raise ValueError(f"invalid parent-session data types: {key}")
                for task in row["tasks"].values():
                    object_value(task)
                    if "authorization" in task:
                        validate_authorization(task["authorization"])
                for review in row["reviews"]:
                    object_value(review)
                    request = object_value(review.get("request"))
                    version = request.get("schema_version")
                    if type(version) is not int or version not in {1, 2}:
                        raise ValueError("unknown or malformed saved review request")
                    if version == 2:
                        validate_authorization(request.get("authorization"))
            owners = [
                r["session"]
                for r in parents.values()
                if r["status"] in {"active", "handoff_pending"}
            ]
            if len(owners) != len(set(owners)):
                raise ValueError(
                    "registry contains multiple active parents for one session"
                )
            self.data["parents"] = parents
            # Keep the envelope readable by active older CLIs. Versioned evidence
            # is additive inside tasks/reviews; never rewrite legacy evidence.
            self._lock = lock
            try:
                yield self
            finally:
                self._lock = None

    @contextmanager
    def unlocked_read(self) -> Iterator[None]:
        """Fetch remotely without blocking peers, then reject a changed registry.

        Compare the durable bytes, not self.data: callers may have unsaved
        changes or hold references into it. Never reload over those references
        or merge stale reservations after another process has saved.
        """
        lock = self._lock
        if lock is None:
            raise ValueError("GitHub read requires an active registry transaction")
        before = (
            self.path.read_bytes()
            if self.path.exists() or self.path.is_symlink()
            else None
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_EX)
            after = (
                self.path.read_bytes()
                if self.path.exists() or self.path.is_symlink()
                else None
            )
            if after != before:
                raise ValueError(
                    "parent registry changed during GitHub retrieval; retry the command to fetch current state"
                )

    @property
    def parents(self) -> dict[str, Any]:
        return object_value(self.data["parents"])

    def save(self) -> None:
        self.write_json(self.path, self.data)

    @staticmethod
    def write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_name(path.name + ".tmp")
        # State is private; never follow a pre-existing temporary symlink.
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
