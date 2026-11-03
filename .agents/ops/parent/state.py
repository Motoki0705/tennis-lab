"""Machine-local ownership: one locked registry, atomic durable writes."""

from __future__ import annotations

import copy
import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from lib.identity import session_identity as session_identity

from . import metrics
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
        self._focus: str | None = None
        self._saved: dict[str, Any] = {}
        self.transaction_epoch = 0

    def focus(self, key: str) -> None:
        self._focus = key.casefold()

    @contextmanager
    def operation(self, key: str) -> Iterator[None]:
        """Serialize a parent's mutations while its registry lock is released."""
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.directory / f"operation-{digest(key)}.lock").open(
            "a", encoding="utf-8"
        ) as lock:
            os.chmod(lock.name, 0o600)
            metrics.flock(lock, fcntl.LOCK_EX)
            yield

    @contextmanager
    def locked(self) -> Iterator[Store]:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.directory / "registry.lock").open("a", encoding="utf-8") as lock:
            os.chmod(lock.name, 0o600)
            metrics.flock(lock, fcntl.LOCK_EX)
            self.transaction_epoch += 1
            self.data = self.read_data()
            self._saved = copy.deepcopy(self.data)
            if (
                set(self.data) != {"version", "parents"}
                or type(self.data["version"]) is not int
                or self.data["version"] not in {1, 2}
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
        before = self.read_data()
        metrics.flock(lock, fcntl.LOCK_UN)
        try:
            yield
        finally:
            metrics.flock(lock, fcntl.LOCK_EX)
            after = self.read_data()
            if after != before:
                key = self._focus
                if (
                    key is None
                    or before.get("version") != after.get("version")
                    or before["parents"].get(key) != after["parents"].get(key)
                ):
                    metrics.count("conflicts")
                    raise ValueError(
                        "parent registry changed during GitHub retrieval; retry the command to fetch current state"
                    )
                # Preserve this parent's live objects, including unsaved changes.
                # Merge only unrelated durable rows; never overwrite them later.
                for peer in set(before["parents"]) | set(after["parents"]):
                    if peer == key:
                        continue
                    if before["parents"].get(peer) != after["parents"].get(peer):
                        if self.parents.get(peer) != before["parents"].get(peer):
                            raise ValueError("concurrent changes to a peer parent")
                        if peer in after["parents"]:
                            self.parents[peer] = copy.deepcopy(after["parents"][peer])
                            self._saved["parents"][peer] = copy.deepcopy(
                                after["parents"][peer]
                            )
                        else:
                            self.parents.pop(peer, None)
                            self._saved["parents"].pop(peer, None)
                owners = [
                    r["session"]
                    for r in self.parents.values()
                    if r["status"] in {"active", "handoff_pending"}
                ]
                if len(owners) != len(set(owners)):
                    raise ValueError("another parent claimed the same session")

    @property
    def parents(self) -> dict[str, Any]:
        return object_value(self.data["parents"])

    def read_data(self) -> dict[str, Any]:
        if not self.path.exists() and not self.path.is_symlink():
            return {"version": 1, "parents": {}}
        data = object_value(
            json.loads(
                self.path.read_text(encoding="utf-8"), object_pairs_hook=unique_object
            )
        )
        if data.get("version") != 2:
            return data
        if set(data) != {"version", "parents"} or type(data["version"]) is not int:
            raise ValueError("unknown or malformed parent-session state")
        parents = {}
        for key, owner in object_value(data["parents"]).items():
            path = self.directory / "parents" / (digest(key) + ".json")
            row = object_value(
                json.loads(
                    path.read_text(encoding="utf-8"), object_pairs_hook=unique_object
                )
            )
            if {
                field: row.get(field) for field in ("session", "status", "generation")
            } != owner:
                raise ValueError(
                    "parent ownership index disagrees with its record; restore the saved migration/transaction backup"
                )
            parents[key] = row
        return {"version": 2, "parents": parents}

    def migrate(self) -> None:
        """Explicit migration under the global lock; never discard old evidence."""
        if self._lock is None:
            raise ValueError("migration requires the registry lock")
        if self.data["version"] == 2:
            return
        backup = self.directory / "registry-v1.backup.json"
        if backup.exists():
            if json.loads(backup.read_text()) != self.data:
                raise ValueError(
                    "migration backup exists for different state; inspect before retry"
                )
        else:
            self.write_json(backup, self.data)
        self.data["version"] = 2
        self.save()

    def save(self) -> None:
        if self._lock is None:
            raise ValueError("save requires an active registry transaction")
        owners = [
            row["session"]
            for row in self.parents.values()
            if row["status"] in {"active", "handoff_pending"}
        ]
        if len(owners) != len(set(owners)):
            raise ValueError(
                "registry contains multiple active parents for one session"
            )
        if self.data["version"] == 1:
            self.write_json(self.path, self.data)
        else:
            ownership_index: dict[str, Any] = {}
            for key, row in self.parents.items():
                path = self.directory / "parents" / (digest(key) + ".json")
                if (
                    self._saved.get("version") != 2
                    or self._saved.get("parents", {}).get(key) != row
                ):
                    self.write_json(path, row)
                ownership_index[key] = {
                    field: row[field] for field in ("session", "status", "generation")
                }
            index = {"version": 2, "parents": ownership_index}
            if not self.path.exists() or json.loads(self.path.read_text()) != index:
                self.write_json(self.path, index)
        self._saved = copy.deepcopy(self.data)

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
