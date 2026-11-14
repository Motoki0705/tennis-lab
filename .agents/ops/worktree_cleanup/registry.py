"""Replay this tool's interrupted updates to the existing split parent registry."""

from __future__ import annotations

import fcntl
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from parent.policy import digest
from parent.state import Store

from .storage import CleanupError, read_json, sync_directory, write_json


class ExpiryStore(Store):
    """Keep Store's schema, locking and validation; journal only expiry writes.

    v2 owns a row and an index in separate files. Durable intent lets cleanup
    repair a process interruption between those writes without relaxing the
    existing CLI's refusal of inconsistent ownership or overwriting newer data.
    """

    write_json = staticmethod(write_json)

    @property
    def journal(self) -> Path:
        return Path(self.directory) / "worktree-expiry-transaction.json"

    def recover(self) -> None:
        if not self.journal.exists():
            return
        value = read_json(self.journal)
        if (
            set(value) != {"schema_version", "changes"}
            or value["schema_version"] != 1
            or not isinstance(value["changes"], list)
        ):
            raise CleanupError("invalid worktree expiry registry transaction")
        for change in value["changes"]:
            if (
                not isinstance(change, dict)
                or set(change) != {"path", "before", "after"}
                or not isinstance(change["path"], str)
                or not isinstance(change["after"], dict)
                or re.fullmatch(
                    r"registry\.json|parents/[0-9a-f]{64}\.json", change["path"]
                )
                is None
            ):
                raise CleanupError("invalid worktree expiry registry transaction path")
            path = self.directory / change["path"]
            current = read_json(path) if path.exists() else None
            if current not in (change["before"], change["after"]):
                raise CleanupError(
                    "expiry transaction conflicts with newer parent state"
                )
        for change in value["changes"]:
            write_json(self.directory / change["path"], change["after"])
        self.journal.unlink()
        sync_directory(self.directory)

    @contextmanager
    def locked(self) -> Iterator[Store]:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(
            self.directory / "registry.lock",
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, "w") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            self.recover()
        with super().locked():
            yield self

    def save(self) -> None:
        self._require_current()
        if self._lock is None:
            raise CleanupError("expiry save requires the registry lock")
        targets: dict[str, Any] = {}
        if self.data["version"] == 1:
            targets["registry.json"] = self.data
        else:
            for key, row in self.parents.items():
                targets[f"parents/{digest(key)}.json"] = row
            targets["registry.json"] = {
                "version": 2,
                "parents": {
                    key: {
                        field: row[field]
                        for field in ("session", "status", "generation")
                    }
                    for key, row in self.parents.items()
                },
            }
        changes = []
        for name, after in targets.items():
            path = self.directory / name
            before = read_json(path) if path.exists() else None
            if before != after:
                changes.append({"path": name, "before": before, "after": after})
        if changes:
            write_json(self.journal, {"schema_version": 1, "changes": changes})
        super().save()
        if self.journal.exists():
            self.journal.unlink()
            sync_directory(self.directory)
