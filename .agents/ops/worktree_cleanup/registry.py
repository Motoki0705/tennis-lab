"""Replay this tool's interrupted updates to the existing split parent registry."""

from __future__ import annotations

import fcntl
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lib.checkout_lock import registry_checkout_lock
from parent.policy import digest
from parent.state import Store

from .storage import CleanupError


class ExpiryStore(Store):
    """Keep Store's schema, locking and validation; journal only expiry writes.

    v2 owns a row and an index in separate files. Durable intent lets cleanup
    repair a process interruption between those writes without relaxing the
    existing CLI's refusal of inconsistent ownership or overwriting newer data.
    """

    @property
    def journal(self) -> Path:
        return Path(self.directory) / "worktree-expiry-transaction.json"

    def recover(self) -> None:
        with self._files() as files:
            self._verify_files(files)
            try:
                value = self._read_object(self.journal, files.root)
            except FileNotFoundError:
                return
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
                    ) is None
                ):
                    raise CleanupError("invalid worktree expiry registry transaction path")
                path = self.directory / change["path"]
                try:
                    current = self._read_object(path, files.directory_for(path))
                except FileNotFoundError:
                    current = None
                if current not in (change["before"], change["after"]):
                    raise CleanupError("expiry transaction conflicts with newer parent state")
            self._verify_files(files)
            for change in value["changes"]:
                path = self.directory / change["path"]
                directory = files.directory_for(path, create=True)
                self._write_json(path, change["after"], directory)
                os.fsync(directory.fd)
            os.unlink(self.journal.name, dir_fd=files.root.fd)
            os.fsync(files.root.fd)

    @contextmanager
    def locked(self) -> Iterator[Store]:
        # Recovery may replay an ownership index write as well as a parent
        # row. It must wait for an in-flight send, just like a fresh release.
        with registry_checkout_lock(), self.ownership(), self._recovered():
            yield self

    @contextmanager
    def _recovered(self) -> Iterator[Store]:
        with self._hold_lock("registry.lock", fcntl.LOCK_EX):
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
        with self._files() as files:
            self._verify_files(files)
            changes = []
            for name, after in targets.items():
                path = self.directory / name
                try:
                    before = self._read_object(path, files.directory_for(path))
                except FileNotFoundError:
                    before = None
                if before != after:
                    changes.append({"path": name, "before": before, "after": after})
            if changes:
                self._write_json(
                    self.journal, {"schema_version": 1, "changes": changes}, files.root
                )
                os.fsync(files.root.fd)
            super().save()
            if changes:
                if files.parents is not None:
                    os.fsync(files.parents.fd)
                os.fsync(files.root.fd)
                os.unlink(self.journal.name, dir_fd=files.root.fd)
                os.fsync(files.root.fd)
