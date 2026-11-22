"""Machine-local ownership: one locked registry, atomic durable writes."""

from __future__ import annotations

import copy
import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, NoReturn, TextIO

from lib.identity import session_identity as session_identity
from lib.state_files import (
    PreparedStateWrite,
    StateDirectory,
    atomic_state_writer,
    open_state_directory,
    open_state_lock,
    prepare_state_write,
    read_state_text,
)

from . import metrics
from .execution import validate_authorization
from .policy import digest, object_value


class RegistryConflict(ValueError):
    """An observation no longer authorizes writes in this transaction."""


class RegistryStateError(ValueError):
    """Persisted registry data failed to load or validate."""

    def __init__(self, reason: str, path: Path, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.path = path


@dataclass
class _RegistryFiles:
    root: StateDirectory
    stack: ExitStack
    index: Path
    parents: StateDirectory | None = None
    lock: TextIO | None = None

    def parent_directory(self, *, create: bool = False) -> StateDirectory:
        if self.parents is None:
            self.parents = self.stack.enter_context(
                open_state_directory(
                    self.root.path / "parents",
                    target=self.index,
                    anchor=self.root,
                    create=create,
                )
            )
        return self.parents

    def verify(self) -> None:
        self.root.verify(self.index)
        if self.parents is not None:
            self.parents.verify(self.index, anchor=self.root)

    def directory_for(self, path: Path, *, create: bool = False) -> StateDirectory:
        if path.parent.absolute() == self.root.path:
            return self.root
        if path.parent.absolute() != self.root.path / "parents":
            raise ValueError("unexpected parent registry record path")
        return self.parent_directory(create=create)


@dataclass(frozen=True)
class Checkpoint:
    version: int
    key: str | None
    parents: dict[str, Any]


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
        self._write_invalid = False
        self._suspended = False
        self._transaction_guard = RLock()
        self._bound_files: ContextVar[_RegistryFiles | None] = ContextVar(
            "parent registry files", default=None
        )
        self.transaction_epoch = 0

    def _require_current(self) -> None:
        if self._write_invalid:
            raise RegistryConflict(
                "parent registry changed or could not be revalidated; retry the command in a new transaction"
            )

    def require_transaction(self) -> None:
        if self._lock is None or self._suspended:
            raise ValueError("I/O requires an active registry transaction")
        self._require_current()

    def require_unlocked(self) -> None:
        if self._lock is not None and not self._suspended:
            raise ValueError(
                "detached client requires the registry lock to be released"
            )

    def conflict(self, message: str) -> NoReturn:
        self._write_invalid = True
        metrics.count("conflicts")
        raise RegistryConflict(message)

    def checkpoint(self) -> Checkpoint:
        self.require_transaction()
        return Checkpoint(
            self.data["version"], self._focus, copy.deepcopy(self.parents)
        )

    def revalidate(self, checkpoint: Checkpoint) -> None:
        self.require_transaction()
        key = checkpoint.key
        if self.data["version"] != checkpoint.version or (
            self.parents != checkpoint.parents
            if key is None
            else self.parents.get(key) != checkpoint.parents.get(key)
        ):
            self.conflict(
                "parent registry changed between transactions; retry the command"
            )

    def focus(self, key: str) -> None:
        self._focus = key.casefold()

    @contextmanager
    def _files(self, *, create: bool = False) -> Iterator[_RegistryFiles]:
        files = self._bound_files.get()
        if files is not None:
            yield files
            return
        with ExitStack() as stack:
            root = stack.enter_context(
                open_state_directory(self.directory, target=self.path, create=create)
            )
            files = _RegistryFiles(root, stack, self.path)
            token = self._bound_files.set(files)
            try:
                yield files
            finally:
                self._bound_files.reset(token)

    def _verify_files(self, files: _RegistryFiles) -> None:
        files.verify()
        if files.lock is not None:
            files.root.verify_lock(self.directory / "registry.lock", files.lock)

    @contextmanager
    def _hold_lock(self, name: str, mode: int) -> Iterator[TextIO]:
        path = self.directory / name
        with (
            self._files(create=True) as files,
            open_state_lock(path, directory=files.root) as lock,
        ):
            metrics.flock(lock, mode)
            files.verify()
            files.root.verify_lock(path, lock)
            if name == "registry.lock":
                files.lock = lock
            try:
                yield lock
            finally:
                if files.lock is lock:
                    files.lock = None

    @contextmanager
    def operation(self, key: str) -> Iterator[None]:
        """Serialize a parent's mutations while its registry lock is released."""
        with self._hold_lock(f"operation-{digest(key)}.lock", fcntl.LOCK_EX):
            yield

    @contextmanager
    def ownership(self) -> Iterator[None]:
        """Fence owner changes and one board send, outside any transaction.

        All worktrees use this one host-state lock. Ordinary registry updates
        do not participate. Never wait for it while holding registry.lock.
        """
        from lib.checkout_lock import registry_checkout_lock

        with registry_checkout_lock(), self._ownership():
            yield

    @contextmanager
    def _ownership(self) -> Iterator[None]:
        # A different thread may be using this Store. Wait for its live
        # references first; only our own nested transaction is an inversion.
        # Release the guard before waiting for the host-wide fence.
        with self._transaction_guard:
            if self._lock is not None:
                raise ValueError(
                    "ownership fence must precede the registry transaction"
                )
        with self._hold_lock("ownership.lock", fcntl.LOCK_EX):
            yield

    def _validate_loaded(self) -> None:
        try:
            self._validate_rows()
        except (ValueError, KeyError, TypeError, OverflowError, RecursionError) as exc:
            raise RegistryStateError("registry_format", self.path, str(exc)) from exc

    def _validate_rows(self) -> None:
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
                if type(version) is not int or version not in {1, 2, 3, 4}:
                    raise ValueError("unknown or malformed saved review request")
                if version in {2, 3, 4}:
                    validate_authorization(request.get("authorization"))
                if version in {3, 4}:
                    from lib.review_settings import validate_selection

                    validate_selection(request.get("review_selection"))
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

    @contextmanager
    def snapshot(self) -> Iterator[Store]:
        """Read a validated ownership snapshot without creating locks or paths."""
        self.data = self.read_data()
        self._validate_loaded()
        yield self

    @contextmanager
    def consistent_snapshot(self) -> Iterator[Store]:
        """Read a validated snapshot that never interleaves with a peer save.

        Every save holds registry.lock exclusively, and a v2 read spans the
        ownership index and each parent record. A shared hold covers only that
        read; callers run their GitHub I/O after it is released, so peers are
        never blocked by remote latency. The snapshot is read-only: it opens no
        transaction and cannot save.
        """
        from lib.checkout_lock import registry_checkout_lock

        with registry_checkout_lock():
            with self._hold_lock("registry.lock", fcntl.LOCK_SH) as lock:
                try:
                    data = self.read_data()
                finally:
                    metrics.flock(lock, fcntl.LOCK_UN)
            self.data = data
            self._validate_loaded()
        yield self

    @contextmanager
    def locked(self) -> Iterator[Store]:
        from lib.checkout_lock import registry_checkout_lock

        # A Store owns live transaction objects. Threads sharing that instance
        # must wait even while its flock is released for another process's I/O.
        with registry_checkout_lock(), self._transaction_guard:
            if self._lock is not None:
                raise ValueError("registry transaction is already active")
            with self._locked() as store:
                yield store

    @contextmanager
    def _locked(self) -> Iterator[Store]:
        with self._hold_lock("registry.lock", fcntl.LOCK_EX) as lock:
            self.transaction_epoch += 1
            self._write_invalid = False
            self._suspended = False
            # Keep the envelope readable by active older CLIs. Versioned evidence
            # is additive inside tasks/reviews; never rewrite legacy evidence.
            self._lock = lock
            try:
                self.data = self.read_data()
                self._saved = copy.deepcopy(self.data)
                self._validate_loaded()
                yield self
            finally:
                self._lock = None
                self._suspended = False

    @contextmanager
    def unlocked_read(self) -> Iterator[None]:
        """Fetch remotely without blocking peers, then reject a changed registry.

        Compare the durable bytes, not self.data: callers may have unsaved
        changes or hold references into it. Never reload over those references
        or merge stale reservations after another process has saved.
        """
        self.require_transaction()
        lock = self._lock
        assert lock is not None
        self._write_invalid = True
        before = self.read_data()
        self._write_invalid = False
        metrics.flock(lock, fcntl.LOCK_UN)
        self._suspended = True
        input_conflict = False
        try:
            yield
        except RegistryConflict:
            input_conflict = True
            raise
        finally:
            metrics.flock(lock, fcntl.LOCK_EX)
            self._suspended = False
            # A caller may catch the retrieval exception to record a failed
            # launch. Never allow that stale in-memory transaction to save.
            self._write_invalid = True
            try:
                after = self.read_data()
            except (ValueError, OSError, KeyError, TypeError) as exc:
                raise RegistryConflict(
                    "parent registry could not be revalidated"
                ) from exc
            if after != before:
                key = self._focus
                if (
                    key is None
                    or before.get("version") != after.get("version")
                    or before["parents"].get(key) != after["parents"].get(key)
                ):
                    metrics.count("conflicts")
                    raise RegistryConflict(
                        "parent registry changed during GitHub retrieval; retry the command to fetch current state"
                    )
                # Preserve this parent's live objects, including unsaved changes.
                # Merge only unrelated durable rows; never overwrite them later.
                for peer in set(before["parents"]) | set(after["parents"]):
                    if peer == key:
                        continue
                    if before["parents"].get(peer) != after["parents"].get(peer):
                        if self.parents.get(peer) != before["parents"].get(peer):
                            raise RegistryConflict(
                                "concurrent changes to a peer parent"
                            )
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
                    raise RegistryConflict("another parent claimed the same session")
            self._write_invalid = input_conflict

    @property
    def parents(self) -> dict[str, Any]:
        return object_value(self.data["parents"])

    def registry_exists(self) -> bool:
        try:
            with self._files() as files:
                self._verify_files(files)
                os.stat(self.path.name, dir_fd=files.root.fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    def read_data(self) -> dict[str, Any]:
        try:
            with self._files() as files:
                self._verify_files(files)
                return self._read_registry(files)
        except FileNotFoundError:
            return {"version": 1, "parents": {}}

    @staticmethod
    def _read_object(path: Path, directory: StateDirectory) -> dict[str, Any]:
        try:
            value = json.loads(
                read_state_text(path, directory=directory),
                object_pairs_hook=unique_object,
            )
        except (ValueError, RecursionError) as exc:
            raise RegistryStateError("registry_json", path, str(exc)) from exc
        if not isinstance(value, dict):
            raise RegistryStateError(
                "registry_format", path, "registry entry must be an object"
            )
        return value

    def _read_registry(self, files: _RegistryFiles) -> dict[str, Any]:
        data = self._read_object(self.path, files.root)
        if (
            set(data) != {"version", "parents"}
            or type(data["version"]) is not int
            or data["version"] not in {1, 2}
            or not isinstance(data["parents"], dict)
        ):
            raise RegistryStateError(
                "registry_format",
                self.path,
                "unknown or malformed parent-session state",
            )
        if data.get("version") != 2:
            return data
        parents = {}
        for key, owner in data["parents"].items():
            if (
                not isinstance(owner, dict)
                or set(owner) != {"session", "status", "generation"}
                or not isinstance(owner["session"], str)
                or not isinstance(owner["status"], str)
                or type(owner["generation"]) is not int
            ):
                raise RegistryStateError(
                    "registry_format", self.path, "invalid ownership index entry"
                )
            path = self.directory / "parents" / (digest(key) + ".json")
            try:
                row = self._read_object(path, files.parent_directory())
            except FileNotFoundError as exc:
                raise RegistryStateError(
                    "registry_record_missing",
                    path,
                    "a parent record listed in the registry is missing",
                ) from exc
            if {
                field: row.get(field) for field in ("session", "status", "generation")
            } != owner:
                raise RegistryStateError(
                    "registry_mismatch",
                    self.path,
                    "parent ownership index disagrees with its record; restore the saved migration/transaction backup",
                )
            parents[key] = row
        return {"version": 2, "parents": parents}

    def migrate(self) -> None:
        """Explicit migration under the global lock; never discard old evidence."""
        self.require_transaction()
        if self.data["version"] == 2:
            return
        backup = self.directory / "registry-v1.backup.json"
        with self._files() as files:
            self._verify_files(files)
            try:
                previous = read_state_text(backup, directory=files.root)
            except FileNotFoundError:
                self._write_json(backup, self.data, files.root)
            else:
                if json.loads(previous) != self.data:
                    raise ValueError(
                        "migration backup exists for different state; inspect before retry"
                    )
        self.data["version"] = 2
        try:
            self.save()
        except (Exception, KeyboardInterrupt, SystemExit):
            self.data["version"] = 1
            raise

    def save(self) -> None:
        self.require_transaction()
        owners = [
            row["session"]
            for row in self.parents.values()
            if row["status"] in {"active", "handoff_pending"}
        ]
        if len(owners) != len(set(owners)):
            raise ValueError(
                "registry contains multiple active parents for one session"
            )
        with self._files() as files:
            self._verify_files(files)
            if self.data["version"] == 1:
                self._write_json(self.path, self.data, files.root)
            else:
                self._save_split(files)
        self._saved = copy.deepcopy(self.data)

    def _save_split(self, files: _RegistryFiles) -> None:
        index = {
            "version": 2,
            "parents": {
                key: {
                    field: row[field] for field in ("session", "status", "generation")
                }
                for key, row in self.parents.items()
            },
        }
        previous_parents = (
            self._saved["parents"] if self._saved.get("version") == 2 else {}
        )
        changed = [
            (key, row)
            for key, row in self.parents.items()
            if previous_parents.get(key) != row
        ]
        try:
            current_index = self._read_object(self.path, files.root)
        except FileNotFoundError:
            current_index = None

        with ExitStack() as stack:
            # Prepare the index first so unsafe entries and write failures are
            # detected before any record is published.
            pending_index = None
            if current_index != index:
                pending_index = stack.enter_context(
                    prepare_state_write(
                        self.path,
                        lambda stream: self._dump_json(stream, index),
                        directory=files.root,
                    )
                )
            pending_rows: list[
                tuple[PreparedStateWrite, PreparedStateWrite | None]
            ] = []
            if changed:
                parent_dir = files.parent_directory(create=True)
                for key, row in changed:
                    path = self.directory / "parents" / (digest(key) + ".json")
                    pending_rows.append(self._prepare_row(stack, path, row, parent_dir))

            self._verify_files(files)
            published = []
            try:
                for update, backup in pending_rows:
                    update.publish()
                    published.append((update, backup))
                if pending_index is not None:
                    pending_index.publish()
            except (Exception, KeyboardInterrupt, SystemExit):
                try:
                    for update, backup in reversed(published):
                        update.restore(backup)
                except OSError:
                    self._write_invalid = True
                    raise
                raise

    def _prepare_row(
        self,
        stack: ExitStack,
        path: Path,
        row: dict[str, Any],
        directory: StateDirectory,
    ) -> tuple[PreparedStateWrite, PreparedStateWrite | None]:
        backup = None
        try:
            previous = read_state_text(path, directory=directory)
        except FileNotFoundError:
            pass
        else:
            backup = stack.enter_context(
                prepare_state_write(
                    path, lambda stream: stream.write(previous), directory=directory
                )
            )
        pending = stack.enter_context(
            prepare_state_write(
                path, lambda stream: self._dump_json(stream, row), directory=directory
            )
        )
        return pending, backup

    @staticmethod
    def _dump_json(stream: TextIO, value: Any) -> None:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    @staticmethod
    def write_json(path: Path, value: Any) -> None:
        Store._write_json(path, value, None)

    @staticmethod
    def _write_json(path: Path, value: Any, directory: StateDirectory | None) -> None:
        with atomic_state_writer(path, directory=directory) as stream:
            Store._dump_json(stream, value)
