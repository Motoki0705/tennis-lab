"""Fail-closed whole-directory publication for fixed scene-stage owners."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.synthetic_data_generation.pipeline.contracts import (
    StageDefinition,
    StageExecutionSummary,
    StagePublicationResult,
)
from src.synthetic_data_generation.pipeline.workspace import SceneWorkspace

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2
_TRANSACTION_SCHEMA = "stage_owner_exchange_v1"
_TRANSACTION_FILE = "transaction.json"


class AtomicPublicationUnavailableError(OSError):
    """The host or workspace filesystem lacks the required rename authority."""


RenameOperation = Callable[[Path, Path, int], None]


@dataclass(frozen=True, slots=True)
class AtomicDirectoryPublication:
    """Publish complete owner snapshots with Linux ``RENAME_EXCHANGE``."""

    rename_operation: RenameOperation = field(
        default=lambda source, destination, flags: _renameat2(
            source,
            destination,
            flags=flags,
        ),
        repr=False,
        compare=False,
    )

    def preflight(
        self,
        workspace: SceneWorkspace,
        definition: StageDefinition[StageExecutionSummary],
    ) -> None:
        """Probe same-filesystem directory exchange before invalidation."""
        publisher = StagePublisher(
            workspace,
            definition,
            rename_operation=self.rename_operation,
        )
        publisher.preflight_atomic_exchange()

    def prepare(
        self,
        workspace: SceneWorkspace,
        definition: StageDefinition[StageExecutionSummary],
    ) -> Path:
        """Create the fixed transaction-root snapshot directory."""
        return StagePublisher(
            workspace,
            definition,
            rename_operation=self.rename_operation,
        ).prepare()

    def publish(
        self,
        workspace: SceneWorkspace,
        definition: StageDefinition[StageExecutionSummary],
    ) -> StagePublicationResult:
        """Exchange the complete staged snapshot with the fixed owner."""
        return StagePublisher(
            workspace,
            definition,
            rename_operation=self.rename_operation,
        ).publish()

    def recover(
        self,
        workspace: SceneWorkspace,
        definition: StageDefinition[StageExecutionSummary],
    ) -> None:
        """Keep whichever complete snapshot is currently at the owner path."""
        StagePublisher(
            workspace,
            definition,
            rename_operation=self.rename_operation,
        ).recover_interrupted_publication()

    def abandon(
        self,
        workspace: SceneWorkspace,
        definition: StageDefinition[StageExecutionSummary],
    ) -> None:
        """Discard partial transaction state while retaining the owner snapshot."""
        StagePublisher(
            workspace,
            definition,
            rename_operation=self.rename_operation,
        ).abandon()

    def invalidate(
        self,
        workspace: SceneWorkspace,
        definition: StageDefinition[StageExecutionSummary],
    ) -> None:
        """Clear transaction residue and physically unpublish the owner."""
        publisher = StagePublisher(
            workspace,
            definition,
            rename_operation=self.rename_operation,
        )
        publisher.recover_interrupted_publication()
        workspace.invalidate_outputs(definition)


@dataclass(frozen=True, slots=True)
class ExternalAtomicPublication:
    """Delegate atomic owner mutation to a stage's public external command."""

    def preflight(
        self,
        workspace: SceneWorkspace,
        definition: StageDefinition[StageExecutionSummary],
    ) -> None:
        """Require only a contained fixed owner for the external authority."""
        workspace.owner_path(definition)

    def prepare(
        self,
        workspace: SceneWorkspace,
        definition: StageDefinition[StageExecutionSummary],
    ) -> Path:
        """Return the fixed owner the external command publishes atomically."""
        owner = workspace.owner_path(definition)
        owner.mkdir(parents=True, exist_ok=True)
        return owner

    def publish(
        self,
        workspace: SceneWorkspace,
        definition: StageDefinition[StageExecutionSummary],
    ) -> StagePublicationResult:
        """Confirm the external command produced its complete fixed inventory."""
        workspace.validate_required_outputs(definition)
        return StagePublicationResult(
            owner_path=workspace.owner_path(definition),
            replaced_existing=True,
        )

    def recover(
        self,
        workspace: SceneWorkspace,
        definition: StageDefinition[StageExecutionSummary],
    ) -> None:
        """Leave the external authority's old-or-new fixed owner untouched."""
        workspace.owner_path(definition)

    def abandon(
        self,
        workspace: SceneWorkspace,
        definition: StageDefinition[StageExecutionSummary],
    ) -> None:
        """Remove an externally failed attempt so it cannot masquerade as valid."""
        workspace.invalidate_outputs(definition)

    def invalidate(
        self,
        workspace: SceneWorkspace,
        definition: StageDefinition[StageExecutionSummary],
    ) -> None:
        """Physically unpublish the fixed external owner."""
        workspace.invalidate_outputs(definition)


@dataclass(frozen=True, slots=True)
class StagePublisher:
    """Operate one fixed transaction and owner pair as complete directories."""

    workspace: SceneWorkspace
    definition: StageDefinition[StageExecutionSummary]
    rename_operation: RenameOperation = field(
        default=lambda source, destination, flags: _renameat2(
            source,
            destination,
            flags=flags,
        ),
        repr=False,
        compare=False,
    )

    @property
    def owner(self) -> Path:
        """Return the one fixed canonical owner directory."""
        return self.workspace.owner_path(self.definition)

    @property
    def transaction(self) -> Path:
        """Return the fixed transaction root for this stage."""
        return self.workspace.stage_transaction_path(self.definition)

    @property
    def staging(self) -> Path:
        """Return the complete replacement snapshot outside the owner."""
        return self.workspace.staging_path(self.definition)

    @property
    def marker(self) -> Path:
        """Return the durable marker outside both exchanged directories."""
        return self.transaction / _TRANSACTION_FILE

    def preflight_atomic_exchange(self) -> None:
        """Verify Linux exchange support on the canonical workspace filesystem."""
        probe = self.workspace.exchange_probe_path
        if probe.is_symlink():
            raise ValueError("Atomic exchange probe root must not be a symlink.")
        if probe.exists():
            _discard_exact(probe)
        left = probe / "left"
        right = probe / "right"
        left.mkdir(parents=True, exist_ok=False)
        right.mkdir(parents=False, exist_ok=False)
        try:
            self.rename_operation(left, right, _RENAME_EXCHANGE)
        finally:
            if probe.exists() and not probe.is_symlink():
                shutil.rmtree(probe)
            _remove_empty_directory(self.workspace.transaction_root)

    def prepare(self) -> Path:
        """Recover residue, then create one fresh complete-snapshot directory."""
        self.recover_interrupted_publication()
        if self.transaction.exists() or self.transaction.is_symlink():
            _discard_exact(self.transaction)
        self.staging.mkdir(parents=True, exist_ok=False)
        return self.staging

    def validate_staging_inventory(self) -> None:
        """Require every output and reject undeclared top-level entries."""
        if self.staging.is_symlink() or not self.staging.is_dir():
            raise ValueError("Publication snapshot must be an ordinary directory.")
        missing = [
            str(path)
            for path in self.definition.required_outputs
            if not (self.staging / path).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Stage {self.definition.name.value} staging is missing outputs: {missing}"
            )
        declared = {path.parts[0] for path in self.definition.required_outputs}
        actual = {path.name for path in self.staging.iterdir()}
        unexpected = sorted(actual - declared)
        if unexpected:
            raise ValueError(
                f"Stage {self.definition.name.value} staging has unexpected "
                f"top-level outputs: {unexpected}"
            )

    def publish(self) -> StagePublicationResult:
        """Atomically install the first owner or exchange a complete replacement."""
        self.validate_staging_inventory()
        self.owner.parent.mkdir(parents=True, exist_ok=True)
        if self.owner.is_symlink():
            raise ValueError("Stage owner must not be a symlink.")
        replaced_existing = self.owner.exists()
        if replaced_existing and not self.owner.is_dir():
            raise ValueError("Stage owner must be an ordinary directory.")
        _write_transaction(
            self.marker,
            stage=self.definition.name,
            owner_relative_path=self.definition.owner_relative_path,
            replaced_existing=replaced_existing,
        )
        _fsync_directory(self.transaction)
        if replaced_existing:
            self.rename_operation(self.staging, self.owner, _RENAME_EXCHANGE)
        else:
            self.rename_operation(self.staging, self.owner, _RENAME_NOREPLACE)
        _fsync_directory(self.owner.parent)
        self._clear_transaction()
        return StagePublicationResult(
            owner_path=self.owner,
            replaced_existing=replaced_existing,
        )

    def recover_interrupted_publication(self) -> None:
        """Keep the complete current owner and discard the non-canonical snapshot."""
        if not self.transaction.exists() and not self.transaction.is_symlink():
            _remove_empty_directory(self.workspace.transaction_root)
            return
        if self.transaction.is_symlink() or not self.transaction.is_dir():
            raise ValueError("Publication transaction root must be an ordinary directory.")
        if not self.marker.exists() and not self.marker.is_symlink():
            self._clear_transaction()
            return
        replaced_existing = _read_transaction(
            self.marker,
            expected_stage=self.definition.name,
            expected_owner=self.definition.owner_relative_path,
        )
        owner_exists = self.owner.exists() and not self.owner.is_symlink()
        staging_exists = self.staging.exists() and not self.staging.is_symlink()
        if self.owner.is_symlink() or self.staging.is_symlink():
            raise ValueError("Publication recovery refuses symlinked snapshots.")
        if owner_exists and not self.owner.is_dir():
            raise ValueError("Publication owner is not an ordinary directory.")
        if staging_exists and not self.staging.is_dir():
            raise ValueError("Publication staging is not an ordinary directory.")
        if replaced_existing and not owner_exists:
            raise ValueError("Publication recovery lost the prior canonical owner.")
        if not replaced_existing and not owner_exists and not staging_exists:
            raise ValueError("Publication recovery lost both initial snapshots.")
        self._clear_transaction()

    def abandon(self) -> None:
        """Recover any exchange and remove all non-canonical transaction residue."""
        self.recover_interrupted_publication()

    def _clear_transaction(self) -> None:
        if self.transaction.is_symlink():
            raise ValueError("Refusing to remove a symlinked publication transaction.")
        if self.transaction.exists():
            shutil.rmtree(self.transaction)
        parent = self.transaction.parent
        _remove_empty_directory(parent)


def _renameat2(source: Path, destination: Path, *, flags: int) -> None:
    """Invoke Linux renameat2 directly; never substitute a sequential fallback."""
    if flags not in {_RENAME_NOREPLACE, _RENAME_EXCHANGE}:
        raise ValueError(f"Unsupported renameat2 flags: {flags}.")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise AtomicPublicationUnavailableError(
            errno.ENOSYS,
            "Linux renameat2 is unavailable; atomic publication is required.",
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        flags,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {
        errno.ENOSYS,
        errno.EXDEV,
        errno.EINVAL,
        errno.EOPNOTSUPP,
        errno.ENOTSUP,
    }:
        raise AtomicPublicationUnavailableError(
            error_number,
            "Workspace filesystem lacks required atomic directory rename authority.",
            str(source),
            str(destination),
        )
    raise OSError(error_number, os.strerror(error_number), str(source), str(destination))


def _write_transaction(
    path: Path,
    *,
    stage: object,
    owner_relative_path: Path,
    replaced_existing: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema": _TRANSACTION_SCHEMA,
                "stage": str(stage),
                "owner_relative_path": owner_relative_path.as_posix(),
                "replaced_existing": replaced_existing,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_transaction(
    path: Path,
    *,
    expected_stage: object,
    expected_owner: Path,
) -> bool:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("Publication transaction lacks its durable marker.")
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "schema",
        "stage",
        "owner_relative_path",
        "replaced_existing",
    }:
        raise ValueError("Publication transaction schema keys are invalid.")
    if raw["schema"] != _TRANSACTION_SCHEMA:
        raise ValueError("Publication transaction schema is unsupported.")
    if raw["stage"] != str(expected_stage):
        raise ValueError("Publication transaction belongs to another stage.")
    if raw["owner_relative_path"] != expected_owner.as_posix():
        raise ValueError("Publication transaction owner authority changed.")
    replaced_existing = raw["replaced_existing"]
    if not isinstance(replaced_existing, bool):
        raise TypeError("Publication transaction replacement flag must be boolean.")
    return replaced_existing


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _discard_exact(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _remove_empty_directory(path: Path) -> None:
    if path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
        path.rmdir()


__all__ = [
    "AtomicDirectoryPublication",
    "AtomicPublicationUnavailableError",
    "ExternalAtomicPublication",
    "StagePublisher",
]
