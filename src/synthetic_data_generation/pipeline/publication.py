"""Stage-local staging and atomic fixed-path publication."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.synthetic_data_generation.pipeline.contracts import StageSpec
from src.synthetic_data_generation.pipeline.workspace import SceneWorkspace

_TRANSACTION_SCHEMA = "stage_publication_transaction_v1"
_TRANSACTION_FILE = "transaction.json"


@dataclass(frozen=True, slots=True)
class StagePublisher:
    """Publish validated outputs from one fixed stage-local staging directory."""

    workspace: SceneWorkspace
    spec: StageSpec

    @property
    def owner(self) -> Path:
        """Return the fixed owner directory."""
        return self.workspace.owner_path(self.spec)

    @property
    def staging(self) -> Path:
        """Return the fixed staging directory for the current attempt."""
        return self.workspace.staging_path(self.spec)

    def prepare(self) -> Path:
        """Clear stale attempt-local staging and create a fresh directory."""
        self.recover_interrupted_publication()
        if self.staging.exists():
            shutil.rmtree(self.staging)
        self.staging.mkdir(parents=True, exist_ok=False)
        return self.staging

    def validate_staging_inventory(self) -> None:
        """Require every declared output and reject undeclared top-level entries."""
        missing = [str(path) for path in self.spec.required_outputs if not (self.staging / path).exists()]
        if missing:
            raise FileNotFoundError(
                f"Stage {self.spec.name.value} staging is missing outputs: {missing}"
            )
        declared = {path.parts[0] for path in self.spec.required_outputs}
        actual = {path.name for path in self.staging.iterdir()}
        unexpected = sorted(actual - declared)
        if unexpected:
            raise ValueError(
                f"Stage {self.spec.name.value} staging has unexpected top-level outputs: {unexpected}"
            )

    def publish(self) -> None:
        """Atomically replace every declared top-level output after validation."""
        self.validate_staging_inventory()
        self.owner.mkdir(parents=True, exist_ok=True)
        declared = sorted({path.parts[0] for path in self.spec.required_outputs})
        backup_root = self.owner / ".publication-backup"
        if backup_root.exists() or backup_root.is_symlink():
            raise ValueError(
                "Interrupted publication must be recovered before a new publish."
            )
        existing = [
            name
            for name in declared
            if (self.owner / name).exists() or (self.owner / name).is_symlink()
        ]
        backup_root.mkdir(parents=False, exist_ok=False)
        _write_transaction(
            backup_root / _TRANSACTION_FILE,
            declared=declared,
            existing=existing,
        )
        try:
            for name in declared:
                destination = self.owner / name
                if destination.exists() or destination.is_symlink():
                    destination.replace(backup_root / name)
            for name in declared:
                (self.staging / name).replace(self.owner / name)
        except BaseException:
            try:
                self.recover_interrupted_publication()
            except BaseException as recovery_error:
                raise RuntimeError(
                    "Stage publication failed and its prior output could not be restored."
                ) from recovery_error
            raise
        else:
            shutil.rmtree(backup_root)
        finally:
            if self.staging.exists():
                shutil.rmtree(self.staging)

    def recover_interrupted_publication(self) -> None:
        """Restore the last complete fixed output set from a durable transaction."""
        backup_root = self.owner / ".publication-backup"
        if not backup_root.exists() and not backup_root.is_symlink():
            return
        if backup_root.is_symlink() or not backup_root.is_dir():
            raise ValueError("Publication backup must be an ordinary directory.")
        declared, existing = _read_transaction(
            backup_root / _TRANSACTION_FILE,
            expected_declared={
                path.parts[0] for path in self.spec.required_outputs
            },
        )
        actual = {path.name for path in backup_root.iterdir()}
        unexpected = actual - existing - {_TRANSACTION_FILE}
        if unexpected:
            raise ValueError(
                f"Publication backup contains unexpected entries: {sorted(unexpected)}."
            )
        for name in declared:
            destination = self.owner / name
            saved = backup_root / name
            if name in existing:
                if saved.exists() or saved.is_symlink():
                    _discard_exact(destination)
                    saved.replace(destination)
                elif not destination.exists() and not destination.is_symlink():
                    raise ValueError(
                        f"Publication recovery lost prior output {name!r}."
                    )
            else:
                _discard_exact(destination)
        shutil.rmtree(backup_root)

    def abandon(self) -> None:
        """Delete partial staging after a failed attempt."""
        if self.staging.exists():
            shutil.rmtree(self.staging)


def _write_transaction(
    path: Path,
    *,
    declared: list[str],
    existing: list[str],
) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema": _TRANSACTION_SCHEMA,
                "declared": declared,
                "existing": existing,
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
    expected_declared: set[str],
) -> tuple[tuple[str, ...], set[str]]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("Publication backup lacks its transaction marker.")
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"schema", "declared", "existing"}:
        raise ValueError("Publication transaction schema keys are invalid.")
    if raw["schema"] != _TRANSACTION_SCHEMA:
        raise ValueError("Publication transaction schema is unsupported.")
    declared = _name_sequence(raw["declared"], name="declared")
    existing = set(_name_sequence(raw["existing"], name="existing"))
    if set(declared) != expected_declared:
        raise ValueError("Publication transaction declared outputs changed.")
    if not existing.issubset(expected_declared):
        raise ValueError("Publication transaction existing outputs are invalid.")
    return declared, existing


def _name_sequence(value: object, *, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or any(
            not isinstance(item, str)
            or not item
            or "/" in item
            or "\\" in item
            or item in {".", ".."}
            for item in value
        )
        or len(value) != len(set(value))
    ):
        raise ValueError(f"Publication transaction {name} is invalid.")
    return tuple(value)


def _discard_exact(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


__all__ = ["StagePublisher"]
