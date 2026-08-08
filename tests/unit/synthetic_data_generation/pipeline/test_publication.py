"""Atomicity and recovery tests for whole-owner stage publication."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.synthetic_data_generation.pipeline import (
    CanonicalStageHandlers,
    SceneWorkspace,
    StageExecutionSummary,
    StageName,
)
from src.synthetic_data_generation.pipeline.contracts import StageExecutionContext
from src.synthetic_data_generation.pipeline.publication import (
    AtomicPublicationUnavailableError,
    StagePublisher,
    _renameat2,
)
from src.synthetic_data_generation.pipeline.registry import canonical_registry
from src.utils.configuration import PathResolver, RuntimePathRoots

_RENAME_EXCHANGE = 2


@dataclass(frozen=True)
class _Handler:
    def preflight(self, context: StageExecutionContext) -> None:
        pass

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        return StageExecutionSummary({})

    def validate(self, context: StageExecutionContext) -> None:
        pass


def _registry():
    handlers = [_Handler() for _ in StageName]
    return canonical_registry(CanonicalStageHandlers(*handlers))


def _workspace(tmp_path: Path) -> SceneWorkspace:
    roots = RuntimePathRoots(
        project_root=tmp_path.resolve(),
        data_root=(tmp_path / "data").resolve(),
        checkpoint_root=(tmp_path / "ckpt").resolve(),
        artifact_root=(tmp_path / "artifacts").resolve(),
        output_root=(tmp_path / "outputs").resolve(),
        cache_root=(tmp_path / "cache").resolve(),
        external_asset_root=(tmp_path / "external").resolve(),
    )
    return SceneWorkspace.resolve(PathResolver(roots), "scene-a")


def _write_output(root: Path, name: str, payload: str) -> None:
    path = root / name
    if name in {"samples", "diagnostics"}:
        path.mkdir(parents=True, exist_ok=True)
        (path / "payload.txt").write_text(payload, encoding="utf-8")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")


def _write_inventory(root: Path, payload: str) -> None:
    for name in ("dataset.json", "samples", "diagnostics"):
        _write_output(root, name, payload)


def _payload(root: Path, name: str) -> str:
    path = root / name
    return (
        (path / "payload.txt").read_text(encoding="utf-8")
        if path.is_dir()
        else path.read_text(encoding="utf-8")
    )


def _snapshot_from_directory_fd(owner: Path) -> tuple[str, str, str]:
    descriptor = os.open(owner, os.O_RDONLY | os.O_DIRECTORY)
    try:
        values: list[str] = []
        for name in ("dataset.json", "samples/payload.txt", "diagnostics/payload.txt"):
            file_descriptor = os.open(name, os.O_RDONLY, dir_fd=descriptor)
            try:
                values.append(os.read(file_descriptor, 64).decode())
            finally:
                os.close(file_descriptor)
        return values[0], values[1], values[2]
    finally:
        os.close(descriptor)


def test_publish_uses_external_transaction_root_and_fixed_owner_reruns(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    definition = _registry().definition(StageName.COURT_DATASET)
    publisher = StagePublisher(workspace, definition)
    first_staging = publisher.prepare()
    _write_inventory(first_staging, "first")

    first = publisher.publish()

    assert first.owner_path == publisher.owner
    assert not first.replaced_existing
    assert first_staging == workspace.transaction_root / "court_dataset/snapshot"
    assert not first_staging.is_relative_to(publisher.owner)
    assert _snapshot_from_directory_fd(publisher.owner) == ("first",) * 3

    second_staging = publisher.prepare()
    _write_inventory(second_staging, "second")
    second = publisher.publish()

    assert second.replaced_existing
    assert publisher.owner == workspace.root / "datasets/court"
    assert _snapshot_from_directory_fd(publisher.owner) == ("second",) * 3
    assert not workspace.transaction_root.exists()
    assert not any("run" in part or "fingerprint" in part for part in publisher.owner.parts)


def test_exchange_observers_see_only_one_complete_owner_snapshot(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    definition = _registry().definition(StageName.COURT_DATASET)
    initial = StagePublisher(workspace, definition)
    staging = initial.prepare()
    _write_inventory(staging, "old")
    initial.publish()

    exchange_started = threading.Event()
    allow_exchange = threading.Event()

    def delayed_exchange(source: Path, destination: Path, flags: int) -> None:
        if flags == _RENAME_EXCHANGE:
            exchange_started.set()
            assert allow_exchange.wait(timeout=5.0)
        _renameat2(source, destination, flags=flags)

    publisher = StagePublisher(
        workspace,
        definition,
        rename_operation=delayed_exchange,
    )
    staging = publisher.prepare()
    _write_inventory(staging, "new")
    errors: list[BaseException] = []

    def publish() -> None:
        try:
            publisher.publish()
        except BaseException as error:  # pragma: no cover - diagnostic capture
            errors.append(error)

    thread = threading.Thread(target=publish)
    thread.start()
    assert exchange_started.wait(timeout=5.0)
    observations = {_snapshot_from_directory_fd(publisher.owner)}
    allow_exchange.set()
    while thread.is_alive():
        observations.add(_snapshot_from_directory_fd(publisher.owner))
    thread.join(timeout=5.0)
    observations.add(_snapshot_from_directory_fd(publisher.owner))

    assert not errors
    assert observations <= {("old",) * 3, ("new",) * 3}
    assert ("old",) * 3 in observations
    assert ("new",) * 3 in observations


def test_publish_removes_stale_entries_with_the_old_snapshot(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    definition = _registry().definition(StageName.COURT_DATASET)
    publisher = StagePublisher(workspace, definition)
    publisher.owner.mkdir(parents=True)
    _write_inventory(publisher.owner, "old")
    stale = publisher.owner / "legacy-chunks"
    stale.mkdir()
    (stale / "stale.json").write_text("stale", encoding="utf-8")
    staging = publisher.prepare()
    _write_inventory(staging, "current")

    publisher.publish()

    assert not stale.exists()
    assert _snapshot_from_directory_fd(publisher.owner) == ("current",) * 3


def test_recovery_after_interrupted_exchange_keeps_a_complete_new_owner(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    definition = _registry().definition(StageName.COURT_DATASET)
    initial = StagePublisher(workspace, definition)
    staging = initial.prepare()
    _write_inventory(staging, "old")
    initial.publish()

    def exchange_then_interrupt(source: Path, destination: Path, flags: int) -> None:
        _renameat2(source, destination, flags=flags)
        if flags == _RENAME_EXCHANGE:
            raise KeyboardInterrupt("simulated process interruption")

    interrupted = StagePublisher(
        workspace,
        definition,
        rename_operation=exchange_then_interrupt,
    )
    staging = interrupted.prepare()
    _write_inventory(staging, "new")

    with pytest.raises(KeyboardInterrupt, match="simulated"):
        interrupted.publish()

    assert _snapshot_from_directory_fd(interrupted.owner) == ("new",) * 3
    assert interrupted.transaction.exists()

    StagePublisher(workspace, definition).recover_interrupted_publication()

    assert _snapshot_from_directory_fd(interrupted.owner) == ("new",) * 3
    assert not workspace.transaction_root.exists()


def test_publish_fails_closed_when_atomic_exchange_authority_is_unavailable(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    definition = _registry().definition(StageName.COURT_DATASET)
    initial = StagePublisher(workspace, definition)
    staging = initial.prepare()
    _write_inventory(staging, "old")
    initial.publish()

    flags_seen: list[int] = []

    def unavailable(source: Path, destination: Path, flags: int) -> None:
        flags_seen.append(flags)
        raise AtomicPublicationUnavailableError("exchange unavailable")

    publisher = StagePublisher(
        workspace,
        definition,
        rename_operation=unavailable,
    )
    staging = publisher.prepare()
    _write_inventory(staging, "new")

    with pytest.raises(AtomicPublicationUnavailableError, match="unavailable"):
        publisher.publish()
    publisher.abandon()

    assert flags_seen == [_RENAME_EXCHANGE]
    assert _snapshot_from_directory_fd(publisher.owner) == ("old",) * 3
    assert not workspace.transaction_root.exists()


def test_recovery_rejects_transaction_that_lost_prior_owner(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    definition = _registry().definition(StageName.COURT_DATASET)
    publisher = StagePublisher(workspace, definition)
    publisher.owner.mkdir(parents=True)
    _write_inventory(publisher.owner, "old")
    staging = publisher.prepare()
    _write_inventory(staging, "new")

    def fail_before_exchange(source: Path, destination: Path, flags: int) -> None:
        raise KeyboardInterrupt

    interrupted = StagePublisher(
        workspace,
        definition,
        rename_operation=fail_before_exchange,
    )
    with pytest.raises(KeyboardInterrupt):
        interrupted.publish()
    for child in publisher.owner.iterdir():
        if child.is_dir():
            for nested in child.iterdir():
                nested.unlink()
            child.rmdir()
        else:
            child.unlink()
    publisher.owner.rmdir()

    with pytest.raises(ValueError, match="lost the prior canonical owner"):
        StagePublisher(workspace, definition).recover_interrupted_publication()
