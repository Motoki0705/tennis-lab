"""Failure-recovery tests for fixed-path stage publication."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.synthetic_data_generation.pipeline import SceneWorkspace, StageName
from src.synthetic_data_generation.pipeline.publication import StagePublisher
from src.synthetic_data_generation.pipeline.registry import canonical_registry
from src.utils.configuration import PathResolver, RuntimePathRoots


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


def _payload(root: Path, name: str) -> str:
    path = root / name
    return (
        (path / "payload.txt").read_text(encoding="utf-8")
        if path.is_dir()
        else path.read_text(encoding="utf-8")
    )


def test_publish_replaces_complete_inventory_and_removes_transaction(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    spec = canonical_registry().spec(StageName.COURT_DATASET)
    publisher = StagePublisher(workspace, spec)
    publisher.owner.mkdir(parents=True)
    for name in ("dataset.json", "samples", "diagnostics"):
        _write_output(publisher.owner, name, "old")
    staging = publisher.prepare()
    for name in ("dataset.json", "samples", "diagnostics"):
        _write_output(staging, name, "new")

    publisher.publish()

    assert all(
        _payload(publisher.owner, name) == "new"
        for name in ("dataset.json", "samples", "diagnostics")
    )
    assert not (publisher.owner / ".publication-backup").exists()
    assert not staging.exists()


def test_prepare_rolls_back_sigterm_interrupted_publication(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    spec = canonical_registry().spec(StageName.COURT_DATASET)
    publisher = StagePublisher(workspace, spec)
    publisher.owner.mkdir(parents=True)
    names = ("dataset.json", "diagnostics", "samples")
    for name in names:
        _write_output(publisher.owner, name, "old")
    backup = publisher.owner / ".publication-backup"
    backup.mkdir()
    (backup / "transaction.json").write_text(
        json.dumps(
            {
                "schema": "stage_publication_transaction_v1",
                "declared": list(names),
                "existing": list(names),
            }
        ),
        encoding="utf-8",
    )
    (publisher.owner / "dataset.json").replace(backup / "dataset.json")
    _write_output(publisher.owner, "dataset.json", "partial-new")
    stale_staging = publisher.staging
    stale_staging.mkdir()
    (stale_staging / "partial.bin").write_bytes(b"partial")

    prepared = publisher.prepare()

    assert _payload(publisher.owner, "dataset.json") == "old"
    assert _payload(publisher.owner, "samples") == "old"
    assert _payload(publisher.owner, "diagnostics") == "old"
    assert not backup.exists()
    assert prepared.is_dir() and not any(prepared.iterdir())


def test_recovery_fails_closed_without_durable_transaction(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    spec = canonical_registry().spec(StageName.BLCS_DATASET)
    publisher = StagePublisher(workspace, spec)
    (publisher.owner / ".publication-backup").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="transaction marker"):
        publisher.prepare()
