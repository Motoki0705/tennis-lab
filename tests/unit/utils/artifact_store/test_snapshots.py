"""A verified snapshot is a coherent generation, not a successful copy command."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from src.utils.artifact_store import ArtifactStoreError, RcloneSnapshotStore


class FilesystemStore(RcloneSnapshotStore):
    def __init__(self, local_root: Path, remote_root: Path, config: Path) -> None:
        super().__init__(local_root, remote="drive", remote_root="runs/proof", config_path=config)
        self.storage = remote_root
        self.failure: str | None = None
        self.commands: list[tuple[str, ...]] = []

    def path(self, value: str) -> Path:
        return self.storage / value.removeprefix("drive:") if value.startswith("drive:") else Path(value)

    def _output(self, *args: str, timeout: int | None = None) -> str:
        del timeout
        self.commands.append(args)
        command = args[0]
        if self.failure == "communication" and command == "copy":
            raise ArtifactStoreError("Injected communication failure")
        if command == "mkdir":
            self.path(args[1]).mkdir(parents=True, exist_ok=True)
        elif command == "lsjson":
            root = self.path(args[1])
            paths = root.rglob("*") if "--recursive" in args else root.iterdir()
            entries = [dict(Path=p.relative_to(root).as_posix(), Name=p.name, IsDir=p.is_dir(),
                            Size=p.stat().st_size if p.is_file() else -1,
                            Hashes={"MD5": hashlib.md5(p.read_bytes()).hexdigest()} if p.is_file() else {}, ID=p.name)
                       for p in sorted(paths)]
            if self.failure == "hash_unavailable":
                for entry in entries:
                    entry["Hashes"] = {}
            if self.failure == "duplicate" and "--recursive" in args and entries:
                entries.append(entries[0])
            return json.dumps(entries)
        elif command in {"copy", "copyto"}:
            source, destination = self.path(args[1]), self.path(args[2])
            if self.failure == "marker" and destination.name == "_SNAPSHOT.json":
                raise ArtifactStoreError("Injected marker publication failure")
            if command == "copyto":
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            else:
                for item in source.rglob("*"):
                    if item.is_file() and not ("--exclude" in args and item.name == "_SNAPSHOT.json"):
                        target = destination / item.relative_to(source)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(item, target)
                if self.failure == "corrupt":
                    target = next(p for p in destination.rglob("*") if p.is_file())
                    target.write_bytes(b"!" + target.read_bytes()[1:])
                if self.failure == "local_mutation":
                    (self.local_root / "late.log").write_text("Written during publication")
        else:
            raise AssertionError(args)
        return ""


@pytest.fixture
def store(tmp_path: Path) -> FilesystemStore:
    config = tmp_path / "rclone.conf"
    config.write_text("[drive]\ntype = drive\n")
    config.chmod(0o600)
    root = tmp_path / "run"
    root.mkdir()
    (root / "checkpoint.pt").write_bytes(b"complete checkpoint")
    (root / "latest.json").write_text('{"checkpoint":"checkpoint.pt"}')
    (root / "train.jsonl").write_text('{"update":1}\n')
    return FilesystemStore(root, tmp_path / "remote", config)


def test_restore_verifies_and_atomically_exposes_original_bytes(store: FilesystemStore) -> None:
    expected = {p.name: p.read_bytes() for p in store.local_root.iterdir()}
    receipt = store.publish("update-1")
    assert receipt["status"] == "verified" and receipt["files"] == 3 and receipt["marker_id"]
    original_root = store.local_root
    store.local_root = original_root.with_name("restored")
    assert store.restore("update-1", manifest_sha256=receipt["manifest_sha256"])["status"] == "verified"
    assert {p.name: p.read_bytes() for p in store.local_root.iterdir()} == expected
    assert not any(command[0] in {"deletefile", "sync", "moveto"} for command in store.commands)


@pytest.mark.parametrize("failure", ["communication", "corrupt", "hash_unavailable", "duplicate", "marker", "local_mutation"])
def test_failed_publication_has_no_verified_receipt_or_completion_marker(store: FilesystemStore, failure: str) -> None:
    store.failure = failure
    with pytest.raises(ArtifactStoreError):
        store.publish("failed")
    assert not (store.storage / "runs/proof/failed/_SNAPSHOT.json").exists()
    assert (store.local_root / "checkpoint.pt").read_bytes() == b"complete checkpoint"


def test_existing_or_partial_generation_cannot_be_overwritten(store: FilesystemStore) -> None:
    store.publish("kept")
    with pytest.raises(ArtifactStoreError, match="already exists"):
        store.publish("kept")
    partial = store.storage / "runs/proof/partial"
    partial.mkdir()
    with pytest.raises(ArtifactStoreError, match="already exists"):
        store.publish("partial")


@pytest.mark.parametrize("damage", ["marker", "payload", "extra", "missing", "local_download"])
def test_restore_rejects_unverified_or_damaged_generations(store: FilesystemStore, damage: str) -> None:
    receipt = store.publish("original")
    remote = store.storage / "runs/proof/original"
    if damage == "marker":
        (remote / "_SNAPSHOT.json").write_text("{}")
    elif damage == "payload":
        (remote / "checkpoint.pt").write_bytes(b"corrupt checkpoint!")
    elif damage == "extra":
        (remote / "extra").write_text("extra")
    elif damage == "missing":
        (remote / "latest.json").unlink()
    else:
        store.failure = "corrupt"
    store.local_root = store.local_root.with_name("restored")
    with pytest.raises(ArtifactStoreError):
        store.restore("original", manifest_sha256=receipt["manifest_sha256"])
    assert not store.local_root.exists()


def test_restore_never_overwrites_an_existing_local_run(store: FilesystemStore) -> None:
    receipt = store.publish("original")
    with pytest.raises(ArtifactStoreError, match="new local"):
        store.restore("original", manifest_sha256=receipt["manifest_sha256"])


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/b", ".", ".."])
def test_generation_cannot_escape_its_remote_root(store: FilesystemStore, name: str) -> None:
    with pytest.raises(ArtifactStoreError):
        store.publish(name)


def test_restoration_rejects_traversal_even_with_a_matching_marker_hash(store: FilesystemStore) -> None:
    store.publish("unsafe")
    marker = store.storage / "runs/proof/unsafe/_SNAPSHOT.json"
    raw: dict[str, Any] = json.loads(marker.read_text())
    raw["files"]["../escape"] = raw["files"].pop("latest.json")
    marker.write_text(json.dumps(raw))
    store.local_root = store.local_root.with_name("restore")
    with pytest.raises(ArtifactStoreError, match="relative paths"):
        store.restore("unsafe", manifest_sha256=hashlib.sha256(marker.read_bytes()).hexdigest())


def test_snapshot_refuses_local_symlinks(store: FilesystemStore) -> None:
    (store.local_root / "alias").symlink_to(store.local_root / "latest.json")
    with pytest.raises(ArtifactStoreError, match="symlink"):
        store.publish("unsafe")


def test_snapshot_restoration_preserves_training_budget_rng_and_samples(store: FilesystemStore, tmp_path: Path) -> None:
    import torch

    from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
    from src.tasks.ball_detection.training.temporal_heatmap.runner import (
        train_temporal_heatmap,
    )
    from tests.support.tasks.ball_detection.temporal_heatmap import (
        prepare_fixture,
        small_model,
        update_budget,
    )

    prepared = prepare_fixture(tmp_path / "inputs")
    options = dict(architecture="hybrid", phase="pretrain", manifest=prepared / "train_T32.json",
                   evaluation_manifest=prepared / "evaluation_T32.json", model_config=small_model("hybrid"),
                   budget=update_budget(), runtime=CoordinateRuntime(cpu_threads=1), device=torch.device("cpu"),
                   augmentation=None, distance_px=20.)
    baseline = tmp_path / "baseline"
    train_temporal_heatmap(output=baseline, **options)
    root = tmp_path / "interrupted"
    store.local_root = root
    receipts: list[dict[str, Any]] = []

    def persist(output: Path, stage: str) -> None:
        assert output == root
        receipts.append(store.publish(f"{stage}-{len(receipts)}"))

    train_temporal_heatmap(output=root, stop_after_update=1, persist=persist, **options)
    root.rename(tmp_path / "discarded-vm")
    saved = receipts[-1]
    store.restore(saved["generation"], manifest_sha256=saved["manifest_sha256"])
    result = train_temporal_heatmap(output=root, resume=root / "checkpoint-update-00000001.pt", persist=persist, **options)
    assert result["complete"] and result["updates_completed"] == result["sample_position"] == 3
    before = [json.loads(line) for line in (baseline / "train.jsonl").read_text().splitlines()]
    after = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
    assert [r["pack_ids"] for r in before] == [r["pack_ids"] for r in after]
    assert [r["loss"] for r in before] == pytest.approx([r["loss"] for r in after], abs=1e-8)
    assert [r["update"] for r in after] == [1, 2, 3]
