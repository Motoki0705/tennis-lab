"""Tests for runner-owned artifact persistence integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from src.tasks.base.training.artifact_store import build_artifact_store
from src.tasks.base.training.artifact_store.checkpoint_io import PublishingCheckpointIO
from src.tasks.base.training.runner import BaseTrainingRunner
from src.utils.artifact_store import (
    ArtifactStore,
    LocalArtifactStore,
    RcloneArtifactStore,
)
from src.utils.configuration import SemanticConfigurationError


class _RecordingStore(ArtifactStore):
    def __init__(self, local_root: Path) -> None:
        super().__init__(local_root)
        self.published: list[Path] = []
        self.removed: list[Path] = []

    @property
    def enabled(self) -> bool:
        return True

    def publish_file(self, path: Path) -> None:
        self.relative_path(path)
        self.published.append(path)

    def fetch_file(self, path: Path) -> None:
        self.relative_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fetched")

    def remove_file(self, path: Path) -> None:
        self.relative_path(path)
        self.removed.append(path)

    def sync_tree(self) -> None:
        return None


def test_local_mode_builds_explicit_noop_backend(make_training_config: Any) -> None:
    config = OmegaConf.create(make_training_config())
    runtime = BaseTrainingRunner().validate_runtime_config(config)

    store = build_artifact_store(
        runtime.run.artifact_store,
        local_root=runtime.run.output_dir,
    )

    assert isinstance(store, LocalArtifactStore)
    assert store.enabled is False


@pytest.mark.parametrize("mode", ["local", "rclone"])
def test_colab_declared_output_rejects_local_or_wrong_remote(
    make_training_config: Any,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    monkeypatch.setenv(
        "TENNIS_COLAB_ARTIFACT_ROOTS", json.dumps(["drive:project/outputs/declared"])
    )
    overrides: dict[str, Any] = {}
    if mode == "rclone":
        overrides = {
            "run": {
                "artifact_store": {
                    "mode": "rclone",
                    "remote": "drive",
                    "remote_root": "project/outputs/different",
                    "sync_interval_seconds": 60,
                }
            }
        }
    runtime = BaseTrainingRunner().validate_runtime_config(
        OmegaConf.create(make_training_config(**overrides))
    )
    with pytest.raises(RuntimeError, match="declared Drive root"):
        build_artifact_store(
            runtime.run.artifact_store, local_root=runtime.run.output_dir
        )


def test_colab_declared_output_accepts_matching_rclone_store(
    make_training_config: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "TENNIS_COLAB_ARTIFACT_ROOTS", json.dumps(["drive:project/outputs/declared"])
    )
    secret = tmp_path / "rclone.conf"
    secret.write_text("[drive]\n")
    secret.chmod(0o600)
    monkeypatch.setenv("RCLONE_CONFIG", str(secret))
    config = OmegaConf.create(
        make_training_config(
            run={
                "artifact_store": {
                    "mode": "rclone",
                    "remote": "drive",
                    "remote_root": "project/outputs/declared",
                    "sync_interval_seconds": 60,
                }
            }
        )
    )
    runtime = BaseTrainingRunner().validate_runtime_config(config)
    store = build_artifact_store(
        runtime.run.artifact_store, local_root=runtime.run.output_dir
    )
    assert isinstance(store, RcloneArtifactStore)
    assert store.remote_uri == "drive:project/outputs/declared"


def test_rclone_mode_requires_explicit_secret_environment(
    make_training_config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = OmegaConf.create(
        make_training_config(
            run={
                "artifact_store": {
                    "mode": "rclone",
                    "remote": "drive",
                    "remote_root": "runs/run-1/training",
                    "sync_interval_seconds": 60,
                }
            }
        )
    )
    runtime = BaseTrainingRunner().validate_runtime_config(config)
    monkeypatch.delenv("RCLONE_CONFIG", raising=False)

    with pytest.raises(RuntimeError, match="RCLONE_CONFIG"):
        build_artifact_store(
            runtime.run.artifact_store,
            local_root=runtime.run.output_dir,
        )


def test_rclone_mode_builds_backend_from_environment(
    make_training_config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = tmp_path / "rclone.conf"
    secret.write_text("[drive]\n", encoding="utf-8")
    secret.chmod(0o600)
    monkeypatch.setenv("RCLONE_CONFIG", str(secret))
    config = OmegaConf.create(
        make_training_config(
            run={
                "artifact_store": {
                    "mode": "rclone",
                    "remote": "drive",
                    "remote_root": "runs/run-1/training",
                    "sync_interval_seconds": 60,
                }
            }
        )
    )
    runtime = BaseTrainingRunner().validate_runtime_config(config)

    store = build_artifact_store(
        runtime.run.artifact_store,
        local_root=runtime.run.output_dir,
    )

    assert isinstance(store, RcloneArtifactStore)
    assert store.remote_uri == "drive:runs/run-1/training"


def test_local_mode_rejects_remote_fields(make_training_config: Any) -> None:
    config = OmegaConf.create(
        make_training_config(
            run={
                "artifact_store": {
                    "mode": "local",
                    "remote": "drive",
                    "remote_root": None,
                    "sync_interval_seconds": None,
                }
            }
        )
    )

    with pytest.raises(SemanticConfigurationError, match="local artifact_store"):
        BaseTrainingRunner().validate_runtime_config(config)


def test_checkpoint_io_publishes_after_local_serialization(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    store = _RecordingStore(root)
    checkpoint_io = PublishingCheckpointIO(store)
    target = root / "checkpoints/model.ckpt"

    checkpoint_io.save_checkpoint({"epoch": 2}, target)

    assert target.is_file()
    assert store.published == [target]
    loaded = checkpoint_io.load_checkpoint(target, weights_only=False)
    assert loaded["epoch"] == 2


def test_checkpoint_io_removes_local_and_remote_file(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    store = _RecordingStore(root)
    checkpoint_io = PublishingCheckpointIO(store)
    target = root / "checkpoints/model.ckpt"
    checkpoint_io.save_checkpoint({"epoch": 1}, target)

    checkpoint_io.remove_checkpoint(target)

    assert not target.exists()
    assert store.removed == [target]
