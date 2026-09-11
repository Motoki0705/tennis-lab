"""Tests for strict rclone artifact publishing."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.artifact_store import ArtifactStoreError, RcloneArtifactStore


class _RecordingRcloneStore(RcloneArtifactStore):
    def __init__(
        self,
        local_root: Path,
        *,
        remote: str,
        remote_root: str,
        config_path: Path,
    ) -> None:
        super().__init__(
            local_root,
            remote=remote,
            remote_root=remote_root,
            config_path=config_path,
        )
        self.commands: list[tuple[str, ...]] = []

    def _run(self, *arguments: str, timeout: int | None = None) -> None:
        del timeout
        self.commands.append(arguments)


def _store(tmp_path: Path) -> _RecordingRcloneStore:
    root = tmp_path / "output"
    root.mkdir()
    config = tmp_path / "rclone.conf"
    config.write_text("[drive]\ntype = drive\n", encoding="utf-8")
    config.chmod(0o600)
    return _RecordingRcloneStore(
        root,
        remote="drive",
        remote_root="team/runs/run-1/training",
        config_path=config,
    )


def test_publish_file_uses_staging_name_then_atomic_remote_move(tmp_path: Path) -> None:
    store = _store(tmp_path)
    checkpoint = store.local_root / "logs/checkpoints/last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    store.publish_file(checkpoint)

    copy, move = store.commands
    assert copy[0:2] == ("copyto", str(checkpoint))
    assert copy[2].startswith(
        "drive:team/runs/run-1/training/logs/checkpoints/last.ckpt.uploading-"
    )
    assert move == (
        "moveto",
        copy[2],
        "drive:team/runs/run-1/training/logs/checkpoints/last.ckpt",
    )


def test_sync_tree_copies_without_remote_deletion(tmp_path: Path) -> None:
    store = _store(tmp_path)

    store.sync_tree()

    assert store.commands == [
        ("copy", str(store.local_root), "drive:team/runs/run-1/training")
    ]


@pytest.mark.parametrize(
    "remote_root", ("/absolute", "parent/../escape", "remote:child", "a//b")
)
def test_remote_root_must_be_normalized_relative_path(
    tmp_path: Path, remote_root: str
) -> None:
    config = tmp_path / "rclone.conf"
    config.write_text("[drive]\n", encoding="utf-8")
    config.chmod(0o600)

    with pytest.raises(ArtifactStoreError, match="remote_root"):
        RcloneArtifactStore(
            tmp_path / "output",
            remote="drive",
            remote_root=remote_root,
            config_path=config,
        )


def test_rejects_group_readable_secret(tmp_path: Path) -> None:
    config = tmp_path / "rclone.conf"
    config.write_text("[drive]\n", encoding="utf-8")
    config.chmod(0o640)

    with pytest.raises(ArtifactStoreError, match="group/other"):
        RcloneArtifactStore(
            tmp_path / "output",
            remote="drive",
            remote_root="runs/run-1",
            config_path=config,
        )
