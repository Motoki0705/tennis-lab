"""Tests for strict rclone artifact publishing."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.artifact_store import ArtifactStoreError, RcloneArtifactStore
from src.utils.artifact_store import rclone as rclone_module

_SOURCE_MISSING = "Source doesn't exist or is a directory and destination is a file"


def _fail_moves(
    store: _RecordingRcloneStore,
    monkeypatch: pytest.MonkeyPatch,
    errors: list[str],
) -> list[float]:
    record = store._run
    pending = iter(errors)
    waits: list[float] = []

    def run(*arguments: str, timeout: int | None = None) -> None:
        record(*arguments, timeout=timeout)
        if arguments[0] == "moveto":
            error = next(pending, None)
            if error is not None:
                raise ArtifactStoreError(error)

    monkeypatch.setattr(store, "_run", run)
    monkeypatch.setattr(rclone_module, "sleep", waits.append)
    return waits


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


def test_publish_file_retries_missing_source_then_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = _store(tmp_path)
    checkpoint = store.local_root / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    waits = _fail_moves(
        store,
        monkeypatch,
        [
            f"rclone moveto failed with exit code 1: {_SOURCE_MISSING}",
            f"rclone moveto failed with exit code 1: 2026/10/31 02:50:12 {_SOURCE_MISSING}",
        ],
    )

    store.publish_file(checkpoint)

    copy, *moves = store.commands
    assert copy[:2] == ("copyto", str(checkpoint))
    assert (
        moves == [("moveto", copy[2], "drive:team/runs/run-1/training/last.ckpt")] * 3
    )
    assert waits == [5.0, 5.0]
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2
    assert "retry 1/12 in 5s" in lines[0]
    assert "retry 2/12 in 5s" in lines[1]
    assert all(copy[2] in line for line in lines)


def test_publish_file_missing_source_exhausts_budget_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = _store(tmp_path)
    checkpoint = store.local_root / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    errors = [
        f"rclone moveto failed with exit code 1: 2026/10/31 02:50:{second:02} {_SOURCE_MISSING}"
        for second in range(13)
    ]
    waits = _fail_moves(store, monkeypatch, errors)

    with pytest.raises(ArtifactStoreError, match="13 attempts") as failure:
        store.publish_file(checkpoint)

    assert errors[-1] in str(failure.value)
    assert str(failure.value.__cause__) == errors[-1]
    copy, *moves, cleanup = store.commands
    assert (
        moves == [("moveto", copy[2], "drive:team/runs/run-1/training/last.ckpt")] * 13
    )
    assert cleanup == ("deletefile", copy[2])
    assert waits == [5.0] * 12
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 12
    assert "retry 12/12 in 5s" in lines[-1]


@pytest.mark.parametrize(
    "error",
    [
        "rclone moveto failed with exit code 1: oauth2: cannot fetch token",
        "rclone moveto failed with exit code 1: googleapi: Error 403: storageQuotaExceeded",
        "rclone moveto failed with exit code 1: dial tcp: i/o timeout",
        f"rclone moveto failed with exit code 1: {_SOURCE_MISSING}\nnetwork error",
        f"rclone moveto failed with exit code 3: {_SOURCE_MISSING}",
        "rclone command could not complete: moveto",
    ],
    ids=["authentication", "quota", "network", "mixed-errors", "exit-code", "timeout"],
)
def test_publish_file_other_move_errors_fail_immediately_and_clean_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: str,
) -> None:
    store = _store(tmp_path)
    checkpoint = store.local_root / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    waits = _fail_moves(store, monkeypatch, [error])

    with pytest.raises(ArtifactStoreError) as failure:
        store.publish_file(checkpoint)

    assert str(failure.value) == error
    copy, move, cleanup = store.commands
    assert move == ("moveto", copy[2], "drive:team/runs/run-1/training/last.ckpt")
    assert cleanup == ("deletefile", copy[2])
    assert waits == []
    assert capsys.readouterr().err == ""


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
