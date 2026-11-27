"""Exercise the public installer using pinned synthetic bytes and an in-memory response."""

from __future__ import annotations

import hashlib
import sys
from io import BytesIO
from pathlib import Path

import pytest

from src.submodules import people_model_assets as assets
from src.submodules.scripts.install_people_model_checkpoint import main


@pytest.fixture
def checkpoint(monkeypatch: pytest.MonkeyPatch) -> assets.PeopleModelCheckpoint:
    checkpoint = assets.PeopleModelCheckpoint(
        "example/model.pth",
        "https://example.test/model.pth",
        hashlib.sha256(b"official bytes").hexdigest(),
    )
    monkeypatch.setitem(assets.PEOPLE_MODEL_CHECKPOINTS, "rfdetr-l", checkpoint)
    return checkpoint


def _run_cli(monkeypatch: pytest.MonkeyPatch, root: Path, *flags: str) -> None:
    monkeypatch.setattr(
        sys, "argv", ["install_people_model_checkpoint", "--model", "rfdetr-l",
                      "--checkpoint-root", str(root), *flags],
    )
    main()


def test_cli_reports_the_source_and_only_downloads_missing_files(
    checkpoint: assets.PeopleModelCheckpoint, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[str] = []

    def download(url: str, *, timeout: int) -> BytesIO:
        requests.append(url)
        return BytesIO(b"official bytes")

    monkeypatch.setattr(assets, "urlopen", download)
    root = tmp_path / "checkpoints"
    path = root / "example/model.pth"
    _run_cli(monkeypatch, root, "--list")
    listing = capsys.readouterr().out
    assert checkpoint.source_url in listing
    assert str(path) in listing
    assert checkpoint.sha256 in listing
    assert not root.exists()
    with pytest.raises(FileNotFoundError, match="Required checkpoint file is missing"):
        _run_cli(monkeypatch, root, "--verify-only")
    assert requests == []
    _run_cli(monkeypatch, root)
    _run_cli(monkeypatch, root, "--verify-only")
    _run_cli(monkeypatch, root)
    assert requests == [checkpoint.source_url]
    assert path.read_bytes() == b"official bytes"


def test_failed_verification_never_publishes_or_replaces_bytes(
    checkpoint: assets.PeopleModelCheckpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def download(url: str, *, timeout: int) -> BytesIO:
        return BytesIO(b"truncated")

    path = tmp_path / "model.pth"
    monkeypatch.setattr(assets, "urlopen", download)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        assets.install_checkpoint(checkpoint, path)
    assert not list(tmp_path.iterdir())
    path.write_bytes(b"another model")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        assets.install_checkpoint(checkpoint, path)
    assert path.read_bytes() == b"another model"


@pytest.mark.parametrize("content", [b"official bytes", b"another model"])
def test_install_verifies_a_concurrent_destination(
    checkpoint: assets.PeopleModelCheckpoint, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, content: bytes,
) -> None:
    def download(url: str, *, timeout: int) -> BytesIO:
        return BytesIO(b"official bytes")

    def racing_link(source: Path, destination: Path) -> None:
        destination.write_bytes(content)
        raise FileExistsError(destination)

    path = tmp_path / "model.pth"
    monkeypatch.setattr(assets, "urlopen", download)
    monkeypatch.setattr(assets.os, "link", racing_link)
    if content == b"official bytes":
        assets.install_checkpoint(checkpoint, path)
    else:
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            assets.install_checkpoint(checkpoint, path)
    assert path.read_bytes() == content
    assert list(tmp_path.iterdir()) == [path]
