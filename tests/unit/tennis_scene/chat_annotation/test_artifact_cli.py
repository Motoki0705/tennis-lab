"""CLI path validation must run before storage or video side effects."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.tennis_scene.chat_annotation.scripts import serve_artifacts, sync_done
from src.utils.configuration.errors import PathContractError


@pytest.mark.parametrize("module", [serve_artifacts, sync_done])
@pytest.mark.parametrize("arguments", [[], ["--root", "relative-output"]])
def test_cli_requires_explicit_absolute_root(module, arguments, monkeypatch) -> None:
    effect = Mock()
    monkeypatch.setattr(
        module, "create_server" if module is serve_artifacts else "sync_done", effect
    )
    monkeypatch.setattr("sys.argv", [module.__name__, *arguments])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    effect.assert_not_called()


def test_server_rejects_filesystem_root_before_starting(monkeypatch) -> None:
    create = Mock()
    monkeypatch.setattr(serve_artifacts, "create_server", create)
    monkeypatch.setattr("sys.argv", ["serve_artifacts", "--root", "/"])
    with pytest.raises(PathContractError, match="filesystem root"):
        serve_artifacts.main()
    create.assert_not_called()


def test_server_uses_validated_directory(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "raw"
    create = Mock()
    monkeypatch.setattr(serve_artifacts, "create_server", create)
    monkeypatch.setenv("ARTIFACT_DOWNLOAD_HOSTS", "files.example.com")
    monkeypatch.setattr("sys.argv", ["serve_artifacts", "--root", str(root)])
    serve_artifacts.main()
    create.assert_called_once_with(
        root, frozenset({"files.example.com"}), "127.0.0.1", 8000
    )
    create.return_value.run.assert_called_once_with(transport="streamable-http")
    assert not root.exists()


def test_completion_requires_existing_root_before_moving(
    tmp_path: Path, monkeypatch
) -> None:
    move = Mock()
    monkeypatch.setattr(sync_done, "sync_done", move)
    monkeypatch.setattr("sys.argv", ["sync_done", "--root", str(tmp_path / "missing")])
    with pytest.raises(PathContractError, match="does not exist"):
        sync_done.main()
    move.assert_not_called()


def test_completion_passes_dry_run_to_validated_root(
    tmp_path: Path, monkeypatch
) -> None:
    move = Mock(return_value={"errors": [], "ready": ["clip"]})
    monkeypatch.setattr(sync_done, "sync_done", move)
    monkeypatch.setattr("sys.argv", ["sync_done", "--root", str(tmp_path), "--dry-run"])
    with pytest.raises(SystemExit) as error:
        sync_done.main()
    assert error.value.code == 0
    move.assert_called_once_with(tmp_path, dry_run=True)
