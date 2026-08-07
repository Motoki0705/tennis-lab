from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.automation.chatgpt_mcp.storage import SqliteStore
from src.automation.chatgpt_mcp.workspace import WorkspaceError, WorkspaceManager


def _run(*arguments: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(arguments),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _repo_with_origin(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source"
    _run("git", "init", "-q", "-b", "main", str(source))
    _run("git", "config", "user.email", "test@example.com", cwd=source)
    _run("git", "config", "user.name", "Test", cwd=source)
    (source / "example.txt").write_text("revision\n", encoding="utf-8")
    _run("git", "add", "example.txt", cwd=source)
    _run("git", "commit", "-qm", "initial", cwd=source)
    revision = _run("git", "rev-parse", "HEAD", cwd=source)

    remote = tmp_path / "origin.git"
    _run("git", "clone", "-q", "--bare", str(source), str(remote))
    repo = tmp_path / "repo"
    _run("git", "clone", "-q", str(remote), str(repo))
    return repo, revision


def _manager(tmp_path: Path) -> tuple[WorkspaceManager, str]:
    repo, revision = _repo_with_origin(tmp_path)
    state = tmp_path / "state"
    store = SqliteStore(state / "gateway.sqlite3")
    return WorkspaceManager(repo, state, store), revision


def test_prepare_revision_fetches_exact_sha_into_detached_worktree(
    tmp_path: Path,
) -> None:
    manager, revision = _manager(tmp_path)

    prepared = manager.prepare_revision(branch="main", expected_sha=revision)
    workspace = manager.get_revision(prepared["workspace_id"])

    assert prepared["revision"] == revision
    assert prepared["branch"] == "main"
    assert workspace.path.parent == manager.workspace_root
    assert _run("git", "rev-parse", "HEAD", cwd=workspace.path) == revision
    assert _run("git", "branch", "--show-current", cwd=workspace.path) == ""
    assert manager.describe_revision(workspace.workspace_id)["tracked_clean"] is True


def test_prepare_revision_rejects_remote_sha_mismatch(tmp_path: Path) -> None:
    manager, revision = _manager(tmp_path)
    wrong = "0" * 40 if revision != "0" * 40 else "1" * 40

    with pytest.raises(WorkspaceError, match="remote revision mismatch"):
        manager.prepare_revision(branch="main", expected_sha=wrong)


def test_execution_requires_registered_id_exact_sha_and_clean_source(
    tmp_path: Path,
) -> None:
    manager, revision = _manager(tmp_path)
    prepared = manager.prepare_revision(branch="main", expected_sha=revision)
    workspace_id = prepared["workspace_id"]

    with pytest.raises(WorkspaceError, match="does not match expected_sha"):
        manager.assert_execution_ready(
            workspace_id=workspace_id,
            expected_sha="f" * 40,
        )

    workspace = manager.get_revision(workspace_id)
    (workspace.path / "example.txt").write_text("modified\n", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="tracked changes"):
        manager.assert_execution_ready(
            workspace_id=workspace_id,
            expected_sha=revision,
        )


def test_arbitrary_filesystem_paths_are_not_workspace_ids(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)

    with pytest.raises(WorkspaceError, match="invalid revision workspace id"):
        manager.get_revision(str(manager.repo_root))
