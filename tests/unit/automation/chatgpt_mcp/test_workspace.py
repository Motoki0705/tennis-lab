from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.automation.chatgpt_mcp.workspace import WorkspaceError, WorkspaceManager


def _git_repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "example.txt").write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "example.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "initial"], check=True)
    return path


def test_read_and_apply_patch_inside_worktree(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    manager = WorkspaceManager(repo)

    content = manager.read_file(str(repo), "example.txt")
    assert content["text"] == "1: before"

    result = manager.apply_patch(
        str(repo),
        """diff --git a/example.txt b/example.txt
index 90be1f3..8306d1e 100644
--- a/example.txt
+++ b/example.txt
@@ -1 +1 @@
-before
+after
""",
    )
    assert " M example.txt" in result["status"]
    assert (repo / "example.txt").read_text(encoding="utf-8") == "after\n"


def test_file_path_cannot_escape_worktree(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    manager = WorkspaceManager(repo)

    with pytest.raises(WorkspaceError, match="escapes"):
        manager.read_file(str(repo), "../outside.txt")
