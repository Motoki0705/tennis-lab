"""Exact-revision workspaces used only for isolated execution and validation."""

from __future__ import annotations

import os
import re
import secrets
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.automation.chatgpt_mcp.storage import SqliteStore

_WORKSPACE_ID = re.compile(r"^rev-[a-f0-9]{16}$")
_GIT_BRANCH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class WorkspaceError(RuntimeError):
    """Raised when a remote revision cannot be prepared or verified safely."""


@dataclass(frozen=True)
class RevisionWorkspace:
    """One detached, exact-SHA source workspace owned by the MCP gateway."""

    workspace_id: str
    path: Path
    branch: str
    revision: str

    def public_dict(self) -> dict[str, str]:
        """Return the non-secret fields exposed through MCP."""

        return {
            "workspace_id": self.workspace_id,
            "branch": self.branch,
            "revision": self.revision,
        }


def _run_git(
    workspace: Path,
    arguments: list[str],
    *,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(workspace), *arguments],
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _checked_git(
    workspace: Path,
    arguments: list[str],
    *,
    message: str,
    timeout: int = 120,
) -> str:
    result = _run_git(workspace, arguments, timeout=timeout)
    if result.returncode != 0:
        raise WorkspaceError(result.stderr.strip() or result.stdout.strip() or message)
    return result.stdout.strip()


def _validate_branch(value: str) -> str:
    branch = value.strip()
    if (
        not _GIT_BRANCH.fullmatch(branch)
        or ".." in branch
        or "//" in branch
        or "@{" in branch
        or branch.endswith("/")
        or branch.startswith("-")
    ):
        raise WorkspaceError(f"invalid remote branch: {value!r}")
    return branch


def _validate_revision(value: str) -> str:
    revision = value.strip().lower()
    if not _GIT_SHA.fullmatch(revision):
        raise WorkspaceError("expected_sha must be a full 40-character commit SHA")
    return revision


def _validate_workspace_id(value: str) -> str:
    workspace_id = value.strip()
    if not _WORKSPACE_ID.fullmatch(workspace_id):
        raise WorkspaceError(f"invalid revision workspace id: {value!r}")
    return workspace_id


class WorkspaceManager:
    """Fetch exact origin revisions and expose only opaque execution workspace IDs."""

    def __init__(
        self,
        repo_root: Path,
        revision_root: Path,
        store: SqliteStore,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.workspace_root = revision_root.resolve()
        if self.workspace_root == self.repo_root or self.workspace_root.is_relative_to(
            self.repo_root
        ):
            raise WorkspaceError(
                "revision workspace storage must be outside the canonical repository"
            )
        self.store = store

    def prepare_revision(self, *, branch: str, expected_sha: str) -> dict[str, str]:
        """Fetch one origin branch and create a detached worktree at an exact SHA."""

        checked_branch = _validate_branch(branch)
        checked_sha = _validate_revision(expected_sha)
        self.workspace_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.workspace_root, 0o700)

        remote_ref = f"refs/remotes/origin/{checked_branch}"
        refspec = f"+refs/heads/{checked_branch}:{remote_ref}"
        _checked_git(
            self.repo_root,
            [
                "fetch",
                "--force",
                "--no-tags",
                "--no-recurse-submodules",
                "origin",
                refspec,
            ],
            message="git fetch failed",
            timeout=300,
        )
        fetched_sha = _checked_git(
            self.repo_root,
            ["rev-parse", f"{remote_ref}^{{commit}}"],
            message="fetched branch did not resolve to a commit",
        ).lower()
        if fetched_sha != checked_sha:
            raise WorkspaceError(
                "remote revision mismatch: "
                f"origin/{checked_branch} is {fetched_sha}, expected {checked_sha}"
            )

        workspace_id = f"rev-{secrets.token_hex(8)}"
        target = (self.workspace_root / workspace_id).resolve()
        if not target.is_relative_to(self.workspace_root) or target.parent != self.workspace_root:
            raise WorkspaceError("revision workspace escaped its configured root")
        result = _run_git(
            self.repo_root,
            ["worktree", "add", "--detach", str(target), checked_sha],
            timeout=300,
        )
        if result.returncode != 0:
            raise WorkspaceError(result.stderr.strip() or "git worktree add failed")
        os.chmod(target, 0o700)

        workspace = RevisionWorkspace(
            workspace_id=workspace_id,
            path=target,
            branch=checked_branch,
            revision=checked_sha,
        )
        try:
            self._verify_materialized_workspace(workspace)
        except BaseException:
            _run_git(
                self.repo_root,
                ["worktree", "remove", "--force", str(target)],
                timeout=120,
            )
            raise

        self.store.put(
            "revision_workspaces",
            workspace_id,
            {
                "workspace_id": workspace_id,
                "path": str(target),
                "branch": checked_branch,
                "revision": checked_sha,
            },
        )
        return workspace.public_dict()

    def get_revision(self, workspace_id: str) -> RevisionWorkspace:
        """Resolve an opaque workspace ID and revalidate its immutable identity."""

        checked_id = _validate_workspace_id(workspace_id)
        payload = self.store.get("revision_workspaces", checked_id)
        if payload is None:
            raise WorkspaceError("revision workspace was not found")
        path = Path(str(payload["path"])).resolve()
        if not path.is_relative_to(self.workspace_root) or path.parent != self.workspace_root:
            raise WorkspaceError("stored revision workspace escaped its configured root")
        workspace = RevisionWorkspace(
            workspace_id=checked_id,
            path=path,
            branch=_validate_branch(str(payload["branch"])),
            revision=_validate_revision(str(payload["revision"])),
        )
        self._verify_materialized_workspace(workspace)
        return workspace

    def assert_execution_ready(
        self,
        *,
        workspace_id: str,
        expected_sha: str,
    ) -> RevisionWorkspace:
        """Require exact SHA binding and a completely clean source before execution."""

        checked_sha = _validate_revision(expected_sha)
        workspace = self.get_revision(workspace_id)
        if workspace.revision != checked_sha:
            raise WorkspaceError(
                "workspace revision does not match expected_sha: "
                f"{workspace.revision} != {checked_sha}"
            )
        status = _checked_git(
            workspace.path,
            ["status", "--porcelain=v1", "--untracked-files=all"],
            message="git status failed",
        )
        if status:
            raise WorkspaceError(
                "revision workspace contains changes; prepare a new workspace"
            )
        return workspace

    def describe_revision(self, workspace_id: str) -> dict[str, Any]:
        """Return exact revision identity and clean state without reading source files."""

        workspace = self.get_revision(workspace_id)
        status = _checked_git(
            workspace.path,
            ["status", "--porcelain=v1", "--untracked-files=all"],
            message="git status failed",
        )
        return {
            **workspace.public_dict(),
            "clean": not bool(status),
        }

    def _verify_materialized_workspace(self, workspace: RevisionWorkspace) -> None:
        if not workspace.path.is_dir():
            raise WorkspaceError("revision workspace directory is missing")
        top_level = _checked_git(
            workspace.path,
            ["rev-parse", "--show-toplevel"],
            message="path is not a git worktree",
        )
        if Path(top_level).resolve() != workspace.path:
            raise WorkspaceError("revision workspace must name its exact git root")
        head = _checked_git(
            workspace.path,
            ["rev-parse", "HEAD^{commit}"],
            message="revision workspace HEAD is unavailable",
        ).lower()
        if head != workspace.revision:
            raise WorkspaceError(
                f"revision workspace moved from {workspace.revision} to {head}"
            )
        git_pointer = workspace.path / ".git"
        if git_pointer.is_symlink() or not git_pointer.is_file():
            raise WorkspaceError("revision workspace .git pointer is not a regular file")
        first_line = git_pointer.read_text(encoding="utf-8").splitlines()[0]
        if not first_line.startswith("gitdir: "):
            raise WorkspaceError("revision workspace .git pointer is malformed")
        git_dir = Path(first_line.removeprefix("gitdir: ")).resolve()
        worktree_metadata_root = Path(
            _checked_git(
                self.repo_root,
                ["rev-parse", "--path-format=absolute", "--git-path", "worktrees"],
                message="repository worktree metadata path is unavailable",
            )
        ).resolve()
        if not git_dir.is_relative_to(worktree_metadata_root):
            raise WorkspaceError("revision workspace git metadata is outside the repository")
