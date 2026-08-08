"""Exact-revision sources backed by the trusted external Git mirror."""

from __future__ import annotations

import os
import re
import secrets
import subprocess
import threading
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
    """One detached exact-SHA source worktree owned by the trusted control plane."""

    workspace_id: str
    path: Path
    branch: str
    revision: str

    def public_dict(self) -> dict[str, str]:
        return {
            "workspace_id": self.workspace_id,
            "branch": self.branch,
            "revision": self.revision,
        }


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
    """Fetch origin revisions into a trusted bare mirror and expose opaque IDs."""

    def __init__(
        self,
        trusted_git_dir: Path,
        state_or_revision_root: Path,
        store: SqliteStore,
    ) -> None:
        self.git_dir = trusted_git_dir.resolve()
        configured_root = state_or_revision_root.resolve()
        self.workspace_root = (
            configured_root
            if configured_root.name == "revisions"
            else (configured_root / "revisions").resolve()
        )
        if self.workspace_root == self.git_dir or self.workspace_root.is_relative_to(
            self.git_dir
        ):
            raise WorkspaceError(
                "revision workspace storage must be separate from trusted Git metadata"
            )
        self.store = store
        self._prepare_lock = threading.Lock()

    @property
    def repo_root(self) -> Path:
        """Compatibility alias for callers that formerly received a checkout."""

        return self.git_dir

    def _git_environment(self) -> dict[str, str]:
        git_home = self.git_dir.parent / "git-home"
        git_home.mkdir(mode=0o700, parents=True, exist_ok=True)
        return {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(git_home),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "1",
        }

    def _git(
        self,
        arguments: list[str],
        *,
        workspace: Path | None = None,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess[str]:
        prefix = ["git", "-c", "core.hooksPath=/dev/null"]
        if workspace is None:
            prefix.extend(["--git-dir", str(self.git_dir)])
        else:
            prefix.extend(["-C", str(workspace)])
        return subprocess.run(
            [*prefix, *arguments],
            env=self._git_environment(),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )

    def _checked_git(
        self,
        arguments: list[str],
        *,
        workspace: Path | None = None,
        message: str,
        timeout: int = 120,
    ) -> str:
        result = self._git(arguments, workspace=workspace, timeout=timeout)
        if result.returncode != 0:
            raise WorkspaceError(
                result.stderr.strip() or result.stdout.strip() or message
            )
        return result.stdout.strip()

    def _validate_trusted_mirror(self) -> None:
        if not self.git_dir.is_dir():
            raise WorkspaceError(f"trusted Git mirror is missing: {self.git_dir}")
        result = self._git(["rev-parse", "--is-bare-repository"])
        if result.returncode != 0 or result.stdout.strip() != "true":
            raise WorkspaceError("trusted Git mirror must be bare")

    def prepare_revision(self, *, branch: str, expected_sha: str) -> dict[str, str]:
        """Fetch one fixed-origin branch and create a detached exact-SHA worktree."""

        self._validate_trusted_mirror()
        checked_branch = _validate_branch(branch)
        checked_sha = _validate_revision(expected_sha)
        self.workspace_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.workspace_root, 0o700)

        with self._prepare_lock:
            remote_ref = f"refs/remotes/origin/{checked_branch}"
            refspec = f"+refs/heads/{checked_branch}:{remote_ref}"
            self._checked_git(
                [
                    "fetch",
                    "--force",
                    "--no-tags",
                    "--no-recurse-submodules",
                    "origin",
                    refspec,
                ],
                message="trusted origin fetch failed",
                timeout=600,
            )
            fetched_sha = self._checked_git(
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
            if (
                not target.is_relative_to(self.workspace_root)
                or target.parent != self.workspace_root
            ):
                raise WorkspaceError("revision workspace escaped its configured root")
            result = self._git(
                ["worktree", "add", "--detach", str(target), checked_sha],
                timeout=600,
            )
            if result.returncode != 0:
                raise WorkspaceError(
                    result.stderr.strip() or "trusted git worktree add failed"
                )

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
            self._git(["worktree", "remove", "--force", str(target)], timeout=120)
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

        self._validate_trusted_mirror()
        checked_id = _validate_workspace_id(workspace_id)
        payload = self.store.get("revision_workspaces", checked_id)
        if payload is None:
            raise WorkspaceError("revision workspace was not found")
        path = Path(str(payload["path"])).resolve()
        if (
            not path.is_relative_to(self.workspace_root)
            or path.parent != self.workspace_root
        ):
            raise WorkspaceError(
                "stored revision workspace escaped its configured root"
            )
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
        """Require exact SHA binding and a completely clean trusted source."""

        checked_sha = _validate_revision(expected_sha)
        workspace = self.get_revision(workspace_id)
        if workspace.revision != checked_sha:
            raise WorkspaceError(
                "workspace revision does not match expected_sha: "
                f"{workspace.revision} != {checked_sha}"
            )
        status = self._checked_git(
            ["status", "--porcelain=v1", "--untracked-files=all"],
            workspace=workspace.path,
            message="git status failed",
        )
        if status:
            raise WorkspaceError(
                "trusted revision source contains changes; prepare a new workspace"
            )
        return workspace

    def describe_revision(self, workspace_id: str) -> dict[str, Any]:
        workspace = self.get_revision(workspace_id)
        status = self._checked_git(
            ["status", "--porcelain=v1", "--untracked-files=all"],
            workspace=workspace.path,
            message="git status failed",
        )
        return {
            **workspace.public_dict(),
            "clean": not bool(status),
            "default_execution_root": "revision",
            "project_root_available": True,
        }

    def _verify_materialized_workspace(self, workspace: RevisionWorkspace) -> None:
        if not workspace.path.is_dir():
            raise WorkspaceError("revision workspace directory is missing")
        top_level = self._checked_git(
            ["rev-parse", "--show-toplevel"],
            workspace=workspace.path,
            message="path is not a git worktree",
        )
        if Path(top_level).resolve() != workspace.path:
            raise WorkspaceError("revision workspace must name its exact git root")
        head = self._checked_git(
            ["rev-parse", "HEAD^{commit}"],
            workspace=workspace.path,
            message="revision workspace HEAD is unavailable",
        ).lower()
        if head != workspace.revision:
            raise WorkspaceError(
                f"revision workspace moved from {workspace.revision} to {head}"
            )
        git_pointer = workspace.path / ".git"
        if git_pointer.is_symlink() or not git_pointer.is_file():
            raise WorkspaceError(
                "revision workspace .git pointer is not a regular file"
            )
        first_line = git_pointer.read_text(encoding="utf-8").splitlines()[0]
        if not first_line.startswith("gitdir: "):
            raise WorkspaceError("revision workspace .git pointer is malformed")
        git_metadata = Path(first_line.removeprefix("gitdir: ")).resolve()
        metadata_root = (self.git_dir / "worktrees").resolve()
        if not git_metadata.is_relative_to(metadata_root):
            raise WorkspaceError(
                "revision workspace Git metadata is outside the trusted mirror"
            )
