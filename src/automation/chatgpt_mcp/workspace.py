"""Strict git-worktree and file operations exposed through MCP tools."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

_WORKTREE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_GIT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")


class WorkspaceError(RuntimeError):
    """Raised when an operation would escape or corrupt an allowed worktree."""


def _run_git(
    workspace: Path,
    arguments: list[str],
    *,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(workspace), *arguments],
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


def _validate_git_name(value: str, label: str) -> str:
    if (
        not _GIT_NAME.fullmatch(value)
        or ".." in value
        or "//" in value
        or "@{" in value
        or value.endswith("/")
    ):
        raise WorkspaceError(f"invalid {label}: {value!r}")
    return value


class WorkspaceManager:
    """Operate only on git worktrees located below one configured repository."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def resolve_workspace(self, value: str) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.repo_root / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.repo_root):
            raise WorkspaceError("workspace must stay inside TENNIS_MCP_REPO_ROOT")
        result = _run_git(resolved, ["rev-parse", "--show-toplevel"])
        if result.returncode != 0:
            raise WorkspaceError(f"not a git worktree: {resolved}")
        top_level = Path(result.stdout.strip()).resolve()
        if top_level != resolved:
            raise WorkspaceError(f"workspace must name its git root: {top_level}")
        return resolved

    def resolve_file(self, workspace: Path, relative_path: str) -> Path:
        requested = Path(relative_path)
        if requested.is_absolute():
            raise WorkspaceError("file path must be relative to the worktree")
        resolved = (workspace / requested).resolve()
        if not resolved.is_relative_to(workspace):
            raise WorkspaceError("file path escapes the worktree")
        return resolved

    def create_worktree(
        self,
        *,
        name: str,
        branch: str,
        base_ref: str = "origin/main",
    ) -> dict[str, str]:
        if not _WORKTREE_NAME.fullmatch(name):
            raise WorkspaceError(
                "worktree name must use lowercase letters, digits, and hyphens"
            )
        branch = _validate_git_name(branch, "branch")
        base_ref = _validate_git_name(base_ref, "base ref")
        target = self.repo_root / ".chatgpt" / "worktrees" / name
        if target.exists():
            raise WorkspaceError(f"worktree path already exists: {target}")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

        result = _run_git(
            self.repo_root,
            ["worktree", "add", "-b", branch, str(target), base_ref],
        )
        if result.returncode != 0:
            raise WorkspaceError(result.stderr.strip() or "git worktree add failed")
        return {"workspace": str(target), "branch": branch, "base_ref": base_ref}

    def list_files(
        self,
        workspace_value: str,
        *,
        path: str = ".",
        limit: int = 500,
    ) -> dict[str, Any]:
        if not 1 <= limit <= 2000:
            raise WorkspaceError("limit must be between 1 and 2000")
        workspace = self.resolve_workspace(workspace_value)
        base = self.resolve_file(workspace, path)
        if not base.exists() or not base.is_dir():
            raise WorkspaceError(f"directory does not exist: {path}")
        files: list[str] = []
        for candidate in sorted(base.rglob("*")):
            if candidate.is_file() and ".git" not in candidate.parts:
                files.append(str(candidate.relative_to(workspace)))
                if len(files) >= limit:
                    break
        return {
            "workspace": str(workspace),
            "files": files,
            "truncated": len(files) == limit,
        }

    def read_file(
        self,
        workspace_value: str,
        relative_path: str,
        *,
        start_line: int = 1,
        max_lines: int = 400,
    ) -> dict[str, Any]:
        if start_line < 1 or not 1 <= max_lines <= 1000:
            raise WorkspaceError("invalid line range")
        workspace = self.resolve_workspace(workspace_value)
        path = self.resolve_file(workspace, relative_path)
        if not path.is_file():
            raise WorkspaceError(f"file does not exist: {relative_path}")
        if path.stat().st_size > 4 * 1024 * 1024:
            raise WorkspaceError("read_file refuses files larger than 4 MiB")
        lines = path.read_text(encoding="utf-8").splitlines()
        selected = lines[start_line - 1 : start_line - 1 + max_lines]
        numbered = "\n".join(
            f"{number}: {line}"
            for number, line in enumerate(selected, start=start_line)
        )
        return {
            "path": relative_path,
            "start_line": start_line,
            "end_line": start_line + len(selected) - 1,
            "total_lines": len(lines),
            "text": numbered,
        }

    def search_code(
        self,
        workspace_value: str,
        query: str,
        *,
        glob: str | None = None,
        max_results: int = 200,
    ) -> dict[str, Any]:
        if not query or len(query) > 500:
            raise WorkspaceError("query must contain 1-500 characters")
        if not 1 <= max_results <= 1000:
            raise WorkspaceError("max_results must be between 1 and 1000")
        workspace = self.resolve_workspace(workspace_value)
        arguments = ["rg", "--line-number", "--color", "never", "--", query]
        if glob:
            if len(glob) > 200:
                raise WorkspaceError("glob is too long")
            arguments = [
                "rg",
                "--line-number",
                "--color",
                "never",
                "--glob",
                glob,
                "--",
                query,
            ]
        result = subprocess.run(
            arguments,
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        if result.returncode not in {0, 1}:
            raise WorkspaceError(result.stderr.strip() or "rg failed")
        lines = result.stdout.splitlines()
        return {
            "query": query,
            "matches": lines[:max_results],
            "truncated": len(lines) > max_results,
        }

    def apply_patch(self, workspace_value: str, patch: str) -> dict[str, Any]:
        if not patch.strip() or len(patch.encode("utf-8")) > 1024 * 1024:
            raise WorkspaceError("patch must contain 1 byte to 1 MiB")
        workspace = self.resolve_workspace(workspace_value)
        check = _run_git(
            workspace,
            ["apply", "--check", "--whitespace=nowarn", "-"],
            input_text=patch,
        )
        if check.returncode != 0:
            raise WorkspaceError(check.stderr.strip() or "patch check failed")
        applied = _run_git(
            workspace,
            ["apply", "--whitespace=nowarn", "-"],
            input_text=patch,
        )
        if applied.returncode != 0:
            raise WorkspaceError(applied.stderr.strip() or "patch apply failed")
        return self.git_status(workspace_value)

    def git_status(self, workspace_value: str) -> dict[str, Any]:
        workspace = self.resolve_workspace(workspace_value)
        status = _run_git(workspace, ["status", "--short", "--branch"])
        if status.returncode != 0:
            raise WorkspaceError(status.stderr.strip())
        return {"workspace": str(workspace), "status": status.stdout.rstrip()}

    def git_diff(self, workspace_value: str, *, staged: bool = False) -> dict[str, Any]:
        workspace = self.resolve_workspace(workspace_value)
        arguments = ["diff", "--no-ext-diff", "--stat"]
        if staged:
            arguments.append("--cached")
        stat = _run_git(workspace, arguments)
        diff_arguments = ["diff", "--no-ext-diff"]
        if staged:
            diff_arguments.append("--cached")
        diff = _run_git(workspace, diff_arguments)
        if stat.returncode != 0 or diff.returncode != 0:
            raise WorkspaceError((stat.stderr or diff.stderr).strip())
        output = diff.stdout
        truncated = len(output) > 200_000
        return {
            "workspace": str(workspace),
            "stat": stat.stdout.rstrip(),
            "diff": output[:200_000],
            "truncated": truncated,
        }

    def commit(self, workspace_value: str, message: str) -> dict[str, str]:
        if not message.strip() or len(message) > 500:
            raise WorkspaceError("commit message must contain 1-500 characters")
        workspace = self.resolve_workspace(workspace_value)
        add = _run_git(workspace, ["add", "--all"])
        if add.returncode != 0:
            raise WorkspaceError(add.stderr.strip())
        commit = _run_git(workspace, ["commit", "-m", message])
        if commit.returncode != 0:
            raise WorkspaceError(commit.stderr.strip() or commit.stdout.strip())
        revision = _run_git(workspace, ["rev-parse", "HEAD"])
        return {"workspace": str(workspace), "commit": revision.stdout.strip()}

    def push(self, workspace_value: str) -> dict[str, str]:
        workspace = self.resolve_workspace(workspace_value)
        branch_result = _run_git(workspace, ["branch", "--show-current"])
        branch = branch_result.stdout.strip()
        _validate_git_name(branch, "branch")
        push = _run_git(workspace, ["push", "--set-upstream", "origin", branch])
        if push.returncode != 0:
            raise WorkspaceError(push.stderr.strip() or push.stdout.strip())
        return {"workspace": str(workspace), "branch": branch, "remote": "origin"}
