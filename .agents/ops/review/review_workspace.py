"""Read-only review inputs: all PR heads plus caller-owned full diff files."""

from __future__ import annotations

import json
import os
import string
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from parent.snapshot import git


def changed_paths(before: str, after: str) -> list[str]:
    raw = git(
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--name-only",
        "-z",
        before,
        after,
        "--",
    )
    if raw and not raw.endswith(b"\0"):
        raise ValueError("git returned incomplete NUL-separated paths")
    return sorted(os.fsdecode(p) for p in raw.split(b"\0")[:-1])


def full_diff(before: str, after: str) -> bytes:
    return git(
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--binary",
        "--full-index",
        "--no-renames",
        "--no-color",
        before,
        after,
        "--",
    )


def write_new(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)


def workspace_path(checkout: Path, identifier: str) -> Path:
    common = Path(
        git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=checkout)
        .decode()
        .strip()
    )
    return common.parent / ".claude/worktrees" / ("review-" + identifier)


def require_clean_workspace(checkout: Path, identifier: str) -> None:
    path = workspace_path(checkout, identifier)
    worktrees = git("worktree", "list", "--porcelain", "-z", cwd=checkout)
    if (
        path.exists()
        or path.is_symlink()
        or any(
            Path(os.fsdecode(field.removeprefix(b"worktree "))).is_relative_to(path)
            for field in worktrees.split(b"\0")
            if field.startswith(b"worktree ")
        )
    ):
        raise ValueError("recovery_required: previous review workspace remains")


@contextmanager
def workspace(
    path: Path,
    request: Mapping[str, Any],
    context: Mapping[str, Any],
    diffs: Mapping[int, bytes],
) -> Iterator[Path]:
    # Metadata lives outside PR checkouts: a PR-provided symlink cannot redirect it.
    path.mkdir(parents=True, mode=0o700, exist_ok=False)
    worktrees: list[Path] = []
    try:
        write_new(
            path / "context.json",
            json.dumps(context, ensure_ascii=False, indent=2).encode(),
        )
        write_new(
            path / "request.json",
            json.dumps(request, ensure_ascii=False, indent=2).encode(),
        )
        for pr in request["prs"]:
            n = pr["number"]
            write_new(path / f"pr-{n}.diff", diffs[n])
            write_new(
                path / f"base-{n}-AGENTS.md", git("show", pr["base_sha"] + ":AGENTS.md")
            )
            head = path / f"pr-{n}"
            git(
                "-c",
                "core.hooksPath=/dev/null",
                "worktree",
                "add",
                "--detach",
                "--quiet",
                str(head),
                pr["head_sha"],
            )
            worktrees.append(head)
        yield path
    finally:
        # Never delete an existing/stale workspace on entry. Cleanup only our own files.
        for head in reversed(worktrees):
            git("worktree", "remove", "--force", str(head))
        for entry in path.iterdir():
            entry.unlink()
        path.rmdir()


def build_prompt(
    request: Mapping[str, Any],
    context: Mapping[str, Any],
    diffs: Mapping[int, bytes],
    template: str,
) -> str:
    targets = [
        {**pr, "checkout": f"pr-{pr['number']}/", "diff": f"pr-{pr['number']}.diff"}
        for pr in request["prs"]
    ]
    return string.Template(template).substitute(
        parent=request["parent_issue"],
        run_id=request["review_run_id"],
        instruction=context.get(
            "instruction", "親・子の要件と対象PRの正しさを確認してください。"
        ),
        targets=json.dumps(targets, ensure_ascii=False, separators=(",", ":")),
    )
