"""Git material for review and worker worktree validation; no merge authority."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any


def git(*args: str, cwd: Path | None = None) -> bytes:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, timeout=60, check=False
    )
    if result.returncode:
        raise ValueError(
            f"git {args[0]} failed: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def validate_worktree(worktree: Path, branch: str) -> None:
    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    common = Path(
        git("rev-parse", "--path-format=absolute", "--git-common-dir").decode().strip()
    )
    main_root = common.parent
    resolved = worktree.resolve(strict=True)
    expected = main_root / ".claude/worktrees"
    if not resolved.is_relative_to(expected) or resolved in (expected, main_root):
        raise ValueError(
            "worker worktree must be under the main checkout's .claude/worktrees/"
        )
    if (
        Path(
            git("rev-parse", "--show-toplevel", cwd=resolved).decode().strip()
        ).resolve()
        != resolved
    ):
        raise ValueError("worker path must be a worktree root")
    worker_common = Path(
        git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=resolved)
        .decode()
        .strip()
    )
    if (
        worker_common != common
        or git("branch", "--show-current", cwd=resolved).decode().strip() != branch
    ):
        raise ValueError("worker must belong to this repository and requested branch")
    if resolved == root.resolve():
        raise ValueError("a child requires its own worktree, separate from the parent")


def snapshot(pr: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    head, base = pr["head"]["sha"], pr["base"]["sha"]
    if any(
        not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha)
        for sha in (head, base)
    ):
        raise ValueError("PR head/base must be full git SHA-1 identifiers")
    for sha in (head, base):
        available = subprocess.run(
            ["git", "cat-file", "-e", sha + "^{commit}"],
            capture_output=True,
            timeout=60,
            check=False,
        )
        if available.returncode:
            git("fetch", "--no-tags", "--no-write-fetch-head", "origin", sha)
    merge_base = git("merge-base", base, head).decode().strip()
    diff = git(
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--binary",
        "--full-index",
        "--no-renames",
        "--no-color",
        merge_base,
        head,
        "--",
    )
    return {
        "number": pr["number"],
        "head_sha": head,
        "base_sha": base,
        "base_ref": pr["base"]["ref"],
        "merge_base_sha": merge_base,
        "branch": pr["head"]["ref"],
    }, diff
