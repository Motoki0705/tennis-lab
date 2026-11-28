"""Git material for review and worker worktree validation; no merge authority."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def git(
    *args: str, cwd: Path | None = None, env: Mapping[str, str] | None = None
) -> bytes:
    result = subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, timeout=60, check=False
    )
    if result.returncode:
        raise ValueError(
            f"git {args[0]} failed: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def worker_location(parent: int, child: int, worker: str) -> tuple[str, Path]:
    common = Path(
        git("rev-parse", "--path-format=absolute", "--git-common-dir").decode().strip()
    )
    name = f"parent-{parent}-issue-{child}-{worker}"
    return "codex/" + name, common.parent / ".claude/worktrees" / name


def prepare_worker_tree(
    tree: Path, branch: str, base_ref: str, state_dir: Path
) -> None:
    """Retry a partially prepared reservation without adopting a foreign tree."""
    from .policy import digest
    from .state import Store

    intent = state_dir / "worker-setup" / (digest(str(tree)) + ".json")
    base = (
        git("rev-parse", "--verify", "--end-of-options", base_ref + "^{commit}")
        .decode()
        .strip()
    )
    expected = {"branch": branch, "worktree": str(tree), "base": base}
    if tree.resolve() != tree:
        raise ValueError("worker checkout contains a symlink")
    if intent.exists():
        if json.loads(intent.read_text()) != expected:
            raise ValueError(
                "worker setup already bound to a different base or worktree"
            )
    else:
        if tree.exists() or git("branch", "--list", branch).strip():
            raise ValueError(
                "automatic worker branch/worktree already exists; inspect before delegating"
            )
        Store.write_json(intent, expected)
    if not tree.exists():
        if git("branch", "--list", branch).strip():
            if git("rev-parse", "refs/heads/" + branch).decode().strip() != base:
                raise ValueError("worker branch moved during setup")
            git("worktree", "add", str(tree), branch)
        else:
            git("worktree", "add", "-b", branch, str(tree), base)
    validate_worktree(tree, branch)
    if (
        git("rev-parse", "HEAD", cwd=tree).decode().strip() != base
        or git("status", "--porcelain", cwd=tree).strip()
    ):
        raise ValueError("worker setup checkout differs from its reserved base")
    ops = Path(__file__).resolve().parents[1]
    main = tree.parents[2]
    prepared = subprocess.run(
        [
            sys.executable,
            str(ops / "checkout_job.py"),
            "--repo",
            str(main),
            "--",
            sys.executable,
            "-m",
            "storage_tier",
            "--repo",
            str(main),
            "prepare-worktree",
            str(tree),
        ],
        env={**os.environ, "PYTHONPATH": str(ops)},
        check=False,
        stdout=subprocess.DEVNULL,
    )
    if prepared.returncode:
        raise ValueError(
            "storage preparation failed; fix the reported cause and retry the same --from"
        )


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
