"""Read-only review inputs: all PR heads plus caller-owned full diff files."""

from __future__ import annotations

import json
import os
import string
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from parent.snapshot import git

MAX_DIFF_CHARS = 200_000


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
    return cast(
        bytes,
        git(
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
        ),
    )


def write_new(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)


@contextmanager
def workspace(
    root: Path,
    request: Mapping[str, Any],
    context: Mapping[str, Any],
    diffs: Mapping[int, bytes],
) -> Iterator[Path]:
    # Metadata lives outside PR checkouts: a PR-provided symlink cannot redirect it.
    path = root / ".claude/worktrees" / ("review-" + request["review_run_id"])
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
    remaining = MAX_DIFF_CHARS
    parts = []
    for pr in request["prs"]:
        n = pr["number"]
        text = diffs[n].decode("utf-8", errors="replace")
        excerpt = text[:remaining]
        remaining -= len(excerpt)
        note = (
            "全文" if len(excerpt) == len(text) else "一部のみ。全文ファイルを必ず読む"
        )
        parts.append(
            f"### PR #{n}: {note}\nhead: pr-{n}/、全文diff: pr-{n}.diff、base規則: base-{n}-AGENTS.md\n```diff\n{excerpt}\n```"
        )
    # Large issue/PR bodies also stay in the full context file; never silently omit them.
    metadata = json.dumps(context, ensure_ascii=False, indent=2)
    if len(metadata) > MAX_DIFF_CHARS:
        metadata = "大きな入力のため context.json の全文を必ず読んでください。"
    return string.Template(template).substitute(
        parent=request["parent_issue"],
        run_id=request["review_run_id"],
        request=json.dumps(request, ensure_ascii=False, indent=2),
        context=metadata,
        diffs="\n\n".join(parts),
    )
