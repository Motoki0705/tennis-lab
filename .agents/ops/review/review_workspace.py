"""Read-only review inputs: all PR heads plus caller-owned full diff files."""

from __future__ import annotations

import json
import os
import string
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from parent.snapshot import git
from parent_review_policy import BaseAdvance, hex_value, snapshots, validate_result


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


def verify_base_advances(
    result: Mapping[str, Any],
    current: Sequence[Mapping[str, Any]],
    merged_prs: Sequence[Mapping[str, Any]],
) -> tuple[BaseAdvance, ...]:
    """Verify every base step against authenticated GitHub GET /pulls/{n} data.

    The caller fetches same-repository peer PRs directly from GitHub, never from
    comments or the PR's payload. Every first-parent commit must be the merge
    commit of a different PR in this run. Unknown commits and rewinds fail closed.
    """
    validate_result(dict(result))
    reviewed = {p["number"]: p for p in result["request"]["prs"]}
    latest = snapshots(list(current))
    merges: dict[str, Mapping[str, Any]] = {}
    repo = result["request"]["repo"].lower()
    for pr in merged_prs:
        n = pr["number"]
        if pr.get("merged") is not True:
            continue
        if (
            n not in reviewed
            or pr["base"]["repo"]["full_name"].lower() != repo
            or pr["base"]["ref"] != reviewed[n]["base_ref"]
            or pr["head"]["ref"] != reviewed[n]["branch"]
        ):
            raise ValueError("base advance evidence is outside this review run/base")
        commit = hex_value(pr.get("merge_commit_sha"), 40)
        if commit in merges:
            raise ValueError("ambiguous merged PR evidence")
        merges[commit] = pr
    advances: set[BaseAdvance] = set()
    for entry in latest:
        original = reviewed[entry["number"]]
        if original["base_ref"] != entry["base_ref"]:
            raise ValueError("retargeted base cannot inherit coverage")
        before, after = original["base_sha"], entry["base_sha"]
        if before == after:
            continue
        git("merge-base", "--is-ancestor", before, after)
        history = (
            git(
                "rev-list",
                "--first-parent",
                "--parents",
                "--reverse",
                f"{before}..{after}",
            )
            .decode()
            .splitlines()
        )
        tip = before
        for row in history:
            commit, *parents = row.split()
            peer = merges.get(commit)
            if (
                not parents
                or parents[0] != tip
                or peer is None
                or peer["number"] == entry["number"]
                or peer["base"]["ref"] != entry["base_ref"]
            ):
                raise ValueError(
                    "base advance contains an unexplained first-parent commit"
                )
            advances.add(BaseAdvance(peer["number"], entry["base_ref"], tip, commit))
            tip = commit
        if tip != after:
            raise ValueError("incomplete first-parent base advance evidence")
    return tuple(sorted(advances, key=lambda step: (step.base_ref, step.before_sha)))


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


def verify_rebased_snapshot(
    original: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    """Only base-touched files may change a PR's own per-file delta."""
    if (
        original["number"] != current["number"]
        or original["branch"] != current["branch"]
    ):
        return False
    base_paths = set(changed_paths(original["base_sha"], current["base_sha"]))

    def deltas(entry: Mapping[str, Any]) -> dict[bytes, bytes]:
        raw = git(
            "diff",
            "--raw",
            "--no-abbrev",
            "--no-renames",
            "--no-ext-diff",
            "--no-textconv",
            "-z",
            entry["merge_base_sha"],
            entry["head_sha"],
            "--",
        )
        parts = raw.split(b"\0")
        if parts[-1] != b"" or (len(parts) - 1) % 2:
            raise ValueError("invalid per-file git delta")
        return dict(zip(parts[1:-1:2], parts[:-1:2], strict=True))

    before, after = deltas(original), deltas(current)
    for path in before.keys() | after.keys():
        if before.get(path) != after.get(path) and os.fsdecode(path) not in base_paths:
            return False
    return True
