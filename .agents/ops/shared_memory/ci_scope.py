"""Decide whether a CI event contains only allowed shared-memory changes."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .git import Git, diff_changes, is_sha


def _revision(value: object) -> str:
    if not isinstance(value, str) or not is_sha(value):
        raise ValueError("event did not supply a full commit SHA")
    return value


def memory_only(
    git: Git, event: str, revision: str, payload: dict[str, object]
) -> bool:
    head = _revision(revision)
    if git.run("rev-parse", "HEAD").decode().strip() != head:
        raise ValueError("scope checkout differs from GITHUB_SHA")
    if event == "push":
        if payload.get("after") != head or payload.get("ref") != "refs/heads/main":
            raise ValueError("push does not identify the checked-out main commit")
        base = _revision(payload.get("before"))
    elif event == "pull_request":
        pr = payload.get("pull_request")
        if not isinstance(pr, dict):
            raise ValueError("missing pull request event")
        base = _revision(pr["base"]["sha"])
        topic = _revision(pr["head"]["sha"])
        parents = git.run("rev-list", "--parents", "-n", "1", head).decode().split()
        if parents != [head, base, topic]:
            raise ValueError("pull request merge commit does not match the event")
    else:
        return False
    raw = git.run(
        "diff",
        "--raw",
        "--abbrev=40",
        "--no-ext-diff",
        "--ignore-submodules=none",
        "--no-renames",
        "-z",
        base,
        head,
        "--",
    )
    return bool(diff_changes(raw))


def main() -> None:
    event = os.environ["GITHUB_EVENT_NAME"]
    exempt = False
    if event in {"push", "pull_request"}:
        try:
            payload = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes())
            if not isinstance(payload, dict):
                raise ValueError("event payload is not an object")
            exempt = memory_only(
                Git(Path.cwd()), event, os.environ["GITHUB_SHA"], payload
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            print(
                f"shared-memory exception did not apply; running full CI: {exc}",
                file=sys.stderr,
            )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        print(f"shared_memory_only={'true' if exempt else 'false'}", file=output)


if __name__ == "__main__":
    main()
