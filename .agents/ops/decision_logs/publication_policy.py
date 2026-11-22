"""Separate, fail-closed exception for new envelopes and regenerated READMEs."""

from __future__ import annotations

from typing import Any

from worker_logs.journal import require
from worker_logs.schema import integer

from .navigation import EVENT, event_path, generate, records


def changes(raw: bytes) -> dict[str, str]:
    """Parse git diff --raw --no-renames -z, checking both modes and all paths."""
    entries = raw.split(b"\0")
    require(entries[-1:] == [b""] and len(entries) % 2 == 1, "unsafe_diff")
    found = {}
    for i in range(0, len(entries) - 1, 2):
        header = entries[i].decode("ascii").split()
        require(len(header) == 5, "unsafe_diff")
        status = header[4]
        require(
            status in {"A", "M"}
            and header[1] == "100644"
            and header[0] == (":000000" if status == "A" else ":100644"),
            "unsafe_diff",
        )
        name = entries[i + 1].decode("utf-8")
        require(name not in found, "unsafe_diff")
        found[name] = status
    return found


def verify(
    before: dict[str, bytes],
    after: dict[str, bytes],
    diff: dict[str, str],
    *,
    parent: int,
    expected: dict[str, bytes],
) -> None:
    """No code, worker-log edits, old event edits, or handwritten README text."""
    require(bool(diff), "unsafe_diff")
    old_records = {event_path(r): before[event_path(r)] for r in records(before)}
    new_records = {event_path(r): after[event_path(r)] for r in records(after)}
    require(
        all(new_records.get(n) == raw for n, raw in old_records.items()),
        "record_already_exists",
    )
    require(
        all(after.get(n) == raw for n, raw in expected.items()),
        "committed_content_changed",
    )
    added = set(new_records) - set(old_records)
    require(added == set(expected) - set(old_records), "unsafe_diff")
    generated = generate(after)
    require(
        all(after.get(n) == raw for n, raw in generated.items()),
        "prepared_content_changed",
    )
    for name, status in diff.items():
        if name in added:
            match = EVENT.fullmatch(name)
            require(
                status == "A" and match is not None and int(match[1]) == parent,
                "unsafe_diff",
            )
        else:
            require(name in generated and name.endswith("/README.md"), "unsafe_diff")
        require(status == ("M" if name in before else "A"), "unsafe_diff")
    needed = {
        name
        for name, raw in {**expected, **generated}.items()
        if before.get(name) != raw
    }
    require(set(diff) == needed, "unsafe_diff")


def pr_changes(actual: list[Any], expected: dict[str, str]) -> None:
    require(len(actual) == len(expected), "unsafe_diff")
    found = {}
    for row in actual:
        require(
            isinstance(row, dict)
            and isinstance(row.get("filename"), str)
            and "previous_filename" not in row
            and row["filename"] not in found,
            "unsafe_diff",
        )
        found[row["filename"]] = row.get("status")
    require(
        found == {n: {"A": "added", "M": "modified"}[s] for n, s in expected.items()},
        "unsafe_diff",
    )


def identity(
    pr: dict[str, Any], row: dict[str, Any], head: str, count: int | None = None
) -> None:
    repo = row["repo"]
    require(
        integer(pr.get("number"))
        and pr.get("html_url") == f"https://github.com/{repo}/pull/{pr['number']}"
        and pr.get("base", {}).get("ref") == "main"
        and pr.get("base", {}).get("repo", {}).get("full_name", "").casefold()
        == repo.casefold()
        and pr.get("head", {}).get("repo", {}).get("full_name", "").casefold()
        == repo.casefold()
        and pr.get("head", {}).get("ref") == row["branch"]
        and pr.get("head", {}).get("sha") == head
        and pr.get("draft") is False
        and pr.get("body") == body(row)
        and (
            count is None
            or type(pr.get("changed_files")) is int
            and pr["changed_files"] == count
        ),
        "pr_identity_changed",
    )


def body(row: dict[str, Any]) -> str:
    return (
        f"<!-- decision-log:v1 parent={row['parent']} batch={row['key']} repair={len(row['superseded_prs'])} -->\n\n"
        f"Refs #{row['parent']}\n\n"
        "検査済みの判断記録と、HEADの正本から決定的に生成したREADME。"
        "専用入口が追加専用差分・branch・marker・HEAD・mainへの保存を照合する。\n"
    )
