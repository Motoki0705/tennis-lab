"""Lossless, selectively readable judgment inputs, separate from the digest snapshot."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from parent.state import Store

from .collect import related
from .schema import guard

TEXT_PART_CHARS = 8000
TEXT_SEGMENT_CHARS = 160
COMMENT_PAGE_CHARS = 12000


def _write_index(path: Path, index: dict[str, Any]) -> None:
    # Keep each summary on one physical line, so Read can show the whole index
    # without spending thousands of lines on small arrays and nested objects.
    def encode(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    fields = []
    for key, value in index.items():
        if key in {"issues", "pulls", "owners"}:
            rows = [f"    {encode(n)}:{encode(row)}" for n, row in value.items()]
            encoded = "{\n" + ",\n".join(rows) + "\n  }"
        else:
            encoded = encode(value)
        fields.append(f"  {encode(key)}:{encoded}")
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write("{\n" + ",\n".join(fields) + "\n}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_agent_input(snapshot: dict[str, Any], directory: Path) -> Path:
    directory.mkdir(mode=0o700, exist_ok=False)
    for name in ("issues", "pulls"):
        (directory / name).mkdir(mode=0o700)

    def write(relative: str, value: Any) -> str:
        Store.write_json(directory / relative, value)
        return relative

    def text_value(text: str, prefix: str) -> dict[str, Any]:
        def segments(part: str) -> dict[str, Any]:
            # Long physical lines are truncated by some Read tools. Preserve the
            # exact string as short JSON segments, including original newlines.
            return {
                "segments": [
                    part[i : i + TEXT_SEGMENT_CHARS]
                    for i in range(0, len(part), TEXT_SEGMENT_CHARS)
                ]
            }

        if len(text) <= TEXT_PART_CHARS:
            return segments(text)
        files: list[str] = []
        for i in range(0, len(text), TEXT_PART_CHARS):
            files.append(
                write(
                    f"{prefix}-{i // TEXT_PART_CHARS + 1:04d}.json",
                    segments(text[i : i + TEXT_PART_CHARS]),
                )
            )
        return {"files": files}

    def write_comments(comments: list[dict[str, Any]], prefix: str) -> list[str]:
        page: list[dict[str, Any]] = []
        files: list[str] = []
        page_chars = 0
        for comment in comments:
            item = {
                **comment,
                "body": text_value(
                    comment["body"], f"{prefix}/comment-{comment['id']}"
                ),
            }
            size = len(json.dumps(item, ensure_ascii=False))
            if page and page_chars + size > COMMENT_PAGE_CHARS:
                files.append(
                    write(f"{prefix}/comments-{len(files) + 1:04d}.json", page)
                )
                page, page_chars = [], 0
            page.append(item)
            page_chars += size
        if page:
            files.append(write(f"{prefix}/comments-{len(files) + 1:04d}.json", page))
        return files

    selected: set[int] = set()
    for n in snapshot["subjects"]:
        selected.update(related(snapshot, n))
    pulls = [p for p in snapshot["pulls"] if set(p["issues"]) & selected]
    owners = {
        n: row
        for n, row in snapshot["owners"].items()
        if int(n) in selected or set(row["tasks"]) & {str(s) for s in selected}
    }
    index: dict[str, Any] = {
        "schema_version": 1,
        "repo": snapshot["repo"],
        "publisher": snapshot["publisher"],
        "input_sha256": snapshot["input_sha256"],
        "code_commit": snapshot["code"]["commit"],
        "subjects": snapshot["subjects"],
        "issues": {},
        "pulls": {},
        "owners": owners,
        "issue_content_complete": True,
        "omissions": [
            {
                "kind": "unrelated_pull_requests",
                "count": len(snapshot["pulls"]) - len(pulls),
                "reason": "subjectsと正式親子・依存・提案出典の閉包にclosing指示がないPR。",
            },
            {
                "kind": "code_tree",
                "reason": "code_commitのclean checkoutがcwd。必要なtrackedファイルを直接調べる。",
            },
            {
                "kind": "execution_inputs",
                "reason": "controller内部の実行根拠用射影。元の本文・コメント・関係は全件収録。",
            },
        ],
    }
    for n in sorted(selected):
        raw = snapshot["issues"][str(n)]
        prefix = f"issues/{n}"
        (directory / prefix).mkdir(mode=0o700)
        summary = {k: v for k, v in raw.items() if k not in {"body", "comments"}}
        if n in snapshot["subjects"]:
            summary["mechanical_guard"] = guard(snapshot, n)
        summary["pulls"] = [p["number"] for p in pulls if n in p["issues"]]
        summary["file"] = write(
            f"{prefix}/issue.json",
            {
                **summary,
                "body": text_value(raw["body"], f"{prefix}/body"),
                "comment_files": write_comments(raw["comments"], prefix),
                "content_complete": True,
                "omissions": [],
            },
        )
        index["issues"][str(n)] = summary
    for pull in pulls:
        n = pull["number"]
        (directory / f"pulls/{n}").mkdir(mode=0o700)
        index["pulls"][str(n)] = {
            "issues": pull["issues"],
            "file": write(
                f"pulls/{n}/pull.json",
                {**pull, "body": text_value(pull["body"], f"pulls/{n}/body")},
            ),
        }
    _write_index(directory / "index.json", index)
    return directory / "index.json"
