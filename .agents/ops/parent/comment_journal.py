"""Check every locally attested comment, independently of its record family."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .judgement_comments import read_json
from .policy import digest, nonempty, number, object_value


def _sha(value: Any) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid comment journal digest")


def load(path: Path, *, update: bool = False) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("comment journal is a symlink")
    try:
        row = object_value(read_json(path.read_text()))
    except OSError as exc:
        raise ValueError("comment journal unavailable") from exc
    common = {"version", "repo", "issue", "publisher", "state", "comment_id"}
    if update:
        fields = common | {"parent", "from_sha256", "to_sha256"}
        valid_version = type(row.get("version")) is int and row["version"] == 1
    else:
        fields = common | {"key", "body_sha256"}
        valid_version = type(row.get("version")) is int and row["version"] in {1, 2}
        if row.get("version") == 2:
            fields |= {"parent", "source"}
    if (
        not valid_version
        or set(row) != fields
        or not isinstance(row["state"], str)
        or row["state"] not in {"sent", "bound"}
    ):
        raise ValueError("invalid comment journal schema")
    if nonempty(row["repo"], "journal repository") != row["repo"].casefold():
        raise ValueError("invalid comment journal repository")
    nonempty(row["publisher"], "journal publisher")
    number(row["issue"])
    if update or row["version"] == 2:
        number(row["parent"])
    if update:
        number(row["comment_id"])
        _sha(row["from_sha256"])
        _sha(row["to_sha256"])
        identity = [row["repo"], row["issue"], row["comment_id"]]
    else:
        nonempty(row["key"], "journal key")
        _sha(row["body_sha256"])
        if row["state"] == "bound":
            number(row["comment_id"])
        elif row["comment_id"] is not None:
            raise ValueError("sent comment journal has an unexpected ID")
        if row["version"] == 2 and (
            not isinstance(row["source"], str)
            or row["source"]
            not in {
                "legacy_parent_comment",
                "evidence_issue",
                "pr",
            }
        ):
            raise ValueError("invalid comment journal source")
        identity = [row["repo"], row["issue"], row["key"]]
    if path.stem != digest(identity):
        raise ValueError("comment journal identity changed")
    return row


def entries(directory: Path, *, update: bool = False) -> list[dict[str, Any]]:
    if directory.is_symlink():
        raise ValueError("comment journal directory is a symlink")
    if directory.exists() and not directory.is_dir():
        raise ValueError("comment journal directory is not a directory")
    if not directory.exists():
        return []
    try:
        paths = sorted(p for p in directory.iterdir() if p.suffix == ".json")
    except OSError as exc:
        raise ValueError("comment journal directory unavailable") from exc
    return [load(path, update=update) for path in paths]


def verify(
    gh: Any,
    directory: Path,
    parent: int,
    login: str,
    rows: dict[int, list[dict[str, Any]]],
    evidence_issue: int | None,
) -> None:
    """Read-only: sent intents can be reconciled from remote bytes, never resent."""
    repo = gh.repo.casefold()
    posts = [
        r
        for r in entries(directory / "comment-outbox")
        if r["repo"] == repo
        and (r.get("parent") == parent if r["version"] == 2 else r["issue"] in rows)
    ]
    updates = {
        (r["issue"], r["comment_id"]): r
        for r in entries(directory / "comment-updates", update=True)
        if r["repo"] == repo and r["parent"] == parent
    }
    for r in [*posts, *updates.values()]:
        if r["publisher"] != login:
            raise ValueError(
                "comment journal publisher differs from authenticated publisher"
            )
        issue = r["issue"]
        if "source" in r and (
            (r["source"] == "legacy_parent_comment" and issue != parent)
            or (r["source"] == "evidence_issue" and issue != evidence_issue)
            or (r["source"] == "pr" and issue in {parent, evidence_issue})
        ):
            raise ValueError("comment journal destination binding changed")
        if "from_sha256" in r and issue not in {parent, evidence_issue}:
            raise ValueError("comment update destination binding changed")
        if issue not in rows:
            rows[issue] = gh.comments(issue)

    def valid(r: dict[str, Any], c: dict[str, Any]) -> bool:
        if c.get("user", {}).get("login") != r["publisher"]:
            return False
        sha = digest(c.get("body") or "")
        u = updates.get((r["issue"], c["id"]))
        if u is None:
            return sha == r["body_sha256"]
        if "body_sha256" in r and r["body_sha256"] not in {
            u["from_sha256"],
            u["to_sha256"],
        }:
            return False
        if r.get("body_sha256") == u["to_sha256"]:
            return sha == u["to_sha256"]
        return sha in (
            {u["to_sha256"]}
            if u["state"] == "bound"
            else {u["from_sha256"], u["to_sha256"]}
        )

    for r in [*posts, *updates.values()]:
        comments = rows[r["issue"]]
        candidates = [
            c
            for c in comments
            if (r["comment_id"] is None or c["id"] == r["comment_id"]) and valid(r, c)
        ]
        if len(candidates) != 1:
            ref = f"https://github.com/{gh.repo}/issues/{r['issue']}"
            if r["comment_id"] is not None:
                ref += f"#issuecomment-{r['comment_id']}"
            raise ValueError(
                f"comment journal: published record missing, changed, duplicate, or uncertain: {ref}"
            )
