"""Shared strict format for publisher-owned judgement display comments."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .policy import digest, number, object_value

PREFIX = "<!-- issue-start-judgement:v1 "
DETAIL_KEYS = {
    "disposition",
    "rationale",
    "facts",
    "missing_information",
    "resume_condition",
    "next_action",
    "sources",
    "target_parent",
}


def read_json(text: str) -> Any:
    from .state import unique_object

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(
        text, object_pairs_hook=unique_object, parse_constant=reject_constant
    )


def comment_body(repo: str, issue: int, detail: Mapping[str, Any]) -> str:
    payload = {
        "schema_version": 1,
        "repo": repo.casefold(),
        "issue": issue,
        "detail": dict(detail),
        "detail_sha256": digest(detail),
    }
    first = (
        PREFIX
        + json.dumps(payload, ensure_ascii=True, sort_keys=True)
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        + " -->"
    )
    target = detail["target_parent"]
    return (
        first
        + "\n\n## 自律着手の判定\n\n"
        + "\n\n".join(
            [
                f"判定: {detail['disposition']}",
                f"管理先: #{target}" if target is not None else "管理先: 対象外",
                f"理由: {detail['rationale']}",
                "確認事実:\n" + "\n".join(f"- {s}" for s in detail["facts"]),
                "不足情報: " + (" / ".join(detail["missing_information"]) or "なし"),
                f"再開条件: {detail['resume_condition']}",
                f"次の対応: {detail['next_action']}",
                "出典:\n" + "\n".join(f"- {s}" for s in detail["sources"]),
                "この記録は着手可否の判断です。人間の要件合意は記録しません。",
            ]
        )
    )


def owned_comment(
    comment: Mapping[str, Any], login: str, repo: str, issue: int
) -> bool:
    from .execution import comment_snapshot

    item = comment_snapshot(comment)
    if item["login"] != login or not item["body"].startswith(PREFIX):
        return False
    first = item["body"].splitlines()[0]
    if not first.endswith(" -->"):
        raise ValueError("malformed judgement comment marker")
    data = object_value(read_json(first[len(PREFIX) : -4]))
    if (
        set(data) != {"schema_version", "repo", "issue", "detail", "detail_sha256"}
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or data["repo"] != repo.casefold()
        or number(data["issue"]) != issue
        or set(object_value(data["detail"])) != DETAIL_KEYS
        or data["detail_sha256"] != digest(data["detail"])
        or comment_body(repo, issue, data["detail"]) != item["body"]
    ):
        raise ValueError(
            "changed or malformed judgement comment; inspect it explicitly"
        )
    return True
