"""One broad review result: approve or major findings. No merge authorization."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from parent.policy import nonempty, object_value
from parent.state import unique_object

PREFIX = "<!-- parent-review:v1 "


def load_json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=unique_object)


def parse_output(text: str, numbers: Sequence[int]) -> dict[str, Any]:
    value = object_value(load_json(text))
    if set(value) != {"verdict", "summary", "major"}:
        raise ValueError("review requires verdict, summary and major")
    nonempty(value["summary"], "review summary")
    if value["verdict"] not in {"approve", "major"} or not isinstance(
        value["major"], list
    ):
        raise ValueError("review must approve or list major findings")
    for finding in value["major"]:
        finding = object_value(finding)
        if set(finding) != {"prs", "title", "detail"}:
            raise ValueError("major finding requires prs, title and detail")
        prs = finding["prs"]
        if (
            not isinstance(prs, list)
            or not prs
            or any(type(n) is not int or n not in numbers for n in prs)
            or len(set(prs)) != len(prs)
        ):
            raise ValueError("major finding refers to unknown or duplicate PRs")
        nonempty(finding["title"], "finding title")
        nonempty(finding["detail"], "finding detail")
    if (value["verdict"] == "approve") != (not value["major"]):
        raise ValueError("review verdict and major findings disagree")
    return value


def marker(record: Mapping[str, Any]) -> str:
    data = json.dumps(dict(record), ensure_ascii=True, sort_keys=True)
    return PREFIX + data.replace("<", r"\u003c").replace(">", r"\u003e") + " -->"


def records(
    comments: Sequence[Mapping[str, Any]], trusted_login: str
) -> list[dict[str, Any]]:
    """Read old records for history without replaying their merge decisions."""
    result = []
    for comment in comments:
        if object_value(comment.get("user")).get("login") != trusted_login:
            continue
        line = (comment.get("body") or "").split("\n", 1)[0].rstrip("\r")
        if line.startswith("<!-- parent-review:v") and not line.startswith(PREFIX):
            raise ValueError("unknown parent-review marker version")
        if not line.startswith(PREFIX):
            continue
        if not line.endswith(" -->"):
            raise ValueError("malformed parent-review marker")
        row = object_value(load_json(line[len(PREFIX) : -4]))
        if (
            type(row.get("schema_version")) is not int
            or row["schema_version"] not in {1, 2}
            or row.get("kind")
            not in {
                "result",
                "pr",
                "resolution",
                "pending",
                "adoption",
                "evidence",
                "broad_review",
            }
        ):
            raise ValueError("unknown parent-review history record")
        result.append(row)
    return result
