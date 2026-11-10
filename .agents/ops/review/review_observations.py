"""Versioned discussion observations; missing old observations stay unknown."""

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from typing import Any

from parent.execution import comment_snapshot, instruction_comments, requirement_body
from parent.policy import digest, number, object_value
from parent.session import ParentSession


def discussion(
    session: ParentSession, kind: str, raw: Mapping[str, Any]
) -> dict[str, Any]:
    n = number(raw["number"])
    groups = {"discussion": session.gh.comments(n)}
    if kind == "pr":
        for name, endpoint in (("inline", "comments"), ("review", "reviews")):
            groups[name] = session.gh.rows(
                f"repos/{session.gh.repo}/pulls/{n}/{endpoint}"
            )
    comments = []
    excluded = []
    for name, rows in groups.items():
        kept = {c["id"] for c in instruction_comments(session, rows)}
        for row in rows:
            if row["id"] in kept:
                comments.append(
                    {
                        "kind": name,
                        **comment_snapshot(row),
                        **{
                            key: row[key]
                            for key in (
                                "state",
                                "path",
                                "line",
                                "start_line",
                                "side",
                                "commit_id",
                                "in_reply_to_id",
                            )
                            if key in row
                        },
                    }
                )
            else:
                excluded.append({"kind": name, "id": row["id"]})
    return {
        "kind": kind,
        "number": n,
        "title": raw["title"],
        "body": raw.get("body") or "",
        "comments": sorted(comments, key=lambda c: (c["kind"], c["id"])),
        "excluded_records": sorted(excluded, key=lambda c: (c["kind"], c["id"])),
    }


def observe(
    session: ParentSession,
    issues: Sequence[Mapping[str, Any]],
    prs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "items": {
            f"{kind}:{raw['number']}": discussion(session, kind, raw)
            for kind, rows in (("issue", issues), ("pr", prs))
            for raw in rows
        },
    }


def baseline(context: Mapping[str, Any], session: ParentSession) -> dict[str, Any]:
    if "review_observations" in context:
        saved = object_value(context["review_observations"])
        if type(saved.get("schema_version")) is not int or saved["schema_version"] != 1:
            raise ValueError("unsupported review observation version")
        return dict(object_value(saved["items"]))
    # Older context recorded bodies and parent comments, but no complete child
    # or PR discussions. Do not manufacture an empty comment baseline.
    result = {}
    for kind, rows in (
        ("issue", [context["parent"], *context["children"]]),
        ("pr", context["prs"]),
    ):
        for raw in rows:
            comments = None
            if (
                kind == "issue"
                and raw["number"] == session.parent
                and "parent_comments" in context
            ):
                comments = [
                    {"kind": "discussion", **c}
                    for c in instruction_comments(session, context["parent_comments"])
                ]
            result[f"{kind}:{raw['number']}"] = {
                "kind": kind,
                "number": raw["number"],
                "title": raw["title"],
                "body": raw.get("body") or "",
                "comments": comments,
            }
    return result


def text_difference(before: str, after: str) -> dict[str, Any]:
    return {
        "before_sha256": digest(before),
        "after_sha256": digest(after),
        "changed": before != after,
        "diff": "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile="reviewed",
                tofile="current",
            )
        ),
    }


def compare_observation(
    before: Mapping[str, Any] | None, after: Mapping[str, Any]
) -> dict[str, Any]:
    if before is None:
        return {"status": "baseline_unavailable"}
    if (before["kind"], before["number"]) != (after["kind"], after["number"]):
        raise ValueError("discussion observation identity mismatch")
    result: dict[str, Any] = {
        "status": "compared",
        "title": text_difference(before["title"], after["title"]),
        "body": text_difference(before["body"], after["body"]),
        "requirements": text_difference(
            requirement_body(before["body"]), requirement_body(after["body"])
        ),
    }
    if before.get("comments") is None:
        result["comments"] = {"status": "baseline_unavailable"}
        return result

    def index(
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[tuple[str, int], Mapping[str, Any]]:
        entries = {(c["kind"], number(c["id"])): c for c in rows}
        if len(entries) != len(rows):
            raise ValueError("duplicate discussion observation")
        return entries

    old, new = index(before["comments"]), index(after["comments"])
    result["comments"] = {
        "status": "compared",
        "added": [new[k] for k in sorted(new.keys() - old.keys())],
        "deleted": [old[k] for k in sorted(old.keys() - new.keys())],
        "edited": [
            {"kind": k[0], "id": k[1], "before": old[k], "after": new[k]}
            for k in sorted(old.keys() & new.keys())
            if old[k] != new[k]
        ],
    }
    return result
