#!/usr/bin/env python3
"""Stateful fake GitHub for rewrite/apply, including edits between read and PATCH."""

import json
import os
import sys
from pathlib import Path
from typing import Any, NoReturn

root = Path(os.environ["REWRITE_FAKE_ROOT"])
args = sys.argv[1:]
state_file = root / "github.json"
state = json.loads(state_file.read_text())
method = args[args.index("--method") + 1]
endpoint = next(x for x in args if x.startswith("repos/") or x == "graphql")
payload: Any = json.load(sys.stdin) if method != "GET" else None
with (root / "gh-calls.jsonl").open("a") as trace:
    trace.write(
        json.dumps({"method": method, "endpoint": endpoint, "payload": payload}) + "\n"
    )


def save() -> None:
    state_file.write_text(json.dumps(state))


def response(value: Any, code: int = 200) -> NoReturn:
    save()
    if "--include" in args:
        print(f"HTTP/1.1 {code} Result\n")
    print(json.dumps(value, ensure_ascii=False))
    raise SystemExit(0 if code < 300 else 1)


def edit(body: str, login: str) -> None:
    if not state["edits"]:
        state["edits"].append(
            {
                "id": "creation",
                "editedAt": state["issue"]["created_at"],
                "deletedAt": None,
                "editor": state["issue"]["user"],
            }
        )
    state["issue"]["body"] = body
    state["edits"].append(
        {
            "id": "edit-" + str(len(state["edits"]) + 1),
            "editedAt": "2026-10-31T00:01:00Z",
            "deletedAt": None,
            "editor": {"login": login},
        }
    )


if state.get("fail_endpoint") == endpoint:
    response({"message": "private-error ghs_" + "a" * 36}, 403)
if endpoint.endswith("/actions/variables/ISSUE_REWRITE_ENABLED_SINCE"):
    response(
        {"name": "ISSUE_REWRITE_ENABLED_SINCE", "value": state["enabled_since"]},
        state.get("variable_status", 200),
    )
if endpoint == "repos/o/r/issues/1" and method == "GET":
    state["issue_reads"] = state.get("issue_reads", 0) + 1
    mutation = state.get("mutation")
    if mutation and state["issue_reads"] == mutation["read"]:
        if "body" in mutation:
            edit(mutation["body"], "human")
        if "comments" in mutation:
            state["comments"] = mutation["comments"]
    response(state["issue"])
if endpoint.startswith("repos/o/r/issues/1/comments"):
    page = int(endpoint.split("page=")[-1])
    response(state["comments"][(page - 1) * 100 : page * 100])
if endpoint == "graphql":
    state["history_reads"] = state.get("history_reads", 0) + 1
    after = payload["variables"]["after"]
    start = int(after) if after is not None else 0
    rows = state["edits"][start : start + 100]
    end = start + len(rows)
    response(
        {
            "data": {
                "viewer": {"login": "github-actions[bot]"},
                "repository": {
                    "issue": {
                        "body": state["issue"]["body"],
                        "createdAt": state["issue"]["created_at"],
                        "author": state["issue"]["user"],
                        "includesCreatedEdit": bool(state["edits"]),
                        "userContentEdits": {
                            "nodes": rows,
                            "totalCount": len(state["edits"]),
                            "pageInfo": {
                                "hasNextPage": end < len(state["edits"]),
                                "endCursor": str(end),
                            },
                        },
                    }
                },
            }
        }
    )
if endpoint == "repos/o/r/issues/1" and method == "PATCH":
    if "body" in payload:
        if state.get("interference") in {"before", "same_author"}:
            editor = (
                "github-actions[bot]"
                if state["interference"] == "same_author"
                else "human"
            )
            edit("concurrent edit", editor)
        if state.get("interference") != "missing_history":
            edit(payload["body"], "github-actions[bot]")
        else:
            state["issue"]["body"] = payload["body"]
        patch_response = json.loads(json.dumps(state["issue"]))
        if state.get("interference") == "after":
            edit("human's latest body", "human")
    if "labels" in payload:
        state["issue"]["labels"] = [{"name": name} for name in payload["labels"]]
    response(patch_response if "body" in payload else state["issue"])
if endpoint.endswith("/issue-rewrite.yml/dispatches") and method == "POST":
    state.setdefault("dispatches", []).append(payload)
    response(None, 204)
raise AssertionError((args, payload))
