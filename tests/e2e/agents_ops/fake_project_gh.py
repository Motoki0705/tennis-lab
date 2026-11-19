"""Disposable GitHub state for the Projects CLI tests; never contacts GitHub."""

from __future__ import annotations

import copy
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    args = sys.argv[1:]
    state_file = Path(os.environ["PROGRESS_GH_STATE"])
    state: dict[str, Any] = json.loads(state_file.read_text())
    graphql = args[:2] == ["api", "graphql"]
    rest_write = "--method" in args and args[args.index("--method") + 1] == "POST"
    payload = json.load(sys.stdin) if graphql or rest_write else None
    if os.environ.get("PROGRESS_ASSERT_OPERATION_PARENT") and (graphql or rest_write):
        import fcntl

        digest = importlib.import_module("parent.policy").digest

        directory = (
            Path(os.environ["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
        )
        for key in (
            f"o/r#{os.environ['PROGRESS_ASSERT_OPERATION_PARENT']}",
            "project:motoki0705#3",
        ):
            with (directory / f"operation-{digest(key)}.lock").open("a") as probe:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    raise AssertionError(f"operation lock missing: {key}")
    with Path(os.environ["PROGRESS_GH_CALLS"]).open("a") as stream:
        stream.write(json.dumps({"args": args, "payload": payload}) + "\n")

    def save() -> None:
        state_file.write_text(json.dumps(state))

    def fail() -> None:
        print("injected fake gh failure", file=sys.stderr)
        raise SystemExit(1)

    def respond(value: Any) -> None:
        value = copy.deepcopy(value)
        if (
            payload is not None
            and os.environ.get("PROGRESS_POST_WRITE_CHANGE")
            and (rest_write or graphql and payload["query"].startswith("mutation"))
        ):
            state["post_write_pending"] = True
            save()
        if "--include" in args:
            print("HTTP/1.1 200 OK\n")
        print(json.dumps(value))
        raise SystemExit(0)

    def connection(
        rows: list[dict[str, Any]], after: str | None = None
    ) -> dict[str, Any]:
        start = int(after or 0)
        end = start + 100
        value: dict[str, Any] = {
            "nodes": rows[start:end],
            "totalCount": len(rows),
            "pageInfo": {
                "hasNextPage": end < len(rows),
                "endCursor": str(min(end, len(rows))),
            },
        }
        if os.environ.get("PROGRESS_BAD_PAGE"):
            value["totalCount"] += 1
            value["pageInfo"]["hasNextPage"] = False
        return value

    def view(raw: dict[str, Any]) -> dict[str, Any]:
        return {
            **raw,
            "configuration": {
                "visibleFields": connection([{"id": n} for n in raw["visible"]])
            },
            "groupByFields": connection([{"name": n} for n in raw["groups"]]),
            "verticalGroupByFields": connection([{"name": n} for n in raw["columns"]]),
        }

    def item(raw: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(raw)
        for key, name in (
            ("stage", "段階"),
            ("wait", "待ち理由"),
            ("next", "次の作業"),
            ("acceptance", "親の受入状態"),
        ):
            value = raw["values"].get(name)
            result[key] = (
                None
                if value is None
                else {"name" if key in {"stage", "acceptance"} else "text": value}
            )
        return result

    if not graphql:
        if rest_write:
            assert (
                f"users/{state['project']['owner']['login']}/projectsV2/3/views" in args
            )
            assert "X-GitHub-Api-Version: 2026-03-10" in args
            assert payload is not None
            if os.environ.get("PROGRESS_FAIL_WRITE") == "createProjectView":
                fail()
            response = {
                **payload,
                **{
                    key: sorted(payload[key])
                    for key in ("visible_fields", "group_by", "vertical_group_by")
                },
            }
            by_id = {raw["databaseId"]: raw for raw in state["fields"]}
            row = {
                "id": "V" + str(len(state["views"]) + 1),
                "number": len(state["views"]) + 1,
                "name": payload["name"],
                "layout": "BOARD_LAYOUT",
                "filter": payload["filter"],
                "visible": [by_id[n]["id"] for n in response["visible_fields"]],
                "groups": [by_id[n]["name"] for n in response["group_by"]],
                "columns": [by_id[n]["name"] for n in response["vertical_group_by"]],
            }
            state["views"].append(row)
            save()
            lost = state.setdefault("lost_responses", [])
            if (
                os.environ.get("PROGRESS_LOSE_RESPONSE") == "createProjectView"
                and "createProjectView" not in lost
            ):
                lost.append("createProjectView")
                save()
                fail()
            mismatch = os.environ.get("PROGRESS_BOARD_RESPONSE_MISMATCH")
            if mismatch:
                response[mismatch] = (
                    [] if isinstance(response[mismatch], list) else "unexpected"
                )
            respond({**response, "node_id": row["id"]})
        if args[:2] == ["repo", "view"]:
            respond({"nameWithOwner": "o/r"})
        assert args[0] == "api", args
        assert args[args.index("--method") + 1] == "GET", args
        if "user" in args:
            respond({"login": "operator"})
        endpoint = next(a for a in args if a.startswith("repos/"))
        if os.environ.get("PROGRESS_FAIL_PULLS"):
            fail()
        timeline = re.fullmatch(
            r"repos/o/r/issues/(\d+)/timeline\?per_page=100&page=(\d+)", endpoint
        )
        pull = re.fullmatch(r"repos/o/r/pulls/(\d+)", endpoint)
        if timeline:
            rows = [
                {
                    "event": "cross-referenced",
                    "created_at": "2026-10-24T00:00:00Z",
                    "source": {
                        "issue": {
                            "id": p["number"] + 10000,
                            "number": p["number"],
                            "body": p["body"],
                            "repository_url": "https://api.github.com/repos/o/r",
                            "pull_request": {"url": "pr"},
                        }
                    },
                }
                for p in state["pulls"]
                if re.search(r"#" + timeline[1] + r"\b", p["body"])
            ]
            page = int(timeline[2])
            respond(rows[(page - 1) * 100 : page * 100])
        if pull:
            if (
                os.environ.get("PROGRESS_PR_RACE")
                and pull[1] == "99"
                and not state.get("pr_raced")
            ):
                old = copy.deepcopy(
                    next(p for p in state["pulls"] if p["number"] == int(pull[1]))
                )
                state["pulls"][0]["state"] = "closed"
                state["pr_raced"] = True
                save()
                respond(old)
            respond(next(p for p in state["pulls"] if p["number"] == int(pull[1])))
        assert endpoint.startswith("repos/o/r/pulls?state=all"), endpoint
        if os.environ.get("PROGRESS_FAIL_PULLS"):
            fail()
        match = re.search(r"[?&]page=(\d+)", endpoint)
        assert match is not None
        page = int(match[1])
        respond(state["pulls"][(page - 1) * 100 : page * 100])
    assert payload is not None
    query, variables = payload["query"], payload["variables"]
    if not query.startswith("mutation"):
        match = re.search(r"query (\w+)", query)
        assert match is not None
        operation = match[1]
        if os.environ.get("PROGRESS_FAIL_READ") == operation:
            fail()
        if os.environ.get("PROGRESS_PARTIAL_ERROR") == operation:
            respond({"data": {}, "errors": [{"message": "partial failure"}]})
        if operation == "ProgressProject":
            if os.environ.get("PROGRESS_SETUP_SNAPSHOT_RACE"):
                state["project_reads"] = state.get("project_reads", 0) + 1
                if state["project_reads"] == 2:
                    state["project"]["title"] += " changed"
                save()
            change = os.environ.get("PROGRESS_POST_WRITE_CHANGE")
            if (
                change
                and state.pop("post_write_pending", False)
                and not state.get("post_write_changed")
            ):
                if change == "project":
                    state["project"]["title"] += " changed"
                elif change == "issue":
                    state["issues"]["1204"]["title"] += " changed"
                elif change == "pr":
                    state["pulls"][0]["state"] = "closed"
                elif change == "field":
                    state["fields"][-1]["id"] += "-replacement"
                elif change == "item":
                    state["items"][0]["id"] += "-replacement"
                elif change == "view":
                    state["views"][-1]["id"] += "-replacement"
                else:
                    raise AssertionError(change)
                state["post_write_changed"] = True
                save()
            assert (
                variables["owner"].casefold()
                == state["project"]["owner"]["login"].casefold()
            )
            assert variables["number"] == 3
            respond({"data": {"user": {"projectV2": state["project"]}}})
        for key in ("fields", "views", "items", "workflows"):
            if operation == f"Progress{key.title()}":
                assert variables["id"] == "P"
                rows = state[key]
                if key == "views":
                    rows = [view(row) for row in rows]
                elif key == "items":
                    rows = [item(row) for row in rows]
                respond(
                    {
                        "data": {
                            "node": {
                                "id": "P",
                                key: connection(rows, variables["after"]),
                            }
                        }
                    }
                )
        assert operation == "ProgressIssue", operation
        n = str(variables["number"])
        assert (variables["owner"], variables["repo"]) == ("o", "r")
        if os.environ.get("PROGRESS_RACE") and n == "1204":
            state["root_reads"] = state.get("root_reads", 0) + 1
            if state["root_reads"] == 2:
                state["issues"][n]["title"] += " (changed)"
            save()
        raw = copy.deepcopy(state["issues"][n])
        raw["subIssues"] = connection(
            [
                {
                    "id": state["issues"][str(child)]["id"],
                    "number": child,
                    "repository": {"nameWithOwner": "o/r"},
                }
                for child in raw.pop("children")
            ],
            variables["after"],
        )
        raw["assignees"] = connection(
            [{"id": "U" + login, "login": login} for login in raw["assignees"]]
        )
        raw["labels"] = connection(
            [{"id": "L" + name, "name": name} for name in raw["labels"]]
        )
        respond({"data": {"repository": {"issue": raw}}})

    match = re.search(r"\{ (\w+)\(input", query)
    assert match is not None
    operation = match[1]
    data = variables["input"]
    assert operation in {
        "createProjectV2Field",
        "createProjectV2View",
        "updateProjectV2View",
        "addProjectV2ItemById",
        "updateProjectV2ItemFieldValue",
        "clearProjectV2ItemFieldValue",
    }, operation
    if os.environ.get("PROGRESS_FAIL_WRITE") == operation:
        fail()
    if operation == "createProjectV2Field":
        row = {
            "id": "F" + str(len(state["fields"])),
            "databaseId": 1000 + len(state["fields"]),
            "name": data["name"],
            "dataType": data["dataType"],
        }
        if data["dataType"] == "SINGLE_SELECT":
            row["options"] = [
                {"id": row["id"] + "O" + str(i), "name": option["name"]}
                for i, option in enumerate(data["singleSelectOptions"])
            ]
        state["fields"].append(row)
        output = {"projectV2Field": row}
    elif operation == "createProjectV2View":
        row = {
            "id": "V" + str(len(state["views"]) + 1),
            "number": len(state["views"]) + 1,
            "name": data["name"],
            "layout": data["layout"],
            "filter": None,
            "visible": data["configuration"]["visibleFieldIds"],
            "groups": [],
            "columns": ["Status"],
        }
        state["views"].append(row)
        output = {"projectV2View": {"id": row["id"]}}
    elif operation == "updateProjectV2View":
        row = next(row for row in state["views"] if row["id"] == data["viewId"])
        row.update(
            layout=data["layout"],
            filter=data["filter"],
            visible=data["configuration"]["visibleFieldIds"],
        )
        output = {"projectV2View": {"id": row["id"]}}
    elif operation == "addProjectV2ItemById":
        issue = next(
            raw for raw in state["issues"].values() if raw["id"] == data["contentId"]
        )
        assert not any(raw["content"]["id"] == issue["id"] for raw in state["items"])
        row = {
            "id": "ITEM" + str(issue["number"]),
            "isArchived": False,
            "content": {
                "__typename": "Issue",
                "id": issue["id"],
                "number": issue["number"],
                "url": issue["url"],
                "repository": {"nameWithOwner": "o/r"},
            },
            "values": {},
        }
        state["items"].append(row)
        output = {"item": {"id": row["id"]}}
    else:
        row = next(raw for raw in state["items"] if raw["id"] == data["itemId"])
        field = next(raw for raw in state["fields"] if raw["id"] == data["fieldId"])
        if operation == "clearProjectV2ItemFieldValue":
            value = None
        elif field["dataType"] == "SINGLE_SELECT":
            value = next(
                raw["name"]
                for raw in field["options"]
                if raw["id"] == data["value"]["singleSelectOptionId"]
            )
        else:
            value = data["value"]["text"]
        row["values"][field["name"]] = value
        output = {"projectV2Item": {"id": row["id"]}}
    if not os.environ.get("PROGRESS_DROP_WRITE"):
        save()
    lost = state.setdefault("lost_responses", [])
    if os.environ.get("PROGRESS_LOSE_RESPONSE") == operation and operation not in lost:
        lost.append(operation)
        save()
        fail()
    respond({"data": {operation: output}})


if __name__ == "__main__":
    main()
