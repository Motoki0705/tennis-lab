"""Disposable GitHub state for the Projects CLI tests; never contacts GitHub."""

from __future__ import annotations

import copy
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
    with Path(os.environ["PROGRESS_GH_CALLS"]).open("a") as stream:
        stream.write(json.dumps({"args": args, "payload": payload}) + "\n")

    def save() -> None:
        state_file.write_text(json.dumps(state))

    def fail() -> None:
        print("injected fake gh failure", file=sys.stderr)
        raise SystemExit(1)

    def respond(value: Any) -> None:
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
