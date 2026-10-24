"""REST fake for repair publication; ci_watch reads use its existing fake."""

from __future__ import annotations

import json
import os
import re
import runpy
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs


def serve() -> int:
    args = sys.argv[1:]
    if "--hostname" in args:
        runpy.run_path(
            str(Path(__file__).with_name("fake_ci_gh.py")), run_name="__main__"
        )
        return 0
    with Path(os.environ["FAKE_FAILURE_CALLS"]).open("a") as fh:
        fh.write(json.dumps(args) + "\n")
    if not args or args[0] != "api" or "--method" not in args:
        raise ValueError(f"unsupported command: {args}")
    method = args[args.index("--method") + 1]
    endpoint = next(a for a in args if a == "user" or a.startswith("repos/"))
    path, _, query = endpoint.partition("?")
    file = Path(os.environ["FAKE_FAILURE_STATE"])
    state = json.loads(file.read_text())
    if state.get("error") == path:
        print("injected GitHub retrieval failure", file=sys.stderr)
        return 1

    def emit(value: Any, code: int = 200) -> int:
        if "--include" in args:
            print(f"HTTP/1.1 {code} Fake\n")
        print(json.dumps(value))
        return 0 if code < 300 else 1

    def page(rows: list[Any]) -> list[Any]:
        params = parse_qs(query)
        index = int(params.get("page", ["1"])[0])
        assert params.get("per_page") == ["100"]
        return rows[(index - 1) * 100 : index * 100]

    def save(value: Any) -> int:
        file.write_text(json.dumps(state))
        if os.environ.get("FAKE_FAILURE_LOSE_RESPONSE") == path:
            print("injected lost response after save", file=sys.stderr)
            return 1
        return emit(value, 201)

    payload: Any = json.load(sys.stdin) if method != "GET" else None
    if path == "user":
        assert method == "GET"
        return emit({"login": "operator"})
    if path == "repos/o/r/pulls":
        assert method == "GET"
        return emit(page(state["pulls"]))
    match = re.fullmatch(r"repos/o/r/actions/jobs/(\d+)/logs", path)
    if match:
        assert method == "GET"
        ci = json.loads(Path(os.environ["FAKE_CI_STATE"]).read_text())
        print(ci["logs"][match[1]], end="")
        return 0
    match = re.fullmatch(
        r"repos/o/r/issues(?:/(\d+))?(?:/(comments|parent|sub_issues|dependencies/blocked_by))?",
        path,
    )
    if not match:
        raise ValueError(f"unsupported endpoint: {endpoint}")
    n, tail = match.groups()
    if n is None:
        if method == "GET":
            rows = [
                i
                for i in state["issues"]
                if parse_qs(query).get("state") != ["open"] or i["state"] == "open"
            ]
            return emit(page(rows))
        assert method == "POST" and set(payload) == {"title", "body", "labels"}
        if pause := os.environ.get("FAKE_FAILURE_CREATE_PAUSE"):
            Path(pause).write_text("ready")
            time.sleep(1)
        number = state["next_number"]
        state["next_number"] += 1
        row = {
            "number": number,
            "id": 10000 + number,
            **payload,
            "repository_url": "https://api.github.com/repos/o/r",
            "state": "open",
            "state_reason": None,
            "assignees": [],
            "html_url": f"https://github.com/o/r/issues/{number}",
        }
        row["labels"] = [{"name": label} for label in row["labels"]]
        state["issues"].append(row)
        return save(row)
    number = int(n)
    if tail == "comments":
        comments = state["comments"].setdefault(n, [])
        if method == "GET":
            return emit(page(comments))
        assert method == "POST" and set(payload) == {"body"}
        row = {
            "id": sum(len(v) for v in state["comments"].values()) + 1,
            "body": payload["body"],
            "user": {"login": "operator"},
        }
        comments.append(row)
        return save(row)
    assert method == "GET"
    if tail == "parent":
        return emit({"message": "No parent issue found"}, 404)
    if tail in {"sub_issues", "dependencies/blocked_by"}:
        return emit([])
    return emit(next(i for i in state["issues"] if i["number"] == number))


if __name__ == "__main__":
    raise SystemExit(serve())
