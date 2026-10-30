#!/usr/bin/env python3
"""Read-only GitHub CLI backed by the managed dispatch fixture's published state."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs


def main() -> None:
    args = sys.argv[1:]
    assert args[0] == "api" and args[args.index("--method") + 1] == "GET"
    endpoint = next(a for a in args if a == "user" or a.startswith("repos/"))
    path, _, query = endpoint.partition("?")
    state = json.loads(Path(os.environ["FAKE_DISPATCH_GITHUB"]).read_text())
    page = int(parse_qs(query).get("page", ["1"])[0])

    def paged(rows: list[Any]) -> list[Any]:
        return rows[(page - 1) * 100 : page * 100]

    result: Any
    code = 200
    if path == "user":
        result = {"login": "operator"}
    elif path == "repos/o/r/pulls":
        result = paged(state["prs"])
    elif path.startswith("repos/o/r/pulls/"):
        result = next(
            pr for pr in state["prs"] if pr["number"] == int(path.rsplit("/", 1)[1])
        )
    else:
        assert path.startswith("repos/o/r/issues/")
        n, _, tail = path.removeprefix("repos/o/r/issues/").partition("/")
        if not tail:
            result = state["issues"][n]
        elif tail == "parent":
            parent = state["parents"].get(n)
            result = (
                state["issues"][str(parent)]
                if parent
                else {"message": "No parent issue found"}
            )
            code = 200 if parent else 404
        elif tail == "sub_issues":
            result = paged(
                [
                    v
                    for k, v in state["issues"].items()
                    if state["parents"].get(k) == int(n)
                ]
            )
        elif tail == "dependencies/blocked_by":
            result = paged(
                [state["issues"][str(k)] for k in state["dependencies"].get(n, [])]
            )
        elif tail == "comments":
            result = paged(state["comments"].get(n, []))
        elif tail == "timeline":
            result = paged(
                [
                    {
                        "id": pr["number"],
                        "event": "cross-referenced",
                        "source": {
                            "issue": {
                                **pr,
                                "repository_url": "https://api.github.com/repos/o/r",
                                "pull_request": {
                                    "url": f"https://api.github.com/repos/o/r/pulls/{pr['number']}"
                                },
                            }
                        },
                    }
                    for pr in state["prs"]
                    if re.search(r"#" + n + r"\b", pr.get("body") or "")
                ]
            )
        else:
            raise AssertionError(endpoint)
    if "--include" in args:
        print(f"HTTP/1.1 {code} {'OK' if code == 200 else 'Not Found'}\n")
    print(json.dumps(result))
    raise SystemExit(0 if code == 200 else 1)


if __name__ == "__main__":
    main()
