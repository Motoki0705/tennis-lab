"""REST-only fake gh for ci_watch; unsupported operations fail closed."""

from __future__ import annotations

import json
import os
import re
import signal
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit


def serve() -> int:
    args = sys.argv[1:]
    with Path(os.environ["FAKE_CI_CALLS"]).open("a") as fh:
        fh.write(json.dumps(args) + "\n")
    if len(args) != 6 or args[:5] != [
        "api",
        "--hostname",
        "github.com",
        "--method",
        "GET",
    ]:
        print(f"unsupported gh arguments: {args}", file=sys.stderr)
        return 2
    endpoint = args[-1]
    state = json.loads(Path(os.environ["FAKE_CI_STATE"]).read_text())
    parsed = urlsplit(endpoint)
    path = parsed.path
    if endpoint == os.environ.get("FAKE_CI_INTERRUPT"):
        os.kill(os.getppid(), signal.SIGINT)
        return 1
    error = state["errors"].get(endpoint, state["errors"].get(path))
    if error:
        print(error, file=sys.stderr)
        return 1
    if endpoint in state["overrides"]:
        print(json.dumps(state["overrides"][endpoint]))
        return 0
    query = parse_qs(parsed.query)

    def page(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        assert query["per_page"] == ["100"]
        index = int(query["page"][0])
        return {"total_count": len(rows), key: rows[(index - 1) * 100 : index * 100]}

    if path == "repos/o/r/git/ref/heads/main":
        result = {
            "ref": "refs/heads/main",
            "object": {"type": "commit", "sha": state["head"]},
        }
    elif match := re.fullmatch(r"repos/o/r/commits/([a-f0-9]{40})", path):
        result = state["commits"][match[1]]
    elif match := re.fullmatch(r"repos/o/r/actions/workflows/([^/]+)/runs", path):
        assert query["branch"] == ["main"]
        rows = state["runs"].get(query["head_sha"][0], {}).get(match[1], [])
        result = page(rows, "workflow_runs")
    elif match := re.fullmatch(
        r"repos/o/r/actions/runs/(\d+)/attempts/(\d+)(/jobs)?", path
    ):
        key = f"{match[1]}/{match[2]}"
        result = (
            page(state["jobs"][key], "jobs") if match[3] else state["attempts"][key]
        )
    elif match := re.fullmatch(r"repos/o/r/actions/jobs/(\d+)/logs", path):
        print(state["logs"][match[1]], end="")
        return 0
    else:
        print(f"unsupported endpoint {endpoint}", file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(serve())
