"""Filesystem fake for the production gh and JSON adapter command boundaries."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit


def main() -> None:
    args = sys.argv[1:]
    root = Path(os.environ["DAILY_CI_FAKE_ROOT"])
    if args[0] == "adapter":
        role = args[1]
        payload = json.loads(Path(args[args.index("--input") + 1]).read_text())
        assert payload["schema_version"] == 1
        assert payload["repository"] == "o/r"
        assert Path(payload["state_dir"]).is_absolute()
        operation = payload["operation"]
        if operation == "probe":
            value = {"schema_version": 1, "ready": True, "role": role}
        elif role == "observer":
            assert payload["cycle"]["cycle_id"].startswith("daily-ci-")
            value = {"schema_version": 1, "results": [], "unresolved": []}
        elif operation == "stop":
            assert payload["cycle"]["status"] == "expired"
            value = {"schema_version": 1, "acknowledged": True, "workers": {}}
        else:
            value = {"schema_version": 1, "acknowledged": True}
        print(json.dumps(value))
        return
    method = args[args.index("--method") + 1]
    endpoint = next(a for a in args if a.startswith("repos/"))
    path = urlsplit(endpoint).path
    payload = json.load(sys.stdin) if "--input" in args else None
    with (root / "calls.jsonl").open("a") as handle:
        handle.write(
            json.dumps({"method": method, "endpoint": endpoint, "input": payload})
            + "\n"
        )
    state_path = root / "remote.json"
    state = json.loads(state_path.read_text())
    rows = state["runs"]
    response: object
    if path.endswith("/git/ref/heads/main"):
        response = {"object": {"sha": state["main"]}}
    elif path.endswith("/dispatches"):
        assert payload is not None
        assert method == "POST" and payload["ref"] == "main"
        assert payload["return_run_details"] is True
        workflow = path.split("/")[-2]
        inputs = payload["inputs"]
        assert set(inputs) == {
            "cycle_id",
            "generation",
            "request_id",
            "target_sha",
            "baseline_sha",
            "deadline_at",
        }
        identifier = len(rows) + 1
        rows.append(
            {
                "id": identifier,
                "run_attempt": 1,
                "event": "workflow_dispatch",
                "path": f".github/workflows/{workflow}",
                "head_branch": "main",
                "head_sha": inputs["target_sha"],
                "display_title": f"daily-ci:{inputs['request_id']}",
                "status": "queued",
                "conclusion": None,
            }
        )
        response = {"workflow_run_id": identifier}
    elif "/workflows/" in path and path.endswith("/runs"):
        workflow = path.split("/")[-2]
        selected = [r for r in rows if r["path"] == f".github/workflows/{workflow}"]
        response = {"total_count": len(selected), "workflow_runs": selected}
    elif path.endswith("/cancel"):
        assert method == "POST"
        identifier = int(path.split("/")[-2])
        next(r for r in rows if r["id"] == identifier).update(
            status="completed", conclusion="cancelled"
        )
        response = None
    else:
        assert method == "GET" and "/actions/runs/" in path
        response = next(r for r in rows if r["id"] == int(path.split("/")[-1]))
    state_path.write_text(json.dumps(state))
    if response is not None:
        print(json.dumps(response))


if __name__ == "__main__":
    main()
