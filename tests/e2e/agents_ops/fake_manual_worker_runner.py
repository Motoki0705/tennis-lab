#!/usr/bin/env python3
"""Offline common-runner boundary: assert durable admission before any work."""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path


def main() -> int:
    args = sys.argv[1:]

    def opt(name: str) -> str:
        return args[args.index(name) + 1]

    directory = Path(opt("--run-dir"))
    if args[0] == "status":
        record = json.loads((directory / "result.json").read_text())
        print(
            json.dumps(
                {
                    **{key: record[key] for key in SUMMARY_KEYS},
                    "runner_alive": False,
                    "agent_alive": record["result"] == "running",
                    "agent_group_alive": record["result"] == "running",
                }
            )
        )
        return 0

    # The parent's final spawn lock is released before it delivers the prompt.
    # This is the work boundary, rather than Python interpreter startup.
    source = opt("--prompt-file")
    raw = sys.stdin.buffer.read() if source == "-" else Path(source).read_bytes()
    prompt = raw.decode("utf-8")
    state_dir = Path(os.environ["MANUAL_STATE_DIR"])
    for path in [state_dir / "registry.lock", *state_dir.glob("operation-*.lock")]:
        with path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    store = importlib.import_module("parent.state").Store(state_dir)
    row = store.read_data()["parents"][os.environ.get("MANUAL_PARENT_KEY", "o/r#1")]
    task = row["tasks"][os.environ.get("MANUAL_CHILD", "10")]
    saved = task["worker_runs"][-1]
    assert saved["delegation_id"] == opt("--task-id") == task["delegation_id"]
    assert state_dir / saved["run_dir"] == directory
    assert "worker_ended" not in task
    assert opt("--role") == "implementation" and opt("--mode") == "write"
    assert opt("--branch") == task["branch"] and opt("--cwd") == task["worktree"]
    assert "--effort" not in args
    resolver = importlib.import_module("lib.agent_settings")
    settings = resolver.resolve(
        resolver.DEFAULT_CONFIG,
        "codex",
        None,
        opt("--model") if "--model" in args else None,
        None,
        "write",
    )
    assert task["model"] == saved["model"] == settings["selected"]["model"]
    assert task["model_reason"] == saved["model_reason"]
    with Path(os.environ["MANUAL_RUNNER_CALLS"]).open("a") as stream:
        stream.write(json.dumps(args) + "\n")
    # Read the instruction as the real runner does (UTF-8 file or stdin).
    if os.environ.get("MANUAL_RUNNER_PROMPTS"):  # Only the producer tests inspect it.
        with Path(os.environ["MANUAL_RUNNER_PROMPTS"]).open("a") as stream:
            stream.write(json.dumps({"prompt_file": source, "prompt": prompt}) + "\n")
    if os.environ.get("MANUAL_PREFLIGHT_FAILURE"):
        print(str(directory), file=sys.stderr)  # Must stay private at the boundary.
        return 2
    code = int(os.environ.get("MANUAL_RUNNER_EXIT", "0"))
    contract = {
        "task_id": opt("--task-id"),
        "cwd": opt("--cwd"),
        "branch": opt("--branch"),
        "mode": "write",
        "settings": settings,
    }
    session_id = "offline-private-session"
    if args[0] == "resume":
        previous = json.loads((Path(opt("--resume-from")) / "result.json").read_text())
        assert previous["contract"] == contract
        session_id = previous["runtime"]["session_id"]
    summary = {
        "schema_version": 1,
        "run_id": directory.name,
        "result": "failed" if code else "completed",
        "turn_end_reason": "cli_error" if code else "completed",
        "exit_code": code,
        "task_success": None,
    }
    directory.mkdir(parents=True, mode=0o700)
    (directory / "prompt.md").write_bytes(raw)
    (directory / "result.json").write_text(
        json.dumps(
            {
                **summary,
                "contract": contract,
                "prompt_sha256": hashlib.sha256(raw).hexdigest(),
                "cleanup_complete": True,
                "started_at": 1794013200,
                "finished_at": 1794013201,
                "runtime": {
                    "session_id": session_id,
                    "effective": {
                        key: {
                            "status": "observed",
                            "value": settings["selected"][key],
                            "source": "offline fixture",
                        }
                        for key in ("model", "effort")
                    },
                },
            }
        )
    )
    (directory / "invocation.json").write_text(
        json.dumps({"schema_version": 1, "contract": contract})
    )
    headings = (
        "確認した根拠",
        "指示との不一致",
        "実装案",
        "テスト設計",
        "制約・未確認事項",
        "親に決めてほしい点",
    )
    (directory / "final.md").write_text(
        os.environ.get(
            "MANUAL_PLANNING_FINAL", "\n\n".join(f"## {h}\n\nなし" for h in headings)
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary))
    return code


SUMMARY_KEYS = (
    "schema_version",
    "run_id",
    "result",
    "turn_end_reason",
    "exit_code",
    "task_success",
)

if __name__ == "__main__":
    raise SystemExit(main())
