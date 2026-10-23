#!/usr/bin/env python3
"""Offline runner boundary for dispatcher input/argument tests, without a real CLI."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def main() -> None:
    args = sys.argv[1:]

    def opt(name: str) -> str:
        return args[args.index(name) + 1]

    runtime = Path(opt("--run-dir"))
    input_dir = Path(opt("--read-dir"))
    assert input_dir.is_absolute() and input_dir.is_dir()
    assert input_dir == runtime.parent / "agent-input"
    assert args.count("--read-dir") == 1
    assert opt("--role") == "parent" and opt("--mode") == "read-only"
    index = json.loads((input_dir / "index.json").read_text())
    prompt = Path(opt("--prompt-file")).read_text()
    with Path(os.environ["FAKE_JUDGEMENT_CALLS"]).open("a") as stream:
        stream.write(json.dumps({"args": args, "prompt": prompt}) + "\n")
    response = Path(os.environ["FAKE_JUDGEMENT_RESPONSE"]).read_text()
    response = response.replace("INPUT_DIGEST", index["input_sha256"])
    response = response.replace("CODE_COMMIT", index["code_commit"])
    runtime.mkdir(mode=0o700)
    (runtime / "final.md").write_text(response)
    summary = {
        "schema_version": 1,
        "run_id": "offline-input-probe",
        "result": "completed",
        "turn_end_reason": "completed",
        "exit_code": 0,
        "task_success": None,
    }
    selected = {"model": "offline-parent"}
    contract = {
        "mode": opt("--mode"),
        "cwd": opt("--cwd"),
        "task_id": opt("--task-id"),
        "read_dirs": [str(input_dir)],
        "json_schema_sha256": hashlib.sha256(
            Path(opt("--json-schema")).read_bytes()
        ).hexdigest(),
        "settings": {
            "role": "parent",
            "agent": "claude",
            "profile": None,
            "overrides": {},
            "selected": selected,
            "defaults": selected,
        },
    }
    scenario = os.environ.get("FAKE_JUDGEMENT_READ_DIRS")
    if scenario == "missing":
        del contract["read_dirs"]
    elif scenario == "wrong":
        contract["read_dirs"] = [str(input_dir.parent)]
    schema_scenario = os.environ.get("FAKE_JUDGEMENT_SCHEMA")
    if schema_scenario == "missing":
        del contract["json_schema_sha256"]
    elif schema_scenario == "wrong":
        contract["json_schema_sha256"] = "f" * 64
    (runtime / "result.json").write_text(
        json.dumps(
            {
                **summary,
                "contract": contract,
                "runtime": {
                    "effective": {
                        "model": {"status": "observed", "value": selected["model"]}
                    },
                    "runtime_errors": [],
                },
                "cleanup_complete": True,
            }
        )
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
