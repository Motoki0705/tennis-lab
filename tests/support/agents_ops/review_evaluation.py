"""Small fixed git/input fixtures and completed runner evidence, without an LLM."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def runner_evidence(
    directory: Path,
    model: str,
    *,
    output: str | None = None,
    cleanup: bool = True,
    usage: dict[str, int] | None = None,
    number: int = 1699,
) -> dict[str, Any]:
    runtime = directory / "agent-runtime"
    runtime.mkdir(parents=True)
    (directory / "reviewer").mkdir(exist_ok=True)
    if not (directory / "prompt.md").exists():
        (directory / "prompt.md").write_text("Frozen comparison prompt")
    text = (
        output
        if output is not None
        else json.dumps(
            {
                "verdict": "approve",
                "summary": "Fixture complete",
                "cross_findings": [],
                "prs": [
                    {
                        "number": number,
                        "coverage": "reviewed",
                        "summary": "Fixture checked",
                        "findings": [],
                        "evidence_waits": [],
                    }
                ],
            }
        )
    )
    (runtime / "final.md").write_text(text)
    (directory / "agent_output.md").write_text(text)
    schema = (
        Path(__file__).resolve().parents[3] / ".agents/ops/review/result_schema.json"
    )
    shutil.copyfile(schema, runtime / "output-schema.json")
    tokens = usage or {
        "input_tokens": 100,
        "cached_input_tokens": 70,
        "output_tokens": 10,
    }
    events = [
        {"type": "thread.started", "thread_id": "evaluation-fixture-session"},
        {"type": "turn.started", "turn_id": "evaluation-fixture-turn"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": text}},
        {"type": "turn.completed", "usage": tokens},
    ]
    (runtime / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    selected = {"model": model, "effort": "xhigh", "service_tier": "default"}
    started_at = time.time()
    record = {
        "schema_version": 1,
        "result": "completed",
        "exit_code": 0,
        "cli_exit_code": 0,
        "turn_end_reason": "completed",
        "agent_spawned": True,
        "cleanup_complete": cleanup,
        "started_at": started_at,
        "finished_at": started_at + 0.01,
        "prompt_sha256": hashlib.sha256(
            (directory / "prompt.md").read_bytes()
        ).hexdigest(),
        "run_dir": str(runtime),
        "output_path": str(directory / "agent_output.md"),
        "final_path": str(runtime / "final.md"),
        "contract": {
            "cwd": str(directory / "reviewer"),
            "mode": "read-only",
            "json_schema_sha256": hashlib.sha256(schema.read_bytes()).hexdigest(),
            "settings": {"agent": "codex", "profile": "review", "selected": selected},
        },
        "runtime": {
            "session_id": "evaluation-fixture-session",
            "turn_id": "evaluation-fixture-turn",
            "requested_session_id": None,
            "runtime_errors": [],
            "effective": {
                key: {"status": "observed", "value": value, "source": "fixture"}
                for key, value in selected.items()
            },
            "usage": {
                "status": "observed",
                "source": "turn.completed",
                "value": tokens,
            },
        },
    }
    (runtime / "result.json").write_text(json.dumps(record))
    return record


def historical_case(tmp_path: Path, c: Any) -> Any:
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Evaluation fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    (repo / "change.py").write_text("answer = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD")
    (repo / "change.py").write_text("answer = 2\n")
    git(repo, "commit", "-qam", "head")
    head = git(repo, "rev-parse", "HEAD")
    (repo / "future-answer.txt").write_text("SPOILER: the corrected answer\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "future fix")
    future = git(repo, "rev-parse", "HEAD")
    import hashlib

    snapshot = {
        "head_sha": head,
        "base_sha": base,
        "merge_base_sha": base,
        "diff_sha256": hashlib.sha256(
            c.git(repo, "diff", *c.DIFF_FLAGS, base, head, "--")
        ).hexdigest(),
        "base_ref": "main",
        "branch": "fixture",
    }
    manifest = copy.deepcopy(c.catalog())
    (data / "inputs").mkdir(parents=True)
    for case in manifest["cases"]:
        case["snapshot"] = {**snapshot, "number": case["pr"]}
        context = {
            "parent": {"number": 1, "body": "Frozen requirements"},
            "children": [],
            "prs": [{"number": case["pr"]}],
            "instruction": "Read the fixed code",
            "parent_comments": [],
        }
        value = {
            "schema_version": 1,
            "case_id": case["case_id"],
            "cutoff": case["cutoff"],
            "context": context,
            "github": {
                "repos/Motoki0705/tennis-lab/issues/1": {
                    "as_of": case["cutoff"],
                    "value": context["parent"],
                }
            },
            "provenance": {"kind": "fixture"},
        }
        case["input_sha256"] = c.digest(value)
        (data / case["input_file"]).write_text(json.dumps(value))
    (data / "manifest.json").write_text(json.dumps(manifest))
    for name in ("gold.json", "protocol.json", "historical_usage.json"):
        (data / name).write_bytes((c.DATA / name).read_bytes())
    state = tmp_path / "private-evaluation"
    return SimpleNamespace(
        repo=repo, data=data, state=state, future=future, case=manifest["cases"][0]
    )
