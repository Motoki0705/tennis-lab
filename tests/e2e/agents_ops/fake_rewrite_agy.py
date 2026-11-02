#!/usr/bin/env python3
"""Finite writer/verifier turns; never executes another agent."""

import importlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

fake_tools = importlib.import_module("fake_agy_tools")

root = Path(os.environ["REWRITE_FAKE_ROOT"])
args = sys.argv[1:]
if args == ["--version"]:
    print("Antigravity CLI 1.4.11")
    raise SystemExit(0)
with (root / "agy-calls.jsonl").open("a") as trace:
    trace.write(json.dumps(args) + "\n")
index = len((root / "agy-calls.jsonl").read_text().splitlines()) - 1
outputs = json.loads((root / "turns.json").read_text())
assert index < len(outputs), "unexpected additional agent call"
log = Path(args[args.index("--log-file") + 1])


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


with (root / "workspace-calls.jsonl").open("a") as trace:
    trace.write(
        json.dumps(
            {
                "cwd": str(Path.cwd()),
                "head": git("rev-parse", "HEAD"),
                "history": git("log", "-2", "--format=%H").splitlines(),
                "shown": git("show", "HEAD:history.txt"),
                "remote": git("config", "--get", "remote.origin.url"),
                "dirty": git(
                    "status", "--porcelain", "--", ".", ":!.agents/hooks.json"
                ),
                "previous_turn_file": (Path.cwd() / "previous-turn.txt").exists(),
                "schema": args[args.index("--json-schema") + 1],
                "prompt": str(log.with_suffix(".md")),
                "log": str(log),
            }
        )
        + "\n"
    )
prompt = (
    json.loads(sys.stdin.read())["message"]["content"]
    if "--input-format" in args
    else args[args.index("-p") + 1]
)
assert log.with_suffix(".md").read_text() == prompt
attacks = (
    json.loads((root / "attacks.json").read_text())
    if (root / "attacks.json").exists()
    else []
)
for attack in attacks:
    if attack["turn"] != index:
        continue
    package = root / "repo/.agents/ops/issue_rewrite"
    target = attack["target"]
    paths = {
        "invariants": package / "invariants.py",
        "prompt": package / "prompts/writer.md",
        "schema": package / "schemas/writer.schema.json",
        "new": root / "repo/planted.py",
        "ignored": root / "repo/planted.ignored",
        "rendered_prompt": log.with_suffix(".md"),
        "rendered_schema": log.with_suffix(".schema.json"),
    }
    if target in paths:
        paths[target].write_text("tampered\n")
    elif target == "head":
        git("-C", str(root / "repo"), "reset", "--hard", "HEAD^")
    elif target == "staged":
        (root / "repo/planted.py").write_text("tampered\n")
        git("-C", str(root / "repo"), "add", "planted.py")
    elif target == "workspace":
        (Path.cwd() / ".agents/ops/issue_rewrite/invariants.py").write_text(
            "tampered\n"
        )
        (Path.cwd() / "previous-turn.txt").write_text("do not carry to another turn\n")
    else:
        raise AssertionError(target)
    if attack.get("exit"):
        raise SystemExit(attack["exit"])
log.write_text(
    'Propagating selected model override to backend: label="Gemini 3.8 Flash (High)"\n'
    "private-log conversation_id=private-conversation /home/private-person/token.json\n"
)
fake_tools.init_stream()
for command in outputs[index].get("commands", []):
    fake_tools.run_command(command)
if outputs[index].get("unhooked_command"):
    fake_tools.run_command(outputs[index]["unhooked_command"], skip_hook=True)
if outputs[index].get("read_fault"):
    fault = outputs[index]["read_fault"]
    fake_tools.read_file(
        "../private-read-input.txt",
        skip_hook=fault == "missing",
        force_execution=fault == "denied_execution",
        hook_step_offset=1 if fault == "step" else 0,
        display_path="../private-other.txt" if fault == "path" else None,
    )
if outputs[index].get("pause"):
    signal.pause()
if outputs[index].get("exit_code"):
    raise SystemExit(outputs[index]["exit_code"])
fake_tools.emit_result(
    {
        "status": "SUCCESS",
        "conversation_id": "private-conversation",
        "duration_seconds": 0.2,
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        **outputs[index],
    }
)
