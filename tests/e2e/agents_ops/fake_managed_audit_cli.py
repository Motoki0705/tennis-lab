#!/usr/bin/env python3
"""Exercise the actual MCP and common runner with offline Claude/Codex streams."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import select
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from uuid import uuid4


def emit(**event: Any) -> None:
    print(json.dumps(event), flush=True)


def pidfd_open(pid: int) -> int:
    # Some Python builds omit os.pidfd_open although the host libc/kernel
    # support it. Keep the same event-based wait in both interpreters.
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = int(libc.pidfd_open(ctypes.c_int(pid), ctypes.c_uint(0)))
    if descriptor < 0:
        code = ctypes.get_errno()
        if code == errno.ESRCH:
            raise ProcessLookupError(code, os.strerror(code))
        raise OSError(code, os.strerror(code))
    return descriptor


def request(server: subprocess.Popen[str], method: str, **params: Any) -> Any:
    assert server.stdin is not None and server.stdout is not None
    server.stdin.write(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        + "\n"
    )
    server.stdin.flush()
    return json.loads(server.stdout.readline())["result"]


def call(server: subprocess.Popen[str], name: str, **arguments: Any) -> Any:
    reply = request(server, "tools/call", name=name, arguments=arguments)
    return json.loads(reply["content"][0]["text"])


def audit(server: subprocess.Popen[str], prompt: str) -> str:
    control = Path(os.environ["FAKE_SCOUT_CONTROL"])
    control.mkdir()
    with ExitStack() as stack:
        channels = {}
        for name in ("first", "independent", "followup", "release"):
            path = control / name
            os.mkfifo(path)
            descriptor = os.open(path, os.O_RDWR | os.O_NONBLOCK)
            stack.callback(os.close, descriptor)
            channels[name] = descriptor
        return audit_tasks(server, prompt, channels)


def audit_tasks(
    server: subprocess.Popen[str], prompt: str, channels: dict[str, int]
) -> str:
    def ok(name: str, **args: Any) -> Any:
        reply = call(server, name, **args)
        assert reply["ok"], reply
        return reply["data"]

    meta = json.loads(prompt.split("```json\n", 1)[1].split("```", 1)[0])
    for task in ("first", "independent", "followup"):
        ok(
            "audit_plan_task",
            id=task,
            question=f"Investigate {task}",
            scope=["big.py"],
            origin={"kind": "goal", "id": meta["slot"]},
        )
    for task in ("first", "independent"):
        ok("audit_start_task", id=task)
    assert ok("audit_scout_result", id="independent")["status"] == "running"
    premature = call(
        server,
        "audit_finish_task",
        id="independent",
        status="completed",
        conclusion="premature",
        evidence=[],
        next_questions=[],
        unconfirmed=[],
    )
    assert (
        not premature["ok"]
        and "terminal execution evidence" in premature["error"]["message"]
    )

    def finish(task: str) -> None:
        assert select.select([channels[task]], [], [], 15)[0], task
        runner_pid = json.loads(os.read(channels[task], 512))
        try:
            finished = pidfd_open(runner_pid)
        except ProcessLookupError:
            pass  # The runner has already exited after sending its notification.
        else:
            try:
                assert select.select([finished], [], [], 15)[0], task
            finally:
                os.close(finished)
        result = ok("audit_scout_result", id=task)
        assert result["status"] == "completed", result
        ok("audit_evidence", id=f"e_{task}", receipt=result["text"].strip())
        ok(
            "audit_finish_task",
            id=task,
            status="completed",
            conclusion=f"Checked {task}",
            evidence=[f"e_{task}"],
            next_questions=[],
            unconfirmed=[],
        )

    finish("first")
    ok("audit_start_task", id="followup")
    finish("followup")
    assert ok("audit_scout_result", id="independent")["status"] == "running"
    os.write(channels["release"], b"1")
    finish("independent")
    ok(
        "audit_end",
        status="completed",
        reason="Verified",
        checked_scope=["big.py"],
        excluded_scope=[],
        unverified_risks=[],
        further_investigation=[],
    )
    return '{"memory_update": null}'


def scout(server: subprocess.Popen[str], prompt: str) -> str:
    tools = request(server, "tools/list")["tools"]
    assert all(not tool["name"].startswith(("audit_", "cpu_")) for tool in tools)
    assert "問い: Investigate" in prompt
    task = prompt.split("問い: Investigate ", 1)[1].splitlines()[0]
    control = Path(os.environ["FAKE_SCOUT_CONTROL"])
    if task == "independent":
        with (control / "release").open("rb", buffering=0) as release:
            assert select.select([release], [], [], 15)[0]
            assert release.read(1) == b"1"
    record = call(server, "code_file", path="big.py", start_line=2, line_count=3)
    assert record["ok"], record
    receipt = record["receipt"]
    assert isinstance(receipt, str)
    with (control / task).open("w") as notification:
        notification.write(json.dumps(os.getppid()) + "\n")
    return receipt


def main() -> None:
    args = sys.argv[1:]
    prompt = sys.stdin.read()
    runtime = Path(os.environ["AGENT_RUNTIME_RUN_DIR"])
    parent = Path(sys.argv[0]).name == "claude"
    model = args[args.index("--model") + 1]
    session = str(uuid4())
    if parent:
        assert (
            model == "claude-opus-5-5" and args[args.index("--effort") + 1] == "xhigh"
        )
        assert "mcp__ops_report__*" in args and "dontAsk" in args
        servers = json.loads(args[args.index("--mcp-config") + 1])["mcpServers"]
        emit(type="system", subtype="init", session_id=session, model=model)
    else:
        assert model == "gpt-6.1-sol" and 'model_reasoning_effort="max"' in args
        assert args[args.index("--sandbox") + 1] == "read-only"
        servers = json.loads((runtime / "mcp-config.json").read_text())
        home = Path(os.environ["CODEX_HOME"]) / "sessions"
        home.mkdir(parents=True, exist_ok=True)
        context = {
            "turn_id": str(uuid4()),
            "cwd": str(Path.cwd()),
            "model": model,
            "effort": "max",
        }
        (home / f"rollout-{session}.jsonl").write_text(
            json.dumps({"type": "session_meta", "payload": {"id": session}})
            + "\n"
            + json.dumps({"type": "turn_context", "payload": context})
            + "\n"
        )
        emit(type="thread.started", thread_id=session)
        emit(type="turn.started")
        if os.environ.get("FAKE_SCOUT_RECONNECT") == "1":
            emit(
                type="error",
                message="Reconnecting... 2/5 (workspace routing discovery timed out)",
            )

    config = servers["ops_report"]
    with subprocess.Popen(
        [config["command"], *config["args"]],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, **config["env"]},
    ) as server:
        try:
            output = audit(server, prompt) if parent else scout(server, prompt)
        finally:
            assert server.stdin is not None
            server.stdin.close()
    assert server.returncode == 0
    if parent:
        emit(
            type="result",
            subtype="success",
            session_id=session,
            is_error=False,
            result=output,
        )
    else:
        Path(args[args.index("--output-last-message") + 1]).write_text(output)
        emit(
            type="item.completed",
            item={"id": "item-1", "type": "agent_message", "text": output},
        )
        emit(type="turn.completed", usage={"input_tokens": 20, "output_tokens": 5})


if __name__ == "__main__":
    main()
