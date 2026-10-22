"""The TUI bridge must clear only its own pane before a subsequent model turn."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from tests.e2e.agents_ops.test_herdr_labels import OPS, Labels
from tests.e2e.agents_ops.test_herdr_labels import labels as labels


def terminal_clear(
    labels: Labels, payload: Any, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(OPS / "herdr_labels/codex_terminal_clear.py")],
        input=json.dumps(payload),
        env=env or labels.env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


def codex_pane(labels: Labels, sid: str) -> dict[str, str]:
    env = {k: v for k, v in labels.env.items() if not k.startswith("CLAUDE_")}
    env.update(CODEX_SESSION_ID=sid, CODEX_THREAD_ID=sid)
    state = labels.read()
    state["panes"]["w1:p1"]["agent"] = "codex"
    labels.write(state)
    return env


def test_native_clear_fences_old_session_and_binds_new_daemon_tools(
    labels: Labels,
) -> None:
    old, new = str(uuid4()), str(uuid4())
    env = codex_pane(labels, old)
    labels.set_registry({"o/r#42": labels.owner(identity=f"codex:{old}")})
    bound = labels.run(
        "bind",
        "--herdr-session",
        "default",
        "--pane",
        "w1:p1",
        "--terminal",
        "term-a",
        env=env,
    )
    assert bound.returncode == 0, bound.stderr
    assert labels.run("sync", env=env).returncode == 0
    registry = labels.registry.read_bytes()
    other = labels.read()["panes"]["w1:p2"]

    result = terminal_clear(
        labels, {"version": 1, "event": "clear", "session_id": new}, env
    )
    assert result.returncode == 0, result.stderr
    state = labels.read()
    assert state["panes"]["w1:p1"].get("display_agent") is None
    assert state["panes"]["w1:p1"]["tokens"] == {"other": "keep"}
    assert state["panes"]["w1:p2"] == other
    assert labels.registry.read_bytes() == registry
    late = labels.run("sync", env=env)
    assert late.returncode == 2 and "no longer the active session" in late.stderr

    # Shared daemon tools do not inherit HERDR_ variables. The trusted TUI saved the
    # actual destination; the new root can register and label itself without guesses.
    daemon_env = {k: v for k, v in env.items() if not k.startswith("HERDR_")}
    daemon_env.update(CODEX_SESSION_ID=new, CODEX_THREAD_ID=new)
    labels.set_registry({"o/r#42": labels.owner(identity=f"codex:{new}")})
    synced = labels.run("sync", "--parent", "42", env=daemon_env)
    assert synced.returncode == 0, synced.stderr
    assert labels.read()["panes"]["w1:p1"]["display_agent"] == "#42 · 親"
    assert labels.read()["panes"]["w1:p2"] == other


@pytest.mark.parametrize("display", ["#42 · 関連", "親未登録", "custom display"])
def test_native_clear_handles_migrated_and_unmanaged_displays(
    labels: Labels, display: str
) -> None:
    sid = str(uuid4())
    env = codex_pane(labels, sid)
    state = labels.read()
    pane = state["panes"]["w1:p1"]
    pane["display_agent"] = display
    if display != "custom display":
        pane["tokens"].update(tl_topic="dsh起動方法", tl_engine="codex")
    labels.write(state)
    result = terminal_clear(
        labels, {"version": 1, "event": "clear", "session_id": sid}, env
    )
    assert result.returncode == 0, result.stderr
    after = labels.read()["panes"]["w1:p1"]
    assert after.get("display_agent") == (
        display if display == "custom display" else None
    )
    assert after["tokens"] == {"other": "keep"}


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"version": True, "event": "clear", "session_id": str(uuid4())},
        {"version": 1, "event": "compact", "session_id": str(uuid4())},
        {"version": 1, "event": "clear", "session_id": "invented-id"},
    ],
)
def test_invalid_or_compact_payload_never_touches_herdr(
    labels: Labels, payload: Any
) -> None:
    before = labels.read()
    result = terminal_clear(labels, payload)
    assert result.returncode == 1
    assert labels.read() == before


def test_common_daemon_cannot_impersonate_the_terminal(labels: Labels) -> None:
    sid = str(uuid4())
    env = codex_pane(labels, sid)
    state = labels.read()
    # PID 1 cannot be the enclosing fake terminal agent. No ancestry proof exists.
    state["pid"] = 1
    labels.write(state)
    result = terminal_clear(
        labels, {"version": 1, "event": "clear", "session_id": sid}, env
    )
    assert result.returncode == 1
    assert any(
        reason in result.stderr
        for reason in ("not a descendant", "nested agent cannot report lifecycle")
    )
    assert not labels.mutations()
    assert not list(
        (labels.registry.parent.parent / "herdr-labels").glob("bindings/*.json")
    )


def test_outside_herdr_is_noop_and_failure_is_recorded(labels: Labels) -> None:
    sid = str(uuid4())
    env = codex_pane(labels, sid)
    before = labels.read()
    outside = {k: v for k, v in env.items() if not k.startswith("HERDR_")}
    payload = {"version": 1, "event": "clear", "session_id": sid}
    result = terminal_clear(labels, payload, outside)
    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "not_configured"
    assert labels.read() == before

    state = labels.read()
    state["reject"] = True
    state["panes"]["w1:p1"]["tokens"]["tl_topic"] = "dsh起動方法"
    labels.write(state)
    failed = terminal_clear(labels, payload, env)
    assert failed.returncode == 1
    path = (
        Path(env["XDG_STATE_HOME"])
        / "tennis-lab-agents/herdr-labels/terminal-clear-error.json"
    )
    error = json.loads(path.read_text())
    assert "failed (exit 1)" in error["error"]
    assert sid not in path.read_text()
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert labels.read()["panes"] == state["panes"]
