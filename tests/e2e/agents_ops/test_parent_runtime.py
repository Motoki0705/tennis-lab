"""Local parent process provenance stays private and is independent of ownership."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


def test_register_keeps_native_runtime_private(parent_env: ParentEnv) -> None:
    env = parent_env
    runtime = {
        "cwd": str(env.repo),
        "process": {"pid": 123, "boot_id": "test", "start_ticks": "456"},
    }
    source = (
        "import json,lib.identity; from parent.cli import main; lib.identity.native_runtime=lambda:"
        + repr(runtime)
        + "; raise SystemExit(main())"
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            source,
            "--repo",
            "o/r",
            "--parent",
            "1",
            "register",
            "--summary",
            "担当来歴の検証",
        ],
        cwd=env.repo,
        env=env.env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    registry = (
        Path(env.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )
    row = json.loads(registry.read_text())["parents"]["o/r#1"]
    assert row["owner_runtime"] == {
        **runtime,
        "session": row["session"],
        "generation": row["generation"],
    }
    assert row["status"] == "active"
    assert "owner_runtime" not in proc.stdout
    assert "start_ticks" not in proc.stdout


def test_native_runtime_uses_agent_ancestor_and_its_own_cwd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / ".agents/ops"))
    identity = importlib.import_module("lib.identity")
    opsproc = importlib.import_module("lib.opsproc")
    monkeypatch.setattr(os, "getppid", lambda: 101)
    records = {
        "/proc/101/stat": "101 (bash) S 100 " + "0 " * 30,
        "/proc/100/stat": "100 (codex) S 99 " + "0 " * 30,
        "/proc/99/stat": "99 (claude) S 1 " + "0 " * 30,
    }
    real = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **kw: (
            records[str(path)] if str(path) in records else real(path, *a, **kw)
        ),
    )
    monkeypatch.setattr(os, "readlink", lambda path: "/repo/.claude/worktrees/parent")
    observed = {"pid": 100, "boot_id": "boot", "start_ticks": "789"}
    monkeypatch.setattr(
        opsproc,
        "process_identity",
        lambda pid: (
            observed
            if pid == 100
            else {"pid": 99, "boot_id": "boot", "start_ticks": "older"}
        ),
    )
    assert identity.native_runtime() == {
        "cwd": "/repo/.claude/worktrees/parent",
        "process": observed,
    }


@pytest.mark.parametrize("new_native", [False, True])
def test_handoff_rebinds_or_invalidates_owner_runtime(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch, new_native: bool
) -> None:
    env = parent_env
    old_cwd = env.repo / ".claude/worktrees/old-owner"
    new_cwd = env.repo / ".claude/worktrees/new-owner"
    runtime = {
        "cwd": str(old_cwd),
        "process": {"pid": 123, "boot_id": "boot", "start_ticks": "456"},
    }

    def cli(*args: str, native: Any, extra: Any = None) -> None:
        source = (
            "import lib.identity; from parent.cli import main; lib.identity.native_runtime=lambda:"
            + repr(native)
            + "; raise SystemExit(main())"
        )
        proc = subprocess.run(
            [sys.executable, "-c", source, "--repo", "o/r", "--parent", "1", *args],
            cwd=env.repo,
            env={**env.env, **(extra or {})},
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        assert proc.returncode == 0, proc.stderr

    cli("register", "--summary", "引継ぎ来歴の検証", native=runtime)
    env.cli("handoff", "--note-file", env.note())
    new_runtime = (
        {
            **runtime,
            "cwd": str(new_cwd),
            "process": {"pid": 789, "boot_id": "boot", "start_ticks": "new"},
        }
        if new_native
        else None
    )
    cli(
        "handoff",
        "--accept",
        "--note-file",
        env.note(),
        native=new_runtime,
        extra={"CLAUDE_CODE_SESSION_ID": "private-session-b-unique"},
    )
    root = Path(env.env["XDG_STATE_HOME"]) / "tennis-lab-agents"
    module = importlib.import_module("parent.state")
    store = module.Store(root / "parent-sessions")
    with store.locked():
        row = store.parents["o/r#1"]
        assert row["generation"] == 2
        if new_native:
            assert new_runtime is not None
            assert row["owner_runtime"] == {
                **new_runtime,
                "session": row["session"],
                "generation": 2,
            }
        else:
            assert "owner_runtime" not in row
        row["tasks"]["2"] = {"worktree": str(new_cwd), "stopped": False}
        store.save()
    lifecycle = importlib.import_module("worktree_cleanup.lifecycle")
    monkeypatch.setattr(lifecycle.processes, "agents", lambda root: [])
    lifecycle.Stopper(env.repo, root=root).registry(old_cwd, [], "2026-11-13T00:00:00Z")
    current = store.read_data()["parents"]["o/r#1"]
    assert current["status"] == "active"
    assert current["tasks"]["2"]["stopped"] is False
