"""Registry lock boundaries with delayed fake GitHub and disposable state."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_project_progress import ProjectEnv
from tests.e2e.agents_ops.test_project_progress import project_env as project_env

# Instrument the real lock in the parent process without adding production
# logging. Times bracket acquired intervals, excluding time spent waiting.
TRACE_PARENT = r"""
import fcntl, json, os, sys, time
from contextlib import contextmanager
from parent.state import Store
from parent.cli import main
original_flock, original_locked = fcntl.flock, Store.locked
active = False
def event(kind):
    with open(os.environ["PARENT_LOCK_TRACE"], "a") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "event": kind, "time": time.monotonic()}) + "\n")
def flock(file, operation):
    global active
    registry = str(getattr(file, "name", "")).endswith("/registry.lock")
    if registry and operation == fcntl.LOCK_UN:
        event("released")
        active = False
    result = original_flock(file, operation)
    if registry and operation == fcntl.LOCK_EX:
        event("acquired")
        active = True
    return result
@contextmanager
def locked(self):
    global active
    with original_locked(self) as store:
        try:
            yield store
        finally:
            if active:
                event("released")
                active = False
fcntl.flock, Store.locked = flock, locked
sys.exit(main())
"""

TRACE_GH = r"""
import fcntl, json, os, pathlib, sys, time
args = sys.argv[1:]
if "--method" in args and args[args.index("--method") + 1] == "GET":
    endpoint = next(a for a in args if a == "user" or a.startswith("repos/"))
    lock_path = pathlib.Path(os.environ["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions/registry.lock"
    held = False
    if lock_path.exists():
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held = True
    started = time.monotonic()
    time.sleep(float(os.environ.get("PARENT_GET_DELAY", "0")))
    if endpoint == os.environ.get("PARENT_GATE_ENDPOINT"):
        gate = pathlib.Path(os.environ["PARENT_GATE"])
        (gate / os.environ["PARENT_GATE_TAG"]).touch()
        deadline = time.monotonic() + 10
        while not (gate / "release").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("test GitHub gate timed out")
            time.sleep(0.01)
    with open(os.environ["PARENT_GET_TRACE"], "a") as stream:
        stream.write(json.dumps({"owner": os.getppid(), "endpoint": endpoint, "held": held, "start": started, "end": time.monotonic()}) + "\n")
"""


def instrument(p: ParentEnv) -> None:
    gh = p.root / "bin/gh"
    shebang, body = gh.read_text().split("\n", 1)
    gh.write_text(shebang + "\n" + TRACE_GH + body)
    p.env.update(
        PARENT_GET_TRACE=str(p.root / "get-trace.jsonl"),
        PARENT_LOCK_TRACE=str(p.root / "lock-trace.jsonl"),
    )


def traced_cli(
    p: ParentEnv,
    *args: str,
    parent: str = "1",
    extra: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.monotonic()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            TRACE_PARENT,
            "--repo",
            "o/r",
            "--parent",
            parent,
            *args,
        ],
        cwd=p.repo,
        env={**p.env, **(extra or {})},
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    return result, time.monotonic() - started


def rows(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def intervals(p: ParentEnv, pid: int | None = None) -> list[tuple[float, float]]:
    active = {}
    result = []
    for row in rows(p.env["PARENT_LOCK_TRACE"]):
        if pid is not None and row["pid"] != pid:
            continue
        owner = row["pid"]
        if row["event"] == "acquired":
            assert owner not in active
            active[owner] = row["time"]
        else:
            result.append((active.pop(owner), row["time"]))
    assert not active
    return sorted(result)


def wait_for(path: Path) -> None:
    deadline = time.monotonic() + 8
    while not path.exists():
        assert time.monotonic() < deadline, "parent did not reach the fake GitHub gate"
        time.sleep(0.01)


@pytest.mark.parametrize("command", ["register", "reconcile", "sync", "status"])
def test_github_reads_do_not_hold_registry_lock(
    parent_env: ParentEnv, command: str
) -> None:
    p = parent_env
    if command != "register":
        p.cli("register")
    instrument(p)
    result, elapsed = traced_cli(p, command, extra={"PARENT_GET_DELAY": "0.05"})
    assert result.returncode == 0, result.stderr + result.stdout
    spans = intervals(p)
    held_seconds = sum(end - start for start, end in spans)
    gets = rows(p.env["PARENT_GET_TRACE"])
    (p.root / "measurement.json").write_text(
        json.dumps({"command": command, "elapsed": elapsed, "held": held_seconds})
    )
    print(
        f"{command}: elapsed={elapsed:.3f}s, registry-held={held_seconds:.3f}s, "
        f"GETs={len(gets)}, GETs-under-lock={sum(row['held'] for row in gets)}"
    )
    assert gets
    assert not any(row["held"] for row in gets)
    for get in gets:
        assert all(end <= get["start"] or start >= get["end"] for start, end in spans)


def test_two_reconciles_fetch_concurrently_and_serialize_local_checks(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.cli("register")
    peer = {"CLAUDE_CODE_SESSION_ID": "test-peer"}
    p.cli("register", parent="2", extra=peer)
    instrument(p)
    gate = p.root / "gate"
    gate.mkdir()
    shared = {
        "PARENT_GATE_ENDPOINT": "user",
        "PARENT_GATE": str(gate),
        "PARENT_GET_DELAY": "0.05",
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            traced_cli, p, "reconcile", extra={**shared, "PARENT_GATE_TAG": "first"}
        )
        second = pool.submit(
            traced_cli,
            p,
            "reconcile",
            parent="2",
            extra={**shared, **peer, "PARENT_GATE_TAG": "second"},
        )
        try:
            wait_for(gate / "first")
            wait_for(gate / "second")
        finally:
            (gate / "release").touch()
        results = [first.result(), second.result()]
    assert all(r.returncode == 0 for r, _ in results), [r.stderr for r, _ in results]
    spans = intervals(p)
    assert len({row["pid"] for row in rows(p.env["PARENT_LOCK_TRACE"])}) == 2
    assert all(
        end <= next_start
        for (_, end), (next_start, _) in zip(spans, spans[1:], strict=False)
    )
    gets = rows(p.env["PARENT_GET_TRACE"])
    for get in gets:
        assert all(
            end <= get["start"] or start >= get["end"]
            for start, end in intervals(p, get["owner"])
        )
    elapsed = max(elapsed for _, elapsed in results)
    get_seconds = sum(row["end"] - row["start"] for row in gets)
    print(f"two reconciles: elapsed={elapsed:.3f}s, total GET time={get_seconds:.3f}s")
    # Both operations spend the injected delay concurrently, not in a serial queue.
    assert elapsed < get_seconds


@pytest.mark.parametrize(
    "change",
    ["session", "generation", "status", "tasks", "reviews", "extension", "peer"],
)
def test_changed_registry_rejects_stale_delegation(
    parent_env: ParentEnv, change: str
) -> None:
    p = parent_env
    p.agreed()
    p.child("lock-scope")
    worktree = p.worktree("lock-scope")
    before_github = p.state()
    instrument(p)
    gate = p.root / "gate"
    gate.mkdir()
    extra = {
        "PARENT_GATE_ENDPOINT": "repos/o/r/issues/1",
        "PARENT_GATE": str(gate),
        "PARENT_GATE_TAG": "fetching",
    }
    state = importlib.import_module("parent.state")
    directory = Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            traced_cli,
            p,
            "delegate",
            "--child",
            "10",
            "--agent",
            "codex",
            "--branch",
            "codex/lock-scope",
            "--worktree",
            str(worktree),
            extra=extra,
        )
        try:
            wait_for(gate / "fetching")
            # A concurrent writer must be able to enter while GET is pending.
            with (directory / "registry.lock").open("a") as probe:
                state.fcntl.flock(probe, state.fcntl.LOCK_EX | state.fcntl.LOCK_NB)
            with state.Store(directory).locked() as store:
                row = store.parents["o/r#1"]
                if change == "session":
                    row["session"] = "claude:test-new-owner"
                elif change == "generation":
                    row["generation"] += 1
                elif change == "status":
                    row["status"] = "released"
                elif change == "tasks":
                    row["tasks"]["10"] = {"delegated": True, "delegation_id": "peer"}
                elif change == "reviews":
                    row["reviews"].append({"request": {"schema_version": 1}})
                elif change == "extension":
                    row["extra_evidence"] = "recorded concurrently"
                else:
                    store.parents["o/r#2"] = {**row, "session": "claude:test-peer"}
                store.save()
                expected = store.path.read_bytes()
        finally:
            (gate / "release").touch()
        result, _ = pending.result()
    assert result.returncode == 2, result.stderr + result.stdout
    assert "registry changed during GitHub retrieval" in result.stderr
    assert (directory / "registry.json").read_bytes() == expected
    assert p.state() == before_github
    assert p.env["CLAUDE_CODE_SESSION_ID"] not in result.stderr
    assert str(directory) not in result.stderr


def test_failed_get_reacquires_and_releases_the_registry_lock(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.cli("register")
    instrument(p)
    result, _ = traced_cli(
        p,
        "reconcile",
        extra={"PARENT_FAIL_GET_PATH": "repos/o/r/issues/1"},
    )
    assert result.returncode == 2
    spans = intervals(p)
    assert len(spans) > 1
    assert not any(get["held"] for get in rows(p.env["PARENT_GET_TRACE"]))
    state = importlib.import_module("parent.state")
    directory = Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    with (directory / "registry.lock").open("a") as probe:
        state.fcntl.flock(probe, state.fcntl.LOCK_EX | state.fcntl.LOCK_NB)
    p.cli("reconcile")


PROJECT_GH_LOCK_CHECK = r"""
import fcntl, io, json, os, pathlib, sys, time
if sys.argv[1:3] == ["api", "graphql"]:
    payload = sys.stdin.read()
    sys.stdin = io.StringIO(payload)
    reading = json.loads(payload)["query"].lstrip().startswith("query ")
    directory = pathlib.Path(os.environ["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    with (directory / "registry.lock").open("a") as probe:
        held = False
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            held = True
        assert held != reading, "unexpected registry lock during GraphQL I/O"
        if reading and os.environ.get("PARENT_TEST_CHANGE_REGISTRY"):
            path = directory / "registry.json"
            state = json.loads(path.read_text())
            state["parents"]["o/r#1204"]["generation"] += 1
            path.write_text(json.dumps(state))
    if reading:
        time.sleep(0.02)
"""


@pytest.mark.parametrize("change_registry", [False, True])
def test_project_query_lock_boundary_and_stale_write_rejection(
    project_env: ProjectEnv, change_registry: bool
) -> None:
    p = project_env
    fake = p.tmp / "bin/gh"
    shebang, body = fake.read_text().split("\n", 1)
    fake.write_text(shebang + "\n" + PROJECT_GH_LOCK_CHECK + body)
    before = p.state()
    result = p.run(
        "project-setup",
        "--apply",
        ok=not change_registry,
        **({"PARENT_TEST_CHANGE_REGISTRY": "1"} if change_registry else {}),
    )
    if change_registry:
        assert "registry changed during GitHub retrieval" in result.stderr
        assert not p.mutations() and p.state() == before
    else:
        assert p.mutations()
        assert any(
            call["payload"] and call["payload"]["query"].startswith("query ")
            for call in p.calls()
            if call["args"][:2] == ["api", "graphql"]
        )
