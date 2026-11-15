"""Expiry does not own registry rows and shares the repo improvement lock."""

from __future__ import annotations

import importlib
import json
import os
import select
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import pytest

from tests.e2e.agents_ops.test_parent_improvements import decide, publish, record
from tests.e2e.agents_ops.test_parent_session import OPS, ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env

NOW = "2099-01-08T00:00:00Z"

REGISTRY_UPDATE = r"""
import json, os, pathlib, runpy
from issues.cli import GitHub as IssueGitHub
from parent.state import Store
original_get, original_save = IssueGitHub.get, Store.save
updated = False
directory = pathlib.Path(os.environ["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
def payloads():
    return {str(p.relative_to(directory)): p.read_bytes().hex()
            for p in directory.rglob("*") if p.is_file() and p.suffix != ".lock"}
def get(self, endpoint, **kwargs):
    global updated
    if not updated and endpoint.startswith("repos/o/r/issues?state=all"):
        updated = True
        with Store(directory).locked() as peer:
            peer.parents["o/r#2"]["generation"] += 1
            peer.parents["o/r#2"]["heartbeat"] = "peer progressed during expiry GET"
            original_save(peer)
        pathlib.Path(os.environ["EXPIRY_EXPECTED_FILES"]).write_text(json.dumps(payloads()))
    return original_get(self, endpoint, **kwargs)
def forbidden_save(self):
    raise AssertionError("expiry must not save any parent registry row")
IssueGitHub.get, Store.save = get, forbidden_save
runpy.run_module("parent", run_name="__main__")
"""

SERIALIZED_ENTRY = r"""
import os, pathlib, runpy
from contextlib import contextmanager
from parent.github import GitHub
from parent.state import Store
from parent import improvement_proposals
from parent.improvements import lock_key
mode = os.environ["EXPIRY_LOCK_MODE"]
done = pathlib.Path(os.environ["EXPIRY_HOLDER_DONE"])
paused = False
def signal(name, value):
    with open(os.environ[name], "wb", buffering=0) as stream:
        stream.write(value)
original_operation, original_write = Store.operation, GitHub.write
@contextmanager
def operation(self, key):
    if mode == "expiry" and key == lock_key("o/r"):
        signal("EXPIRY_LOCK_EVENTS", b"L")
    with original_operation(self, key):
        yield
        if mode != "expiry" and paused and key == lock_key("o/r"):
            done.touch()  # Business writes are complete, while the lock is held.
def write(self, method, endpoint, payload):
    global paused
    result = original_write(self, method, endpoint, payload)
    extraction = (mode == "extract" and method == "POST" and endpoint == "repos/o/r/issues"
                  and "parent-improvement-extraction:v1" in payload.get("body", ""))
    projection = mode == "propose" and method == "PATCH" and endpoint == "repos/o/r/issues/10"
    if not paused and (extraction or projection):
        paused = True
        signal("EXPIRY_HOLDER_READY", b"H")
        with open(os.environ["EXPIRY_HOLDER_RELEASE"], "rb", buffering=0) as stream:
            assert stream.read(1) == b"R"
    return result
original_expire = improvement_proposals.expire
def expire(*args, **kwargs):
    signal("EXPIRY_LOCK_EVENTS", b"E")
    assert done.exists(), "expiry entered before the competing improvement completed"
    return original_expire(*args, **kwargs)
Store.operation, GitHub.write = operation, write
if mode == "expiry":
    improvement_proposals.expire = expire
runpy.run_module("parent", run_name="__main__")
"""


def state_directory(p: ParentEnv) -> Path:
    return Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"


def private_payloads(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): path.read_bytes().hex()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix != ".lock"
    }


def run_cli(
    p: ParentEnv, entry: str, args: list[str], extra: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    expiry = args[0] == "improvement-expire"
    launcher = (
        [sys.executable, str(OPS / "checkout_job.py"), "--repo", str(p.repo), "--"]
        if expiry
        else []
    )
    return subprocess.run(
        [
            *launcher,
            sys.executable,
            "-c",
            entry,
            "--repo",
            "o/r",
            *([] if expiry else ["--parent", "1"]),
            *args,
        ],
        cwd=p.repo,
        env={**p.env, **extra},
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )


@pytest.mark.parametrize("migrate", [False, True], ids=["registry-v1", "registry-v2"])
@pytest.mark.parametrize("dry_run", [False, True], ids=["close", "dry-run"])
def test_expiry_preserves_peer_registry_updates_during_get(
    parent_env: ParentEnv, migrate: bool, dry_run: bool
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    publish(p)
    module = importlib.import_module("parent.state")
    directory = state_directory(p)
    with module.Store(directory).locked() as store:
        store.parents["o/r#2"] = {
            "session": "claude:expiry-peer-registry-session",
            "generation": 1,
            "status": "active",
            "tasks": {},
            "reviews": [],
        }
        store.save()
        if migrate:
            store.migrate()
    before = p.state()
    expected = p.root / "expected-private-files.json"
    args = ["improvement-expire", "--now", NOW, *(["--dry-run"] if dry_run else [])]
    result = run_cli(p, REGISTRY_UPDATE, args, {"EXPIRY_EXPECTED_FILES": str(expected)})
    assert result.returncode == 0, result.stderr + result.stdout
    data = json.loads(result.stdout)
    assert data == {
        "closed": [] if dry_run else [10],
        "pending": [10],
        "errors": [],
        "dry_run": dry_run,
    }
    assert private_payloads(directory) == json.loads(expected.read_text())
    assert module.Store(directory).read_data()["parents"]["o/r#2"]["generation"] == 2
    # Expiry does not append business decisions to the source's evidence graph.
    assert all(
        p.state()["comments"][key] == value
        for key, value in before["comments"].items()
        if key != "10"
    )
    if dry_run:
        assert p.state() == before
    else:
        assert p.state()["issues"]["10"]["state_reason"] == "not_planned"
    replay = json.loads(p.cli(*args).stdout)
    assert replay["errors"] == [] and replay["closed"] == []
    assert private_payloads(directory) == json.loads(expected.read_text())
    assert len(p.state()["comments"].get("10", [])) == (0 if dry_run else 1)


def receive(fd: int) -> bytes:
    ready, _, _ = select.select([fd], [], [], 30)
    assert ready, "CLI did not reach its synchronization boundary"
    return os.read(fd, 1)


@pytest.mark.parametrize("competitor", ["extract", "propose"])
@pytest.mark.parametrize("dry_run", [False, True], ids=["close", "dry-run"])
def test_expiry_serializes_with_competing_improvement(
    parent_env: ParentEnv, competitor: str, dry_run: bool
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    publish(p)
    decide(p, "approve")
    if competitor == "propose":
        record(p, fixed_key="second")
    paths = {
        name: p.root / name
        for name in ("holder-ready", "holder-release", "expiry-events")
    }
    with ExitStack() as stack:
        fds = {}
        for name, path in paths.items():
            os.mkfifo(path)
            fds[name] = os.open(path, os.O_RDWR | os.O_NONBLOCK)
            stack.callback(os.close, fds[name])
        done = p.root / "holder-done"
        extra = {
            "EXPIRY_HOLDER_READY": str(paths["holder-ready"]),
            "EXPIRY_HOLDER_RELEASE": str(paths["holder-release"]),
            "EXPIRY_LOCK_EVENTS": str(paths["expiry-events"]),
            "EXPIRY_HOLDER_DONE": str(done),
        }
        args = (
            ["improvement-propose"]
            if competitor == "propose"
            else ["improvement-extract", "--candidates", "I-1-result-access"]
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            holder = pool.submit(
                run_cli,
                p,
                SERIALIZED_ENTRY,
                args,
                {**extra, "EXPIRY_LOCK_MODE": competitor},
            )
            try:
                assert receive(fds["holder-ready"]) == b"H"
                expiry = pool.submit(
                    run_cli,
                    p,
                    SERIALIZED_ENTRY,
                    [
                        "improvement-expire",
                        "--now",
                        NOW,
                        *(["--dry-run"] if dry_run else []),
                    ],
                    {**extra, "EXPIRY_LOCK_MODE": "expiry"},
                )
                assert receive(fds["expiry-events"]) == b"L"
                assert not done.exists() and not holder.done()
            finally:
                os.write(fds["holder-release"], b"R")
            published = holder.result()
            closed = expiry.result()
    assert published.returncode == 0, published.stderr + published.stdout
    assert closed.returncode == 0, closed.stderr + closed.stdout
    result = json.loads(closed.stdout)
    assert result["errors"] == []
    if competitor == "extract":
        assert result["closed"] == result["pending"] == []
        assert p.state()["issues"]["10"]["state_reason"] == "completed"
        assert p.state()["issues"]["11"]["labels"] == [
            {"name": "enhancement"},
            {"name": "issue:parent"},
        ]
        assert not any(
            "parent-improvement-expired:v1" in c["body"]
            for c in p.state()["comments"].get("10", [])
        )
    else:
        assert result["pending"] == [10]
        assert result["closed"] == ([] if dry_run else [10])
        assert "I-1-second" in p.state()["issues"]["10"]["body"]
        assert p.state()["issues"]["10"]["state"] == ("open" if dry_run else "closed")
