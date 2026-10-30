"""Public collector contracts, using private disposable registry/run fixtures.

Design: protect the eight issue states, identity/freshness boundaries, public
redaction, read-only files and explicit GET-only GitHub enrichment. In particular,
a reused PID, a stale delegation and exit 0 must never imply task completion.
These API/CLI e2e cases complement test_agent_runtime's fake CLI coverage and
test_parent_launch's existing launch-plan coverage. No real agents or GitHub.
Budget: fewer than 50 expansions per function; local cases should take <1s,
GitHub fixture cases <5s. Measure setup/call/teardown with --durations=0.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"
NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)
DELEGATION = "a" * 32


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(OPS))
    monkeypatch.setenv("PATH", str(tmp_path))  # local reads must never run gh
    api = importlib.import_module("worker_status")
    row: dict[str, Any] = {
        "session": "claude:private-owner-session",
        "status": "active",
        "generation": 2,
        "reviews": [],
        "tasks": {"11": {}},
    }
    state = tmp_path / "parent"
    state.mkdir()
    path = state / "registry.json"

    class Fixture:
        def __init__(self) -> None:
            self.api, self.row, self.state = api, row, state
            self.task = row["tasks"]["11"]
            self.run = state / "private-run-directory"

        def save(self) -> None:
            path.write_text(json.dumps({"version": 1, "parents": {"o/r#1": row}}))

        def collect(self, **kwargs: Any) -> dict[str, Any]:
            self.save()
            result: dict[str, Any] = api.collect_status(
                state_dir=state, now=NOW, **kwargs
            )
            return result

        def child(self, **kwargs: Any) -> dict[str, Any]:
            result: dict[str, Any] = self.collect(**kwargs)["parents"][0]["children"][0]
            return result

        def record(self, result: str = "completed", **kwargs: Any) -> None:
            self.task.update(delegated=True, delegation_id=DELEGATION)
            self.task.setdefault("worktree", str(tmp_path / "private-worktree"))
            self.task.setdefault("branch", "codex/fixture-worker")
            self.task["worker_runs"] = [
                {
                    "delegation_id": DELEGATION,
                    "run_dir": self.run.name,
                    "kind": "initial",
                    "recorded_at": NOW.isoformat(),
                }
            ]
            self.run.mkdir(exist_ok=True)
            runtime = {
                "schema_version": 1,
                "run_id": "private-run-token",
                "run_dir": str(self.run),
                "session_root": str(self.run),
                "result": result,
                "turn_end_reason": "completed" if result == "completed" else None,
                "runner": None,
                "process": None,
                "started_at": NOW.timestamp() - 1000,
                "finished_at": NOW.timestamp() if result != "running" else None,
                "contract": {
                    "task_id": DELEGATION,
                    "cwd": self.task["worktree"],
                    "branch": self.task["branch"],
                    "settings": {"agent": "codex"},
                },
                "runtime": {
                    "session_id": "private-child-session",
                    "turn_id": "private-turn",
                },
                **kwargs,
            }
            (self.run / "result.json").write_text(json.dumps(runtime))
            (self.run / "session.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "session_id": "private-child-session",
                        "last_turn_id": "private-turn",
                        "latest_run": str(self.run),
                    }
                )
            )
            self.events([])

        def events(
            self, items: list[dict[str, Any]], *, age: float = 0, receipts: bool = True
        ) -> None:
            offset = 0
            timing = []
            raw = bytearray()
            for event in items:
                line = json.dumps(event).encode()
                raw.extend(line + b"\n")
                timing.append(
                    {
                        "schema_version": 1,
                        "event_start": offset,
                        "event_end": len(raw),
                        "sha256": hashlib.sha256(line).hexdigest(),
                        "received_at": (NOW - timedelta(seconds=age)).isoformat(),
                    }
                )
                offset = len(raw)
            (self.run / "events.jsonl").write_bytes(raw)
            times = self.run / "event-receipts.jsonl"
            if receipts:
                times.write_text("".join(json.dumps(t) + "\n" for t in timing))
            else:
                times.unlink(missing_ok=True)

    return Fixture()


def item(kind: str, **fields: Any) -> dict[str, Any]:
    return {
        "type": "item.completed",
        "item": {"id": kind + "-1", "type": kind, **fields},
    }


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        ("running", "running"),
        ("completed", "turn_completed"),
        ("failed", "failed"),
        ("stopped", "stopped"),
        ("dependency", "dependency_wait"),
        ("reserved", "not_started"),
        ("undelegated", "not_delegated"),
        ("lost", "unknown"),
    ],
)
def test_states_have_sources_and_never_imply_task_acceptance(
    fixture: Any, setup: str, expected: str
) -> None:
    f = fixture
    if setup in {"running", "completed", "failed", "lost"}:
        identity = importlib.import_module("lib.opsproc").process_identity(os.getpid())
        f.record(
            "running" if setup in {"running", "lost"} else setup,
            runner=identity if setup == "running" else None,
            turn_end_reason="usage_limit"
            if setup == "failed"
            else "completed"
            if setup == "completed"
            else None,
        )
    elif setup == "stopped":
        f.task.update(
            stopped=True,
            last_status={
                "reason": "仕様待ち",
                "resume_condition": "回答後",
                "next_action": "親へ確認",
            },
        )
    elif setup == "dependency":
        f.task["dependencies"] = [12]
        f.row["tasks"]["12"] = {"stopped": True}
    elif setup == "reserved":
        f.task.update(delegated=True, delegation_id=DELEGATION, worker_runs=[])
    child = f.child()
    assert child["status"] == expected
    assert child["evidence"] and all(
        e["source"] and e["fields"] for e in child["evidence"]
    )
    assert child["github"]["status"] == "not_requested"
    if setup == "failed":
        assert child["run"]["turn_end_reason"] == "usage_limit"


@pytest.mark.parametrize(
    "damage",
    [
        "pid_reused",
        "runner_missing",
        "bad_json",
        "version",
        "session_mismatch",
        "old_delegation",
        "partial_run",
        "events_corrupt",
    ],
)
def test_untrusted_records_never_show_running(fixture: Any, damage: str) -> None:
    f = fixture
    identity = importlib.import_module("lib.opsproc").process_identity(os.getpid())
    assert identity is not None
    f.record("running", runner=identity)
    if damage == "pid_reused":
        identity["start_ticks"] = "0"
        f.record("running", runner=identity)
    elif damage == "runner_missing":
        f.record("running", runner=None)
    elif damage == "bad_json":
        (f.run / "result.json").write_text("{")
    elif damage == "version":
        f.record("running", runner=identity, schema_version=77)
    elif damage == "session_mismatch":
        (f.run / "session.json").write_text(
            '{"schema_version":1,"session_id":"wrong","latest_run":"wrong"}'
        )
    elif damage == "old_delegation":
        f.task["delegation_id"] = "b" * 32
    elif damage == "partial_run":
        (f.run / "result.json").unlink()
    else:
        (f.run / "events.jsonl").write_text('{"type":wrong}\n')
    child = f.child()
    assert child["status"] == (
        "not_started" if damage == "old_delegation" else "unknown"
    )
    assert child["evidence"]
    if damage == "version":
        assert child["evidence"][-1]["code"] == "unsupported_schema"
    elif damage == "bad_json":
        assert child["evidence"][-1]["code"] == "corrupt_json"
    elif damage == "pid_reused":
        assert child["run"]["identity_status"] == "identity_mismatch"


@pytest.mark.parametrize("record", ["result", "invocation"])
@pytest.mark.parametrize("field", ["task_id", "cwd", "branch"])
@pytest.mark.parametrize("missing", [False, True])
def test_runner_contract_must_match_the_current_delegation(
    fixture: Any, record: str, field: str, missing: bool
) -> None:
    f = fixture
    f.record(
        "running",
        runner=importlib.import_module("lib.opsproc").process_identity(os.getpid()),
    )
    path = f.run / "result.json"
    value = json.loads(path.read_text())
    if missing:
        value["contract"].pop(field)
    else:
        value["contract"][field] = {
            "task_id": "b" * 32,
            "cwd": str(f.state / "other-worktree"),
            "branch": "codex/other-worker",
        }[field]
    if record == "invocation":
        path.unlink()
        path = f.run / "invocation.json"
    path.write_text(json.dumps(value))
    child = f.child()
    assert child["status"] == "unknown"
    assert child["evidence"][-1] == {
        "source": f"runner.{record}",
        "fields": [f"contract.{field}"],
        "code": "contract_missing" if missing else "contract_mismatch",
    }
    assert value["contract"].get(field, "not-in-output") not in json.dumps(child)


def test_legacy_delegation_without_run_tracking_is_unknown(fixture: Any) -> None:
    f = fixture
    f.task.update(delegated=True, delegation_id=DELEGATION)
    child = f.child()
    assert child["status"] == "unknown"
    assert child["evidence"][-1]["code"] == "no_run_record"
    assert "worker_runs" not in f.task


def test_missing_result_only_uses_invocation_to_verify_contract(fixture: Any) -> None:
    f = fixture
    f.record()
    result = f.run / "result.json"
    (f.run / "invocation.json").write_bytes(result.read_bytes())
    result.unlink()
    child = f.child()
    assert child["status"] == "unknown"
    assert child["evidence"][-1]["code"] == "record_missing"


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("delegation_id", "delegation_id"),
        ("worktree", "contract.cwd"),
        ("branch", "contract.branch"),
    ],
)
def test_current_delegation_contract_fields_are_required(
    fixture: Any, field: str, expected: str
) -> None:
    f = fixture
    f.record()
    f.task.pop(field)
    child = f.child()
    assert child["status"] == "unknown"
    assert child["evidence"][-1]["code"] == "contract_missing"
    assert child["evidence"][-1]["fields"] == [expected]


def test_corrupt_result_does_not_fall_back_to_matching_invocation(fixture: Any) -> None:
    f = fixture
    f.record()
    path = f.run / "result.json"
    (f.run / "invocation.json").write_bytes(path.read_bytes())
    path.write_text("{")
    child = f.child()
    assert child["status"] == "unknown"
    assert child["evidence"][-1]["source"] == "runner.result"
    assert child["evidence"][-1]["code"] == "corrupt_json"


def test_staleness_uses_received_time_and_never_stops_a_live_runner(
    fixture: Any,
) -> None:
    f = fixture
    identity = importlib.import_module("lib.opsproc").process_identity(os.getpid())
    f.record("running", runner=identity)
    f.events([item("command_execution", command="pytest -q")], age=300)
    child = f.child(stale_after=300)
    assert child["status"] == "running" and child["updates_stalled"] is False
    assert child["last_activity_at"] == (NOW - timedelta(seconds=300)).isoformat()
    assert f.child(stale_after=299)["updates_stalled"] is True
    assert f.child(stale_after=301)["updates_stalled"] is False


def test_summary_last_message_recent_tools_and_legacy_times(fixture: Any) -> None:
    f = fixture
    f.record()
    f.events(
        [
            item("agent_message", text="obsolete"),
            item("reasoning", text="never publish this"),
            item(
                "command_execution",
                command="pytest -q\nsecond line",
                aggregated_output="private output",
            ),
            item("mcp_tool_call", server="docs", tool="search"),
            item("file_change", changes=[{"path": "/private/file", "kind": "add"}]),
            item("agent_message", text="完了" * 200),
        ]
    )
    summary = f.child(tool_limit=2)["summary"]
    assert summary["message"]["text"] == ("完了" * 200)[:300]
    assert summary["message"]["truncated"] is True
    assert [t["kind"] for t in summary["tools"]] == ["mcp_tool_call", "file_change"]
    assert summary["tools"][0]["server"]["text"] == "docs"
    assert summary["tools"][1]["change_count"] == 1
    assert all(t["received_at"] == NOW.isoformat() for t in summary["tools"])
    serialized = json.dumps(summary)
    assert "never publish" not in serialized and "private output" not in serialized
    f.events([item("command_execution", command="pytest -q")], receipts=False)
    legacy = f.child()
    assert legacy["activity_time_status"] == "unavailable"
    assert legacy["last_activity_at"] is None
    assert legacy["summary"]["tools"][0]["time_status"] == "unavailable"


@pytest.mark.parametrize(
    "private",
    [
        "/home/private/file",
        "private-child-session",
        DELEGATION,
        "private-run-directory",
        "secret-environment-value",
        "thread_id=opaque",
        "ghp_" + "x" * 36,
    ],
)
def test_private_text_is_hidden_before_truncating(
    fixture: Any, monkeypatch: pytest.MonkeyPatch, private: str
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "secret-environment-value")
    f = fixture
    f.record()
    f.events(
        [
            item("agent_message", text="safe prefix " * 40 + private),
            item("command_execution", command="pytest\n" + private),
        ]
    )
    child = f.child()
    assert child["summary"]["message"]["status"] == "hidden"
    assert child["summary"]["message"]["text"] is None
    assert child["summary"]["tools"][0]["command"]["status"] == "hidden"
    assert private not in json.dumps(child, ensure_ascii=False)


@pytest.mark.parametrize("github", [False, True])
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("GITHUB_EVENT_NAME", "pull_request"),
        ("USER", "completed"),
        ("LOGNAME", "observed"),
        ("WSL_DISTRO_NAME", "Ubuntu-24.04"),
        ("LANG", "en_US.UTF-8"),
    ],
)
def test_environment_metadata_only_hides_free_text(
    fixture: Any, monkeypatch: pytest.MonkeyPatch, github: bool, key: str, value: str
) -> None:
    f = fixture
    monkeypatch.setenv(key, value)
    f.record()
    f.events(
        [
            item("agent_message", text="safe prefix " * 40 + value),
            item("command_execution", command="pytest\n" + value),
        ]
    )
    calls = []

    def snapshot(repo: str, parent: int, store: Any, guard: Any) -> dict[int, Any]:
        calls.append(parent)
        return {
            11: {
                "status": "observed",
                "issue": {"state": "open", "state_reason": None},
                "pull_requests": [],
                "stopped": False,
                "stop_record": {},
                "launch": {"reasons": []},
            }
        }

    monkeypatch.setattr(
        importlib.import_module("worker_status.collector"), "parent_snapshot", snapshot
    )
    output = f.collect(github=github)
    assert calls == ([1] if github else [])
    child = output["parents"][0]["children"][0]
    assert child["status"] == "turn_completed"
    assert child["summary"]["message"]["status"] == "hidden"
    assert child["summary"]["message"]["text"] is None
    assert child["summary"]["tools"][0]["command"]["status"] == "hidden"
    # #1569 adds these fields, then uses this same board public validator.
    output["parents"][0]["roster"] = {
        "name": "Test parent",
        "identifier": "Worker",
        "state": "unknown",
        "source": "board_names / parent_registry / Herdr",
    }
    output["github_refresh"] = {
        "epoch": "public-epoch",
        "status": "idle",
        "revision": 1,
        "result": "succeeded",
    }
    importlib.import_module("board.safety").public(output, {"o/r#1": f.row}, os.environ)


@pytest.mark.parametrize(
    "key", ["CLAUDE_CODE_PARENT_SESSION_ID", "AGENT_RUNTIME_TASK_ID"]
)
def test_identity_environment_values_remain_private_in_structured_output(
    fixture: Any, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    f = fixture
    secret = "opaque-private-identity"
    monkeypatch.setenv(key, secret)
    f.record()
    f.events([item("agent_message", text=secret)])
    assert f.child()["summary"]["message"]["status"] == "hidden"
    f.task.update(stopped=True, last_status={secret: "safe text"})
    with pytest.raises(f.api.StatusError, match="^unsafe_public_field$"):
        f.child()


def tree_snapshot(path: Path) -> dict[str, Any]:
    return {
        str(p.relative_to(path)): (
            p.read_bytes(),
            p.stat().st_mtime_ns,
            p.stat().st_mode,
        )
        for p in path.rglob("*")
        if p.is_file()
    }


def test_cli_is_read_only_and_public_even_while_writer_lock_is_held(
    fixture: Any,
) -> None:
    import fcntl

    f = fixture
    f.record()
    f.save()
    with (f.state / "registry.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = tree_snapshot(f.state)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "worker_status",
                "--state-dir",
                str(f.state),
                "--parent",
                "1",
            ],
            env={**os.environ, "PYTHONPATH": str(OPS)},
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert tree_snapshot(f.state) == before
    output = json.loads(proc.stdout)
    assert output["dispatcher"]["status"] == "unused"
    assert output["parents"][0]["children"][0]["status"] == "turn_completed"
    for secret in (
        str(f.state),
        f.run.name,
        "private-owner-session",
        "private-child-session",
        DELEGATION,
    ):
        assert secret not in proc.stdout
    importlib.import_module("board.safety").public(output, {"o/r#1": f.row}, os.environ)


def test_inactive_parents_and_unknown_registry_are_explicit(fixture: Any) -> None:
    f = fixture
    f.row["status"] = "ended"
    assert f.collect()["parents"] == []
    assert len(f.collect(include_inactive=True)["parents"]) == 1
    (f.state / "registry.json").write_text('{"version":77,"parents":{}}')
    with pytest.raises(f.api.StatusError):
        f.api.collect_status(state_dir=f.state)


def write_registry(state: Path, parents: dict[str, Any], version: int = 1) -> None:
    """Publish coherent disposable v1/v2 rows, without production mutations."""
    entries = parents
    if version == 2:
        digest = importlib.import_module("parent.policy").digest
        directory = state / "parents"
        directory.mkdir(exist_ok=True)
        for key, row in parents.items():
            (directory / (digest(key) + ".json")).write_text(json.dumps(row))
        entries = {
            key: {field: row[field] for field in ("session", "status", "generation")}
            for key, row in parents.items()
        }
    (state / "registry.json").write_text(
        json.dumps({"version": version, "parents": entries})
    )


@pytest.mark.parametrize("github", [False, True])
@pytest.mark.parametrize("excluded_by", ["parent", "repo", "inactive"])
def test_updates_to_excluded_parents_do_not_invalidate_observation(
    fixture: Any, monkeypatch: pytest.MonkeyPatch, github: bool, excluded_by: str
) -> None:
    f = fixture
    collector = importlib.import_module("worker_status.collector")
    peer = deepcopy(f.row)
    peer["session"] = "claude:private-excluded-owner"
    peer_key = "other/repo#2" if excluded_by == "repo" else "o/r#2"
    if excluded_by == "inactive":
        peer["status"] = "ended"
    parents = {"o/r#1": f.row, peer_key: peer}
    write_registry(f.state, parents)
    queries: dict[str, Any] = (
        {"parent": 1}
        if excluded_by == "parent"
        else {"repo": "o/r"}
        if excluded_by == "repo"
        else {}
    )
    calls = []
    if github:

        def snapshot(repo: str, parent: int, store: Any, guard: Any) -> dict[int, Any]:
            calls.append(parent)
            peer["generation"] += 1
            write_registry(f.state, parents)
            return {}

        monkeypatch.setattr(collector, "parent_snapshot", snapshot)
    else:
        original = collector.child_status

        def child_status(*args: Any, **kwargs: Any) -> dict[str, Any]:
            peer["generation"] += 1
            write_registry(f.state, parents)
            result: dict[str, Any] = original(*args, **kwargs)
            return result

        monkeypatch.setattr(collector, "child_status", child_status)
    result = f.api.collect_status(state_dir=f.state, github=github, now=NOW, **queries)
    assert [p["parent"] for p in result["parents"]] == [1]
    assert result["parents"][0]["generation"] == 2
    assert peer["generation"] == 3
    assert calls == ([1] if github else [])


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("changes", [1, 2])
def test_changed_parent_is_retrieved_once_without_mixing_generations(
    fixture: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    version: int,
    changes: int,
) -> None:
    f = fixture
    collector = importlib.import_module("worker_status.collector")
    stable = deepcopy(f.row)
    stable.update(session="claude:private-stable-owner", generation=5)
    parents = {"o/r#1": stable, "o/r#2": f.row}
    write_registry(f.state, parents, version)
    calls = []
    stable_calls = []
    old_owner = "private-owner-session"

    def snapshot(repo: str, parent: int, store: Any, guard: Any) -> dict[int, Any]:
        row = store.parents[f"o/r#{parent}"]
        if parent == 1:
            stable_calls.append(parent)
            return {}
        calls.append(row["generation"])
        child = int(next(iter(row["tasks"])))
        remote = {
            child: {
                "status": "observed",
                "issue": {
                    "state": "open" if row["generation"] == 2 else "closed",
                    "state_reason": None if row["generation"] == 2 else "completed",
                },
                "stopped": True,
                "stop_record": {
                    "reason": guard.text(row["session"].partition(":")[2]),
                    "facts": guard.text(old_owner),
                },
                "launch": {"reasons": []},
            }
        }
        if len(calls) <= changes:
            f.row.update(
                generation=f.row["generation"] + 1,
                session="claude:private-new-owner",
                tasks={"12": {}},
            )
            write_registry(f.state, parents, version)
        return remote

    monkeypatch.setattr(collector, "parent_snapshot", snapshot)
    if changes == 1:
        result = f.api.collect_status(state_dir=f.state, github=True, now=NOW)
        assert result["parents"][0]["generation"] == 5
        parent = result["parents"][1]
        assert parent["parent"] == 2
        assert parent["generation"] == 3
        assert [c["child"] for c in parent["children"]] == [12]
        child = parent["children"][0]
        assert child["github"]["issue"] == {
            "state": "closed",
            "state_reason": "completed",
        }
        assert child["stop_record"]["reason"]["status"] == "hidden"
        assert child["stop_record"]["facts"]["status"] == "hidden"
        assert "private-new-owner" not in json.dumps(result)
        assert result["github"]["status"] == "observed"
    else:
        cli = importlib.import_module("worker_status.__main__")
        assert cli.main(["--state-dir", str(f.state), "--github"]) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert json.loads(captured.err) == {"error": "registry_changed_retry"}
    assert calls == [2, 3]
    assert stable_calls == [1]


def test_each_parent_is_checked_before_fetching_the_next_parent(
    fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = fixture
    collector = importlib.import_module("worker_status.collector")
    peer = deepcopy(f.row)
    peer["session"] = "claude:private-peer-owner"
    parents = {"o/r#1": f.row, "o/r#2": peer}
    write_registry(f.state, parents)
    calls = []

    def snapshot(repo: str, parent: int, store: Any, guard: Any) -> dict[int, Any]:
        row = store.parents[f"o/r#{parent}"]
        calls.append((parent, row["generation"]))
        if parent == 1:
            peer["generation"] = 7
        else:
            f.row["generation"] = 9
        write_registry(f.state, parents)
        return {}

    monkeypatch.setattr(collector, "parent_snapshot", snapshot)
    result = f.api.collect_status(state_dir=f.state, github=True, now=NOW)
    assert [(p["parent"], p["generation"]) for p in result["parents"]] == [
        (1, 2),
        (2, 7),
    ]
    assert calls == [(1, 2), (2, 7)]
    assert f.row["generation"] == 9


def test_local_change_to_selected_parent_still_fails(
    fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = fixture
    f.save()
    collector = importlib.import_module("worker_status.collector")
    original = collector.child_status

    def child_status(*args: Any, **kwargs: Any) -> dict[str, Any]:
        f.row["generation"] += 1
        f.save()
        result: dict[str, Any] = original(*args, **kwargs)
        return result

    monkeypatch.setattr(collector, "child_status", child_status)
    with pytest.raises(f.api.StatusError, match="^registry_changed_retry$"):
        f.api.collect_status(state_dir=f.state, github=False, now=NOW)


def test_github_is_explicit_get_only_and_separates_turn_from_acceptance(
    parent_env: ParentEnv, fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    p, f = parent_env, fixture
    gh_state = p.state()
    policy = importlib.import_module("parent.policy")
    agreement = {
        k: "fixture agreement"
        for k in ("purpose", "scope", "behavior", "acceptance", "human_confirmation")
    }
    gh_state["comments"]["1"] = [
        {
            "id": 1,
            "user": {"login": "operator"},
            "body": policy.marker(
                {
                    "event": "agreed",
                    "parent": 1,
                    "agreement": agreement,
                    "agreement_sha256": policy.digest(agreement),
                }
            ),
        }
    ]
    stop = {
        "reason": "仕様の確認待ち",
        "facts": "再現を確認済み",
        "impact": "依存先",
        "resume_condition": "判断の合意を記録",
        "next_action": "親へ確認",
    }
    for n in (10, 11, 12):
        sections = {name: "fixture requirements" for name in p.formats.CHILD_REQUIRED}
        sections["Acceptance checklist"] = (
            "- [ ] 動作する\n  文脈: 親の目的。\n  下流への影響: 状態表示。\n  検証方法・証拠: fake gh。"
        )
        sections["依存関係"] = "#11" if n == 12 else "なし。"
        if n == 10:
            sections["作業状況・停止事項"] = (
                "再開\n\n<!-- parent-task-status:v1 state=resumed detail="
                + "d" * 64
                + " -->"
            )
        if n == 11:
            sections["作業状況・停止事項"] = (
                "\n\n".join(f"{k}: {v}" for k, v in stop.items())
                + "\n\n<!-- parent-task-status:v1 state=stopped detail="
                + "d" * 64
                + " -->"
            )
        gh_state["issues"][str(n)] = {
            **gh_state["issues"]["1"],
            "id": 10000 + n,
            "number": n,
            "title": f"child {n}",
            "labels": [{"name": "issue:child"}],
            "body": p.formats.render_child_body(1, sections),
        }
        gh_state["parents"][str(n)] = 1
    gh_state["prs"] = [
        {
            "number": 100,
            "body": "Closes #10",
            "state": "open",
            "merged_at": None,
            "html_url": "https://github.com/o/r/pull/100",
            "head": {
                "sha": "a" * 40,
                "ref": "codex/work",
                "repo": {"full_name": "o/r"},
            },
        },
        {"number": 101, "body": "Refs #10", "state": "open"},
    ]
    p.save(gh_state)
    for key, value in p.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    f.row["tasks"] = {"10": f.task}
    f.record()
    f.task.update(stopped=True, last_status={"reason": "古い停止理由"})
    f.events([item("agent_message", text="変更を提出しました")])
    f.save()
    files_before = tree_snapshot(f.state)
    remote_before = p.state_path.read_bytes()
    calls_before = len(p.calls())
    local = f.api.collect_status(state_dir=f.state, parent=1)
    assert len(p.calls()) == calls_before
    assert local["github"]["status"] == "not_requested"
    assert local["parents"][0]["children"][0]["status"] == "stopped"
    remote = f.api.collect_status(state_dir=f.state, parent=1, github=True)
    children = {c["child"]: c for c in remote["parents"][0]["children"]}
    assert children[10]["status"] == "turn_completed"
    assert children[10]["github"]["issue"] == {"state": "open", "state_reason": None}
    assert children[10]["github"]["pull_requests"] == [
        {
            "number": 100,
            "state": "open",
            "merged": False,
            "url": "https://github.com/o/r/pull/100",
        }
    ]
    assert children[11]["status"] == "stopped"
    assert children[11]["stop_record"]["reason"]["text"] == "仕様の確認待ち"
    assert children[11]["stop_record"]["resume_condition"]["text"] == "判断の合意を記録"
    assert children[12]["status"] == "dependency_wait"
    assert "dependency_blocked" in {
        r["code"] for r in children[12]["github"]["launch"]["reasons"]
    }
    assert remote["github"]["fetched_at"] is not None
    assert all(
        args[0] == "api" and args[args.index("--method") + 1] == "GET"
        for args in p.calls()[calls_before:]
    )
    assert p.state_path.read_bytes() == remote_before
    assert tree_snapshot(f.state) == files_before

    monkeypatch.setenv("PARENT_FAIL_GET_PATH", "repos/o/r/issues/1")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "worker_status",
            "--state-dir",
            str(f.state),
            "--github",
        ],
        env={**os.environ, "PYTHONPATH": str(OPS)},
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert proc.returncode == 2 and proc.stdout == ""
    assert json.loads(proc.stderr) == {"error": "github_observation_failed"}
    assert str(f.state) not in proc.stderr


def test_latest_current_run_survives_handoff_and_v2_storage(fixture: Any) -> None:
    f = fixture
    f.record("failed", turn_end_reason="usage_limit")
    old_run = f.task["worker_runs"][0].copy()
    f.task["worker_runs"].insert(
        0, {**old_run, "delegation_id": "c" * 32, "run_dir": "missing-old-run"}
    )
    f.task["worker_runs"].append({**old_run, "kind": "resume"})
    f.row.update(
        session="claude:new-private-owner", generation=3, status="handoff_pending"
    )
    f.save()
    policy = importlib.import_module("parent.policy")
    parent_file = f.state / "parents" / (policy.digest("o/r#1") + ".json")
    parent_file.parent.mkdir()
    parent_file.write_text(json.dumps(f.row))
    (f.state / "registry.json").write_text(
        json.dumps(
            {
                "version": 2,
                "parents": {
                    "o/r#1": {k: f.row[k] for k in ("session", "status", "generation")}
                },
            }
        )
    )
    before = tree_snapshot(f.state)
    result = f.api.collect_status(state_dir=f.state, now=NOW)
    child = result["parents"][0]["children"][0]
    assert child["status"] == "failed" and child["run"]["alias"] == "run-3"
    assert child["run"]["kind"] == "resume"
    assert result["parents"][0]["assignment"] == "handoff_pending"
    assert result["parents"][0]["generation"] == 3
    assert tree_snapshot(f.state) == before


def test_bounded_tail_drops_large_reasoning_and_reports_omission(fixture: Any) -> None:
    f = fixture
    f.record()
    f.events(
        [
            item("reasoning", text="private reasoning " * 100000),
            item("agent_message", text="提出済み"),
        ]
    )
    child = f.child(tail_bytes=1024)
    assert child["status"] == "turn_completed"
    assert child["summary"]["message"]["text"] == "提出済み"
    assert child["summary"]["tail_truncated"] is True
    assert child["last_activity_at"] == NOW.isoformat()
    f.events([item("reasoning", text="private reasoning " * 100000)])
    child = f.child(tail_bytes=1024)
    assert child["summary"]["message"]["status"] == "unavailable"
    assert child["last_activity_at"] == NOW.isoformat()


@pytest.mark.parametrize("damage", ["digest", "json", "schema"])
def test_corrupt_event_receipts_are_unknown(fixture: Any, damage: str) -> None:
    f = fixture
    f.record(
        "running",
        runner=importlib.import_module("lib.opsproc").process_identity(os.getpid()),
    )
    f.events([item("agent_message", text="safe")])
    path = f.run / "event-receipts.jsonl"
    receipt = json.loads(path.read_text())
    if damage == "digest":
        receipt["sha256"] = "0" * 64
    elif damage == "schema":
        receipt["schema_version"] = 77
    path.write_text("{\n" if damage == "json" else json.dumps(receipt) + "\n")
    assert f.child()["status"] == "unknown"


def test_stop_record_is_independent_of_damaged_runner_and_hides_private_notes(
    fixture: Any,
) -> None:
    f = fixture
    f.record()
    (f.run / "result.json").write_text("{")
    f.task.update(
        stopped=True,
        last_status={
            "reason": "/private/location",
            "resume_condition": "親の確認後",
            "next_action": "再調査",
        },
    )
    child = f.child()
    assert child["status"] == "stopped"
    assert child["stop_record"]["reason"]["status"] == "hidden"
    assert child["stop_record"]["next_action"]["text"] == "再調査"


def test_standard_run_directory_name_does_not_mask_the_run_kind(fixture: Any) -> None:
    f = fixture
    f.run = f.state / "initial"
    f.record()
    child = f.child()
    assert child["status"] == "turn_completed" and child["run"]["kind"] == "initial"


def test_structural_private_field_fails_without_partial_stdout(fixture: Any) -> None:
    f = fixture
    f.task.update(stopped=True, last_status={"/private/key": "安全な内容"})
    f.save()
    proc = subprocess.run(
        [sys.executable, "-m", "worker_status", "--state-dir", str(f.state)],
        env={**os.environ, "PYTHONPATH": str(OPS)},
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert proc.returncode == 2 and proc.stdout == ""
    assert json.loads(proc.stderr) == {"error": "unsafe_public_field"}


def test_dispatcher_current_child_job_and_unused_schedule_are_distinct(
    fixture: Any,
) -> None:
    f = fixture
    f.record()
    f.task.pop("worker_runs")
    directory = f.state.parent / "dispatcher"
    directory.mkdir()
    schedule = importlib.import_module("dispatcher.schedule").Schedule(directory)
    schedule.initialize(
        {
            "repo": "o/r",
            "cwd": str(f.state),
            "runner": str(f.state / "runner"),
            "max_parents": 1,
            "max_children": 2,
            "judgement_timeout": 60,
        },
        NOW,
    )
    job_root = f.state / "job"
    f.run.rename(job_root)
    runtime_dir = job_root / "runtime"
    runtime_dir.mkdir()
    for path in list(job_root.iterdir()):
        if path.is_file():
            path.rename(runtime_dir / path.name)
    for name in ("result.json", "session.json"):
        path = runtime_dir / name
        path.write_text(path.read_text().replace(str(f.run), str(runtime_dir)))
    result_path = runtime_dir / "result.json"
    result = json.loads(result_path.read_text())
    result["contract"]["task_id"] = "private-dispatcher-job"
    result_path.write_text(json.dumps(result))
    f.row["dispatch"] = {
        "schema_version": 1,
        "jobs": {
            "old": {
                "sequence": 1,
                "role": "implementation",
                "child": 11,
                "delegation_id": "b" * 32,
            },
            "current": {
                "sequence": 2,
                "role": "implementation",
                "child": 11,
                "delegation_id": DELEGATION,
                "directory": str(job_root),
                "phase": "ended",
                "task_id": "private-dispatcher-job",
            },
        },
    }
    f.save()
    before = tree_snapshot(f.state.parent)
    value = f.api.collect_status(state_dir=f.state, now=NOW)
    child = value["parents"][0]["children"][0]
    assert value["dispatcher"]["status"] == "available"
    assert child["status"] == "turn_completed"
    assert child["run"]["alias"] == "dispatcher-run-2"
    assert child["run"]["source"] == "parent_registry.dispatch.jobs"
    assert tree_snapshot(f.state.parent) == before
