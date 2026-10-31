"""Web reads and human writes through temporary state and fake notifier routes."""

from __future__ import annotations

import importlib
import json
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.e2e.agents_ops.test_board import board as board
from tests.e2e.agents_ops.test_board import (
    prepare_roster,
    private_output_absent,
    run_board,
)
from tests.e2e.agents_ops.test_board_messages import World
from tests.e2e.agents_ops.test_board_messages import world as world
from tests.e2e.agents_ops.test_herdr_labels import labels as labels


@pytest.fixture
def web(world: World) -> SimpleNamespace:
    return SimpleNamespace(
        app=importlib.import_module("board.web"),
        data=importlib.import_module("board.web_data"),
    )


@pytest.fixture
def client(world: World, web: SimpleNamespace) -> Iterator[TestClient]:
    with TestClient(
        web.app.create_app(world.env(), notification_routes=world.routes),
        base_url="http://127.0.0.1",
    ) as client:
        yield client


# Execution test design: preserve the collector's states and independent Issue
# acceptance, explicit-only GitHub/Herdr reads, cache/failure semantics and public
# GET boundaries. Extend the existing HTTP/read-only/security tests below, and
# exercise the production scripts with a virtual clock (board_execution.cjs).
# Budget: <50 expansions/function, <5s/case; measure via --durations=0. No agents,
# GPU, live GitHub or extra browser dependencies are needed for these checks.
@pytest.fixture
def execution(world: World, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    api = importlib.import_module("board.execution_data")
    now = "2026-10-30T12:00:00+00:00"

    def text(value: str | None, status: str = "visible") -> dict[str, Any]:
        return {"status": status, "text": value, "truncated": False}

    children = []
    for number, (status, label) in enumerate(
        (
            ("running", "実行中"),
            ("turn_completed", "ターン終了"),
            ("failed", "失敗"),
            ("stopped", "停止中"),
            ("dependency_wait", "依存待ち"),
            ("not_started", "委任予約済み・未起動"),
            ("not_delegated", "未委任"),
            ("unknown", "状態不明"),
        ),
        11,
    ):
        children.append(
            {
                "child": number,
                "status": status,
                "label": label,
                "delegation": "delegation-display",
                "run": None,
                "observed_at": now,
                "last_activity_at": now,
                "activity_time_status": "observed",
                "updates_stalled": status == "running",
                "evidence": [
                    {
                        "source": "runner.result",
                        "code": "identity_verified",
                        "fields": ["result", "runner.start_ticks"],
                    }
                ],
                "summary": {
                    "message": {
                        **text("<img src=x onerror=alert(1)> を文字として表示"),
                        "received_at": now,
                        "time_status": "observed",
                    },
                    "tools": [
                        {
                            "kind": "command_execution",
                            "command": text(None, "hidden"),
                            "received_at": now,
                            "time_status": "observed",
                        },
                        {
                            "kind": "mcp_tool_call",
                            "server": text("lookup"),
                            "tool": text("search"),
                            "received_at": None,
                            "time_status": "unavailable",
                        },
                        {
                            "kind": "file_change",
                            "change_count": 2,
                            "received_at": now,
                            "time_status": "observed",
                        },
                    ],
                    "tail_truncated": True,
                },
                "stop_record": {
                    "reason": text("依存PRを待っています。"),
                    "next_action": text("親が差分を確認する。"),
                }
                if status == "stopped"
                else {},
                "dependencies": [],
                "github": {"status": "not_requested"},
            }
        )
    snapshot = {
        "schema_version": 1,
        "observed_at": now,
        "stale_after_seconds": 300,
        "dispatcher": {"status": "unused", "label": "自律dispatcher未使用"},
        "github": {"status": "not_requested", "fetched_at": None},
        "parents": [
            {
                "repo": "o/r",
                "parent": 1,
                "assignment": "active",
                "generation": 1,
                "source": "parent_registry",
                "children": children,
            }
        ],
    }
    fake = SimpleNamespace(
        api=api, snapshot=snapshot, calls=[], fail=False, fetched_at=now
    )

    def collect(**kwargs: Any) -> dict[str, Any]:
        fake.calls.append(kwargs)
        assert kwargs["state_dir"] == world.labels.registry.parent
        if fake.fail:
            raise ValueError("private runtime details")
        result: dict[str, Any] = deepcopy(fake.snapshot)
        if kwargs["github"]:
            result["github"] = {"status": "observed", "fetched_at": fake.fetched_at}
            for parent in result["parents"]:
                for child in parent["children"]:
                    child["github"] = {
                        "status": "observed",
                        "issue": {"state": "open", "state_reason": None},
                        "pull_requests": [
                            {
                                "number": 88,
                                "state": "open",
                                "merged": False,
                                "url": "https://github.com/o/r/pull/88",
                            }
                        ],
                        "launch": {
                            "action": "wait",
                            "state": "waiting",
                            "dependencies": [],
                            "reasons": [],
                        },
                        "stopped": False,
                        "stop_record": {},
                    }
        return result

    monkeypatch.setattr(api, "collect_status", collect)
    return fake


def test_execution_explicit_refresh_cache_and_fresh_local_state(
    world: World, client: TestClient, execution: SimpleNamespace
) -> None:
    calls = len(world.labels.read()["calls"])
    initial = client.get("/api/execution").json()
    parent = initial["parents"][0]
    assert parent["roster"]["name"] == world.assignments[1].name
    assert parent["roster"]["state"] == "idle"
    assert parent["children"][1]["status"] == "turn_completed"
    assert parent["children"][1]["github"] == {"status": "not_requested"}
    assert len(world.labels.read()["calls"]) == calls
    refreshed = client.get("/api/execution?refresh=true").json()
    child = refreshed["parents"][0]["children"][1]
    assert child["status"] == "turn_completed"
    assert child["github"]["issue"]["state"] == "open"
    assert child["github"]["fetched_at"] == refreshed["github"]["fetched_at"]
    assert len(world.labels.read()["calls"]) > calls
    calls = len(world.labels.read()["calls"])
    execution.snapshot["parents"][0]["children"][1].update(
        status="running", label="実行中"
    )
    cached = client.get("/api/execution").json()
    assert cached["parents"][0]["children"][1]["status"] == "running"
    assert cached["parents"][0]["children"][1]["github"] == child["github"]
    assert cached["github"] == refreshed["github"]
    assert cached["roster_observed_at"] == refreshed["roster_observed_at"]
    assert len(world.labels.read()["calls"]) == calls
    assert [call["github"] for call in execution.calls] == [False, True, False]


# Concurrency contract: hold either external source on an Event, then require
# local GET and duplicate refresh to finish before releasing it. This exposes
# both reader locks without sleep-based timing or real GitHub/Herdr latency.
# Two expansions; reuse the real HTTP route, cache and SQLite projection.
@pytest.mark.parametrize("phase", ["github", "roster"])
def test_execution_local_reads_and_duplicate_refresh_do_not_wait_for_external_io(
    world: World,
    client: TestClient,
    execution: SimpleNamespace,
    web: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    previous = client.get("/api/execution?refresh=true").json()
    entered, release = Event(), Event()
    original_collect = execution.api.collect_status
    original_agents = web.data.agents

    def pause() -> None:
        entered.set()
        assert release.wait(5), "test did not release the external observation"

    def collect(**kwargs: Any) -> dict[str, Any]:
        if phase == "github" and kwargs["github"]:
            pause()
        result: dict[str, Any] = original_collect(**kwargs)
        return result

    def agents(*args: Any) -> dict[str, Any]:
        if phase == "roster":
            pause()
        result: dict[str, Any] = original_agents(*args)
        return result

    monkeypatch.setattr(execution.api, "collect_status", collect)
    monkeypatch.setattr(web.data, "agents", agents)
    execution.snapshot["parents"][0]["children"][0]["label"] = "新しいローカル観測"
    execution.fetched_at = "2026-10-30T12:01:00+00:00"
    before = world.labels.registry.read_bytes(), world.store.path.read_bytes()
    with ThreadPoolExecutor(max_workers=3) as pool:
        refreshing = pool.submit(client.get, "/api/execution?refresh=true")
        try:
            assert entered.wait(2)
            local = pool.submit(client.get, "/api/execution").result(timeout=2)
            assert local.status_code == 200
            data = local.json()
            assert data["parents"][0]["children"][0]["label"] == "新しいローカル観測"
            assert data["github"] == previous["github"]
            assert data["github_refresh"]["status"] == "running"
            duplicate = pool.submit(client.get, "/api/execution?refresh=true").result(
                timeout=2
            )
            assert duplicate.status_code == 202
            assert duplicate.json()["github_refresh"] == data["github_refresh"]
            assert not refreshing.done()
        finally:
            release.set()
        completed = refreshing.result(timeout=2)
    assert completed.status_code == 200
    finished = client.get("/api/execution").json()
    assert finished["github"]["fetched_at"] == execution.fetched_at
    assert finished["github_refresh"]["status"] == "idle"
    assert finished["github_refresh"]["result"] == "succeeded"
    assert finished["github_refresh"]["revision"] > data["github_refresh"]["revision"]
    assert sum(call["github"] for call in execution.calls) == 2
    assert (world.labels.registry.read_bytes(), world.store.path.read_bytes()) == before


def test_execution_failures_keep_last_cache_and_never_expose_errors(
    client: TestClient,
    execution: SimpleNamespace,
    web: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = client.get("/api/execution?refresh=true").json()
    execution.fail = True
    for url in ("/api/execution", "/api/execution?refresh=true"):
        failed = client.get(url)
        assert failed.status_code == 503
        assert "private runtime details" not in failed.text
    execution.fail = False

    def no_roster(*args: Any) -> None:
        raise web.app.BoardError("private roster details")

    with monkeypatch.context() as patch:
        patch.setattr(web.data, "agents", no_roster)
        assert client.get("/api/execution?refresh=true").status_code == 503
    cached = client.get("/api/execution").json()
    state = cached.pop("github_refresh")
    assert state["status"] == "idle" and state["result"] == "failed"
    assert state["revision"] > previous.pop("github_refresh")["revision"]
    assert cached == previous
    recovered = client.get("/api/execution?refresh=true")
    assert recovered.status_code == 200
    assert recovered.json()["github_refresh"]["result"] == "succeeded"


def test_execution_cached_github_children_and_handoff_do_not_reuse_old_identity(
    world: World, client: TestClient, execution: SimpleNamespace
) -> None:
    parent = execution.snapshot["parents"][0]
    parent["children"][-1]["delegation"] = None
    client.get("/api/execution?refresh=true").raise_for_status()
    removed = parent["children"].pop()
    retained = client.get("/api/execution").json()["parents"][0]["children"][-1]
    assert retained["child"] == removed["child"]
    assert retained["cached_local_observation"] is True
    assert retained["observed_at"] == removed["observed_at"]
    # Removing a real local delegation must not leave a cached running worker.
    running = parent["children"].pop(0)
    children = client.get("/api/execution").json()["parents"][0]["children"]
    assert running["child"] not in {row["child"] for row in children}
    parent["children"][0]["delegation"] = "new-delegation"
    assert client.get("/api/execution").json()["parents"][0]["children"][0][
        "github"
    ] == {"status": "not_requested"}
    parent["generation"] = 2
    fresh = client.get("/api/execution").json()["parents"][0]
    assert fresh["roster"]["state"] == "unknown"
    assert len(fresh["children"]) == len(parent["children"])
    absent = deepcopy(parent)
    absent.update(parent=5, generation=1, assignment="handoff_pending")
    execution.snapshot["parents"].append(absent)
    roster = client.get("/api/execution").json()["parents"][1]["roster"]
    assert roster["name"] == world.assignments[5].name
    assert roster["state"] == "unknown"


@pytest.mark.parametrize(
    "private",
    [
        "/private/run-directory",
        "session_id=private",
        "01234567-89ab-cdef-0123-456789abcdef",
    ],
)
def test_execution_revalidates_public_response_and_keeps_safe_cache(
    client: TestClient, execution: SimpleNamespace, private: str
) -> None:
    previous = client.get("/api/execution?refresh=true").json()
    message = execution.snapshot["parents"][0]["children"][0]["summary"]["message"]
    saved = message["text"]
    message["text"] = private
    for url in ("/api/execution", "/api/execution?refresh=true"):
        response = client.get(url)
        assert response.status_code == 503
        assert private not in response.text
    message["text"] = saved
    cached = client.get("/api/execution").json()
    assert cached.pop("github_refresh")["result"] == "failed"
    previous.pop("github_refresh")
    assert cached == previous


def test_execution_scripts_render_and_poll_only_the_visible_tab(
    world: World, client: TestClient, execution: SimpleNamespace
) -> None:
    local = client.get("/api/execution").json()
    refreshed = client.get("/api/execution?refresh=true").json()
    result = subprocess.run(
        [
            "node",
            str(Path(__file__).with_name("board_execution.cjs")),
            str(Path(__file__).resolve().parents[3] / ".agents/ops/board/static"),
        ],
        input=json.dumps(
            {"local": local, "refreshed": refreshed, "post": world.post()}
        ),
        env=world.env(),
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# Exercise the Web projection at the producer/collector boundary. Register a
# matching run through the actual producer, then model one missing/mismatched
# contract field or a new reservation retaining the old run. Four expansions;
# the collector's field-by-field validation remains in its own tests.
@pytest.mark.parametrize(
    ("case", "expected_status", "expected_code"),
    [
        ("matching", "turn_completed", "contract_verified"),
        ("missing", "unknown", "contract_missing"),
        ("mismatch", "unknown", "contract_mismatch"),
        ("redelegated", "not_started", "reserved"),
    ],
)
def test_execution_reads_real_collector_records_without_changing_any_file(
    world: World,
    web: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_status: str,
    expected_code: str,
) -> None:
    state = world.labels.registry.parent
    run = state / "private-worker-run"
    run.mkdir()
    worktree = state.parent / "private-worker-worktree"
    worktree.mkdir()
    delegation = "a" * 32
    branch = "codex/board-contract-fixture"
    runtime: dict[str, Any] = {
        "schema_version": 1,
        "run_id": "private-run-token",
        "run_dir": str(run),
        "session_root": str(run),
        "result": "completed",
        "turn_end_reason": "completed",
        "runner": None,
        "process": None,
        "started_at": 1793360000,
        "finished_at": 1793360200,
        "contract": {
            "task_id": delegation,
            "cwd": str(worktree),
            "branch": branch,
            "settings": {"agent": "codex"},
        },
        "runtime": {"session_id": "private-child-session", "turn_id": "private-turn"},
    }
    (run / "result.json").write_text(json.dumps(runtime))
    (run / "session.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": "private-child-session",
                "last_turn_id": "private-turn",
                "latest_run": str(run),
            }
        )
    )
    (run / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "message",
                    "type": "agent_message",
                    "text": "/private/hidden-output",
                },
            }
        )
        + "\n"
    )
    parents = json.loads(world.labels.registry.read_text())["parents"]
    parents["o/r#1"]["tasks"]["11"] = {
        "agent": "codex",
        "delegated": True,
        "delegation_id": delegation,
        "worktree": str(worktree),
        "branch": branch,
        "worker_runs": [],
    }
    world.labels.set_registry(parents)

    class FixtureGitHub:
        repo = "o/r"

        def get(self, endpoint: str) -> dict[str, str]:
            assert endpoint == "user"  # Session construction only; no network.
            return {"login": "fixture-user"}

    parent_state = importlib.import_module("parent.state")
    parent_session = importlib.import_module("parent.session")
    manual = importlib.import_module("parent.manual_workers")
    with parent_state.Store(state).locked() as registry:
        session = parent_session.ParentSession(
            FixtureGitHub(), registry, 1, parents["o/r#1"]["session"]
        )
        manual.record_worker_run(session, 11, run)
        [record] = registry.parents["o/r#1"]["tasks"]["11"]["worker_runs"]
        assert set(record) == {"delegation_id", "run_dir", "kind", "recorded_at"}
    if case == "missing":
        runtime["contract"].pop("branch")
    elif case == "mismatch":
        runtime["contract"]["branch"] = "codex/another-delegation"
    elif case == "redelegated":
        # delegate preserves prior worker_runs when reserving a new delegation.
        with parent_state.Store(state).locked() as registry:
            registry.parents["o/r#1"]["tasks"]["11"]["delegation_id"] = "b" * 32
            registry.save()
    (run / "result.json").write_text(json.dumps(runtime))
    paths = [world.labels.registry, world.store.path, *run.iterdir()]
    wal = Path(str(world.store.path) + "-wal")
    if wal.exists():
        paths.append(wal)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_mode)
        for path in paths
    }

    def no_write(*args: Any, **kwargs: Any) -> None:
        pytest.fail("execution attempted a board transaction")

    monkeypatch.setattr(world.board.storage.Store, "transaction", no_write)
    with TestClient(
        web.app.create_app(world.env()), base_url="http://localhost"
    ) as client:
        response = client.get("/api/execution")
        assert response.status_code == 200
        child = response.json()["parents"][0]["children"][0]
        assert child["status"] == expected_status
        assert expected_code in {row["code"] for row in child["evidence"]}
        if case == "matching":
            assert child["run"]["kind"] == "registered"
            assert child["summary"]["message"]["status"] == "hidden"
            assert child["summary"]["message"]["text"] is None
        for private in (
            str(run),
            str(worktree),
            run.name,
            "private-child-session",
            "private-run-token",
            "a" * 32,
            "b" * 32,
            branch,
            "/private/hidden-output",
        ):
            assert private not in response.text
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_mode)
        for path in paths
    } == before


def test_roster_posts_threads_and_delivery_problems(
    world: World, client: TestClient
) -> None:
    root = world.post(to=["all"], body="@all 共有ファイルを変更します。")
    reply = world.board.messages.reply(
        world.env(2), post_id=root["id"], body="確認しました。", notifiers=world.routes
    )["post"]
    held = world.post(to=[world.assignments[4].name])
    failed = world.post(notifiers={})
    world.fake.accepting = False
    refused = world.post()
    result = client.get("/api/changes").json()
    assert result["reset"] is True
    assert [post["id"] for post in result["posts"]] == [
        refused["id"],
        failed["id"],
        held["id"],
        reply["id"],
        root["id"],
    ]
    roster = {row["parent"]: row for row in result["agents"]}
    assert set(roster) == {1, 2, 3}
    assert roster[2]["identifier"] is None
    assert roster[2]["engine"] == "codex"
    assert roster[2]["unread_count"] == 2
    assert roster[2]["last_activity"] == reply["created_at"]
    posts = {post["id"]: post for post in result["posts"]}
    assert posts[root["id"]]["broadcast"] is True
    assert [row["parent"] for row in posts[root["id"]]["recipients"]] == [2, 3]
    assert (
        posts[held["id"]]["recipients"][0]["delivery_message"] == "担当不在・配送保留"
    )
    assert (
        posts[failed["id"]]["recipients"][0]["delivery_reason"]
        == "notifier_unconfigured"
    )
    refused_row = posts[refused["id"]]["recipients"][0]
    assert refused_row["delivery_state"] == "delivery_failed"
    assert refused_row["delivery_message"] == "受信設定が accept ではありません"
    assert refused_row["attempts"] == 1 and refused_row["attempted_at"] is not None
    thread = client.get(f"/api/thread/{reply['id']}").json()
    assert thread["thread_id"] == root["id"]
    assert [post["id"] for post in thread["posts"]] == [root["id"], reply["id"]]
    assert thread["posts"][1]["in_reply_to"] == root["id"]
    # The other recipient still needs to respond to the broadcast question.
    assert thread["posts"][0]["recipients"][1]["response_state"] is None


def test_runtime_includes_unregistered_agents_without_changing_membership(
    world: World,
    client: TestClient,
    web: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    secret = "f08ef56a-0296-4706-9a8c-88a9366ef922"

    class RuntimeHerdr:
        def __init__(self, env: Any, session: str | None = None) -> None:
            self.session = session

        def call(self, *args: str) -> dict[str, Any]:
            observed.append((self.session, args))
            if args[0] == "session":
                return {
                    "sessions": [
                        {"name": "main", "running": True},
                        {"name": "other", "running": True},
                        {"name": "stopped", "running": False},
                    ]
                }
            return {
                "agents": [
                    {
                        "agent": "codex",
                        "agent_status": "working",
                        "terminal_title_stripped": "実験の確認"
                        if self.session == "main"
                        else f"/private/work {secret}",
                        "terminal_id": "private-terminal",
                        "pane_id": "private-pane",
                        "agent_session": {"value": secret},
                    }
                ]
            }

    before = world.message_rows()
    registry = world.labels.registry.read_bytes()
    monkeypatch.setattr(web.app.runtime, "Herdr", RuntimeHerdr)
    response = client.get("/api/runtime")
    assert response.status_code == 200
    data = response.json()
    assert data["observed_at"]
    assert [row["label"] for row in data["agents"]] == ["codex 1", "codex 2"]
    assert data["agents"][0]["title"] == "実験の確認"
    assert data["agents"][1]["title"] == ""
    assert data["agents"][1]["title_status"] == "redacted"
    assert all(row["state"] == "working" for row in data["agents"])
    assert secret not in response.text and "private-" not in response.text
    assert not any(session == "stopped" for session, _ in observed)
    assert world.message_rows() == before
    assert world.labels.registry.read_bytes() == registry


@pytest.mark.parametrize(
    "payload",
    [
        {"sessions": None},
        {"sessions": [{"running": "yes"}]},
        {"sessions": [{"running": True, "name": "broken"}]},
    ],
)
def test_runtime_source_failure_is_not_an_empty_success(
    client: TestClient,
    web: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
) -> None:
    def call(self: Any, *args: str) -> dict[str, Any]:
        if args[0] == "agent":
            raise ValueError("private failure details")
        return payload

    monkeypatch.setattr(web.app.runtime.Herdr, "call", call)
    response = client.get("/api/runtime")
    assert response.status_code == 503
    assert "private failure" not in response.text


def test_stopped_parent_is_historical_recipient_but_not_live_roster_or_broadcast(
    world: World, web: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = world
    parents = json.loads(w.labels.registry.read_text())["parents"]
    parents["o/r#7"] = w.labels.owner(identity="claude:stopped-owner")
    w.labels.set_registry(parents)
    before = w.labels.registry.read_bytes()
    monkeypatch.setattr(
        w.board.service.GitHub,
        "issue",
        lambda _github, number: {
            "title": f"[担当{number}] タスク",
            "labels": [{"name": "issue:parent"}],
        },
    )
    synced = w.board.service.sync_parent(w.env(), "o/r", 1)
    assert synced["additional_parents"][0]["parent"] == 7
    with w.store.read() as connection:
        assignment = w.board.names.for_parent(connection, "o/r", 7)
    w.fake.fail_once_for.add(assignment.id)
    broadcast = w.post(to=["all"])
    assert {row["parent"] for row in broadcast["recipients"]} == {2, 3}
    root = w.post(to=[assignment.name])
    status = w.own_state(root["id"])
    with TestClient(web.app.create_app(w.env()), base_url="http://127.0.0.1") as client:
        changes = client.get("/api/changes")
        assert changes.status_code == 200
        data = changes.json()
        thread = client.get(f"/api/thread/{root['id']}")
        assert thread.status_code == 200
    assert {row["parent"] for row in data["agents"]} == {1, 2, 3}
    for view in (status, data["posts"][0], thread.json()["posts"][0]):
        failed = next(row for row in view["recipients"] if row["parent"] == 7)
        assert failed["name"] == assignment.name
        assert failed["delivery_state"] == failed["state"] == "delivery_failed"
        assert failed["delivery_reason"] == "route_unavailable"
        assert failed["delivery_message"] == "通知経路が利用できません（未送信）"
        assert failed["retryable"] is True and failed["read_at"] is None
    with w.store.read() as connection:
        assert connection.execute(
            "SELECT state, reason, retryable FROM deliveries WHERE post_id=? AND assignment_id=?",
            (root["id"], assignment.id),
        ).fetchone() == ("delivery_failed", "route_unavailable", 1)
    assert w.labels.registry.read_bytes() == before


@pytest.mark.parametrize("broadcast", [False, True])
def test_saved_broadcast_is_returned_without_body_or_recipient_inference(
    world: World, client: TestClient, broadcast: bool
) -> None:
    # Both choices resolve to exactly the same recipients; neither body mentions all.
    root = world.post(
        to=["all"]
        if broadcast
        else [world.assignments[2].name, world.assignments[3].name],
    )
    reply = world.board.messages.reply(
        world.env(2), post_id=root["id"], body="確認しました。", notifiers=world.routes
    )["post"]
    posts = client.get("/api/changes").json()["posts"]
    assert posts[1]["broadcast"] is broadcast
    assert [row["parent"] for row in posts[1]["recipients"]] == [2, 3]
    assert posts[0]["broadcast"] is False  # A reply does not inherit all.
    thread = client.get(f"/api/thread/{reply['id']}").json()["posts"]
    assert [post["broadcast"] for post in thread] == [broadcast, False]
    assert all("mentions_all" not in post for post in posts + thread)


def test_rename_updates_roster_but_preserves_read_only_post_snapshots(
    world: World, client: TestClient
) -> None:
    w = world
    root = w.post()
    first = client.get("/api/changes").json()
    rename = importlib.import_module("board.renaming").rename
    result = rename(w.env(), parent=2)
    before = w.message_rows()
    changed = client.get("/api/changes", params={"cursor": first["cursor"]}).json()
    assert changed["posts"] == []
    recipient = next(row for row in changed["agents"] if row["parent"] == 2)
    assert recipient["name"] == result["new_name"]
    assert recipient["unread_count"] == 1
    thread = client.get(f"/api/thread/{root['id']}").json()["posts"]
    assert thread[0]["recipients"][0]["name"] == w.assignments[2].name
    assert thread[0]["recipients"][0]["assignment_id"] == w.assignments[2].id
    assert thread[0]["recipients"][0]["read_at"] is None
    historical = client.get(
        "/api/changes", params={"name": result["old_name"], "direction": "received"}
    ).json()
    assert [post["id"] for post in historical["posts"]] == [root["id"]]
    current = client.get(
        "/api/changes", params={"name": result["new_name"], "direction": "received"}
    ).json()
    assert current["posts"] == []
    assert w.message_rows() == before


def test_unassigned_roster_is_preserved(world: World, web: SimpleNamespace) -> None:
    parents = json.loads(world.labels.registry.read_text())["parents"]
    parents["o/r#7"] = world.labels.owner(identity="codex:unassigned-owner")
    world.labels.set_registry(parents)
    world.live_parent(7)
    with TestClient(
        web.app.create_app(world.env()), base_url="http://localhost"
    ) as client:
        rows = client.get("/api/changes").json()["agents"]
    unassigned = next(row for row in rows if row["parent"] == 7)
    assert unassigned["name"] is None
    assert unassigned["identifier"] is None
    assert unassigned["name_status"] == "unassigned"
    assert unassigned["unread_count"] == 0


def test_name_filters_include_only_matching_sent_and_received_posts(
    world: World, client: TestClient
) -> None:
    root = world.post()
    reply = world.board.messages.reply(
        world.env(2), post_id=root["id"], body="確認しました。", notifiers=world.routes
    )["post"]
    other = world.post(to=[world.assignments[3].name])
    name = world.assignments[2].name.lower()
    received = client.get(
        "/api/changes", params={"name": name, "direction": "received"}
    ).json()
    assert [post["id"] for post in received["posts"]] == [root["id"]]
    sent = client.get("/api/changes", params={"name": name, "direction": "sent"}).json()
    assert [post["id"] for post in sent["posts"]] == [reply["id"]]
    both = client.get("/api/changes", params={"name": name}).json()
    assert [post["id"] for post in both["posts"]] == [reply["id"], root["id"]]
    assert all(post["id"] != other["id"] for post in both["posts"])
    changed_filter = client.get(
        "/api/changes",
        params={"name": name, "direction": "sent", "cursor": received["cursor"]},
    ).json()
    assert changed_filter["reset"] is True
    assert [post["id"] for post in changed_filter["posts"]] == [reply["id"]]


def test_cursor_reports_new_posts_and_old_recipient_changes(
    world: World, client: TestClient, web: SimpleNamespace
) -> None:
    root = world.post(notifiers={})
    initial = client.get("/api/changes").json()
    empty = client.get("/api/changes", params={"cursor": initial["cursor"]}).json()
    assert empty["posts"] == [] and empty["agents"] is None
    assert empty["cursor"] == initial["cursor"]
    assert empty["reset"] is False
    newest = world.post(kind="info")
    world.board.messages.ack(world.env(2), post_id=root["id"], state="accepted")
    updated = client.get("/api/changes", params={"cursor": initial["cursor"]}).json()
    assert [post["id"] for post in updated["posts"]] == [newest["id"], root["id"]]
    recipient = updated["posts"][1]["recipients"][0]
    assert (
        recipient["read_at"] is not None and recipient["response_state"] == "accepted"
    )
    assert recipient["delivery_state"] == "delivery_failed"
    assert updated["agents"] is not None
    # A lagging client can still receive all changes, even after another poll.
    again = client.get("/api/changes", params={"cursor": initial["cursor"]}).json()
    assert again["posts"] == updated["posts"]
    with TestClient(
        web.app.create_app(world.env()), base_url="http://localhost"
    ) as restarted:
        reset = restarted.get(
            "/api/changes", params={"cursor": updated["cursor"]}
        ).json()
    assert reset["reset"] is True and len(reset["posts"]) == 2


def test_open_thread_changes_are_reported_outside_the_name_filter(
    world: World, client: TestClient
) -> None:
    root = world.post(to=["all"])
    world.board.messages.reply(
        world.env(2), post_id=root["id"], body="確認しました。", notifiers=world.routes
    )
    query = {"name": world.assignments[2].name, "direction": "sent"}
    initial = client.get("/api/changes", params=query).json()
    world.board.messages.reply(
        world.env(3),
        post_id=root["id"],
        body="こちらも確認しました。",
        notifiers=world.routes,
    )
    changed = client.get(
        "/api/changes", params={**query, "cursor": initial["cursor"]}
    ).json()
    assert changed["posts"] == []
    assert changed["changed_threads"] == [root["id"]]
    thread = client.get(f"/api/thread/{root['id']}").json()
    assert len(thread["posts"]) == 3
    assert thread["posts"][-1]["sender"]["parent"] == 3


def test_cursor_tracks_reaction_adds_removals_and_duplicate_noops(
    world: World, client: TestClient
) -> None:
    root = world.post()
    world.post(kind="info")  # A reaction to an older post must still be returned.
    first = client.get("/api/changes").json()
    assert all(post["reactions"] == [] for post in first["posts"])
    before = world.message_rows()
    previous = first
    for parent, remove, actor_count, mutated in (
        (2, False, 1, True),
        (2, False, 1, False),
        (1, False, 2, True),
        (2, True, 1, True),
        (2, True, 1, False),
        (1, True, 0, True),
    ):
        world.board.reactions.react(
            world.env(parent), post_id=root["id"], emoji="+1", remove=remove
        )
        response = client.get("/api/changes", params={"cursor": previous["cursor"]})
        assert response.status_code == 200
        changed = response.json()
        assert changed["reset"] is False and changed["agents"] is None
        thread = client.get(f"/api/thread/{root['id']}").json()["posts"][0]
        if actor_count:
            assert len(thread["reactions"][0]["reactors"]) == actor_count
        else:
            assert thread["reactions"] == []
        if not mutated:
            assert changed["posts"] == [] and changed["cursor"] == previous["cursor"]
            assert changed["changed_threads"] == []
        else:
            assert changed["cursor"] != previous["cursor"]
            assert [post["id"] for post in changed["posts"]] == [root["id"]]
            assert changed["posts"][0]["reactions"] == thread["reactions"]
            assert changed["changed_threads"] == [root["id"]]
        previous = changed
    assert world.message_rows() == before


def test_reaction_changes_refresh_threads_outside_the_name_filter(
    world: World, client: TestClient
) -> None:
    root = world.post()
    query = {"name": world.assignments[3].name, "direction": "sent"}
    initial = client.get("/api/changes", params=query).json()
    assert initial["posts"] == []
    world.board.reactions.react(world.env(2), post_id=root["id"], emoji="eyes")
    changed = client.get(
        "/api/changes", params={**query, "cursor": initial["cursor"]}
    ).json()
    assert changed["posts"] == []
    assert changed["changed_threads"] == [root["id"]]
    assert changed["cursor"] != initial["cursor"]
    thread = client.get(f"/api/thread/{root['id']}").json()
    assert thread["posts"][0]["reactions"][0]["reactors"][0]["parent"] == 2


def test_gets_and_disallowed_methods_leave_bytes_and_unread_unchanged(
    world: World, web: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = world.post()
    paths = [world.store.path, Path(str(world.store.path) + "-wal")]
    before = {path: path.read_bytes() for path in paths if path.exists()}
    registry_before = world.labels.registry.read_bytes()
    connects = []
    connect = sqlite3.connect

    def read_connect(path: str, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        connects.append(path)
        assert path.endswith("?mode=ro") and kwargs["uri"] is True
        connection: sqlite3.Connection = connect(path, *args, **kwargs)
        return connection

    def no_write(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Web UI attempted a board transaction")

    monkeypatch.setattr(sqlite3, "connect", read_connect)
    monkeypatch.setattr(world.board.storage.Store, "transaction", no_write)
    with TestClient(
        web.app.create_app(world.env()), base_url="http://localhost"
    ) as client:
        for url in (
            "/",
            "/static/app.js",
            "/static/execution.js",
            "/static/style.css",
            "/api/changes",
            "/api/changes?refresh_roster=true",
            "/api/execution",
            f"/api/thread/{post['id']}",
        ):
            assert client.get(url).status_code == 200
        for method in ("POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE"):
            for url in (
                "/api/changes",
                "/api/execution",
                "/api/execution?refresh=true",
                f"/api/thread/{post['id']}",
                "/",
            ):
                response = client.request(method, url)
                assert response.status_code == 405
                assert response.headers["allow"] == "GET"
    assert connects
    assert all(path.read_bytes() == value for path, value in before.items())
    assert world.labels.registry.read_bytes() == registry_before
    with world.store.read() as connection:
        assert connection.execute(
            "SELECT read_at, response_state FROM recipients"
        ).fetchall() == [(None, None)]


def test_security_headers_hosts_assets_and_html_text(
    world: World, client: TestClient
) -> None:
    payload = '<script>window.boardXss=1\n<img src=x onerror="window.boardXss=2">'
    root = world.post(body=payload, subject="<b>件名")
    response = client.get("/api/changes")
    assert response.json()["posts"][0]["body"] == payload
    assert "<script>" not in response.text and "<img" not in response.text
    assert "\\u003cscript\\u003e" in response.text
    for url in (
        "/",
        "/api/changes",
        "/api/execution",
        f"/api/thread/{root['id']}",
        "/missing",
        "/static/missing.js",
    ):
        headers = client.get(url).headers
        assert "default-src 'none'" in headers["content-security-policy"]
        assert "script-src 'self'" in headers["content-security-policy"]
        assert "connect-src 'self'" in headers["content-security-policy"]
        assert "'unsafe-inline'" not in headers["content-security-policy"]
        assert "https:" not in headers["content-security-policy"]
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["cache-control"] == "no-store"
    for host in ("evil.example", "localhost.evil.example", "testserver", "0.0.0.0"):
        refused = client.get("/api/changes", headers={"host": host})
        assert refused.status_code == 400
        assert "content-security-policy" in refused.headers
    assert client.get("/", headers={"host": "localhost:8790"}).status_code == 200
    for url in (
        "/static/index.html",
        "/static/names.txt",
        "/static/board.sqlite3",
        "/static/%2e%2e%2fstorage.py",
        "/docs",
        "/redoc",
        "/openapi.json",
    ):
        assert client.get(url).status_code == 404


@pytest.mark.parametrize("human_sender", [False, True])
def test_human_posts_render_in_timeline_and_thread(
    world: World, client: TestClient, human_sender: bool
) -> None:
    if human_sender:
        response = client.post(
            "/api/posts",
            headers=write_headers(client),
            json={
                "kind": "question",
                "subject": "Human question",
                "body": f"@{world.assignments[2].name} Please check this post.",
            },
        )
        assert response.status_code == 200
    else:
        world.post(to=["human"], body="@human Please check this post.")
    posts = client.get("/api/changes").json()["posts"]
    assert len(posts) == 1
    result = subprocess.run(
        [
            "node",
            str(Path(__file__).with_name("board_rendering.cjs")),
            str(
                Path(__file__).resolve().parents[3] / ".agents/ops/board/static/app.js"
            ),
        ],
        input=json.dumps(posts[0]),
        env=world.env(),
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_no_external_source_polling_without_explicit_roster_refresh(
    world: World, web: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = world.post()
    app = web.app.create_app(world.env())

    def fail(*args: Any, **kwargs: Any) -> None:
        pytest.fail("The web poll queried a registry or runtime source")

    monkeypatch.setattr(web.data, "agents", fail)
    monkeypatch.setattr(web.data, "registry", fail)
    monkeypatch.setattr(world.board.service, "registry", fail)
    monkeypatch.setattr(world.board.service, "_runtime_states", fail)
    with TestClient(app, base_url="http://localhost") as client:
        first = client.get("/api/changes").json()
        assert (
            client.get("/api/changes", params={"cursor": first["cursor"]}).status_code
            == 200
        )
        assert client.get(f"/api/thread/{root['id']}").status_code == 200
        assert (
            client.get("/api/changes", params={"direction": "sent"}).status_code == 200
        )


def test_explicit_roster_refresh_updates_registry_and_runtime_read_only(
    world: World, web: SimpleNamespace
) -> None:
    prepare_roster(world.board, world.labels)
    with TestClient(
        web.app.create_app(world.labels.env), base_url="http://localhost"
    ) as client:
        initial = client.get("/api/changes", params={"refresh_roster": "true"}).json()
        assert [(row["parent"], row["state"]) for row in initial["agents"]] == [
            (42, "working"),
            (43, "idle"),
        ]
        parents = json.loads(world.labels.registry.read_text())["parents"]
        parents["o/r#42"]["status"] = "released"
        parents["o/r#44"]["status"] = "active"
        world.labels.set_registry(parents)
        runtime = world.labels.read()
        runtime["panes"]["w1:p2"]["state"] = "blocked"
        world.labels.write(runtime)
        registry_before = world.labels.registry.read_bytes()
        db_before = world.store.path.read_bytes()
        owners = world.labels.registry.parent.parent / "herdr-labels/owners"
        owners_before = {path: path.read_bytes() for path in owners.iterdir()}

        cached = client.get("/api/changes", params={"cursor": initial["cursor"]}).json()
        assert cached["agents"] is None
        assert cached["roster_observed_at"] == initial["roster_observed_at"]
        assert world.labels.read()["calls"] == runtime["calls"]

        refreshed = client.get(
            "/api/changes",
            params={"cursor": initial["cursor"], "refresh_roster": "true"},
        ).json()
        assert refreshed["reset"] is False
        assert refreshed["posts"] == []
        assert [
            (row["parent"], row["engine"], row["state"]) for row in refreshed["agents"]
        ] == [
            (43, "codex", "blocked"),
        ]
        assert refreshed["roster_observed_at"] > initial["roster_observed_at"]
        assert len(world.labels.read()["calls"]) > len(runtime["calls"])
        # A repeated page load still observes current sources, even with no row changes.
        reloaded = client.get("/api/changes", params={"refresh_roster": "true"}).json()
        assert reloaded["agents"] == refreshed["agents"]
        assert reloaded["roster_observed_at"] > refreshed["roster_observed_at"]
        assert world.labels.registry.read_bytes() == registry_before
        assert world.store.path.read_bytes() == db_before
        assert all(path.read_bytes() == value for path, value in owners_before.items())
        assert world.labels.read()["panes"] == runtime["panes"]
        assert all(
            call
            in (
                ["session", "list", "--json"],
                ["--session", "default", "agent", "list"],
            )
            for call in world.labels.read()["calls"]
        )
        private_output_absent(json.dumps(reloaded), world.labels.env)


def test_herdr_outage_is_explicit_and_keeps_last_observed_roster(
    world: World, web: SimpleNamespace
) -> None:
    prepare_roster(world.board, world.labels)
    with TestClient(
        web.app.create_app(world.labels.env), base_url="http://localhost"
    ) as client:
        initial = client.get("/api/changes").json()
        assert initial["agents"][0]["state"] == "working"
        (world.labels.root / "bin/herdr").write_text(
            "#!/bin/sh\necho 'private runtime details' >&2\nexit 3\n"
        )
        before = world.labels.registry.read_bytes()
        response = client.get("/api/changes", params={"refresh_roster": "true"})
        assert response.status_code == 503
        cached = client.get("/api/changes").json()
        assert cached["agents"] == initial["agents"]
        assert cached["roster_observed_at"] == initial["roster_observed_at"]
        assert "private runtime details" not in response.text
        assert world.labels.registry.read_bytes() == before


def test_failed_roster_refresh_is_explicit_and_preserves_last_observation(
    world: World, client: TestClient
) -> None:
    initial = client.get("/api/changes").json()
    saved = world.labels.registry.read_bytes()
    world.labels.registry.write_text("invalid registry with private details")
    response = client.get("/api/changes", params={"refresh_roster": "true"})
    assert response.status_code == 503
    assert "private details" not in response.text
    cached = client.get("/api/changes").json()
    assert cached["agents"] == initial["agents"]
    assert cached["roster_observed_at"] == initial["roster_observed_at"]
    world.labels.registry.write_bytes(saved)
    retry = client.get("/api/changes", params={"refresh_roster": "true"}).json()
    assert retry["agents"] == initial["agents"]
    assert retry["roster_observed_at"] > initial["roster_observed_at"]


@pytest.mark.parametrize(
    "params", [{"cursor": "bad"}, {"name": "/private/file"}, {"direction": "other"}]
)
def test_invalid_queries_have_no_input_in_errors(
    client: TestClient, params: dict[str, str]
) -> None:
    response = client.get("/api/changes", params=params)
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid board query."}
    assert client.get("/api/thread/invalid").status_code == 422
    assert client.get("/api/thread/" + "0" * 32).status_code == 404


def test_private_sources_and_corrupt_database_never_leak_or_become_empty(
    world: World, client: TestClient, web: SimpleNamespace
) -> None:
    world.post()
    for url in ("/api/changes", "/"):
        response = client.get(url)
        private_output_absent(response.text, world.env())
        assert "board-owner-" not in response.text
        assert str(world.store.path) not in response.text
    with world.store.transaction() as connection:
        connection.execute(
            "UPDATE name_assignments SET identifier=? WHERE parent=1",
            ("/private/runtime.sock",),
        )
    rejected = client.get("/api/changes")
    assert rejected.status_code == 503
    assert "runtime.sock" not in rejected.text
    with world.store.transaction() as connection:
        connection.execute("UPDATE name_assignments SET identifier=NULL")
        connection.execute("PRAGMA user_version=999")
    before = world.store.path.read_bytes()
    assert client.get("/api/changes").status_code == 503
    assert world.store.path.read_bytes() == before


def test_web_binds_only_loopback_and_cli_imports_web_lazily(
    world: World, web: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    uvicorn = importlib.import_module("uvicorn")
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append(kwargs))
    web.app.serve(world.env())
    web.app.serve(world.env(), port=8791)
    assert calls == [
        {"host": "127.0.0.1", "port": 8790, "access_log": False},
        {"host": "127.0.0.1", "port": 8791, "access_log": False},
    ]
    code = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'fastapi', 'starlette', 'uvicorn'}:
        raise AssertionError('Core CLI imported web dependencies')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['board', 'agents']
runpy.run_module('board', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=world.env(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["agents"]) == 3
    cli = importlib.import_module("board.__main__")
    assert cli.main(["web", "--port", "0"]) == 2
    assert cli.main(["web", "--port", "65536"]) == 2
    assert cli.main(["web", "--host", "0.0.0.0"]) == 2


def write_headers(client: TestClient) -> dict[str, str]:
    page = client.get("/")
    token = page.headers["x-csrf-token"]
    assert f'<meta name="board-csrf-token" content="{token}">' in page.text
    return {"Origin": str(client.base_url).rstrip("/"), "X-CSRF-Token": token}


def saved_rows(world: World) -> dict[str, list[tuple[Any, ...]]]:
    with world.store.read() as connection:
        return {
            name: connection.execute(f"SELECT * FROM {name}").fetchall()
            for name in (
                "posts",
                "recipients",
                "deliveries",
                "reactions",
                "name_assignments",
            )
        }


def test_human_posts_and_replies_reach_parent_inbox_and_fake_bells(
    world: World, client: TestClient
) -> None:
    w = world
    headers = write_headers(client)
    registry_before = w.labels.registry.read_bytes()
    response = client.post(
        "/api/posts",
        headers=headers,
        json={
            "subject": "人間からの質問",
            "body": f"宛先は@{w.assignments[2].name.lower()}です。`@{w.assignments[4].name}`は例です。",
            "kind": "question",
            "to": [w.assignments[3].name, w.assignments[2].name.upper()],
        },
    )
    assert response.status_code == 200, response.text
    root = response.json()["post"]
    assert root["sender"] == {
        "assignment_id": None,
        "name": "human",
        "repo": None,
        "parent": None,
    }
    w.assert_delivered(root, [2, 3])
    assert all(
        bell.sender_name == "human" and "None" not in bell.bell for bell in w.fake.calls
    )
    assert all(
        root["body"] not in bell.bell and root["subject"] not in bell.bell
        for bell in w.fake.calls
    )
    assert w.board.messages.inbox(w.env(2), unread=True)["posts"][0]["id"] == root["id"]
    answer = w.board.messages.reply(
        w.env(2), post_id=root["id"], body="回答します。", notifiers=w.routes
    )["post"]
    assert [row["name"] for row in answer["recipients"]] == ["human"]
    assert answer["recipients"][0]["delivery_state"] == "available"
    followup = client.post(
        f"/api/posts/{answer['id']}/replies",
        headers=headers,
        json={"body": f"@{w.assignments[3].name}も確認してください。"},
    )
    assert followup.status_code == 200, followup.text
    reply = followup.json()["post"]
    w.assert_delivered(reply, [2, 3])
    assert reply["sender"]["name"] == "human"
    assert reply["thread_id"] == root["id"] and reply["in_reply_to"] == answer["id"]
    assert client.get("/api/changes").json()["human"]["unread_count"] == 0
    thread = client.get(f"/api/thread/{root['id']}").json()["posts"]
    assert [row["id"] for row in thread] == [root["id"], answer["id"], reply["id"]]
    assert thread[1]["recipients"][0]["response_state"] == "answered"
    assert w.labels.registry.read_bytes() == registry_before
    with w.store.read() as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM name_assignments WHERE lower(name)='human'"
            ).fetchall()
            == []
        )


@pytest.mark.parametrize(
    "body,to", [("@HuMaN 確認してください。", []), ("確認してください。", ["HUMAN"])]
)
def test_cli_human_recipient_unread_response_counts_and_read_api(
    world: World, client: TestClient, tmp_path: Path, body: str, to: list[str]
) -> None:
    body_file = tmp_path / "message.md"
    body_file.write_text(body, encoding="utf-8")
    initial = client.get("/api/changes").json()
    args = [
        "post",
        "--kind",
        "question",
        "--subject",
        "人間への質問",
        "--body-file",
        str(body_file),
    ]
    for name in to:
        args.extend(["--to", name])
    result = run_board(world.env(), *args)
    assert result.returncode == 0, result.stderr
    post = json.loads(result.stdout)["post"]
    assert len(post["recipients"]) == 1
    recipient = post["recipients"][0]
    assert recipient["name"] == "human" and recipient["assignment_id"] is None
    assert recipient["repo"] is None and recipient["parent"] is None
    assert recipient["delivery_state"] == "available" and recipient["attempts"] == 0
    changed = client.get("/api/changes", params={"cursor": initial["cursor"]}).json()
    assert changed["human"]["unread_count"] == 1
    assert changed["human"]["awaiting_response_count"] == 1
    assert changed["cursor"] != initial["cursor"]
    assert [row["id"] for row in changed["posts"]] == [post["id"]]
    assert (
        client.get(
            "/api/changes", params={"name": "human", "direction": "received"}
        ).json()["posts"][0]["id"]
        == post["id"]
    )
    headers = write_headers(client)
    before = saved_rows(world)
    assert client.get(f"/api/thread/{post['id']}").status_code == 200
    assert saved_rows(world) == before  # Viewing a thread still never marks it read.
    read = client.post(f"/api/posts/{post['id']}/read", headers=headers)
    assert read.status_code == 200, read.text
    receipt = read.json()["post"]["recipients"][0]
    assert receipt["read_at"] is not None and receipt["response_state"] is None
    assert receipt["delivery_state"] == "available"
    repeated = client.post(f"/api/posts/{post['id']}/read", headers=headers).json()
    assert repeated["post"]["recipients"][0]["read_at"] == receipt["read_at"]
    updated = client.get("/api/changes", params={"cursor": changed["cursor"]}).json()
    assert updated["human"]["unread_count"] == 0
    assert updated["human"]["awaiting_response_count"] == 1
    answered = client.post(
        f"/api/posts/{post['id']}/replies",
        headers=headers,
        json={"body": "確認しました。"},
    )
    assert answered.status_code == 200, answered.text
    counts = client.get("/api/changes").json()["human"]
    assert counts["unread_count"] == counts["awaiting_response_count"] == 0
    assert len(world.fake.calls) == 1 and world.fake.calls[0].parent == 1


def test_human_can_join_visible_parent_thread_and_react_without_reading_it(
    world: World, client: TestClient
) -> None:
    root = world.post()
    world.board.reactions.react(world.env(1), post_id=root["id"], emoji="+1")
    headers = write_headers(client)
    before = world.message_rows()
    bell_count = len(world.fake.calls)
    initial = client.get("/api/changes").json()
    url = f"/api/posts/{root['id']}/reactions/+1"
    added = client.put(url, headers=headers)
    assert added.status_code == 200, added.text
    reactors = added.json()["post"]["reactions"][0]["reactors"]
    assert len(reactors) == 2
    human = next(row for row in reactors if row["type"] == "human")
    assert human["assignment_id"] is None and human["name"] == "human"
    assert human["repo"] is human["parent"] is None
    assert (
        client.put(url, headers=headers).json()["post"]["reactions"][0]["reactors"]
        == reactors
    )
    changed = client.get("/api/changes", params={"cursor": initial["cursor"]}).json()
    assert changed["changed_threads"] == [root["id"]]
    assert changed["posts"][0]["reactions"][0]["reactors"] == reactors
    removed = client.delete(url, headers=headers)
    assert removed.status_code == 200
    assert [
        row["type"] for row in removed.json()["post"]["reactions"][0]["reactors"]
    ] == ["parent"]
    assert client.delete(url, headers=headers).json() == removed.json()
    assert world.message_rows() == before and len(world.fake.calls) == bell_count
    reply = client.post(
        f"/api/posts/{root['id']}/replies",
        headers=headers,
        json={"body": "人間から補足します。"},
    )
    assert reply.status_code == 200, reply.text
    assert [row["parent"] for row in reply.json()["post"]["recipients"]] == [1]


@pytest.mark.parametrize(
    "method,suffix",
    [
        ("POST", ""),
        ("POST", "/replies"),
        ("POST", "/read"),
        ("PUT", "/reactions/eyes"),
        ("DELETE", "/reactions/eyes"),
    ],
)
@pytest.mark.parametrize(
    "failure",
    [
        "no_token",
        "wrong_token",
        "no_origin",
        "other_origin",
        "other_port",
        "null_origin",
        "bad_host",
        "duplicate_origin",
        "duplicate_token",
    ],
)
def test_rejected_web_writes_preserve_all_rows(
    world: World, client: TestClient, method: str, suffix: str, failure: str
) -> None:
    root = world.post()
    url = "/api/posts" + (f"/{root['id']}" + suffix if suffix else "")
    headers = write_headers(client)
    if failure == "no_token":
        del headers["X-CSRF-Token"]
    elif failure == "wrong_token":
        headers["X-CSRF-Token"] = "invalid"
    elif failure == "no_origin":
        del headers["Origin"]
    elif failure == "other_origin":
        headers["Origin"] = "http://evil.example"
    elif failure == "other_port":
        headers["Origin"] += ":8791"
    elif failure == "null_origin":
        headers["Origin"] = "null"
    elif failure == "bad_host":
        headers.update(Host="evil.example", Origin="http://evil.example")
    request_headers = list(headers.items())
    if failure == "duplicate_origin":
        request_headers.append(("Origin", headers["Origin"]))
    elif failure == "duplicate_token":
        request_headers.append(("X-CSRF-Token", headers["X-CSRF-Token"]))
    before = saved_rows(world)
    registry_before = world.labels.registry.read_bytes()
    calls = len(world.fake.calls)
    response = client.request(
        method,
        url,
        headers=request_headers,
        json={
            "kind": "info",
            "subject": "hello",
            "body": f"@{world.assignments[2].name} hello",
        },
    )
    assert response.status_code == (400 if failure == "bad_host" else 403)
    assert saved_rows(world) == before
    assert world.labels.registry.read_bytes() == registry_before
    assert len(world.fake.calls) == calls


@pytest.mark.parametrize(
    "value",
    [
        "/private/file",
        "~/secrets",
        r"C:\private\file",
        "session_id=private",
        "board-owner-2",
        "12345678-1234-1234-1234-123456789abc",
        "control\x01",
    ],
)
@pytest.mark.parametrize("field", ["subject", "body"])
def test_private_human_input_is_rejected_before_any_save(
    world: World, client: TestClient, value: str, field: str
) -> None:
    headers = write_headers(client)
    payload = {
        "kind": "info",
        "subject": "安全な件名",
        "body": f"@{world.assignments[2].name} 確認します。",
    }
    payload[field] = value
    before = saved_rows(world)
    response = client.post("/api/posts", headers=headers, json=payload)
    assert response.status_code == 503
    assert value not in response.text
    assert saved_rows(world) == before and world.fake.calls == []


def test_process_token_and_human_identity_do_not_depend_on_parent_environment(
    world: World, web: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def configured(env: dict[str, str]) -> dict[str, Any]:
        calls.append(env)
        return dict(world.routes)

    monkeypatch.setattr(web.app.notifiers, "configured", configured)
    env = {
        **world.labels.env,
        "CLAUDE_CODE_SESSION_ID": "unregistered-web-process",
        "CODEX_SESSION_ID": "ambiguous-process",
    }
    with (
        TestClient(web.app.create_app(env), base_url="http://localhost:8791") as one,
        TestClient(web.app.create_app(env), base_url="http://localhost:8791") as two,
    ):
        headers = write_headers(one)
        assert write_headers(one) == headers
        other = write_headers(two)
        assert other["X-CSRF-Token"] != headers["X-CSRF-Token"]
        payload = {
            "kind": "info",
            "subject": "Webから",
            "body": f"@{world.assignments[2].name} 確認してください。",
        }
        before = saved_rows(world)
        assert two.post("/api/posts", headers=headers, json=payload).status_code == 403
        assert saved_rows(world) == before
        response = one.post("/api/posts", headers=headers, json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["post"]["sender"]["name"] == "human"
        assert len(world.fake.calls) == 1 and world.fake.calls[0].parent == 2
    assert calls == [env, env]


@pytest.mark.parametrize(
    "payload",
    [
        {"subject": "hello", "body": "hello"},
        {"kind": "info", "subject": "hello", "body": 42},
        {"kind": "info", "subject": "hello", "body": "hello", "sender": "human"},
        {"kind": "info", "subject": "hello", "body": "hello", "to": "all"},
    ],
)
def test_write_payload_validation_never_echoes_or_saves_input(
    world: World, client: TestClient, payload: dict[str, Any]
) -> None:
    before = saved_rows(world)
    response = client.post("/api/posts", headers=write_headers(client), json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid board query."}
    assert saved_rows(world) == before


def test_human_broadcast_uses_all_active_parents_without_a_sender_session(
    world: World, client: TestClient
) -> None:
    response = client.post(
        "/api/posts",
        headers=write_headers(client),
        json={
            "kind": "info",
            "subject": "全員への連絡",
            "body": "@all 確認してください。",
        },
    )
    assert response.status_code == 200, response.text
    post = response.json()["post"]
    assert post["broadcast"] is True
    world.assert_delivered(post, [1, 2, 3])
    assert all(row["name"] != "human" for row in post["recipients"])


def test_human_receipts_reactions_and_counts_are_separate_from_parent_receipts(
    world: World, client: TestClient
) -> None:
    post = world.post(
        to=["human", world.assignments[2].name], body="@HuMaN 確認してください。"
    )
    assert [row["name"] for row in post["recipients"]] == [
        "human",
        world.assignments[2].name,
    ]
    assert [bell.parent for bell in world.fake.calls] == [2]
    headers = write_headers(client)
    before = world.message_rows()
    assert (
        client.put(
            f"/api/posts/{post['id']}/reactions/eyes", headers=headers
        ).status_code
        == 200
    )
    assert world.message_rows() == before
    human = client.get("/api/changes").json()["human"]
    assert human["unread_count"] == human["awaiting_response_count"] == 1
    read = client.post(f"/api/posts/{post['id']}/read", headers=headers).json()["post"]
    assert read["recipients"][0]["read_at"] is not None
    assert read["recipients"][1]["read_at"] is None
    world.board.messages.read(world.env(2), post["id"])
    human = client.get("/api/changes").json()["human"]
    assert human["unread_count"] == 0 and human["awaiting_response_count"] == 1
    assert len(world.fake.calls) == 1


@pytest.mark.parametrize(
    "body",
    ["@human @all 自分宛て", "`@all` の例だけ", "@Adam 未割当の名前", "/private/file"],
)
def test_invalid_human_post_or_reply_cannot_save_or_mark_the_original_read(
    world: World, client: TestClient, body: str
) -> None:
    root = world.post(to=["human"])
    headers = write_headers(client)
    before = saved_rows(world)
    for url, payload in (
        ("/api/posts", {"subject": "不正入力", "kind": "info", "body": body}),
        (f"/api/posts/{root['id']}/replies", {"body": body}),
    ):
        # A reply automatically has its original author as a recipient, so a
        # code-only mention is allowed there. Root posts still need a recipient.
        if "/replies" in url and body.startswith("`"):
            continue
        response = client.post(url, headers=headers, json=payload)
        assert response.status_code == 503
        assert saved_rows(world) == before
    assert world.fake.calls == []


@pytest.mark.parametrize(
    "suffix,allowed",
    [
        ("", "POST"),
        ("/replies", "POST"),
        ("/read", "POST"),
        ("/reactions/eyes", "DELETE, PUT"),
    ],
)
def test_write_paths_only_accept_their_declared_methods(
    world: World, client: TestClient, suffix: str, allowed: str
) -> None:
    root = world.post()
    url = "/api/posts" + (f"/{root['id']}" + suffix if suffix else "")
    headers = write_headers(client)
    before = saved_rows(world)
    for method in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "TRACE"):
        if method in allowed.split(", "):
            continue
        response = client.request(method, url, headers=headers)
        assert response.status_code == 405
        assert response.headers["allow"] == allowed
    assert saved_rows(world) == before


def test_web_writes_do_not_initialize_or_migrate_a_database(
    world: World, client: TestClient
) -> None:
    headers = write_headers(client)
    payload = {
        "subject": "hello",
        "kind": "info",
        "body": f"@{world.assignments[2].name} hello",
    }
    with world.store.transaction() as connection:
        connection.execute("PRAGMA user_version=4")
    assert client.post("/api/posts", headers=headers, json=payload).status_code == 503
    with sqlite3.connect(world.store.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert connection.execute("SELECT COUNT(*) FROM posts").fetchone()[0] == 0
    world.store.path.unlink()
    assert client.post("/api/posts", headers=headers, json=payload).status_code == 503
    assert not world.store.path.exists() and world.fake.calls == []


def test_browser_human_conversation_workflow(
    world: World, web: SimpleNamespace, tmp_path: Path, execution: SimpleNamespace
) -> None:
    """Optional real Chromium check; enable with Playwright on NODE_PATH."""
    import os
    import socket
    import threading
    import time

    import uvicorn

    available = subprocess.run(
        ["node", "-e", "require.resolve('playwright')"],
        capture_output=True,
        check=False,
        timeout=15,
    )
    if available.returncode:
        unavailable = pytest.fail if os.environ.get("CI_REQUIRE_BROWSER") == "1" else pytest.skip
        unavailable("Install the optional Playwright package and set NODE_PATH")
    root = world.post(
        to=["human", world.assignments[2].name],
        body='@human <img src=x onerror="alert(1)"> を文字として表示',
    )
    world.board.messages.reply(
        world.env(2),
        post_id=root["id"],
        body="元の質問を確認しました。",
        notifiers=world.routes,
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            web.app.create_app(world.env(), notification_routes=world.routes),
            log_level="error",
            access_log=False,
        )
    )
    worker = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True
    )
    worker.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and worker.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        result = subprocess.run(
            ["node", str(Path(__file__).with_name("board_browser.cjs"))],
            input=json.dumps(
                {
                    "url": f"http://127.0.0.1:{port}",
                    "rootId": root["id"],
                    "body": root["body"],
                    "sender": world.assignments[1].name,
                    "recipient": world.assignments[2].name,
                }
            ),
            env={**os.environ, "XDG_STATE_HOME": str(tmp_path / "browser-state")},
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    finally:
        server.should_exit = True
        worker.join(timeout=10)
        sock.close()
        assert not worker.is_alive()
