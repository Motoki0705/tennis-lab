"""Read-only web contracts through fixture SQLite, registry and fake routes."""

from __future__ import annotations

import importlib
import json
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.e2e.agents_ops.test_board import board as board
from tests.e2e.agents_ops.test_board import prepare_roster, private_output_absent
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
        web.app.create_app(world.env()), base_url="http://127.0.0.1"
    ) as client:
        yield client


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


def test_broadcast_to_newly_named_stopped_parent_exposes_unavailable_route(
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
    root = w.post(to=["all"])
    assert {row["parent"] for row in root["recipients"]} == {2, 3, 7}
    status = w.own_state(root["id"])
    with TestClient(web.app.create_app(w.env()), base_url="http://127.0.0.1") as client:
        changes = client.get("/api/changes")
        assert changes.status_code == 200
        data = changes.json()
        thread = client.get(f"/api/thread/{root['id']}")
        assert thread.status_code == 200
    roster = next(row for row in data["agents"] if row["parent"] == 7)
    assert roster["name"] == assignment.name
    assert roster["name_status"] == "assigned" and roster["unread_count"] == 1
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


def test_db_read_mode_and_all_gets_leave_bytes_and_unread_unchanged(
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
            "/static/style.css",
            "/api/changes",
            "/api/changes?refresh_roster=true",
            f"/api/thread/{post['id']}",
        ):
            assert client.get(url).status_code == 200
        for method in ("POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE"):
            for url in ("/api/changes", f"/api/thread/{post['id']}", "/"):
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
            (44, "claude", "unknown"),
        ]
        assert refreshed["agents"][1]["name_status"] == "unassigned"
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


def test_explicit_roster_refresh_shows_unknown_when_herdr_is_unavailable(
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
        assert response.status_code == 200
        refreshed = response.json()
        assert [row["state"] for row in refreshed["agents"]] == ["unknown", "unknown"]
        assert refreshed["roster_observed_at"] > initial["roster_observed_at"]
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
