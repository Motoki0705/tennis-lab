"""Web reads and human writes through temporary state and fake notifier routes."""

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
    world: World, web: SimpleNamespace, tmp_path: Path
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
        pytest.skip("Install the optional Playwright package and set NODE_PATH")
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
