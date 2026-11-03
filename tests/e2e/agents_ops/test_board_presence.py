"""Live Herdr membership, frozen history and broadcast resolution boundaries."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.e2e.agents_ops.test_board import board as board
from tests.e2e.agents_ops.test_board_messages import World
from tests.e2e.agents_ops.test_board_messages import world as world
from tests.e2e.agents_ops.test_board_web import web as web
from tests.e2e.agents_ops.test_herdr_labels import labels as labels


def test_departures_remove_members_and_new_broadcast_targets_but_keep_history(
    world: World,
    web: Any,
) -> None:
    w = world
    root = w.post(to=["all"])
    registry_before = w.labels.registry.read_bytes()
    with TestClient(web.app.create_app(w.env()), base_url="http://localhost") as client:
        first = client.get("/api/changes").json()
        assert {row["parent"] for row in first["agents"]} == {1, 2, 3}
        runtime = w.labels.read()
        del runtime["panes"]["w1:board-2"]
        runtime["panes"]["w1:board-3"]["tokens"]["tl_owner"] = "replacement"
        w.labels.write(runtime)
        current = client.get("/api/changes?refresh_roster=true").json()
        assert [row["parent"] for row in current["agents"]] == [1]
        assert current["human"]["name"] == "human"
        thread = client.get(f"/api/thread/{root['id']}").json()["posts"]
        assert [row["parent"] for row in thread[0]["recipients"]] == [2, 3]
        human_post = w.board.messages.post(
            w.env(),
            human=True,
            kind="info",
            subject="Live",
            body="@all hello",
            notifiers=w.routes,
        )["post"]
        assert [row["parent"] for row in human_post["recipients"]] == [1]
        runtime = w.labels.read()
        del runtime["panes"]["w1:board-1"]
        w.labels.write(runtime)
        empty = client.get("/api/changes?refresh_roster=true").json()
        assert empty["agents"] == [] and empty["human"]["name"] == "human"
        assert len(empty["posts"]) == 2
        before = w.message_rows()
        with pytest.raises(w.board.errors.BoardError, match="at least one recipient"):
            w.board.messages.post(
                w.env(),
                human=True,
                kind="info",
                subject="Empty",
                body="@all hello",
                notifiers=w.routes,
            )
        assert w.message_rows() == before
        # The same confirmed conversation can return without inventing an identity.
        w.live_parent(2, status="blocked")
        back = client.get("/api/changes?refresh_roster=true").json()
        assert [(row["parent"], row["state"]) for row in back["agents"]] == [
            (2, "blocked")
        ]
    assert w.labels.registry.read_bytes() == registry_before


def test_broadcast_runtime_failure_is_atomic(world: World) -> None:
    w = world
    (w.labels.root / "bin/herdr").write_text("#!/bin/sh\nexit 3\n")
    before = w.message_rows()
    with pytest.raises(w.board.errors.BoardError, match="Herdr runtime is unavailable"):
        w.post(to=["all"])
    assert w.message_rows() == before and w.fake.calls == []


@pytest.mark.parametrize("sessions", [None, {}, [None], [{"running": True}]])
def test_malformed_server_list_is_not_an_empty_roster(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
    sessions: Any,
) -> None:
    service = world.board.service
    monkeypatch.setattr(service.Herdr, "call", lambda *args: {"sessions": sessions})
    with pytest.raises(world.board.errors.BoardError, match="Herdr"):
        service.agents(world.env())


def test_broadcast_rejects_assignment_change_during_live_lookup(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    w = world
    lookup = w.board.messages._runtime_states

    def replaced(env: Any, identities: list[str]) -> dict[str, str]:
        states: dict[str, str] = lookup(env, identities)
        w.change(2, session="codex:replacement", generation=2)
        return states

    monkeypatch.setattr(w.board.messages, "_runtime_states", replaced)
    before = w.message_rows()
    with pytest.raises(w.board.errors.BoardError, match="registry changed"):
        w.post(to=["all"])
    assert w.message_rows() == before and w.fake.calls == []


@pytest.mark.parametrize("replaced", [False, True])
def test_tokenless_current_herdr_requires_matching_foreground_process(
    world: World,
    replaced: bool,
) -> None:
    w = world
    runtime = w.labels.read()
    pane = runtime["panes"]["w1:board-1"]
    pane["tokens"] = {}
    w.labels.write(runtime)
    service = w.board.service
    identity = json.loads(w.labels.registry.read_text())["parents"]["o/r#1"]["session"]
    processes = service.process_stamp(service.Herdr(w.env()), pane)
    if replaced:
        processes[0]["start"] = "stale-process-start"
    path = w.labels.registry.parent.parent / "herdr-labels/panes"
    path.mkdir(exist_ok=True)
    key = service.owner_token(
        service.owner_token("/fake/herdr.sock") + ":" + pane["terminal_id"]
    )
    (path / f"{key}.json").write_text(
        json.dumps(
            {
                "version": 1,
                "owner": service.owner_token(identity),
                "state": "active",
                "processes": processes,
            }
        )
    )
    roster = service.agents(w.env())["agents"]
    assert {row["parent"] for row in roster} == ({2, 3} if replaced else {1, 2, 3})
