"""Real controller outbox, registry and receiver transports with disposable peers."""

from __future__ import annotations

import importlib
import json
import socket
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_agy_parent import _agy_env, _pre_invocation
from tests.e2e.agents_ops.test_board_doorbell import (
    THREAD,
    Bells,
    inbox_socket,
    wait_received,
)
from tests.e2e.agents_ops.test_board_doorbell import (
    bells as bells,
)
from tests.e2e.agents_ops.test_board_messages import board as board
from tests.e2e.agents_ops.test_board_messages import labels as labels
from tests.e2e.agents_ops.test_board_messages import world as world
from tests.support.agents_ops.agy_tui import OTHER, ROOT, foreground
from tests.support.agents_ops.locks import available


def engine_for(
    bells: Bells,
    monkeypatch: pytest.MonkeyPatch,
    *,
    parent: int = 3,
    prs: tuple[int, ...] = (11,),
    suffix: str = "a",
) -> Any:
    progression = importlib.import_module("lib.ops_review.progression")
    backend = importlib.import_module("lib.ops_review.progression_backend")
    path = bells.w.labels.registry
    monkeypatch.setenv("XDG_STATE_HOME", bells.w.labels.env["XDG_STATE_HOME"])
    store = progression.Store(path.parent)
    with store.snapshot():
        row = store.parents[f"o/r#{parent}"]
    identifier = f"parent-{parent}-{suffix * 32}"
    state = progression.new_state(
        {
            "schema_version": 4,
            "review_run_id": identifier,
            "repo": "o/r",
            "parent_issue": parent,
            "prs": [{"number": pr} for pr in prs],
        },
        owner=row["session"],
        generation=row["generation"],
    )
    state_path = path.parent / "progressions" / identifier / "state.json"
    engine = progression.Engine(
        state_path, backend.LiveBackend(state, store), state=state
    )
    engine.save()
    return engine


def saved_notices(engine: Any) -> list[dict[str, Any]]:
    with engine.backend.store.snapshot() as store:
        row = store.parents[engine.backend.key]
    return sorted(
        row.get("progression_notifications", {}).values(), key=lambda n: n["event_id"]
    )


def replay_saved_events(engine: Any) -> None:
    for event in engine.state["events"]:
        event["delivered"] = False
    engine.save()
    restarted = type(engine)(engine.path, engine.backend)
    restarted.deliver()


def test_claude_bells_are_durable_private_and_idempotent(
    bells: Bells, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = engine_for(bells, monkeypatch)
    notifiers = bells.modules.notifiers
    send = notifiers.claude
    attempts = []

    def check_before_send(env: dict[str, str], bell: Any) -> Any:
        state_dir = engine.backend.store.directory
        assert available(state_dir / "registry.lock")
        assert not available(state_dir / "ownership.lock")
        record = saved_notices(engine)[-1]
        assert record["delivery"]["state"] == "notification_uncertain"
        attempts.append(bell.idempotency_key)
        return send(env, bell)

    monkeypatch.setattr(notifiers, "claude", check_before_send)
    before = bells.w.message_rows()
    with inbox_socket() as (path, received, condition):
        env = bells.claude(3, path)
        credential = env["CLAUDE_CODE_MESSAGING_TOKEN"]
        engine.emit("major", reason="major findings require parent instruction")
        engine.pause(
            engine.state["targets"]["11"], f"observation_failed: {path} {credential}"
        )
        engine.emit("publication_warning", reason=f"sensitive diagnostic {credential}")
        engine.state["status"] = "completed"
        engine.emit("completed")
        wait_received(received, condition, 4)
        replay_saved_events(engine)
        records = saved_notices(engine)
        assert [record["kind"] for record in records] == [
            "major",
            "parent_wait",
            "publication_warning",
            "completed",
        ]
        assert [record["delivery"]["state"] for record in records] == ["notified"] * 4
        assert records[1]["reason"] == "observation_failed"
        assert records[2]["reason"] == "details_in_merge_status"
        assert [message[1]["msg_id"] for message in received] == attempts
        assert len(set(attempts)) == 4
        assert all(
            message[0] == {"type": "auth", "token": credential} for message in received
        )
        messages = [message[1]["message"]["content"] for message in received]
        run = engine.state["id"]
        assert messages[1] == (
            f"[merge-progression:{run}#2] parent #3 PR #11 parent_wait (observation_failed): "
            f"parent --parent 3 merge-status --run-id {run}"
        )
        assert " PR #" not in messages[-1]
        public = json.dumps(records + messages)
        for private in (str(path), credential, env["CLAUDE_CODE_SESSION_ID"]):
            assert private not in public
    assert bells.w.message_rows() == before


def test_codex_uses_the_bound_thread_and_sends_one_line_once(
    bells: Bells, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, _, calls = bells.codex()
    engine = engine_for(bells, monkeypatch, parent=2)
    config = Path(env["CODEX_HOME"]) / "fake.json"
    data = json.loads(config.read_text())
    data["private"] = str(bells.modules.receivers.root(env) / "progression-queue")
    config.write_text(json.dumps(data))
    engine.state["status"] = "completed"
    engine.emit("completed")
    replay_saved_events(engine)
    (call,) = [json.loads(line) for line in calls.read_text().splitlines()]
    run = engine.state["id"]
    assert call["args"][4:] == [
        THREAD,
        "--message",
        f"[merge-progression:{run}#1] parent #2 completed: parent --parent 2 merge-status --run-id {run}",
    ]
    assert call["cwd"] == str(bells.home / "original cwd")
    assert call["phases"] == ["uncertain"]
    (notice,) = saved_notices(engine)
    assert notice["delivery"]["state"] == "notified"
    assert THREAD not in json.dumps(notice)


def test_agy_defers_to_its_foreground_hook(
    bells: Bells, monkeypatch: pytest.MonkeyPatch
) -> None:
    bells.w.change(3, session=f"agy:{ROOT}")
    engine = engine_for(bells, monkeypatch)
    engine.pause(engine.state["targets"]["11"], "ci_timeout")
    (notice,) = saved_notices(engine)
    assert notice["delivery"]["state"] == "awaiting_hook"
    assert not (bells.w.store.directory / "receivers").exists()
    with foreground(bells.home / "agy") as tui:
        assert tui.lock(ROOT)
        tui.metadata(ROOT)
        env = _agy_env(bells.w.labels.env)
        injected = _pre_invocation(tui, env, bells.home)
        (step,) = injected["injectSteps"]
        line = step["systemMessage"]["systemMessage"]
        assert (
            f"未確認通知:1件（PR #11 parent_wait / run={engine.state['id']}）" in line
        )
        assert ROOT not in line
        assert tui.lock(OTHER)
        assert _pre_invocation(tui, env, bells.home) == {}
        assert _pre_invocation(tui, env, bells.home, OTHER) == {}


@pytest.mark.parametrize(
    ("route", "expected"),
    [
        ("no_board", "route_unavailable"),
        ("unregistered", "route_unavailable"),
        ("stopped", "route_unavailable"),
        ("refused", "receiver_not_accepting"),
        ("exception", "notification_uncertain"),
    ],
)
def test_unavailable_bell_still_appears_in_hook_context(
    bells: Bells, monkeypatch: pytest.MonkeyPatch, route: str, expected: str
) -> None:
    engine = engine_for(bells, monkeypatch)
    display = bells.modules.display
    env = bells.w.env(3)
    board_context = display.context(env)
    if route == "no_board":
        bells.w.store.path.unlink()
        board_context = ""
    if route == "exception":

        def raise_private(*args: Any) -> Any:
            raise RuntimeError("private credential with /private/path")

        monkeypatch.setattr(bells.modules.notifiers, "claude", raise_private)
    with tempfile.TemporaryDirectory(prefix="progression-bell-") as short:
        path = Path(short) / "inbox.sock"
        if route in {"stopped", "refused"}:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(path))
                server.listen()
                env = bells.claude(
                    3, path, accept="hold" if route == "refused" else "accept"
                )
            assert path.is_socket()
        engine.pause(engine.state["targets"]["11"], "ci_timeout")
        assert engine.state["events"][0]["delivered"]
        (notice,) = saved_notices(engine)
        assert notice["delivery"]["state"] == notice["delivery"]["reason"] == expected
        assert "private credential" not in json.dumps(notice)
        assert display.context(env).splitlines() == [
            *board_context.splitlines(),
            f"[merge-progression] 未確認通知:1件（PR #11 parent_wait / run={engine.state['id']}）。"
            f"parent --parent 3 merge-status --run-id {engine.state['id']} で確認。",
        ]


def test_no_bell_to_a_stale_generation_or_a_replaced_owner(
    bells: Bells, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = engine_for(bells, monkeypatch)
    routing = importlib.import_module("lib.ops_review.progression_notify")
    with inbox_socket() as (path, received, _):
        env = bells.claude(3, path)
        context = bells.modules.display.context(env)
        bells.w.change(3, generation=old.state["generation"] + 1)
        old.pause(old.state["targets"]["11"], "ci_timeout")
        assert saved_notices(old) == []
        current = engine_for(bells, monkeypatch, suffix="b")
        current.pause(current.state["targets"]["11"], "ci_timeout")
        assert saved_notices(current)[0]["delivery"]["state"] == "route_unavailable"
        bells.claude(3, path)
        deliver = routing.deliver_bell

        def replace_owner(
            env: dict[str, str], store: Any, notice: dict[str, Any], owner: str
        ) -> str:
            assert (
                saved_notices(current)[-1]["delivery"]["state"]
                == "notification_uncertain"
            )
            with store.ownership(), store.locked():
                store.parents["o/r#3"].update(session="claude:new-owner", generation=3)
                store.save()
            return str(deliver(env, store, notice, owner))

        monkeypatch.setattr(routing, "deliver_bell", replace_owner)
        current.emit("parent_wait", reason="recovery_required")
        assert saved_notices(current)[-1]["delivery"]["state"] == "owner_changed"
        assert received == []
        assert bells.modules.display.context(env) == ""
        assert bells.modules.display.context(bells.w.env(3)) == context


def test_partial_continuation_hides_only_the_resolved_notice(
    bells: Bells, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = engine_for(bells, monkeypatch, prs=(11, 12))
    monkeypatch.setenv("AGENT_AUTO_MERGE", "1")
    monkeypatch.setattr(engine.backend, "changes", lambda _: {})
    env = bells.w.env(3)
    display = bells.modules.display
    initial = display.context(env)
    first, second = engine.state["targets"]["11"], engine.state["targets"]["12"]
    engine.pause(first, "ci_timeout")
    engine.pause(second, "ci_timeout")
    assert "未確認通知:2件" in display.context(env)
    engine.continue_run("1 件目だけを再開する。", prs=[11])
    assert "未確認通知:1件（PR #12 parent_wait" in display.context(env)
    engine.pause(first, "ci_timeout")
    assert "未確認通知:2件（PR #11 parent_wait" in display.context(env)
    engine.continue_run("残りを再開する。")
    assert display.context(env) == initial
    engine.pause(first, "ci_timeout")
    bells.w.change(3, status="released")
    assert display.context(env) == ""
