"""Messages through private SQLite, temporary registry and fake notifier routes."""

from __future__ import annotations

import importlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.e2e.agents_ops.test_board import NOW, private_output_absent, run_board
from tests.e2e.agents_ops.test_board import board as board
from tests.e2e.agents_ops.test_herdr_labels import Labels
from tests.e2e.agents_ops.test_herdr_labels import labels as labels


@dataclass
class World:
    board: SimpleNamespace
    labels: Labels
    assignments: dict[int, Any]
    fake: Any

    def env(self, parent: int = 1) -> dict[str, str]:
        parents = json.loads(self.labels.registry.read_text())["parents"]
        identity = parents[f"o/r#{parent}"]["session"]
        engine, _, session = identity.partition(":")
        env = {
            key: value
            for key, value in self.labels.env.items()
            if not key.startswith(("CLAUDE_", "CODEX_"))
        }
        env["CLAUDE_CODE_SESSION_ID" if engine == "claude" else "CODEX_SESSION_ID"] = (
            session
        )
        if engine == "codex":
            env["CODEX_THREAD_ID"] = session
        return env

    @property
    def store(self) -> Any:
        return self.board.storage.Store(self.board.storage.directory(self.labels.env))

    @property
    def routes(self) -> dict[str, Any]:
        return {"claude": self.fake, "codex": self.fake}

    def change(self, parent: int, **values: Any) -> None:
        data = json.loads(self.labels.registry.read_text())
        data["parents"][f"o/r#{parent}"].update(values)
        self.labels.set_registry(data["parents"])

    def post(self, **values: Any) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self.board.messages.post(
                self.env(),
                **{
                    "to": [self.assignments[2].name],
                    "kind": "question",
                    "subject": "Dependency",
                    "body": "Please check the dependency.",
                    "notifiers": self.routes,
                    "now": NOW,
                    **values,
                },
            )["post"],
        )

    def own_state(self, post_id: str, parent: int = 1) -> dict[str, Any]:
        return next(
            row
            for row in self.board.messages.status(self.env(parent))["posts"]
            if row["id"] == post_id
        )

    def message_rows(self) -> dict[str, list[tuple[Any, ...]]]:
        with self.store.read() as connection:
            return {
                table: connection.execute(f"SELECT * FROM {table}").fetchall()
                for table in ("posts", "recipients", "deliveries")
            }

    def assert_delivered(self, post: dict[str, Any], parents: list[int]) -> None:
        assert [row["parent"] for row in post["recipients"]] == parents
        assert [row["name"] for row in post["recipients"]] == [
            self.assignments[parent].name for parent in parents
        ]
        assert all(row["delivery_state"] == "notified" for row in post["recipients"])
        calls = [call.parent for call in self.fake.calls if call.post_id == post["id"]]
        assert calls == parents
        with self.store.read() as connection:
            for table in ("recipients", "deliveries"):
                assert connection.execute(
                    f"SELECT assignment_id FROM {table} WHERE post_id=? ORDER BY assignment_id",
                    (post["id"],),
                ).fetchall() == [(self.assignments[parent].id,) for parent in parents]


@pytest.fixture
def world(board: SimpleNamespace, labels: Labels) -> World:
    for name in ("messages", "delivery", "records", "redelivery"):
        setattr(board, name, importlib.import_module("board." + name))
    statuses = ("active", "active", "active", "released", "handoff_pending", "ended")
    labels.set_registry(
        {
            f"o/r#{parent}": labels.owner(
                status=status, identity=f"{engine}:board-owner-{parent}"
            )
            for parent, (status, engine) in enumerate(
                zip(
                    statuses,
                    ("claude", "codex", "claude", "codex", "claude", "claude"),
                    strict=True,
                ),
                1,
            )
        }
    )
    assignments = {}
    with board.storage.Store(
        board.storage.directory(labels.env)
    ).transaction() as connection:
        for parent in range(1, 7):
            assignments[parent] = board.names.ensure_name(
                connection, "o/r", parent, None, now=NOW
            )
        board.names.retire_name(connection, "o/r", 6, now=NOW)
    return World(board, labels, assignments, board.delivery.FakeNotifier())


def test_post_reply_thread_inbox_and_recipient_response(world: World) -> None:
    w = world
    root = w.post(
        to=[w.assignments[2].name.lower(), w.assignments[2].name.upper()],
        body=f"@{w.assignments[2].name.lower()}, please check #10.",
    )
    assert root["sender"] == {
        "assignment_id": w.assignments[1].id,
        "name": w.assignments[1].name,
        "repo": "o/r",
        "parent": 1,
    }
    assert root["recipients"][0]["state"] == "notified"
    assert len(root["recipients"]) == len(w.fake.accepted) == 1
    bell = w.fake.calls[0]
    assert bell.engine == "codex"
    assert root["body"] not in bell.bell
    assert root["id"] in bell.bell and "board read" in bell.bell
    previews = w.board.messages.inbox(w.env(2), unread=True)["posts"]
    assert [item["id"] for item in previews] == [root["id"]]
    assert "body" not in previews[0]
    assert w.own_state(root["id"])["recipients"][0]["state"] == "notified"
    result = w.board.messages.read(w.env(2), root["id"])["post"]
    assert result["body"] == root["body"]
    assert result["recipients"][0]["state"] == "read"
    assert result["recipients"][0]["delivery_state"] == "notified"
    assert w.board.messages.inbox(w.env(2), unread=True) == {"posts": []}
    response = w.board.messages.reply(
        w.env(2),
        post_id=root["id"],
        to=[w.assignments[3].name],
        body="The dependency is ready.",
        notifiers=w.routes,
        now=NOW,
    )["post"]
    assert response["kind"] == "reply"
    assert response["in_reply_to"] == root["id"] == response["thread_id"]
    assert {row["parent"] for row in response["recipients"]} == {1, 3}
    assert root["id"] < response["id"]
    assert w.own_state(root["id"])["recipients"][0]["state"] == "answered"
    conversation = w.board.messages.thread(w.env(3), response["id"])
    assert conversation["thread_id"] == root["id"]
    assert [row["body"] for row in conversation["posts"]] == [
        root["body"],
        response["body"],
    ]
    assert w.board.messages.inbox(w.env(3), unread=True) == {"posts": []}
    nested = w.board.messages.reply(
        w.env(1), post_id=response["id"], body="Thanks.", notifiers=w.routes, now=NOW
    )["post"]
    assert (
        nested["recipients"][0]["parent"] == 2
    )  # Immediate author, not the root author.
    assert nested["thread_id"] == root["id"]
    assert [root["id"], response["id"], nested["id"]] == sorted(
        [root["id"], response["id"], nested["id"]]
    )
    private_output_absent(json.dumps(conversation), w.env())


def test_renamed_parents_keep_message_identity_state_and_reply_routes(
    world: World,
) -> None:
    w = world
    rename = importlib.import_module("board.renaming").rename
    unread = w.post()
    read = w.post()
    acknowledged = w.post(kind="request")
    read_at = w.board.messages.read(w.env(2), read["id"])["post"]["recipients"][0][
        "read_at"
    ]
    w.board.messages.ack(w.env(2), post_id=acknowledged["id"], state="accepted")
    outgoing = w.board.messages.post(
        w.env(2),
        to=[w.assignments[1].name],
        kind="info",
        subject="確認結果",
        body="確認しました。",
        notifiers=w.routes,
        now=NOW,
    )["post"]
    before = w.message_rows()
    with w.store.read() as connection:
        activity = w.board.activity.summary(connection, w.assignments[2].id)
    assert activity.unread_count == 1 and activity.awaiting_response
    changed = {
        parent: rename(w.env(), parent=parent, now=NOW + timedelta(days=1))
        for parent in (1, 2)
    }
    assert w.message_rows() == before
    for parent in (1, 2):
        assert changed[parent]["assignment_id"] == w.assignments[parent].id
        assert changed[parent]["old_name"] == w.assignments[parent].name
        assert changed[parent]["new_name"] != w.assignments[parent].name
    assert w.board.service.whoami(w.env(2))["name"] == changed[2]["new_name"]
    with w.store.read() as connection:
        assert w.board.activity.summary(connection, w.assignments[2].id) == activity
    inbox = w.board.messages.inbox(w.env(2), unread=True)["posts"]
    assert [post["id"] for post in inbox] == [unread["id"]]
    sent = w.own_state(outgoing["id"], parent=2)
    assert sent["sender"]["name"] == w.assignments[2].name
    root = w.own_state(unread["id"])
    assert root["sender"]["name"] == w.assignments[1].name
    assert root["recipients"][0]["name"] == w.assignments[2].name
    acknowledged_row = w.own_state(acknowledged["id"])["recipients"][0]
    assert acknowledged_row["state"] == "accepted"
    read_again = w.board.messages.read(w.env(2), read["id"])["post"]
    assert read_again["recipients"][0]["read_at"] == read_at
    thread = w.board.messages.thread(w.env(2), unread["id"])["posts"]
    assert thread[0]["sender"]["name"] == w.assignments[1].name
    reply = w.board.messages.reply(
        w.env(2), post_id=unread["id"], body="対応します。", notifiers=w.routes
    )["post"]
    assert reply["sender"]["name"] == changed[2]["new_name"]
    assert reply["sender"]["assignment_id"] == w.assignments[2].id
    recipient = reply["recipients"][0]
    assert recipient["name"] == changed[1]["new_name"]
    assert recipient["assignment_id"] == w.assignments[1].id
    assert w.own_state(unread["id"])["recipients"][0]["response_state"] == "answered"


@pytest.mark.parametrize("command", ["post", "reply"])
@pytest.mark.parametrize("audience", ["named", "all"])
def test_body_mentions_without_to_are_saved_and_delivered(
    world: World, command: str, audience: str
) -> None:
    w = world
    root = w.post() if command == "reply" else None
    body = (
        f"Ask @{w.assignments[3].name.swapcase()}." if audience == "named" else "@aLl"
    )
    if root:
        post = w.board.messages.reply(
            w.env(2), post_id=root["id"], body=body, notifiers=w.routes
        )["post"]
        expected = [1, 3]  # A mention adds to the immediate author's assignment.
        assert w.own_state(root["id"])["recipients"][0]["state"] == "answered"
    else:
        post = w.board.messages.post(
            w.env(), kind="info", subject="Mention only", body=body, notifiers=w.routes
        )["post"]
        expected = [3] if audience == "named" else [2, 3]
    assert post["broadcast"] is (audience == "all")
    assert post["body"] == body
    w.assert_delivered(post, expected)


@pytest.mark.parametrize("command", ["post", "reply"])
def test_to_and_body_mentions_form_a_case_insensitive_union(
    world: World, command: str
) -> None:
    w = world
    root = w.post() if command == "reply" else None
    other = 1 if root else 2
    name = w.assignments[other].name
    additional = w.assignments[3].name
    arguments = {
        "to": [name.lower(), name.upper()],
        "body": f"@{name.swapcase()}, ask @{additional} and @{additional.lower()}.",
        "notifiers": w.routes,
    }
    if root:
        post = w.board.messages.reply(w.env(2), post_id=root["id"], **arguments)["post"]
    else:
        post = w.post(**arguments)
    assert post["broadcast"] is False
    w.assert_delivered(post, [other, 3])


@pytest.mark.parametrize("command", ["post", "reply"])
@pytest.mark.parametrize(
    "audience", ["named", "all", "mixed", "body-named", "body-all", "body-mixed"]
)
def test_cli_broadcast_is_persisted_and_returned_in_every_post_view(
    world: World, command: str, audience: str
) -> None:
    w = world
    root = w.post(to=["all"]) if command == "reply" else None
    caller = 2 if root else 1
    recipients = [parent for parent in (1, 2, 3) if parent != caller]
    names = {
        "named": [w.assignments[parent].name for parent in recipients],
        "all": ["all"],
        "mixed": [w.assignments[3].name.lower(), "aLl"],
        "body-named": [],
        "body-all": [],
        "body-mixed": [w.assignments[3].name.lower()],
    }[audience]
    body_file = w.labels.root / "input.md"
    body = {
        "body-named": " ".join(
            f"@{w.assignments[parent].name.swapcase()}" for parent in recipients
        ),
        "body-all": "@aLl",
        "body-mixed": f"@ALL @{w.assignments[3].name.upper()}",
    }.get(audience, "A public update without body mentions.")
    body_file.write_text(body)
    args = [command, "--body-file", str(body_file)]
    if root:
        args += ["--post", root["id"]]
    else:
        args += ["--kind", "info", "--subject", "Public"]
    for name in names:
        args += ["--to", name]
    created = run_board(w.env(caller), *args)
    assert created.returncode == 0, created.stderr
    post = json.loads(created.stdout)["post"]
    broadcast = audience not in {"named", "body-named"}
    assert post["broadcast"] is broadcast
    # Explicit names and all expand to the same recipients, but keep their intent.
    assert [row["parent"] for row in post["recipients"]] == recipients
    with w.store.read() as connection:
        assert connection.execute(
            "SELECT broadcast, typeof(broadcast) FROM posts WHERE id=?",
            (post["id"],),
        ).fetchone() == (int(broadcast), "integer")
    for env, operation in [
        (w.env(3), ["inbox", "--unread"]),
        (w.env(caller), ["status"]),
        (w.env(3), ["read", post["id"]]),
        (w.env(3), ["thread", post["id"]]),
        (w.env(3), ["ack", "--post", post["id"], "--state", "accepted"]),
    ]:
        result = run_board(env, *operation)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        views = [data["post"]] if "post" in data else data["posts"]
        target = next(view for view in views if view["id"] == post["id"])
        assert target["broadcast"] is broadcast
        private_output_absent(result.stdout + result.stderr, env)


@pytest.mark.parametrize("command", ["post", "reply"])
@pytest.mark.parametrize("audience", ["self", "other-self", "all-self"])
def test_cli_self_recipient_is_rejected_atomically(
    world: World, command: str, audience: str
) -> None:
    w = world
    root = w.post() if command == "reply" else None
    caller = 2 if root else 1
    names = {
        "self": [],
        "other-self": [w.assignments[3].name],
        "all-self": ["all"],
    }[audience] + [w.assignments[caller].name.swapcase()]
    before = w.message_rows()
    calls = list(w.fake.calls)
    body_file = w.labels.root / "input.md"
    body_file.write_text("A public update.")
    args = [command, "--body-file", str(body_file)]
    if root:
        args += ["--post", root["id"]]
    else:
        args += ["--kind", "info", "--subject", "Public"]
    for name in names:
        args += ["--to", name]

    with pytest.raises(w.board.errors.BoardError, match="sender cannot be a recipient"):
        if root:
            w.board.messages.reply(
                w.env(caller),
                post_id=root["id"],
                to=names,
                body="A public update.",
                notifiers=w.routes,
            )
        else:
            w.post(to=names)
    result = run_board(w.env(caller), *args)
    assert result.returncode == 2 and result.stdout == ""
    assert "sender cannot be a recipient" in result.stderr
    assert w.message_rows() == before
    assert w.fake.calls == calls
    assert w.board.messages.inbox(w.env(caller), unread=True) == {
        "posts": [w.own_state(root["id"])] if root else []
    }


def test_cli_reply_to_own_post_without_to_or_body_mentions_is_rejected(
    world: World,
) -> None:
    w = world
    root = w.post(to=["all"])
    before = w.message_rows()
    calls = list(w.fake.calls)
    with pytest.raises(w.board.errors.BoardError, match="at least one recipient"):
        w.board.messages.reply(
            w.env(), post_id=root["id"], body="An update.", notifiers=w.routes
        )
    body_file = w.labels.root / "input.md"
    body_file.write_text("An update.")
    result = run_board(
        w.env(), "reply", "--post", root["id"], "--body-file", str(body_file)
    )
    assert result.returncode == 2 and result.stdout == ""
    assert "at least one recipient" in result.stderr
    assert w.message_rows() == before
    assert w.fake.calls == calls
    assert w.board.messages.inbox(w.env(), unread=True) == {"posts": []}


@pytest.mark.parametrize("audience", ["named", "all", "body-named", "body-all"])
def test_reply_to_own_post_uses_to_or_mentions_without_inheriting_recipients(
    world: World, audience: str
) -> None:
    w = world
    root = w.post()
    target = w.assignments[3].name if audience.endswith("named") else "all"
    names = [] if audience.startswith("body-") else [target]
    expected = [3] if audience.endswith("named") else [2, 3]
    reply = w.board.messages.reply(
        w.env(),
        post_id=root["id"],
        to=names,
        body=f"An update for @{target}." if not names else "An update.",
        notifiers=w.routes,
    )["post"]
    assert [row["parent"] for row in reply["recipients"]] == expected
    assert reply["in_reply_to"] == reply["thread_id"] == root["id"]
    assert [call.parent for call in w.fake.calls if call.post_id == reply["id"]] == (
        expected
    )
    assert w.board.messages.inbox(w.env(), unread=True) == {"posts": []}
    assert [post["id"] for post in w.board.messages.inbox(w.env(2))["posts"]] == (
        [root["id"]] if audience.endswith("named") else [root["id"], reply["id"]]
    )
    original_recipient = w.own_state(root["id"])["recipients"][0]
    assert original_recipient["read_at"] is None
    assert original_recipient["response_state"] is None
    with w.store.read() as connection:
        assert connection.execute(
            "SELECT assignment_id FROM recipients WHERE post_id=? ORDER BY assignment_id",
            (reply["id"],),
        ).fetchall() == [(w.assignments[parent].id,) for parent in expected]
        assert connection.execute(
            "SELECT assignment_id FROM deliveries WHERE post_id=? ORDER BY assignment_id",
            (reply["id"],),
        ).fetchall() == [(w.assignments[parent].id,) for parent in expected]


@pytest.mark.parametrize("command", ["post", "reply"])
@pytest.mark.parametrize("audience", ["named", "all", "mention-only"])
def test_cli_self_mention_is_rejected_as_self_addressing_atomically(
    world: World, command: str, audience: str
) -> None:
    w = world
    root = w.post() if command == "reply" else None
    names = {
        "named": [w.assignments[3].name],
        "all": ["all"],
        "mention-only": [],
    }[audience]
    body = f"An update from @{w.assignments[1].name.swapcase()}."
    before = w.message_rows()
    calls = list(w.fake.calls)
    with pytest.raises(w.board.errors.BoardError, match="sender cannot be a recipient"):
        if root:
            w.board.messages.reply(
                w.env(),
                post_id=root["id"],
                to=names,
                body=body,
                notifiers=w.routes,
            )
        else:
            w.post(to=names, body=body)
    body_file = w.labels.root / "input.md"
    body_file.write_text(body)
    args = [command, "--body-file", str(body_file)]
    for name in names:
        args += ["--to", name]
    args += (
        ["--post", root["id"]] if root else ["--kind", "info", "--subject", "Public"]
    )
    result = run_board(w.env(), *args)
    assert result.returncode == 2 and result.stdout == ""
    assert "sender cannot be a recipient" in result.stderr
    assert w.message_rows() == before
    assert w.fake.calls == calls
    assert w.board.messages.inbox(w.env(), unread=True) == {"posts": []}


@pytest.mark.parametrize("kind", ["info", "question", "request"])
def test_all_freezes_only_active_others_and_response_waiting(
    world: World, kind: str
) -> None:
    w = world
    root = w.post(to=["ALL"], kind=kind, body="@all: Please check this change.")
    assert [r["parent"] for r in root["recipients"]] == [2, 3]
    with w.store.read() as connection:
        for parent in (2, 3):
            summary = w.board.activity.summary(connection, w.assignments[parent].id)
            assert summary.unread_count == 1
            assert summary.awaiting_response is (kind != "info")
        assert not w.board.activity.has_pending(connection, w.assignments[1].id)
    w.change(4, status="active")
    assert [r["parent"] for r in w.own_state(root["id"])["recipients"]] == [2, 3]
    assert w.board.messages.inbox(w.env(4)) == {"posts": []}


def test_all_rejects_unassigned_active_parent_atomically(world: World) -> None:
    w = world
    data = json.loads(w.labels.registry.read_text())["parents"]
    data["o/r#7"] = w.labels.owner(identity="codex:board-owner-7")
    w.labels.set_registry(data)
    with pytest.raises(w.board.errors.BoardError, match="o/r#7 has no live board name"):
        w.post(to=["all"])
    assert w.board.messages.status(w.env()) == {"posts": []}
    assert w.fake.calls == []


@pytest.mark.parametrize("parent", [4, 5])
def test_held_visible_to_sender_and_resumes_with_same_assignment(
    world: World, parent: int
) -> None:
    w = world
    root = w.post(to=[w.assignments[parent].name])
    recipient = root["recipients"][0]
    assert recipient["state"] == recipient["delivery_state"] == "held"
    assert recipient["delivery_reason"] == "recipient_unavailable"
    assert recipient["delivery_message"] == "担当不在・配送保留"
    assert w.fake.calls == []
    held = w.board.delivery.held_for_name(w.env(), w.assignments[parent].name.lower())
    assert [row["post_id"] for row in held] == [root["id"]]
    assert w.own_state(root["id"])["recipients"][0]["state"] == "held"
    old_env = w.env(parent)
    w.change(
        parent, status="active", session=f"codex:board-successor-{parent}", generation=2
    )
    w.board.delivery.dispatch_held(w.env(parent), w.assignments[parent].name, w.routes)
    w.board.delivery.dispatch_held(w.env(parent), w.assignments[parent].name, w.routes)
    assert len(w.fake.calls) == 1
    assert w.fake.calls[0].assignment_id == w.assignments[parent].id
    assert w.fake.calls[0].engine == "codex"
    assert w.board.delivery.held_for_name(w.env(), w.assignments[parent].name) == []
    assert (
        w.board.messages.inbox(w.env(parent), unread=True)["posts"][0]["id"]
        == root["id"]
    )
    with pytest.raises(w.board.errors.BoardError, match="registered active"):
        w.board.messages.read(old_env, root["id"])


def test_each_recipient_has_independent_delivery_read_ack_and_answer(
    world: World,
) -> None:
    w = world
    root = w.post(to=["all"], kind="request")
    accepted = w.board.messages.ack(w.env(2), post_id=root["id"], state="accepted")[
        "post"
    ]
    first_stamp = accepted["recipients"][0]["responded_at"]
    duplicate = w.board.messages.ack(w.env(2), post_id=root["id"], state="accepted")[
        "post"
    ]
    assert duplicate["recipients"][0]["responded_at"] == first_stamp
    assert duplicate["recipients"][1]["state"] == "notified"
    w.board.messages.ack(w.env(2), post_id=root["id"], state="done")
    w.board.messages.ack(w.env(3), post_id=root["id"], state="declined")
    assert [r["state"] for r in w.own_state(root["id"])["recipients"]] == [
        "done",
        "declined",
    ]
    w.board.messages.read(w.env(2), root["id"])
    w.board.messages.reply(
        w.env(3),
        post_id=root["id"],
        body="I have other priorities.",
        notifiers=w.routes,
    )
    assert [r["state"] for r in w.own_state(root["id"])["recipients"]] == [
        "done",
        "declined",
    ]
    with w.store.read() as connection:
        assert not w.board.activity.has_pending(connection, w.assignments[2].id)
        assert not w.board.activity.has_pending(connection, w.assignments[3].id)
    with pytest.raises(w.board.errors.BoardError, match="already final"):
        w.board.messages.ack(w.env(2), post_id=root["id"], state="declined")


def test_partial_failure_retry_preserves_keys_and_never_resends_success(
    world: World,
) -> None:
    w = world
    w.fake.fail_once_for.add(w.assignments[3].id)
    root = w.post(to=["all"])
    assert [r["state"] for r in root["recipients"]] == ["notified", "delivery_failed"]
    assert root["recipients"][1]["delivery_reason"] == "route_unavailable"
    original_key = w.fake.calls[1].idempotency_key
    w.board.delivery.dispatch(w.env(), w.routes, post_id=root["id"])
    w.board.delivery.dispatch(w.env(), w.routes, post_id=root["id"])
    assert [r["state"] for r in w.own_state(root["id"])["recipients"]] == [
        "notified",
        "notified",
    ]
    assert len(w.fake.calls) == 3 and len(w.fake.accepted) == 2
    assert w.fake.calls[-1].idempotency_key == original_key
    assert w.fake.calls[0].idempotency_key != original_key


def test_sender_dispatch_retries_only_failed_recipients_with_the_same_key(
    world: World,
) -> None:
    w = world
    w.fake.fail_once_for.add(w.assignments[3].id)
    root = w.post(to=[w.assignments[2].name, w.assignments[3].name])
    original_key = w.fake.calls[-1].idempotency_key
    result = w.board.redelivery.dispatch_post(w.env(), w.routes, post_id=root["id"])
    assert result["status"] == "notified"
    notified, retried = result["recipients"]
    assert notified["outcome"] == "skipped"
    assert notified["skip_reason"] == "already_notified"
    assert not notified["attempted"] and notified["attempts"] == 1
    assert retried["outcome"] == "notified" and retried["attempted"]
    assert retried["skip_reason"] is None and retried["attempts"] == 2
    assert len(w.fake.calls) == 3 and w.fake.calls[-1].idempotency_key == original_key
    assert all(r["read_at"] is None for r in result["recipients"])
    again = w.board.redelivery.dispatch_post(w.env(), w.routes, post_id=root["id"])
    assert again["status"] == "no_targets" and len(w.fake.calls) == 3
    private_output_absent(json.dumps(result), w.env())


def test_sender_dispatch_selects_saved_recipient_and_sends_pending_without_retry_flag(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    w = world
    # Simulate interruption after the post commit and before initial dispatch.
    with monkeypatch.context() as patch:
        patch.setattr(w.board.messages, "dispatch", lambda *_args, **_kwargs: [])
        root = w.post(to=[w.assignments[2].name, w.assignments[3].name])
    assert all(
        r["delivery_state"] == "pending" and not r["retryable"]
        for r in root["recipients"]
    )
    result = w.board.redelivery.dispatch_post(
        w.env(), w.routes, post_id=root["id"], to=w.assignments[3].name.swapcase()
    )
    assert result["status"] == "notified" and len(result["recipients"]) == 1
    assert result["recipients"][0]["parent"] == 3
    assert len(w.fake.calls) == 1 and w.fake.calls[0].parent == 3
    assert [r["delivery_state"] for r in w.own_state(root["id"])["recipients"]] == [
        "pending",
        "notified",
    ]


@pytest.mark.parametrize("state", ["notified", "read", "uncertain", "terminal", "held"])
def test_sender_dispatch_reports_skips_without_notifying(
    world: World, state: str
) -> None:
    w = world
    if state == "held":
        root = w.post(to=[w.assignments[4].name])
        w.change(
            4, status="active"
        )  # Only the restoration hook may dispatch held rows.
    elif state in {"uncertain", "terminal"}:
        reason = (
            "notification_uncertain"
            if state == "uncertain"
            else "notification_rejected"
        )
        root = w.post(
            notifiers={
                "codex": lambda _n: w.board.delivery.NotificationResult(False, reason)
            }
        )
    else:
        if state == "read":
            w.fake.accepting = False
        root = w.post()
        if state == "read":
            w.board.messages.read(w.env(2), root["id"])
    before = w.message_rows()
    calls = list(w.fake.calls)
    result = w.board.redelivery.dispatch_post(w.env(), w.routes, post_id=root["id"])
    assert result["status"] == "no_targets"
    recipient = result["recipients"][0]
    assert recipient["outcome"] == "skipped" and not recipient["attempted"]
    assert (
        recipient["skip_reason"]
        == {
            "notified": "already_notified",
            "read": "already_read",
            "uncertain": "notification_uncertain",
            "terminal": "not_retryable",
            "held": "held",
        }[state]
    )
    assert w.message_rows() == before and w.fake.calls == calls


def test_sender_dispatch_reports_retry_failure_and_keeps_it_retryable(
    world: World,
) -> None:
    w = world
    w.fake.accepting = False
    root = w.post()
    result = w.board.redelivery.dispatch_post(w.env(), w.routes, post_id=root["id"])
    assert result["status"] == "failed"
    recipient = result["recipients"][0]
    assert recipient["attempted"] and recipient["outcome"] == "failed"
    assert (
        recipient["delivery_reason"] == "receiver_not_accepting"
        and recipient["retryable"]
    )
    assert recipient["attempts"] == len(w.fake.calls) == 2
    assert w.fake.calls[0].idempotency_key == w.fake.calls[1].idempotency_key


@pytest.mark.parametrize("status", ["handoff_pending", "ended"])
def test_sender_dispatch_rechecks_recipient_before_attempt(
    world: World, status: str
) -> None:
    w = world
    root = w.post(notifiers={})
    w.change(2, status=status)
    result = w.board.redelivery.dispatch_post(w.env(), w.routes, post_id=root["id"])
    assert result["status"] == "failed" and w.fake.calls == []
    recipient = result["recipients"][0]
    assert recipient["outcome"] == "failed" and not recipient["attempted"]
    assert (
        recipient["skip_reason"]
        == recipient["delivery_reason"]
        == ("recipient_retired" if status == "ended" else "recipient_unavailable")
    )


@pytest.mark.parametrize("entrypoint", ["api", "cli"])
@pytest.mark.parametrize(
    "change", ["released", "ended", "replaced", "assignment", "identity"]
)
def test_sender_dispatch_rechecks_authority_after_notification_lock_wait(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    entrypoint: str,
    change: str,
) -> None:
    w = world
    w.fake.fail_once_for.update(w.assignments[p].id for p in (2, 3))
    root = w.post(to=[w.assignments[p].name for p in (2, 3)])
    w.fake.calls.clear()
    before = w.message_rows()
    env = w.env()  # Keep the original caller even after the owner is replaced.
    waiting = threading.Event()
    original_lock = w.board.delivery.locked

    @contextmanager
    def observed_lock(path: Path) -> Iterator[None]:
        assert path == w.store.directory / "notifications.lock"
        waiting.set()  # dispatch_post has released its initial registry lock.
        with original_lock(path):
            yield

    monkeypatch.setattr(w.board.delivery, "locked", observed_lock)
    cli = importlib.import_module("board.__main__")
    monkeypatch.setattr(cli, "os", SimpleNamespace(environ=env))
    monkeypatch.setattr(cli, "configured", lambda _env: w.routes)

    def retry() -> Any:
        if entrypoint == "cli":
            return cli.main(["dispatch", "--post", root["id"]])
        return w.board.redelivery.dispatch_post(env, w.routes, post_id=root["id"])

    with ThreadPoolExecutor(max_workers=1) as pool:
        with original_lock(w.store.directory / "notifications.lock"):
            future = pool.submit(retry)
            assert waiting.wait(timeout=5)
            assert not future.done()
            if change == "replaced":
                w.change(1, session="claude:replacement-owner", generation=2)
            elif change == "assignment":
                with w.store.transaction() as connection:
                    w.board.names.retire_name(connection, "o/r", 1, now=NOW)
                    connection.execute(
                        """INSERT INTO name_assignments
                           (name, repo, parent, identifier, assigned_at, retired_at)
                           VALUES (?, 'o/r', 1, NULL, ?, NULL)""",
                        (w.board.names.load_names()[10], w.board.names.timestamp(NOW)),
                    )
            elif change == "identity":
                env.update(w.env(3))
            else:
                w.change(1, status=change)
        if entrypoint == "cli":
            assert future.result(timeout=5) == 2
            output = capsys.readouterr()
            assert output.out == ""
            assert "only the active post sender" in output.err
            private_output_absent(output.err, env)
        else:
            with pytest.raises(
                w.board.errors.BoardError, match="only the active post sender"
            ):
                future.result(timeout=5)
    assert w.fake.calls == []
    assert w.message_rows() == before  # Includes state, attempts, reason and times.


@pytest.mark.parametrize("change", ["released", "replaced"])
def test_sender_dispatch_rechecks_authority_before_each_recipient(
    world: World, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    w = world
    root = w.post(to=[w.assignments[p].name for p in (2, 3)], notifiers={})
    before = w.message_rows()["deliveries"]
    env = w.env()
    original_parent_state = w.board.delivery.parent_state
    completed = 0

    @contextmanager
    def change_after_first_recipient(values: Any) -> Iterator[dict[str, Any]]:
        nonlocal completed
        with original_parent_state(values) as parents:
            yield parents
        completed += 1
        if completed == 1:
            if change == "replaced":
                w.change(1, session="claude:replacement-owner", generation=2)
            else:
                w.change(1, status=change)

    monkeypatch.setattr(w.board.delivery, "parent_state", change_after_first_recipient)
    with pytest.raises(w.board.errors.BoardError, match="only the active post sender"):
        w.board.redelivery.dispatch_post(env, w.routes, post_id=root["id"])
    assert [n.assignment_id for n in w.fake.calls] == [w.assignments[2].id]
    after = w.message_rows()["deliveries"]
    assert after[0] != before[0]  # The first attempt was still authorized.
    assert after[1:] == before[1:]


def test_concurrent_sender_dispatch_reports_only_its_own_attempt(world: World) -> None:
    w = world
    root = w.post(notifiers={})
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: w.board.redelivery.dispatch_post(
                    w.env(), w.routes, post_id=root["id"]
                ),
                range(4),
            )
        )
    assert len(w.fake.calls) == 1
    assert sorted(r["status"] for r in results) == ["no_targets"] * 3 + ["notified"]
    assert sum(r["recipients"][0]["attempted"] for r in results) == 1


@pytest.mark.parametrize("caller", [2, 3])
def test_cli_dispatch_rejects_recipients_and_other_parents(
    world: World, caller: int
) -> None:
    w = world
    root = w.post(notifiers={})
    before = w.message_rows()
    result = run_board(w.env(caller), "dispatch", "--post", root["id"])
    assert result.returncode == 2 and result.stdout == ""
    assert "only the post sender" in result.stderr
    assert w.message_rows() == before
    private_output_absent(result.stderr, w.env(caller))


@pytest.mark.parametrize("post_id", ["f" * 32, "invalid"])
def test_cli_dispatch_rejects_missing_or_invalid_post(
    world: World, post_id: str
) -> None:
    w = world
    before = w.message_rows()
    result = run_board(w.env(), "dispatch", "--post", post_id)
    assert result.returncode == 2 and result.stdout == ""
    assert (
        "does not exist" if len(post_id) == 32 else "invalid board post ID"
    ) in result.stderr
    assert w.message_rows() == before


@pytest.mark.parametrize("to", ["all", "unknown", "self", "nonrecipient"])
def test_cli_dispatch_rejects_names_outside_saved_recipients(
    world: World, to: str
) -> None:
    w = world
    root = w.post(notifiers={})
    name = {"self": w.assignments[1].name, "nonrecipient": w.assignments[3].name}.get(
        to, to
    )
    before = w.message_rows()
    result = run_board(w.env(), "dispatch", "--post", root["id"], "--to", name)
    assert result.returncode == 2 and result.stdout == ""
    assert "saved post recipient" in result.stderr and w.message_rows() == before


@pytest.mark.parametrize("crash", [False, True])
def test_ambiguous_acceptance_or_process_interrupt_is_durable_and_not_retried(
    world: World, crash: bool
) -> None:
    w = world
    calls = []

    def uncertain(notification: Any) -> Any:
        calls.append(notification)
        if crash:
            raise KeyboardInterrupt()
        raise RuntimeError("private transport endpoint")

    if crash:
        with pytest.raises(KeyboardInterrupt):
            w.post(notifiers={"codex": uncertain})
        root = w.board.messages.status(w.env())["posts"][0]
    else:
        root = w.post(notifiers={"codex": uncertain})
    recipient = root["recipients"][0]
    assert recipient["state"] == "delivery_failed"
    assert recipient["delivery_reason"] == "notification_uncertain"
    assert recipient["retryable"] is False
    w.board.delivery.dispatch(w.env(), {"codex": uncertain})
    assert len(calls) == 1
    assert "private transport" not in json.dumps(root)


def test_concurrent_dispatch_is_serialized_and_fake_deduplicates(world: World) -> None:
    w = world
    root = w.post(notifiers={})
    assert root["recipients"][0]["delivery_reason"] == "notifier_unconfigured"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda _: w.board.delivery.dispatch(
                    w.env(), w.routes, post_id=root["id"]
                ),
                range(4),
            )
        )
    assert len(w.fake.calls) == len(w.fake.accepted) == 1
    same = w.fake.calls[0]
    assert w.fake(same).accepted
    assert len(w.fake.accepted) == 1


def test_claude_receiver_must_accept_and_notified_does_not_read_or_answer(
    world: World,
) -> None:
    w = world
    w.fake.accepting = False
    root = w.post(to=[w.assignments[3].name])
    recipient = root["recipients"][0]
    assert recipient["delivery_state"] == "delivery_failed"
    assert recipient["delivery_reason"] == "receiver_not_accepting"
    assert "accept" in recipient["delivery_message"]
    assert not recipient["read_at"] and not recipient["response_state"]
    w.fake.accepting = True
    w.board.delivery.dispatch(w.env(), w.routes, post_id=root["id"])
    state = w.own_state(root["id"])["recipients"][0]
    assert state["state"] == "notified"
    assert state["read_at"] is None and state["response_state"] is None
    with w.store.read() as connection:
        assert w.board.activity.summary(
            connection, w.assignments[3].id
        ).awaiting_response


@pytest.mark.parametrize(
    "body,parents,broadcast",
    [
        (
            "Contact help@{other}.example.test; @NotInTheCatalog is a handle.",
            [2],
            False,
        ),
        (
            "The relative file is src/lib/example.py. See https://example.test/docs.",
            [2],
            False,
        ),
        ("Addressed @{target} and #1395.", [2], False),
        ("Addressed @{upper}.", [2], False),
        ("宛先は@{target}です。", [2], False),
        ("追加の宛先@{other}へ質問。", [2, 3], False),
        ("An info / question / request with a / b calculation.", [2], False),
        ("Please ask @{other}.", [2, 3], False),
        ("```python\n# Example @{other}\n```", [2], False),
        ("Inline `@{other}` is documentation, not a destination.", [2], False),
        ("Notify @aLl.", [2, 3], True),
        ("x@{other} _@{other} .@{other} +@{other} src/@{other}", [2], False),
        ("@{other}_handle @{other}-handle @{other}@example.test", [2], False),
    ],
)
def test_prose_mentions_add_destinations_but_code_email_and_unknown_handles_do_not(
    world: World, body: str, parents: list[int], broadcast: bool
) -> None:
    body = body.format(
        target=world.assignments[2].name.lower(),
        upper=world.assignments[2].name.upper(),
        other=world.assignments[3].name,
    )
    post = world.post(body=body)
    assert post["body"] == body
    assert post["broadcast"] is broadcast
    world.assert_delivered(post, parents)


@pytest.mark.parametrize(
    "code",
    [
        "Inline `{mentions}` is an example.",
        "Inline ``{mentions} with `nested` ticks`` is an example.",
        "Inline `{mentions}\nover multiple lines` is an example.",
        "```python\n{mentions}\n```",
        "~~~text\n{mentions}\n~~~",
        "   ```text\n{mentions}\n   ```  \n",
        "````text\n{mentions}\n```\n{mentions}\n````",
        "```text\n{mentions}\n~~~\n{mentions}\n```",
        "```text\n{mentions}",
    ],
)
def test_code_mentions_are_neither_destinations_nor_recipient_errors(
    world: World, code: str
) -> None:
    w = world
    mentions = " ".join(
        f"@{name}"
        for name in (
            w.assignments[1].name,
            w.assignments[3].name,
            w.assignments[6].name,
            w.board.names.load_names()[10],
            "NotInTheCatalog",
            "aLl",
        )
    )
    body = code.format(mentions=mentions)
    post = w.post(body=body)
    assert post["body"] == body
    assert post["broadcast"] is False
    w.assert_delivered(post, [2])
    before = w.message_rows()
    calls = list(w.fake.calls)
    with pytest.raises(w.board.errors.BoardError, match="at least one recipient"):
        w.post(to=[], body=body)
    assert w.message_rows() == before
    assert w.fake.calls == calls


@pytest.mark.parametrize("command", ["post", "reply"])
def test_code_mentions_do_not_hide_adjacent_prose_destinations(
    world: World, command: str
) -> None:
    w = world
    root = w.post() if command == "reply" else None
    body = (
        f"`@{w.assignments[2 if root else 1].name}` is an example.\n"
        f"```text\n@all @{w.assignments[6].name}\n```\n"
        f"宛先は@{w.assignments[3].name}です。"
    )
    if root:
        post = w.board.messages.reply(
            w.env(2), post_id=root["id"], body=body, notifiers=w.routes
        )["post"]
        expected = [1, 3]
    else:
        post = w.post(to=[], body=body)
        expected = [3]
    assert post["broadcast"] is False
    w.assert_delivered(post, expected)


@pytest.mark.parametrize("body", ["An unmatched ` @{name}.", r"Escaped \` @{name} \`."])
def test_unmatched_or_escaped_backticks_leave_mentions_in_prose(
    world: World, body: str
) -> None:
    post = world.post(to=[], body=body.format(name=world.assignments[3].name))
    world.assert_delivered(post, [3])


@pytest.mark.parametrize(
    "value",
    [
        "/private/project/file.txt",
        "See [/private/project](https://example.test).",
        "key=/private/state",
        "path: /etc/config",
        "参照先：/private/state",
        "**/private/state**",
        "参照先は/private/stateです",
        "/",
        "~/private/file",
        "C:\\private\\file",
        "\\\\server\\private",
        "Unknown 01234567-89ab-cdef-0123-456789abcdef",
        "session_id=private-session",
        "A null\x00byte",
        "https://example.test/?file=/private/state",
    ],
)
def test_private_body_and_subject_are_rejected_without_echo(
    world: World, value: str
) -> None:
    w = world
    for field in ("body", "subject"):
        with pytest.raises(w.board.errors.BoardError):
            w.post(**{field: value})
    body_file = w.labels.root / "input.md"
    body_file.write_text(value)
    response = run_board(
        w.env(),
        "post",
        "--to",
        w.assignments[2].name,
        "--kind",
        "info",
        "--subject",
        "Public",
        "--body-file",
        str(body_file),
    )
    assert response.returncode == 2 and response.stdout == ""
    assert response.stderr.startswith("board: ")
    assert value not in response.stderr
    private_output_absent(response.stderr, w.env())
    assert w.board.messages.status(w.env()) == {"posts": []}


def test_known_other_parent_session_and_identifier_output_are_guarded(
    world: World,
) -> None:
    w = world
    for parent in (1, 2, 6):
        env = w.env(parent)
        secret = env.get("CLAUDE_CODE_SESSION_ID", env.get("CODEX_SESSION_ID"))
        with pytest.raises(w.board.errors.BoardError, match="private session"):
            w.post(body=f"Reference {secret}")


@pytest.mark.parametrize("recipient", ["unknown", "retired", "unassigned"])
def test_cli_recipient_rejections_are_exit_2_and_atomic(
    world: World, recipient: str
) -> None:
    w = world
    name = {
        "unknown": "NotInCatalog",
        "retired": w.assignments[6].name,
        "unassigned": w.board.names.load_names()[10],
    }[recipient]
    before = w.message_rows()
    body_file = w.labels.root / "input.md"
    body_file.write_text("A public update.")
    result = run_board(
        w.env(),
        "post",
        "--to",
        name,
        "--kind",
        "info",
        "--subject",
        "Public",
        "--body-file",
        str(body_file),
    )
    assert (
        result.returncode == 2
        and result.stdout == ""
        and result.stderr.startswith("board: ")
    )
    private_output_absent(result.stderr, w.env())
    assert w.message_rows() == before
    assert w.fake.calls == []


@pytest.mark.parametrize("command", ["post", "reply"])
@pytest.mark.parametrize("recipient", ["retired", "unassigned", "unregistered"])
def test_invalid_known_body_mention_rejects_all_destinations_atomically(
    world: World, command: str, recipient: str
) -> None:
    w = world
    root = w.post() if command == "reply" else None
    caller = 2 if root else 1
    name = {
        "retired": w.assignments[6].name,
        "unassigned": w.board.names.load_names()[10],
        "unregistered": w.assignments[4].name,
    }[recipient]
    if recipient == "unregistered":
        parents = json.loads(w.labels.registry.read_text())["parents"]
        del parents["o/r#4"]
        w.labels.set_registry(parents)
    before = w.message_rows()
    calls = list(w.fake.calls)
    body = f"Ask @{w.assignments[3].name} and @{name.swapcase()}."
    with pytest.raises(w.board.errors.BoardError):
        if root:
            w.board.messages.reply(
                w.env(caller), post_id=root["id"], body=body, notifiers=w.routes
            )
        else:
            w.post(to=[], body=body)
    body_file = w.labels.root / "input.md"
    body_file.write_text(body)
    args = [command, "--body-file", str(body_file)]
    args += (
        ["--post", root["id"]] if root else ["--kind", "info", "--subject", "Public"]
    )
    result = run_board(w.env(caller), *args)
    assert result.returncode == 2 and result.stdout == ""
    private_output_absent(result.stderr, w.env(caller))
    assert w.message_rows() == before
    assert w.fake.calls == calls


@pytest.mark.parametrize(
    "body", ["No recipients.", "@NotInTheCatalog", "`@all`", "```text\n@all\n```"]
)
def test_post_without_to_or_prose_board_mentions_has_no_recipient_and_saves_nothing(
    world: World, body: str
) -> None:
    w = world
    before = w.message_rows()
    with pytest.raises(w.board.errors.BoardError, match="at least one recipient"):
        w.post(to=[], body=body)
    body_file = w.labels.root / "input.md"
    body_file.write_text(body)
    result = run_board(
        w.env(),
        "post",
        "--kind",
        "info",
        "--subject",
        "Public",
        "--body-file",
        str(body_file),
    )
    assert result.returncode == 2 and result.stdout == ""
    assert "at least one recipient" in result.stderr
    assert w.message_rows() == before
    assert w.fake.calls == []


@pytest.mark.parametrize(
    "identity",
    [
        "missing",
        "unregistered",
        "codex-child",
        "claude-child",
        "nested",
        "released",
        "ended",
        "spoof",
    ],
)
def test_cli_identity_rejects_every_message_mutation(
    world: World, identity: str
) -> None:
    w = world
    root = w.post()
    body_file = w.labels.root / "input.md"
    body_file.write_text("A public update.")
    env = w.env()
    extra = []
    if identity == "missing":
        env.pop("CLAUDE_CODE_SESSION_ID")
    elif identity == "unregistered":
        env["CLAUDE_CODE_SESSION_ID"] = "unregistered-caller"
    elif identity == "codex-child":
        env = w.env(2)
        env["CODEX_THREAD_ID"] = "nested-thread"
    elif identity == "claude-child":
        env["CLAUDE_CODE_AGENT_ID"] = "nested-agent"
    elif identity == "nested":
        env["CLAUDE_CODE_PARENT_SESSION_ID"] = "nested-parent"
    elif identity in {"released", "ended"}:
        env = w.env(4 if identity == "released" else 6)
    elif identity == "spoof":
        extra = ["--sender", w.assignments[1].name]
    for args in [
        [
            "post",
            "--to",
            w.assignments[2].name,
            "--kind",
            "info",
            "--subject",
            "Public",
            "--body-file",
            str(body_file),
        ],
        ["reply", "--post", root["id"], "--body-file", str(body_file)],
        ["read", root["id"]],
        ["ack", "--post", root["id"], "--state", "done"],
    ]:
        result = run_board(env, *args, *extra)
        assert (
            result.returncode == 2
            and result.stdout == ""
            and result.stderr.startswith("board: ")
        )
        private_output_absent(result.stderr, env)
    assert len(w.board.messages.status(w.env())["posts"]) == 1
    assert w.own_state(root["id"])["recipients"][0]["state"] == "notified"


def test_unrelated_parent_cannot_read_reply_ack_or_access_thread(world: World) -> None:
    w = world
    root = w.post()
    for action in (
        lambda: w.board.messages.read(w.env(3), root["id"]),
        lambda: w.board.messages.reply(w.env(3), post_id=root["id"], body="Unrelated"),
        lambda: w.board.messages.ack(w.env(3), post_id=root["id"], state="done"),
        lambda: w.board.messages.thread(w.env(3), root["id"]),
        lambda: w.board.messages.ack(w.env(), post_id=root["id"], state="done"),
    ):
        with pytest.raises(w.board.errors.BoardError):
            action()
    assert w.own_state(root["id"])["recipients"][0]["state"] == "notified"


def test_cli_all_commands_and_safe_errors(world: World) -> None:
    w = world
    body_file = w.labels.root / "input.md"
    body_file.write_text("Public CLI message.")
    created = run_board(
        w.env(),
        "post",
        "--to",
        w.assignments[2].name.lower(),
        "--kind",
        "request",
        "--subject",
        "CLI",
        "--body-file",
        str(body_file),
    )
    assert created.returncode == 0, created.stderr
    root = json.loads(created.stdout)["post"]
    assert root["recipients"][0]["delivery_reason"] == "route_unavailable"
    for env, args in [
        (w.env(2), ["inbox", "--unread"]),
        (w.env(2), ["read", root["id"]]),
        (w.env(2), ["ack", "--post", root["id"], "--state", "accepted"]),
        (w.env(2), ["reply", "--post", root["id"], "--body-file", str(body_file)]),
        (w.env(), ["thread", root["id"]]),
        (w.env(), ["status"]),
    ]:
        result = run_board(env, *args)
        assert result.returncode == 0, result.stderr
        json.loads(result.stdout)
        private_output_absent(result.stdout + result.stderr, env)
    for args in [
        ["read", "/private/invalid"],
        ["read", "f" * 32],
        ["post", "--to", w.assignments[2].name, "--kind", "/private/kind"],
        ["ack", "--post", root["id"], "--state", "/private/state"],
        [
            "reply",
            "--post",
            root["id"],
            "--body-file",
            str(w.labels.root / "missing.md"),
        ],
    ]:
        result = run_board(w.env(), *args)
        assert result.returncode == 2 and result.stdout == ""
        assert "/private/" not in result.stderr
        private_output_absent(result.stderr, w.env())


def test_read_failed_delivery_remains_distinct_and_skips_unnecessary_bell(
    world: World,
) -> None:
    w = world
    root = w.post(notifiers={})
    read = w.board.messages.read(w.env(2), root["id"])["post"]["recipients"][0]
    assert read["state"] == "read" and read["delivery_state"] == "delivery_failed"
    w.board.delivery.dispatch(w.env(), w.routes)
    assert w.fake.calls == []
    with w.store.read() as connection:
        assert w.board.activity.summary(
            connection, w.assignments[2].id
        ).awaiting_response


def test_real_unread_and_response_waiting_block_reuse_then_history_stays_fixed(
    world: World,
) -> None:
    w = world
    root = w.post()
    target = w.assignments[2]
    with w.store.transaction() as connection:
        w.board.names.retire_name(connection, target.repo, target.parent, now=NOW)
        with pytest.raises(w.board.errors.BoardError, match="exhausted"):
            w.board.names.ensure_name(
                connection,
                "o/r",
                7,
                None,
                now=NOW + timedelta(days=30),
                catalog=[target.name],
            )
    # Resume this exact assignment, explicitly read, then still block on its question.
    with w.store.transaction() as connection:
        resumed = w.board.names.ensure_name(connection, "o/r", 2, None, now=NOW)
        assert resumed.id == target.id
    w.board.messages.read(w.env(2), root["id"])
    with w.store.transaction() as connection:
        w.board.names.retire_name(connection, "o/r", 2, now=NOW)
        with pytest.raises(w.board.errors.BoardError, match="exhausted"):
            w.board.names.ensure_name(
                connection,
                "o/r",
                7,
                None,
                now=NOW + timedelta(days=30),
                catalog=[target.name],
            )
        w.board.names.ensure_name(connection, "o/r", 2, None, now=NOW)
    w.board.messages.reply(
        w.env(2), post_id=root["id"], body="Answered.", notifiers=w.routes
    )
    with w.store.transaction() as connection:
        w.board.names.retire_name(connection, "o/r", 2, now=NOW)
        replacement = w.board.names.ensure_name(
            connection,
            "o/r",
            7,
            None,
            now=NOW + timedelta(days=30),
            catalog=[target.name],
        )
        assert replacement.id != target.id
        assert w.board.activity.summary(connection, replacement.id).unread_count == 0
    recipient = w.own_state(root["id"])["recipients"][0]
    assert (recipient["assignment_id"], recipient["parent"], recipient["name"]) == (
        target.id,
        2,
        target.name,
    )


def test_reply_never_redirects_to_reused_original_author_name(world: World) -> None:
    w = world
    root = w.post()
    author = w.assignments[1]
    with w.store.transaction() as connection:
        w.board.names.retire_name(connection, "o/r", 1, now=NOW)
        replacement = w.board.names.ensure_name(
            connection,
            "o/r",
            7,
            None,
            now=NOW + timedelta(days=30),
            catalog=[author.name],
        )
        assert replacement.id != author.id
    with pytest.raises(w.board.errors.BoardError, match="retired"):
        w.board.messages.reply(
            w.env(2),
            post_id=root["id"],
            body="Must not be redirected.",
            notifiers=w.routes,
        )
    assert len(w.fake.calls) == 1


def test_roster_gets_real_unread_counts_and_last_post_time(world: World) -> None:
    w = world
    root = w.post()
    roster = {row["parent"]: row for row in w.board.service.agents(w.env())["agents"]}
    assert roster[1]["last_activity"] == root["created_at"]
    assert roster[2]["last_activity"] is None and roster[2]["unread_count"] == 1
    assert roster[1]["identifier"] is None
    w.board.messages.read(w.env(2), root["id"])
    roster = {row["parent"]: row for row in w.board.service.agents(w.env())["agents"]}
    assert roster[2]["unread_count"] == 0


def test_notifier_observes_committed_post_without_sqlite_writer_lock(
    world: World,
) -> None:
    w = world
    seen = []

    def route(notification: Any) -> Any:
        # A callback may read the board and briefly acquire a SQLite writer;
        # dispatch must have committed both the post and its uncertainty marker.
        with w.store.transaction() as connection:
            assert connection.execute(
                "SELECT body FROM posts WHERE id=?", (notification.post_id,)
            ).fetchone()[0]
            state = connection.execute(
                "SELECT state, reason FROM deliveries WHERE post_id=?",
                (notification.post_id,),
            ).fetchone()
            assert state == ("delivery_failed", "notification_uncertain")
        seen.append(notification)
        return w.board.delivery.NotificationResult(True)

    root = w.post(notifiers={"codex": route})
    assert len(seen) == 1
    assert root["recipients"][0]["state"] == "notified"


def test_recipient_ending_or_leaving_before_retry_is_rechecked(world: World) -> None:
    w = world
    root = w.post(notifiers={})
    w.change(2, status="handoff_pending")
    w.board.delivery.dispatch(w.env(), w.routes, post_id=root["id"])
    assert w.own_state(root["id"])["recipients"][0]["state"] == "held"
    assert w.fake.calls == []
    w.change(2, status="ended")  # Even if name synchronization has not run yet.
    w.board.delivery.dispatch(w.env(), w.routes, post_id=root["id"])
    assert (
        w.own_state(root["id"])["recipients"][0]["delivery_reason"]
        == "recipient_retired"
    )
    with pytest.raises(w.board.errors.BoardError, match="retired"):
        w.post()
    assert w.fake.calls == []


def test_v1_migration_preserves_name_lifetimes_and_nullable_identifier(
    board: SimpleNamespace, tmp_path: Path
) -> None:
    store = board.storage.Store(tmp_path / "board")
    store.directory.mkdir()
    with sqlite3.connect(store.path) as connection:
        board.storage.MIGRATIONS[0](connection)
        connection.execute("PRAGMA user_version=1")
        connection.execute(
            "INSERT INTO name_assignments VALUES(?,?,?,?,?,?,?)",
            (42, "Aaron", "o/r", 123, None, board.names.timestamp(NOW), None),
        )
        before = connection.execute("SELECT * FROM name_assignments").fetchall()
    with pytest.raises(board.errors.BoardError, match="migration"), store.read():
        pass
    with store.transaction() as connection:
        assert (
            connection.execute("PRAGMA user_version").fetchone()[0]
            == board.storage.SCHEMA_VERSION
        )
        assert connection.execute("SELECT * FROM name_assignments").fetchall() == before
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT COUNT(*) FROM posts").fetchone()[0] == 0
        columns = {
            row[1]: row for row in connection.execute("PRAGMA table_info(posts)")
        }
        assert columns["broadcast"][2:4] == ("INTEGER", 1)


@pytest.mark.parametrize("broadcast", [None, -1, 2])
def test_posts_reject_non_boolean_broadcast(
    world: World, broadcast: int | None
) -> None:
    w = world
    root = w.post()
    with pytest.raises(w.board.errors.BoardError), w.store.transaction() as connection:
        connection.execute(
            """INSERT INTO posts
               SELECT ?, created_at, sender_assignment_id, sender_name, sender_repo,
                      sender_parent, kind, ?, subject, body, in_reply_to, thread_id
               FROM posts WHERE id=?""",
            ("f" * 32, broadcast, root["id"]),
        )
    with w.store.read() as connection:
        assert connection.execute("SELECT COUNT(*) FROM posts").fetchone()[0] == 1


def test_v2_schema_without_broadcast_is_rejected(world: World) -> None:
    w = world
    with sqlite3.connect(w.store.path) as connection:
        connection.execute("ALTER TABLE posts DROP COLUMN broadcast")
        connection.execute("DROP TABLE name_renames")
        connection.execute("PRAGMA user_version=2")
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
    with pytest.raises(w.board.errors.BoardError, match="migration"), w.store.read():
        pass
    with (
        pytest.raises(w.board.errors.BoardError, match="inconsistent"),
        w.store.transaction(),
    ):
        pass
    with sqlite3.connect(w.store.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='name_renames'"
            ).fetchall()
            == []
        )


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE posts SET body='overwrite'",
        "UPDATE posts SET sender_name='Barry'",
        "DELETE FROM posts",
        "DELETE FROM recipients",
        "DELETE FROM deliveries",
        "UPDATE recipients SET parent=999",
        "UPDATE recipients SET name='Barry'",
        "UPDATE deliveries SET idempotency_key='replacement'",
        "INSERT OR REPLACE INTO posts SELECT * FROM posts",
        "INSERT OR REPLACE INTO recipients SELECT * FROM recipients",
        "INSERT OR REPLACE INTO deliveries SELECT * FROM deliveries",
        "UPDATE name_renames SET old_name='Barry'",
        "DELETE FROM name_renames",
        "INSERT OR REPLACE INTO name_renames SELECT * FROM name_renames",
    ],
)
def test_history_cannot_be_edited_deleted_or_retargeted(
    world: World, statement: str
) -> None:
    w = world
    root = w.post()
    importlib.import_module("board.renaming").rename(w.env(), parent=1)
    with pytest.raises(w.board.errors.BoardError), w.store.transaction() as connection:
        connection.execute(statement)
    assert w.board.messages.read(w.env(), root["id"])["post"]["body"] == root["body"]
