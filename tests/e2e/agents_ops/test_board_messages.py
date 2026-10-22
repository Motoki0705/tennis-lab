"""Messages through private SQLite, temporary registry and fake notifier routes."""

from __future__ import annotations

import importlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
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


@pytest.fixture
def world(board: SimpleNamespace, labels: Labels) -> World:
    for name in ("messages", "delivery", "records"):
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
    with pytest.raises(w.board.errors.BoardError, match="no live board name"):
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
    "body,allowed",
    [
        ("Contact help@Adam.example.test; @NotInTheCatalog is a code handle.", True),
        (
            "The relative file is src/lib/example.py. See https://example.test/docs.",
            True,
        ),
        ("Addressed @adam and #1395.", True),
        ("Addressed @ADAM.", True),
        ("宛先は@Adamです。", True),
        ("宛先外の@Adrianへ質問。", False),
        ("An info / question / request with a / b calculation.", True),
        ("Please ask @Adrian.", False),
        ("```python\n# Ask @Adrian\n```", False),
        ("Inline `@Adrian` is checked too.", False),
        ("Do not imply @all delivery.", False),
    ],
)
def test_known_name_mentions_include_code_but_ignore_email_and_unknown_handles(
    world: World, body: str, allowed: bool
) -> None:
    if allowed:
        assert world.post(body=body)["body"] == body
    else:
        with pytest.raises(
            world.board.errors.BoardError, match="outside the recipients"
        ):
            world.post(body=body)
        assert world.board.messages.status(world.env()) == {"posts": []}


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


@pytest.mark.parametrize(
    "recipient", ["unknown", "retired", "unassigned", "outside-body"]
)
def test_cli_recipient_rejections_are_exit_2_and_atomic(
    world: World, recipient: str
) -> None:
    w = world
    name = {
        "unknown": "NotInCatalog",
        "retired": w.assignments[6].name,
        "unassigned": w.board.names.load_names()[10],
        "outside-body": w.assignments[2].name,
    }[recipient]
    body_file = w.labels.root / "input.md"
    body_file.write_text(
        f"Ask @{w.assignments[3].name}"
        if recipient == "outside-body"
        else "A public update."
    )
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
    assert w.board.messages.status(w.env()) == {"posts": []}
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
    assert root["recipients"][0]["delivery_reason"] == "notifier_unconfigured"
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
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("SELECT * FROM name_assignments").fetchall() == before
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT COUNT(*) FROM posts").fetchone()[0] == 0


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE posts SET body='overwrite'",
        "DELETE FROM posts",
        "DELETE FROM recipients",
        "DELETE FROM deliveries",
        "UPDATE recipients SET parent=999",
        "UPDATE deliveries SET idempotency_key='replacement'",
        "INSERT OR REPLACE INTO posts SELECT * FROM posts",
        "INSERT OR REPLACE INTO recipients SELECT * FROM recipients",
        "INSERT OR REPLACE INTO deliveries SELECT * FROM deliveries",
    ],
)
def test_history_cannot_be_edited_deleted_or_retargeted(
    world: World, statement: str
) -> None:
    w = world
    root = w.post()
    with pytest.raises(w.board.errors.BoardError), w.store.transaction() as connection:
        connection.execute(statement)
    assert w.board.messages.read(w.env(), root["id"])["post"]["body"] == root["body"]
