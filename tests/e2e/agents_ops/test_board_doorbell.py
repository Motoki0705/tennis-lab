"""Bells through disposable AF_UNIX servers and fake CLI executables only."""

from __future__ import annotations

import importlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.e2e.agents_ops.test_board import OPS, private_output_absent, run_board
from tests.e2e.agents_ops.test_board_messages import World
from tests.e2e.agents_ops.test_board_messages import board as board
from tests.e2e.agents_ops.test_board_messages import labels as labels
from tests.e2e.agents_ops.test_board_messages import world as world
from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env

THREAD = "11111111-2222-4333-8444-555555555555"

FAKE_CODEX = r"""
import json, os, sys, time
from pathlib import Path
config = json.loads((Path(os.environ['CODEX_HOME']) / 'fake.json').read_text())
args = sys.argv[1:]
assert args[:4] == ['--disable', 'daemon_auto_start', 'queue', '--thread']
assert args[4] == config['thread'] and args[5] == '--message'
states = list(Path(config['private']).glob('*/task.json'))
phases = [json.loads(p.read_text())['deliveries'][-1]['phase'] for p in states]
with Path(config['calls']).open('a') as stream:
    stream.write(json.dumps({'args': args, 'cwd': os.getcwd(), 'phases': phases,
                            'home': os.environ['CODEX_HOME'],
                            'sqlite': os.environ['CODEX_SQLITE_HOME']}) + '\n')
mode = config.get('mode', 'accepted')
if mode == 'completed':
    with Path(config['rollout']).open('a') as stream:
        for event in [dict(type='task_started', turn_id='bell-turn'),
                      dict(type='user_message', message=args[6]),
                      dict(type='task_complete', turn_id='bell-turn')]:
            stream.write(json.dumps({'type':'event_msg','payload':event,
                                     'timestamp':'2026-10-23T00:00:00Z'})+'\n')
if mode in ('accepted', 'completed', 'ack_timeout'):
    print('Queued message fake-item for thread ' + config['thread'] + '.', flush=True)
if mode in ('timeout', 'ack_timeout'): time.sleep(10)
if mode == 'rejected':
    print('failure with /private/path credential-secret', file=sys.stderr)
    sys.exit(1)
if mode == 'uncertain': print('unrecognized receipt')
"""


@contextmanager
def inbox_socket() -> Iterator[
    tuple[Path, list[list[dict[str, Any]]], threading.Condition]
]:
    # Linux AF_UNIX has a short path limit; pytest's parameterized directory
    # names can exceed it. This directory contains only a disposable fake.
    with tempfile.TemporaryDirectory(prefix="board-bell-") as short:
        path = Path(short) / "inbox.sock"
        received: list[list[dict[str, Any]]] = []
        condition = threading.Condition()
        stopping = threading.Event()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(path))
            path.chmod(0o600)
            server.listen(8)
            server.settimeout(0.1)

            def receive() -> None:
                while not stopping.is_set():
                    try:
                        peer, _ = server.accept()
                    except TimeoutError:
                        continue
                    with peer, peer.makefile("rb") as stream:
                        lines = [json.loads(line) for line in stream]
                    with condition:
                        received.append(lines)
                        condition.notify_all()

            worker = threading.Thread(target=receive, daemon=True)
            worker.start()
            try:
                yield path, received, condition
            finally:
                stopping.set()
                worker.join(timeout=3)
                assert not worker.is_alive()


def wait_received(
    received: list[Any], condition: threading.Condition, count: int = 1
) -> None:
    with condition:
        assert condition.wait_for(lambda: len(received) >= count, timeout=5)


@dataclass
class Bells:
    w: World
    modules: SimpleNamespace
    home: Path

    def claude(
        self,
        parent: int,
        path: Path,
        *,
        accept: Any = "accept",
        source: str = "startup",
    ) -> dict[str, str]:
        config = self.home / f"claude-{parent}"
        config.mkdir(exist_ok=True)
        (config / "settings.json").write_text(
            json.dumps({"crossSessionInbound": accept})
        )
        env = {
            **self.w.env(parent),
            "CLAUDE_CONFIG_DIR": str(config),
            "CLAUDE_CODE_MESSAGING_SOCKET": str(path),
            "CLAUDE_CODE_MESSAGING_TOKEN": "private-fixture-credential",
        }
        self.modules.receivers.capture_claude(
            env, source, self.modules.receivers.process(os.getpid())
        )
        return env

    def codex(self, mode: str = "accepted") -> tuple[dict[str, str], Path, Path]:
        self.w.change(2, session="codex:" + THREAD)
        home = self.home / "codex"
        sessions = home / "sessions"
        sessions.mkdir(parents=True, exist_ok=True)
        rollout = sessions / f"rollout-fixture-{THREAD}.jsonl"
        cwd = self.home / "original cwd"
        cwd.mkdir(exist_ok=True)
        rollout.write_text(
            json.dumps(
                {"type": "session_meta", "payload": {"id": THREAD, "cwd": str(cwd)}}
            )
            + "\n"
        )
        executable = self.w.labels.root / "bin/codex"
        executable.write_text(f"#!{sys.executable}\n" + FAKE_CODEX)
        executable.chmod(0o700)
        calls = home / "calls.jsonl"
        config = {
            "thread": THREAD,
            "mode": mode,
            "calls": str(calls),
            "rollout": str(rollout),
            "private": str(self.modules.receivers.root(self.w.env()) / "queue"),
        }
        (home / "fake.json").write_text(json.dumps(config))
        env = {
            **self.w.env(2),
            "CODEX_HOME": str(home),
            "CODEX_SQLITE_HOME": str(home / "db"),
        }
        assert self.modules.receivers.bind_receiver(env) == {"status": "bound"}
        return env, rollout, calls

    def post(self, parent: int = 3) -> dict[str, Any]:
        return self.w.post(
            to=[self.w.assignments[parent].name],
            notifiers=self.modules.notifiers.configured(self.w.env()),
        )

    def dispatch(self, post: dict[str, Any]) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]],
            self.w.board.delivery.dispatch(
                self.w.env(),
                self.modules.notifiers.configured(self.w.env()),
                post_id=post["id"],
            ),
        )


@pytest.fixture
def bells(world: World, tmp_path: Path) -> Bells:
    # These tests bind notification routes separately from Herdr display owners.
    # Start with unconfigured displays; confirm_parent_display opts in explicitly.
    owners = world.labels.registry.parent.parent / "herdr-labels/owners"
    parents = json.loads(world.labels.registry.read_text())["parents"]
    runtime = world.labels.read()
    for parent in (1, 2, 3):
        identity = parents[f"o/r#{parent}"]["session"]
        (owners / f"{world.board.service.owner_token(identity)}.json").unlink()
        del runtime["panes"][f"w1:board-{parent}"]
    world.labels.write(runtime)
    modules = SimpleNamespace(
        **{
            name: importlib.import_module("board." + name)
            for name in ("receivers", "notifiers", "display")
        }
    )
    return Bells(world, modules, tmp_path)


def test_claude_socket_cli_post_and_reply_stop_at_notified(bells: Bells) -> None:
    b = bells
    with inbox_socket() as (path, received, condition):
        env = b.claude(3, path)
        body = b.home / "body.md"
        body.write_text("Private body stays on the board.")
        result = run_board(
            b.w.env(),
            "post",
            "--to",
            b.w.assignments[3].name,
            "--kind",
            "question",
            "--subject",
            "Dependency",
            "--body-file",
            str(body),
        )
        assert result.returncode == 0, result.stderr
        post = json.loads(result.stdout)["post"]
        wait_received(received, condition)
        auth, message = received[0]
        assert auth == {"type": "auth", "token": env["CLAUDE_CODE_MESSAGING_TOKEN"]}
        assert message["type"] == "user"
        assert set(message) == {"type", "message", "from", "uuid", "msg_id"}
        assert "board read " + post["id"] in message["message"]["content"]
        assert post["body"] not in message["message"]["content"]
        assert post["subject"] not in message["message"]["content"]
        recipient = post["recipients"][0]
        assert (
            recipient["delivery_state"] == "notified" and recipient["read_at"] is None
        )
        b.dispatch(post)
        assert len(received) == 1
        private_output_absent(result.stdout + result.stderr, env)
        assert env["CLAUDE_CODE_MESSAGING_TOKEN"] not in result.stdout + result.stderr
        # Reply uses the actual Claude adapter too; capture another receiver.
        b.claude(1, path)
        reply = run_board(env, "reply", "--post", post["id"], "--body-file", str(body))
        assert reply.returncode == 0, reply.stderr
        wait_received(received, condition, 2)
        assert (
            json.loads(reply.stdout)["post"]["recipients"][0]["delivery_state"]
            == "notified"
        )


@pytest.mark.parametrize("setting", [None, "hold", "refuse", True])
def test_claude_user_setting_required_without_alternative_route(
    bells: Bells, setting: Any
) -> None:
    b = bells
    with inbox_socket() as (path, received, _):
        env = b.claude(3, path, accept=setting)
        project = b.home / ".claude"
        project.mkdir()
        (project / "settings.json").write_text('{"crossSessionInbound":"accept"}')
        original = (Path(env["CLAUDE_CONFIG_DIR"]) / "settings.json").read_bytes()
        post = b.post()
        recipient = post["recipients"][0]
        assert recipient["delivery_state"] == "delivery_failed"
        assert recipient["delivery_reason"] == "receiver_not_accepting"
        assert recipient["retryable"]
        assert received == []
        assert (
            Path(env["CLAUDE_CONFIG_DIR"]) / "settings.json"
        ).read_bytes() == original
        assert (
            b.w.own_state(post["id"])["recipients"][0]["delivery_reason"]
            == "receiver_not_accepting"
        )
        assert "配送失敗:1" in b.modules.display.context(env)


def test_claude_connection_failure_is_known_unsent_and_can_retry(bells: Bells) -> None:
    b = bells
    with inbox_socket() as (path, received, condition):
        b.claude(3, path)
        hidden = path.with_name("temporarily-hidden.sock")
        path.rename(hidden)
        post = b.post()
        assert post["recipients"][0]["delivery_reason"] == "route_unavailable"
        assert post["recipients"][0]["retryable"]
        hidden.rename(path)
        result = run_board(
            b.w.env(),
            "dispatch",
            "--post",
            post["id"],
            "--to",
            b.w.assignments[3].name.lower(),
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["status"] == "notified"
        assert (
            data["recipients"][0]["attempted"]
            and data["recipients"][0]["attempts"] == 2
        )
        wait_received(received, condition)
        again = run_board(b.w.env(), "dispatch", "--post", post["id"])
        assert again.returncode == 1, again.stderr
        data = json.loads(again.stdout)
        assert data["status"] == "no_targets"
        assert data["recipients"][0]["skip_reason"] == "already_notified"
        assert len(received) == 1
        private_output_absent(
            result.stdout + result.stderr + again.stdout + again.stderr, b.w.env()
        )


def test_cli_dispatch_reports_partial_failure(bells: Bells) -> None:
    b = bells
    with inbox_socket() as (path, received, condition):
        b.claude(3, path)
        post = b.w.post(
            to=[b.w.assignments[2].name, b.w.assignments[3].name], notifiers={}
        )
        result = run_board(b.w.env(), "dispatch", "--post", post["id"])
        assert result.returncode == 1, result.stderr
        data = json.loads(result.stdout)
        assert data["status"] == "failed"
        failed, notified = data["recipients"]
        assert (
            failed["outcome"] == "failed"
            and failed["delivery_reason"] == "route_unavailable"
        )
        assert notified["outcome"] == "notified" and notified["attempted"]
        wait_received(received, condition)
        assert len(received) == 1
        private_output_absent(result.stdout + result.stderr, b.w.env())


def test_claude_refused_connection_does_not_call_codex(
    bells: Bells, monkeypatch: pytest.MonkeyPatch
) -> None:
    b = bells
    calls = []
    monkeypatch.setattr(b.modules.notifiers, "codex", lambda *args: calls.append(args))
    with tempfile.TemporaryDirectory(prefix="board-bell-") as short:
        path = Path(short) / "closed.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as closed:
            closed.bind(str(path))
        b.claude(3, path)
        post = b.post()
        assert post["recipients"][0]["delivery_reason"] == "route_unavailable"
        assert post["recipients"][0]["retryable"]
        assert calls == []


@pytest.mark.parametrize("stage", ["connect", "send"])
def test_claude_timeouts_do_not_resend_ambiguous_writes(
    bells: Bells, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    b = bells
    with inbox_socket() as (path, received, condition):
        b.claude(3, path)
        real_socket = socket.socket

        class TimedOutSocket:
            def __enter__(self) -> TimedOutSocket:
                self.client = real_socket(socket.AF_UNIX, socket.SOCK_STREAM)
                return self

            def __exit__(self, *_args: Any) -> None:
                self.client.close()

            def settimeout(self, seconds: float) -> None:
                assert seconds > 0

            def connect(self, address: str) -> None:
                if stage == "connect":
                    raise TimeoutError("private socket credential")
                self.client.connect(address)

            def sendall(self, data: bytes) -> None:
                self.client.sendall(data)
                raise TimeoutError("peer might have accepted")

        with monkeypatch.context() as patch:
            patch.setattr(
                b.modules.notifiers,
                "socket",
                SimpleNamespace(
                    AF_UNIX=socket.AF_UNIX,
                    SOCK_STREAM=socket.SOCK_STREAM,
                    socket=lambda *_args: TimedOutSocket(),
                ),
            )
            post = b.post()
        state = post["recipients"][0]
        if stage == "send":
            wait_received(received, condition)
            assert (
                state["delivery_reason"] == "notification_uncertain"
                and not state["retryable"]
            )
            result = run_board(b.w.env(), "dispatch", "--post", post["id"])
            assert result.returncode == 1, result.stderr
            assert (
                json.loads(result.stdout)["recipients"][0]["skip_reason"]
                == "notification_uncertain"
            )
            assert len(received) == 1
        else:
            assert (
                state["delivery_reason"] == "route_unavailable" and state["retryable"]
            )
            assert received == []


@pytest.mark.parametrize("same_identity", [False, True])
def test_clear_fences_same_socket_token_and_conversation(
    bells: Bells, same_identity: bool
) -> None:
    b = bells
    with inbox_socket() as (path, received, _):
        env = b.claude(3, path)
        next_env = (
            env
            if same_identity
            else {**env, "CLAUDE_CODE_SESSION_ID": "different-conversation"}
        )
        b.modules.receivers.capture_claude(
            next_env, "clear", b.modules.receivers.process(os.getpid())
        )
        post = b.post()
        assert post["recipients"][0]["delivery_reason"] == "route_unavailable"
        assert not received
        if same_identity:
            assert b.modules.receivers.bind_receiver(env)["status"] == "bound"
            assert b.dispatch(post)[0]["delivery_state"] == "notified"


def test_resume_replaces_epoch_and_receiver_permissions_are_private(
    bells: Bells,
) -> None:
    b = bells
    with inbox_socket() as (path, _, _):
        env = b.claude(3, path)
        root = b.modules.receivers.root(env)
        owner = importlib.import_module("herdr_labels.coordination").owner_token(
            "claude:board-owner-3"
        )
        record_path = root / f"{owner}.json"
        before = b.modules.receivers.read(record_path)
        b.modules.receivers.capture_claude(
            env, "resume", b.modules.receivers.process(os.getpid())
        )
        after = b.modules.receivers.read(record_path)
        assert before["epoch"] != after["epoch"]
        assert before["binding"] == after["binding"]
        assert root.stat().st_mode & 0o777 == 0o700
        for item in root.rglob("*"):
            assert item.stat().st_mode & 0o777 == (0o700 if item.is_dir() else 0o600)


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("accepted", "notified"),
        ("completed", "notified"),
        ("uncertain", "delivery_failed"),
        ("rejected", "delivery_failed"),
        ("timeout", "delivery_failed"),
        ("ack_timeout", "notified"),
    ],
)
def test_codex_fake_runtime_and_durable_no_duplicate(
    bells: Bells, monkeypatch: pytest.MonkeyPatch, mode: str, expected: str
) -> None:
    b = bells
    alternative = []
    monkeypatch.setattr(
        b.modules.notifiers, "claude", lambda *args: alternative.append(args)
    )
    receiver, _, calls = b.codex(mode)
    if "timeout" in mode:
        monkeypatch.setattr(b.modules.notifiers, "TRANSPORT_TIMEOUT", 0.4)
    post = b.post(2)
    recipient = post["recipients"][0]
    assert recipient["delivery_state"] == expected
    assert recipient["read_at"] is None
    b.dispatch(post)
    b.dispatch(post)
    sent = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(sent) == 1
    assert sent[0]["phases"] == ["uncertain"]
    assert sent[0]["home"] == receiver["CODEX_HOME"]
    assert sent[0]["sqlite"] == receiver["CODEX_SQLITE_HOME"]
    assert sent[0]["cwd"] == str(b.home / "original cwd")
    assert sent[0]["args"][4] == THREAD
    if expected == "delivery_failed":
        assert (
            recipient["delivery_reason"] == "notification_uncertain"
            and not recipient["retryable"]
        )
    # Even a direct adapter repeat uses the common receipt instead of queuing.
    with b.w.store.read() as connection:
        key = connection.execute(
            "SELECT idempotency_key FROM deliveries WHERE post_id=?", (post["id"],)
        ).fetchone()[0]
    notification = b.w.board.delivery.Notification(
        post["id"],
        b.w.assignments[2].id,
        b.w.assignments[2].name,
        "o/r",
        2,
        "codex",
        key,
        b.w.assignments[1].name,
        "o/r",
        1,
        "question",
    )
    b.modules.notifiers.codex(b.w.env(), notification)
    assert len(calls.read_text().splitlines()) == 1
    assert alternative == []


def test_codex_phases_are_separate_from_board_read_and_ack(bells: Bells) -> None:
    b = bells
    receiver, rollout, _ = b.codex()
    post = b.post(2)
    receipts = b.modules.notifiers.receipts(b.w.env())["receipts"]
    assert receipts[0]["phase"] == "queued"
    task = next((b.modules.receivers.root(receiver) / "queue").glob("*/task.json"))
    marker = json.loads(task.read_text())["deliveries"][-1]["marker"]
    with rollout.open("a") as stream:
        for event in [
            {"type": "task_started", "turn_id": "matched-turn"},
            {"type": "user_message", "message": marker},
        ]:
            stream.write(json.dumps({"type": "event_msg", "payload": event}) + "\n")
    assert b.modules.notifiers.receipts(receiver)["receipts"][0]["phase"] == "started"
    with rollout.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "type": "event_msg",
                    "timestamp": "2026-10-23T00:00:00Z",
                    "payload": {"type": "task_complete", "turn_id": "matched-turn"},
                }
            )
            + "\n"
        )
    result = run_board(receiver, "receipts")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["receipts"][0]["phase"] == "completed"
    assert b.w.own_state(post["id"])["recipients"][0]["state"] == "notified"
    private_output_absent(result.stdout + result.stderr, receiver)


def test_codex_ambiguous_rollout_and_stale_parent_generation_fail_closed(
    bells: Bells,
) -> None:
    b = bells
    receiver, rollout, calls = b.codex()
    duplicate = rollout.with_name("other-" + rollout.name)
    duplicate.write_bytes(rollout.read_bytes())
    with pytest.raises(b.w.board.errors.BoardError, match="ambiguous"):
        b.modules.receivers.bind_receiver(receiver)
    b.w.change(2, generation=2)
    post = b.post(2)
    assert post["recipients"][0]["delivery_reason"] == "route_unavailable"
    assert not calls.exists()


def test_codex_receiver_binding_must_match_current_parent_generation(
    bells: Bells,
) -> None:
    b = bells
    receiver, _, calls = b.codex()
    b.w.change(2, generation=2)
    post = b.post(2)
    assert post["recipients"][0]["delivery_reason"] == "route_unavailable"
    assert not calls.exists()
    b.modules.receivers.bind_receiver(receiver)
    assert b.dispatch(post)[0]["delivery_state"] == "notified"
    assert len(calls.read_text().splitlines()) == 1


def hook(
    env: dict[str, str], event: str = "SessionStart", **fields: Any
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(OPS / "board/claude_hook.py")],
        env=env,
        input=json.dumps(
            {
                "hook_event_name": event,
                "session_id": env["CLAUDE_CODE_SESSION_ID"],
                **fields,
            }
        ),
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )


def test_hooks_capture_real_payload_and_report_failures_without_private_values(
    bells: Bells,
) -> None:
    b = bells
    with inbox_socket() as (path, _, _):
        env = b.claude(3, path, accept="hold")
        b.post()
        before = b.w.labels.registry.read_bytes()
        for event in ("SessionStart", "UserPromptSubmit"):
            result = hook(env, event, source="resume")
            assert result.returncode == 0, result.stderr
            output = json.loads(result.stdout)["hookSpecificOutput"]
            assert output["hookEventName"] == event
            assert (
                "未読:1" in output["additionalContext"]
                and "配送失敗:1" in output["additionalContext"]
            )
            for private in (
                env["CLAUDE_CODE_MESSAGING_TOKEN"],
                str(path),
                env["CLAUDE_CODE_SESSION_ID"],
            ):
                assert private not in result.stdout + result.stderr
        assert b.w.labels.registry.read_bytes() == before


@pytest.mark.parametrize(
    "extra,payload",
    [
        ({"CLAUDE_CODE_PARENT_SESSION_ID": "nested"}, {}),
        ({"CLAUDE_CODE_AGENT_ID": "nested"}, {}),
        ({"CODEX_SESSION_ID": "other"}, {}),
        ({}, {"agent_id": "child"}),
    ],
)
def test_hook_rejects_marked_children_without_receiver_changes(
    bells: Bells, extra: dict[str, str], payload: dict[str, str]
) -> None:
    b = bells
    with inbox_socket() as (path, _, _):
        env = b.claude(3, path)
        root = b.modules.receivers.root(env)
        before = {p: p.read_bytes() for p in root.rglob("*.json")}
        result = hook({**env, **extra}, **payload)
        assert result.returncode == 1
        assert {p: p.read_bytes() for p in root.rglob("*.json")} == before
        assert env["CLAUDE_CODE_MESSAGING_TOKEN"] not in result.stdout + result.stderr


def test_hook_rejects_unmarked_nested_and_unrelated_foreground(bells: Bells) -> None:
    b = bells
    with inbox_socket() as (path, _, _):
        env = b.claude(3, path)
        script = """
import ctypes,json,subprocess,sys
ctypes.CDLL(None).prctl(15,b'claude',0,0,0)
r=subprocess.run([sys.executable,sys.argv[1]],input=json.dumps({'hook_event_name':'SessionStart','session_id':'unmarked-child'}),text=True)
raise SystemExit(r.returncode)
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(OPS / "board/claude_hook.py")],
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
        )
        assert result.returncode == 1
        state = b.w.labels.read()
        state["pid"] = 1
        b.w.labels.write(state)
        assert hook(env).returncode == 1


def confirm_parent_display(bells: Bells) -> dict[str, Any]:
    b = bells
    # Existing #1353 owner mapping confirms only parent 1 / term-a.
    state = b.w.labels.read()
    owner_token = importlib.import_module("herdr_labels.coordination").owner_token
    pane = state["panes"]["w1:p1"]
    pane["display_agent"] = "Parent label"
    pane["tokens"].update(
        tl_parent="#1",
        tl_topic="topic",
        tl_engine="claude",
        tl_owner=owner_token("claude:board-owner-1"),
    )
    b.w.labels.write(state)
    # Replace fixture's original owner mapping with the world identity.
    owners = (
        Path(b.w.labels.env["XDG_STATE_HOME"]) / "tennis-lab-agents/herdr-labels/owners"
    )
    (owners / f"{owner_token('claude:board-owner-1')}.json").write_text(
        json.dumps(
            {
                "version": 1,
                "scope": owner_token("/fake/herdr.sock"),
                "terminal_id": "term-a",
            }
        )
    )
    return cast(dict[str, Any], pane)


@pytest.mark.parametrize("reading", ["read", "thread", "ack", "reply"])
def test_herdr_unread_events_preserve_parent_tokens_and_do_not_poll(
    bells: Bells, reading: str
) -> None:
    b = bells
    pane = confirm_parent_display(b)
    post = b.w.board.messages.post(
        b.w.env(3),
        to=[b.w.assignments[1].name],
        kind="info",
        subject="Notice",
        body="An update.",
        notifiers=b.w.routes,
    )["post"]
    after = b.w.labels.read()["panes"]["w1:p1"]
    assert (
        after["tokens"]["tl_inbox"] == "1"
        and after["tokens"]["tl_inbox_label"] == "未読:1"
    )
    writes = [x for x in b.w.labels.mutations() if "tennis-board" in x]
    assert len(writes) == 1 and "--ttl-ms" not in writes[0]
    b.w.board.messages.inbox(b.w.env())
    assert len([x for x in b.w.labels.mutations() if "tennis-board" in x]) == 1
    if reading == "ack":
        b.w.board.messages.ack(b.w.env(), post_id=post["id"], state="accepted")
    elif reading == "reply":
        b.w.board.messages.reply(
            b.w.env(), post_id=post["id"], body="Received.", notifiers=b.w.routes
        )
    else:
        getattr(b.w.board.messages, reading)(b.w.env(), post["id"])
    b.w.board.messages.read(b.w.env(), post["id"])
    after = b.w.labels.read()["panes"]["w1:p1"]
    assert (
        after["tokens"]["tl_inbox"] == "0" and "tl_inbox_label" not in after["tokens"]
    )
    assert {
        k: v for k, v in after["tokens"].items() if not k.startswith("tl_inbox")
    } == pane["tokens"]
    assert after["display_agent"] == "Parent label"
    assert len([x for x in b.w.labels.mutations() if "tennis-board" in x]) == 2


@pytest.mark.parametrize(
    "failure", ["reject", "drop_write", "timeout", "wrong_owner", "wrong_native"]
)
def test_herdr_failures_are_explicit_and_never_select_another_pane(
    bells: Bells, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    b = bells
    confirm_parent_display(b)
    state = b.w.labels.read()
    if failure in {"reject", "drop_write"}:
        state[failure] = True
    elif failure == "wrong_owner":
        state["panes"]["w1:p1"]["tokens"]["tl_owner"] = "replacement-owner"
    elif failure == "wrong_native":
        state["panes"]["w1:p1"]["agent_session"] = {
            "kind": "id",
            "agent": "claude",
            "value": "replacement-conversation",
        }
    b.w.labels.write(state)
    if failure == "timeout":
        original = b.modules.display.Herdr.call

        def timeout(client: Any, *args: str) -> dict[str, Any]:
            if args[:2] == ("pane", "report-metadata"):
                raise subprocess.TimeoutExpired("fake-herdr", 5)
            return cast(dict[str, Any], original(client, *args))

        monkeypatch.setattr(b.modules.display.Herdr, "call", timeout)
    result = b.w.board.messages.post(
        b.w.env(3),
        to=[b.w.assignments[1].name],
        kind="info",
        subject="Notice",
        body="An update.",
        notifiers=b.w.routes,
    )
    display = next(row for row in result["inbox_display"] if row["parent"] == 1)
    assert display["status"] == "error" and display["reason"] == "route_unavailable"
    assert result["post"]["recipients"][0]["delivery_state"] == "notified"
    assert len(b.w.fake.accepted) == 1
    assert "tl_inbox" not in b.w.labels.read()["panes"]["w1:p2"].get("tokens", {})
    assert all("w1:p2" not in args for args in b.w.labels.mutations())


def test_missing_herdr_owner_mapping_is_not_guessed(bells: Bells) -> None:
    result = bells.post(2)
    assert bells.w.labels.mutations() == []
    assert result["id"]


def test_compact_preserves_receiver_generation_and_user_prompt_does_not_rewrite(
    bells: Bells,
) -> None:
    b = bells
    with inbox_socket() as (path, _, _):
        env = b.claude(3, path)
        root = b.modules.receivers.root(env)
        before = {p: p.read_bytes() for p in root.rglob("*.json")}
        assert hook(env, source="compact").returncode == 0
        assert hook(env, "UserPromptSubmit").returncode == 0
        assert {p: p.read_bytes() for p in root.rglob("*.json")} == before


def test_standard_user_settings_and_invalid_settings_revoke_old_accept(
    bells: Bells,
) -> None:
    b = bells
    with inbox_socket() as (path, received, condition):
        env = b.claude(3, path)
        env.pop("CLAUDE_CONFIG_DIR")
        env["HOME"] = str(b.home)
        settings = b.home / ".claude/settings.json"
        settings.parent.mkdir()
        settings.write_text('{"crossSessionInbound":"accept"}')
        b.modules.receivers.capture_claude(
            env, "resume", b.modules.receivers.process(os.getpid())
        )
        assert b.post()["recipients"][0]["delivery_state"] == "notified"
        wait_received(received, condition)
        settings.write_text("invalid settings with private-fixture-credential")
        b.modules.receivers.capture_claude(
            env, "resume", b.modules.receivers.process(os.getpid())
        )
        post = b.post()
        assert post["recipients"][0]["delivery_reason"] == "receiver_not_accepting"
        assert len(received) == 1


def test_codex_receipt_history_corruption_blocks_without_resend(bells: Bells) -> None:
    b = bells
    _, rollout, calls = b.codex()
    post = b.post(2)
    rollout.write_text(rollout.read_text().replace("original cwd", "changed cwd!"))
    receipt = b.modules.notifiers.receipts(b.w.env())["receipts"][0]
    assert receipt["history_blocked"] and receipt["phase"] == "queued"
    b.dispatch(post)
    assert len(calls.read_text().splitlines()) == 1


def test_headless_root_hook_uses_native_ancestor_and_rejects_unowned_process(
    bells: Bells,
) -> None:
    b = bells
    with inbox_socket() as (path, _, _):
        env = {k: v for k, v in b.claude(3, path).items() if not k.startswith("HERDR_")}
        script = """
import ctypes,json,subprocess,sys
ctypes.CDLL(None).prctl(15,b'claude',0,0,0)
r=subprocess.run([sys.executable,sys.argv[1]],input=json.dumps({'hook_event_name':'SessionStart','session_id':'board-owner-3','source':'resume'}),text=True)
raise SystemExit(r.returncode)
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(OPS / "board/claude_hook.py")],
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr
        assert hook(env).returncode == 1


@pytest.mark.parametrize("transition", ["register", "accept"])
def test_held_dispatch_after_parent_cli_restoration_once(
    parent_env: ParentEnv, board: SimpleNamespace, transition: str
) -> None:
    p = parent_env
    receivers = importlib.import_module("board.receivers")
    messages = importlib.import_module("board.messages")
    p.cli("register")
    old = "private-receiver-old"
    old_env = {**p.env, "CLAUDE_CODE_SESSION_ID": old}
    p.cli("register", parent="2", extra={"CLAUDE_CODE_SESSION_ID": old})
    with board.storage.Store(board.storage.directory(p.env)).read() as connection:
        name = board.names.for_parent(connection, "o/r", 2).name
    p.cli(
        "release" if transition == "register" else "handoff",
        "--note-file",
        p.note(),
        parent="2",
        extra={"CLAUDE_CODE_SESSION_ID": old},
    )
    post = messages.post(
        p.env, to=[name], kind="question", subject="Question", body="Please check."
    )["post"]
    assert post["recipients"][0]["delivery_state"] == "held"
    recipient_env = (
        old_env
        if transition == "register"
        else {**p.env, "CLAUDE_CODE_SESSION_ID": "private-receiver-new"}
    )
    with inbox_socket() as (path, received, condition):
        config = p.root / "claude-user"
        config.mkdir()
        (config / "settings.json").write_text('{"crossSessionInbound":"accept"}')
        recipient_env.update(
            CLAUDE_CONFIG_DIR=str(config),
            CLAUDE_CODE_MESSAGING_SOCKET=str(path),
            CLAUDE_CODE_MESSAGING_TOKEN="private-token",
        )
        receivers.capture_claude(
            recipient_env, "startup", receivers.process(os.getpid())
        )
        extra = {k: v for k, v in recipient_env.items() if k.startswith("CLAUDE_")}
        args = (
            ["register"]
            if transition == "register"
            else ["handoff", "--accept", "--note-file", p.note()]
        )
        result = p.cli(*args, parent="2", extra=extra)
        data = json.loads(result.stdout)
        assert (
            data["parent_action_applied"] and data["board_delivery"]["status"] == "ok"
        )
        assert data["board_delivery"]["deliveries"][0]["delivery_state"] == "notified"
        wait_received(received, condition)
        p.cli("register", parent="2", extra=extra)
        assert len(received) == 1
        assert messages.status(p.env)["posts"][0]["recipients"][0]["read_at"] is None
        private_output_absent(result.stdout + result.stderr, recipient_env)


def test_held_failure_after_register_keeps_assignment_and_reports_next_start(
    parent_env: ParentEnv, board: SimpleNamespace
) -> None:
    p = parent_env
    receivers = importlib.import_module("board.receivers")
    messages = importlib.import_module("board.messages")
    p.cli("register")
    receiver = {**p.env, "CLAUDE_CODE_SESSION_ID": "private-receiver"}
    p.cli("register", parent="2", extra={"CLAUDE_CODE_SESSION_ID": "private-receiver"})
    with board.storage.Store(board.storage.directory(p.env)).read() as connection:
        name = board.names.for_parent(connection, "o/r", 2).name
    p.cli(
        "release",
        "--note-file",
        p.note(),
        parent="2",
        extra={"CLAUDE_CODE_SESSION_ID": "private-receiver"},
    )
    post = messages.post(
        p.env, to=[name], kind="info", subject="Notice", body="Please check."
    )["post"]
    with inbox_socket() as (path, received, _):
        config = p.root / "receiver-settings"
        config.mkdir()
        (config / "settings.json").write_text('{"crossSessionInbound":"hold"}')
        receiver.update(
            CLAUDE_CONFIG_DIR=str(config),
            CLAUDE_CODE_MESSAGING_SOCKET=str(path),
            CLAUDE_CODE_MESSAGING_TOKEN="private-token",
        )
        receivers.capture_claude(receiver, "resume", receivers.process(os.getpid()))
        extra = {k: v for k, v in receiver.items() if k.startswith("CLAUDE_")}
        result = p.cli("register", parent="2", extra=extra, ok=False)
        data = json.loads(result.stdout)
        assert result.returncode == 1 and data["parent_action_applied"]
        assert (
            data["board_delivery"]["deliveries"][0]["delivery_reason"]
            == "receiver_not_accepting"
        )
        assert data["board_delivery"]["inbox"]["delivery_failed_count"] == 1
        assert (
            messages.status(p.env)["posts"][0]["recipients"][0]["state"]
            == "delivery_failed"
        )
        again = json.loads(p.cli("register", parent="2", extra=extra).stdout)
        assert again["board_delivery"]["inbox"]["delivery_failed_count"] == 1
        assert again["board_delivery"]["inbox"]["unread_count"] == 1
        assert messages.inbox(receiver)["posts"][0]["id"] == post["id"]
        assert received == []


def test_board_hook_configuration_is_additive() -> None:
    settings = json.loads((OPS.parents[1] / ".claude/settings.json").read_text())
    hooks = settings["hooks"]
    for event in ("SessionStart", "UserPromptSubmit"):
        assert any(
            "board/claude_hook.py" in command["command"]
            for group in hooks[event]
            for command in group["hooks"]
        )
    assert any(
        "herdr_labels/claude_hook.py" in command["command"]
        for group in hooks["SessionStart"]
        for command in group["hooks"]
    )
    assert "crossSessionInbound" not in settings
