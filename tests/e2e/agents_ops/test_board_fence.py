"""Owner changes wait for disposable board sends while ordinary updates proceed."""

from __future__ import annotations

import copy
import importlib
import json
import subprocess
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.e2e.agents_ops.test_board_doorbell import (
    Bells,
    inbox_socket,
    wait_received,
)
from tests.e2e.agents_ops.test_board_doorbell import bells as bells
from tests.e2e.agents_ops.test_board_messages import World
from tests.e2e.agents_ops.test_board_messages import board as board
from tests.e2e.agents_ops.test_board_messages import labels as labels
from tests.e2e.agents_ops.test_board_messages import message_world as message_world
from tests.e2e.agents_ops.test_board_messages import world as world
from tests.e2e.agents_ops.test_git_sync import wait_for_flock
from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.support.agents_ops.locks import available


def lock_paths(w: World) -> tuple[Path, Path]:
    return (
        w.labels.registry.with_name("registry.lock"),
        w.labels.registry.with_name("ownership.lock"),
    )


def test_owner_fence_must_precede_registry(tmp_path: Path, board: Any) -> None:
    store = importlib.import_module("parent.state").Store(tmp_path / "parents")
    with (
        store.locked(),
        pytest.raises(ValueError, match="must precede"),
        store.ownership(),
    ):
        pytest.fail("an inverted ownership lock order was admitted")
    assert not (store.directory / "ownership.lock").exists()


def test_owner_fence_allows_another_threads_registry_transaction(
    tmp_path: Path, board: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = importlib.import_module("parent.state").Store(tmp_path / "parents")
    entered, release = threading.Event(), threading.Event()
    owner = threading.get_ident()
    original = store._transaction_guard

    class Guard:
        def __enter__(self) -> Any:
            if threading.get_ident() == owner:
                release.set()  # Ownership admission has reached the shared guard.
            return original.__enter__()

        def __exit__(self, *args: Any) -> None:
            original.__exit__(*args)

    monkeypatch.setattr(store, "_transaction_guard", Guard())

    def write() -> None:
        with store.locked():
            entered.set()
            assert release.wait(timeout=5)
            store.save()

    with ThreadPoolExecutor(max_workers=1) as pool:
        writer = pool.submit(write)
        try:
            assert entered.wait(timeout=5)
            with store.ownership():
                assert not available(store.directory / "ownership.lock")
        finally:
            release.set()
        writer.result(timeout=5)


@pytest.mark.parametrize(
    "action", ["register", "release", "handoff", "recover", "expiry", "journal"]
)
def test_owner_changes_wait_for_send_and_then_proceed(
    message_world: World,
    parent_env: ParentEnv,
    tmp_path: Path,
    action: str,
) -> None:
    w, p = message_world, parent_env
    if action == "register":
        # Even an unrelated, previously unassigned parent must wait.
        parents = json.loads(w.labels.registry.read_text())["parents"]
        parents.pop("o/r#2")
        w.labels.set_registry(parents)
    expired = tmp_path / "expired"
    expired.mkdir()
    if action == "expiry":
        w.change(
            1,
            owner_runtime={
                "session": "claude:board-owner-1",
                "generation": 1,
                "cwd": str(expired),
                "process": None,
            },
        )
    entered, release = threading.Event(), threading.Event()
    registry, fence = lock_paths(w)

    def send(notification: Any) -> Any:
        assert available(registry), "notifier holds registry.lock"
        assert not available(fence), "notifier did not fence owner changes"
        with w.store.read() as connection:
            assert connection.execute(
                "SELECT state, reason, retryable FROM deliveries WHERE post_id=?",
                (notification.post_id,),
            ).fetchone() == ("delivery_failed", "notification_uncertain", 0)
        entered.set()
        assert release.wait(timeout=20), "test did not release the fake send"
        return w.board.delivery.NotificationResult(True)

    def post() -> Any:
        return w.board.messages.post(
            w.env(3),
            to=[w.assignments[1].name],
            kind="info",
            subject="A fenced send",
            body="The post and uncertainty marker must already be committed.",
            notifiers={"claude": send},
        )

    env = dict(p.env)
    env["XDG_STATE_HOME"] = w.env()["XDG_STATE_HOME"]
    env["CLAUDE_CODE_SESSION_ID"] = (
        "successor" if action in {"register", "recover"} else "board-owner-1"
    )
    target = "2" if action == "register" else "1"
    command = [sys.executable, "-m", "parent", "--repo", "o/r", "--parent", target]
    note = tmp_path / "note.md"
    note.write_text("The prior owner and its handoff evidence were inspected.")
    if action == "register":
        command += ["register", "--summary", "Another parent"]
    elif action == "release":
        command += ["release", "--note-file", str(note)]
    elif action in {"handoff", "recover"}:
        command += ["handoff", "--note-file", str(note)]
        if action == "recover":
            command += ["--accept", "--recover", "--summary", "New owner"]
    elif action == "expiry":
        command = [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; from worktree_cleanup.lifecycle import Stopper; "
            "Stopper(Path(sys.argv[1]), root=Path(sys.argv[2])).registry(Path(sys.argv[3]), [], 'expiry-test')",
            str(p.repo),
            str(w.labels.registry.parent.parent),
            str(expired),
        ]
    else:
        command = [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; from worktree_cleanup.registry import ExpiryStore; "
            "store=ExpiryStore(Path(sys.argv[1])); "
            'exec(\'with store.locked():\\n    assert store.parents["o/r#1"]["status"] == "released"\')',
            str(w.labels.registry.parent),
        ]

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(post)
        try:
            if not entered.wait(timeout=5):
                pending.result(timeout=1)
                pytest.fail("delivery did not reach the fake notifier")
            # An ordinary writer must finish while the send is still blocked.
            update = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys; from pathlib import Path; from parent.state import Store; "
                    "store=Store(Path(sys.argv[1])); "
                    'exec(\'with store.locked():\\n    store.parents["o/r#1"]["ordinary_update"] = 42\\n    store.save()\')',
                    str(w.labels.registry.parent),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=5,
            )
            assert update.returncode == 0, update.stdout + update.stderr
            before = json.loads(w.labels.registry.read_text())
            if action == "journal":
                after = copy.deepcopy(before)
                after["parents"]["o/r#1"]["status"] = "released"
                w.labels.registry.with_name(
                    "worktree-expiry-transaction.json"
                ).write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "changes": [
                                {
                                    "path": "registry.json",
                                    "before": before,
                                    "after": after,
                                }
                            ],
                        }
                    )
                )
            with subprocess.Popen(
                command,
                env=env,
                cwd=p.repo,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ) as change:
                try:
                    wait_for_flock(change)
                    assert available(registry), (
                        "owner change waited inside registry.lock"
                    )
                    assert json.loads(w.labels.registry.read_text()) == before
                    if action == "release":
                        status = subprocess.run(
                            [*command[:7], "status"],
                            env=env,
                            cwd=p.repo,
                            capture_output=True,
                            text=True,
                            timeout=8,
                        )
                        assert status.returncode == 0, status.stdout + status.stderr
                finally:
                    release.set()
                stdout, stderr = change.communicate(timeout=20)
                assert change.returncode in (
                    {0, 1}
                    if action in {"register", "release", "handoff", "recover"}
                    else {0}
                ), stdout + stderr
            result = pending.result(timeout=20)
        finally:
            release.set()
    assert result["post"]["recipients"][0]["delivery_state"] == "notified"
    parents = json.loads(w.labels.registry.read_text())["parents"]
    assert parents["o/r#1"]["ordinary_update"] == 42
    changed = parents[f"o/r#{target}"]
    assert changed["status"] == (
        "active"
        if action in {"register", "recover"}
        else "handoff_pending"
        if action == "handoff"
        else "released"
    )
    if action in {"register", "recover"}:
        assert changed["session"] == "claude:successor"
        assert changed["generation"] == (1 if action == "register" else 2)
    assert available(fence) and available(registry)


def test_delivery_waiting_for_fence_captures_the_new_owner(
    message_world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = message_world
    root = w.post(notifiers={})
    state = importlib.import_module("parent.state").Store(w.labels.registry.parent)
    waiting = threading.Event()
    original = w.board.delivery.ownership_fence

    @contextmanager
    def observed(env: Any) -> Iterator[None]:
        waiting.set()
        with original(env):
            yield

    monkeypatch.setattr(w.board.delivery, "ownership_fence", observed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with state.ownership():
            delivery = pool.submit(
                w.board.delivery.dispatch, w.env(), w.routes, post_id=root["id"]
            )
            assert waiting.wait(timeout=5), "delivery bypassed the ownership fence"
            w.change(2, session="claude:new-owner", generation=2)
        delivery.result(timeout=10)
    assert len(w.fake.calls) == 1
    sent = w.fake.calls[0]
    assert (sent.engine, sent.owner_session, sent.owner_generation) == (
        "claude",
        "claude:new-owner",
        2,
    )


@pytest.mark.parametrize("route_update", ["bind", "hook"])
def test_route_update_waits_outside_registry_during_transport(
    bells: Bells, monkeypatch: pytest.MonkeyPatch, route_update: str
) -> None:
    b = bells
    receiver, _, _ = b.codex()
    if route_update == "hook":
        receiver = b.claude(3, b.home / "unused.sock")
    entered, release = threading.Event(), threading.Event()
    original = b.modules.notifiers._command

    def command(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(timeout=10)
        return original(*args, **kwargs)

    monkeypatch.setattr(b.modules.notifiers, "_command", command)
    script = (
        "import os; from board.receivers import bind_receiver; bind_receiver(os.environ)"
        if route_update == "bind"
        else "import os; from board.receivers import capture_claude, process; "
        "capture_claude(os.environ, 'resume', process(os.getpid()))"
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        sending = pool.submit(b.post, 2)
        try:
            if not entered.wait(timeout=5):
                sending.result(timeout=1)
                pytest.fail("Codex send never reached its transport")
            with subprocess.Popen(
                [sys.executable, "-c", script],
                env=receiver,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ) as update:
                try:
                    wait_for_flock(update)
                    assert available(lock_paths(b.w)[0]), (
                        "route update waited for transport inside registry.lock"
                    )
                finally:
                    release.set()
                stdout, stderr = update.communicate(timeout=10)
                assert update.returncode == 0, stdout + stderr
            result = sending.result(timeout=10)
        finally:
            release.set()
    assert result["recipients"][0]["delivery_state"] == "notified"


@pytest.mark.parametrize("engine", ["claude", "codex"])
@pytest.mark.parametrize("changed", [False, True])
def test_transport_runs_without_registry_and_rejects_replaced_owner(
    bells: Bells, monkeypatch: pytest.MonkeyPatch, engine: str, changed: bool
) -> None:
    b = bells
    registry, fence = lock_paths(b.w)
    original_claude = b.modules.notifiers.claude
    original_codex = b.modules.notifiers.codex
    original_socket = b.modules.notifiers.socket
    original_command = b.modules.notifiers._command
    reached = []

    def probe() -> None:
        assert available(registry), "transport holds registry.lock"
        assert not available(fence), "transport did not pin its owner"
        reached.append(True)

    class Socket:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.client = original_socket.socket(*args, **kwargs)

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            self.client.close()

        def __getattr__(self, name: str) -> Any:
            return getattr(self.client, name)

        def sendall(self, data: bytes) -> None:
            probe()
            self.client.sendall(data)

    def command(*args: Any, **kwargs: Any) -> Any:
        probe()
        return original_command(*args, **kwargs)

    def route(notification: Any) -> Any:
        if changed:
            # Simulate an out-of-protocol index edit at the adapter boundary.
            # A compliant handoff cannot pass the held fence (tested above).
            b.w.change(notification.parent, generation=2)
            # Even a valid newly bound route cannot replace the pinned generation.
            path = b.modules.receivers.root(b.w.env()) / (
                b.w.board.service.owner_token(notification.owner_session) + ".json"
            )
            record = b.modules.receivers.read(path)
            record["binding"]["generation"] = 2
            b.modules.receivers.write(path, record)
        return (original_claude if engine == "claude" else original_codex)(
            b.w.env(), notification
        )

    monkeypatch.setattr(
        b.modules.notifiers,
        "socket",
        SimpleNamespace(
            socket=Socket,
            AF_UNIX=original_socket.AF_UNIX,
            SOCK_STREAM=original_socket.SOCK_STREAM,
        ),
    )
    monkeypatch.setattr(b.modules.notifiers, "_command", command)
    with inbox_socket() as (path, received, condition):
        if engine == "claude":
            b.claude(3, path)
            target, calls = 3, None
        else:
            _, _, calls = b.codex()
            target = 2
        result = b.w.board.messages.post(
            b.w.env(),
            to=[b.w.assignments[target].name],
            kind="info",
            subject="Pinned owner",
            body="Only this owner may receive the bell.",
            notifiers={engine: route},
        )
        recipient = result["post"]["recipients"][0]
        assert recipient["delivery_state"] == (
            "delivery_failed" if changed else "notified"
        )
        if changed:
            assert (
                not reached and not received and (calls is None or not calls.exists())
            )
            assert recipient["delivery_reason"] == "route_unavailable"
        else:
            assert reached == [True]
            if engine == "claude":
                wait_received(received, condition)
            else:
                assert calls is not None and len(calls.read_text().splitlines()) == 1
