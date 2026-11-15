"""External board I/O and stale projections against disposable local state."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_board_doorbell import Bells, confirm_parent_display
from tests.e2e.agents_ops.test_board_doorbell import bells as bells
from tests.e2e.agents_ops.test_board_messages import board as board
from tests.e2e.agents_ops.test_board_messages import labels as labels
from tests.e2e.agents_ops.test_board_messages import world as world

LOCK_PROBE = """
import fcntl, pathlib, sys
with pathlib.Path(sys.argv[1]).open('a') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('held')
    else:
        print('free')
"""


def available(path: Path) -> bool:
    result = subprocess.run(
        [sys.executable, "-c", LOCK_PROBE, str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.stdout.strip() in {"held", "free"}
    return result.stdout.strip() == "free"


def registry_lock(b: Bells) -> Path:
    return Path(b.w.labels.registry.parent) / "registry.lock"


def change_owner(b: Bells, parent: int, **values: Any) -> None:
    script = """
import json, pathlib, sys
from parent.state import Store
with Store(pathlib.Path(sys.argv[1])).locked() as state:
    state.parents['o/r#' + sys.argv[2]].update(json.loads(sys.argv[3]))
    state.save()
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(b.w.labels.registry.parent),
            str(parent),
            json.dumps(values),
        ],
        env=b.w.env(),
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )


@pytest.mark.parametrize("renamed", [False, True])
def test_herdr_display_io_releases_registry(
    bells: Bells, monkeypatch: pytest.MonkeyPatch, renamed: bool
) -> None:
    b = bells
    confirm_parent_display(b)
    if renamed:
        change_owner(b, 1, display_summary="topic")
        runtime = b.w.labels.read()
        runtime["panes"]["w1:p1"]["tokens"]["tl_parent"] = "o/r#1"
        b.w.labels.write(runtime)
    original = b.modules.display.Herdr.call
    calls = []

    def checked(client: Any, *args: str) -> Any:
        assert available(registry_lock(b)), "Herdr I/O holds registry.lock"
        calls.append(args[:2])
        return original(client, *args)

    monkeypatch.setattr(b.modules.display.Herdr, "call", checked)
    result = b.modules.display.update(
        b.w.env(), [b.w.assignments[1].id], renamed=renamed
    )
    assert result[0]["status"] == "updated"
    assert ("pane", "report-metadata") in calls


@pytest.mark.parametrize("when", ["before", "after"])
@pytest.mark.parametrize("change", ["generation", "destination", "native"])
def test_display_rechecks_fixed_owner_without_undoing_post(
    bells: Bells,
    monkeypatch: pytest.MonkeyPatch,
    when: str,
    change: str,
) -> None:
    b = bells
    confirm_parent_display(b)
    original = b.modules.display.Herdr.call
    changed = False

    def raced(client: Any, *args: str) -> Any:
        nonlocal changed
        assert available(registry_lock(b)), "Herdr I/O holds registry.lock"
        result = original(client, *args)
        trigger = ("agent", "list") if when == "before" else ("pane", "report-metadata")
        if not changed and args[:2] == trigger:
            changed = True
            if change == "generation":
                change_owner(b, 1, generation=2)
            elif change == "native":
                runtime = b.w.labels.read()
                runtime["panes"]["w1:p1"]["agent_session"] = {
                    "kind": "id",
                    "agent": "claude",
                    "value": "replacement-session",
                }
                b.w.labels.write(runtime)
            else:
                token = b.w.board.service.owner_token("claude:board-owner-1")
                owner = (
                    b.w.labels.registry.parent.parent
                    / "herdr-labels/owners"
                    / f"{token}.json"
                )
                record = json.loads(owner.read_text())
                record["terminal_id"] = "term-b"
                owner.write_text(json.dumps(record))
        return result

    monkeypatch.setattr(b.modules.display.Herdr, "call", raced)
    result = b.w.board.messages.post(
        b.w.env(3),
        to=[b.w.assignments[1].name],
        kind="info",
        subject="Notice",
        body="The committed notice.",
        notifiers=b.w.routes,
    )
    assert changed
    assert result["post"]["body"] == "The committed notice."
    assert result["post"]["recipients"][0]["delivery_state"] == "notified"
    display = next(row for row in result["inbox_display"] if row["parent"] == 1)
    assert display["status"] == "error"
    assert all("w1:p2" not in args for args in b.w.labels.mutations())


def test_display_serializes_before_snapshot_and_keeps_latest_count(
    bells: Bells,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b = bells
    confirm_parent_display(b)
    post = b.w.board.messages.post(
        b.w.env(3),
        to=[b.w.assignments[1].name],
        kind="info",
        subject="Notice",
        body="Unread.",
        notifiers=b.w.routes,
    )["post"]
    # Make the old projection need an external write, then pause at that write.
    runtime = b.w.labels.read()
    runtime["panes"]["w1:p1"]["tokens"].pop("tl_inbox")
    b.w.labels.write(runtime)
    entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
    original_call = b.modules.display.Herdr.call
    coordination = __import__("herdr_labels.coordination", fromlist=["locked"])
    original_lock = coordination.locked
    old_thread: list[int] = []
    held = threading.local()
    original_capture = b.modules.display._capture

    def captured(*args: Any, **kwargs: Any) -> Any:
        assert getattr(held, "value", False), "snapshot predates display serialization"
        return original_capture(*args, **kwargs)

    @contextmanager
    def observed_lock(path: Path) -> Iterator[None]:
        if old_thread and threading.get_ident() != old_thread[0]:
            assert not available(path), "a newer display bypassed the old projection"
            waiting.set()
        with original_lock(path):
            held.value = True
            try:
                yield
            finally:
                held.value = False

    def delayed(client: Any, *args: str) -> Any:
        if (
            args[:2] == ("pane", "report-metadata")
            and threading.get_ident() == old_thread[0]
        ):
            entered.set()
            assert release.wait(timeout=10)
        return original_call(client, *args)

    monkeypatch.setattr(b.modules.display, "locked", observed_lock, raising=False)
    monkeypatch.setattr(b.modules.display, "_capture", captured)
    monkeypatch.setattr(b.modules.display.Herdr, "call", delayed)

    def old_update() -> Any:
        old_thread.append(threading.get_ident())
        return b.modules.display.update(b.w.env(), [b.w.assignments[1].id])

    with ThreadPoolExecutor(max_workers=2) as pool:
        old = pool.submit(old_update)
        try:
            if not entered.wait(timeout=5):
                old.result(timeout=1)
                raise AssertionError("old projection did not reach the write")
            new = pool.submit(b.w.board.messages.read, b.w.env(), post["id"])
            assert waiting.wait(timeout=5), (
                "new projection did not wait outside registry"
            )
            assert available(registry_lock(b))
        finally:
            release.set()
        assert old.result(timeout=10)[0]["status"] == "error"
        assert new.result(timeout=10)["inbox_display"][0]["status"] == "updated"
    assert b.w.labels.read()["panes"]["w1:p1"]["tokens"]["tl_inbox"] == "0"


def test_codex_discovery_releases_registry_and_routes(
    bells: Bells, monkeypatch: pytest.MonkeyPatch
) -> None:
    b = bells
    env, _, _ = b.codex()
    receivers = b.modules.receivers
    original_glob, original_history = Path.rglob, receivers.codex_queue.history
    calls = []

    def check(kind: str) -> None:
        assert available(registry_lock(b)), "Codex discovery holds registry.lock"
        assert available(receivers.root(env) / "routes.lock"), (
            "Codex discovery holds routes.lock"
        )
        calls.append(kind)

    def glob(path: Path, pattern: str) -> Iterator[Path]:
        check("rglob")
        yield from original_glob(path, pattern)

    def history(*args: Any, **kwargs: Any) -> Any:
        check("history")
        return original_history(*args, **kwargs)

    monkeypatch.setattr(Path, "rglob", glob)
    monkeypatch.setattr(receivers.codex_queue, "history", history)
    assert receivers.bind_receiver(env)["status"] == "bound"
    assert calls == ["rglob", "history"]


def test_codex_binding_keeps_registry_before_routes_lock_order(
    bells: Bells,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b = bells
    env, _, _ = b.codex()
    receivers = b.modules.receivers
    original = receivers.locked
    acquisitions = []

    @contextmanager
    def checked(path: Path) -> Iterator[None]:
        assert not available(registry_lock(b))
        acquisitions.append(path.name)
        with original(path):
            yield

    monkeypatch.setattr(receivers, "locked", checked)
    assert receivers.bind_receiver(env)["status"] == "bound"
    assert acquisitions == ["routes.lock", "routes.lock"]


def test_codex_discovery_cannot_bind_changed_generation(
    bells: Bells, monkeypatch: pytest.MonkeyPatch
) -> None:
    b = bells
    env, _, _ = b.codex()
    receivers, original = b.modules.receivers, b.modules.receivers._codex

    def discover(*args: Any) -> Any:
        assert available(registry_lock(b))
        record = original(*args)
        change_owner(b, 2, generation=2)
        return record

    monkeypatch.setattr(receivers, "_codex", discover)
    with pytest.raises(b.w.board.errors.BoardError, match="changed"):
        receivers.bind_receiver(env)
    path = (
        receivers.root(env)
        / f"{b.w.board.service.owner_token('codex:' + env['CODEX_SESSION_ID'])}.json"
    )
    assert receivers.read(path)["binding"] is None


@pytest.mark.parametrize("failed", [False, True])
def test_old_codex_discovery_never_overwrites_newer_epoch(
    bells: Bells,
    monkeypatch: pytest.MonkeyPatch,
    failed: bool,
) -> None:
    b = bells
    env, _, _ = b.codex()
    receivers, original = b.modules.receivers, b.modules.receivers._codex
    path = (
        receivers.root(env)
        / f"{b.w.board.service.owner_token('codex:' + env['CODEX_SESSION_ID'])}.json"
    )
    calls, newer = [], []

    def discover(*args: Any) -> Any:
        assert available(registry_lock(b))
        assert available(receivers.root(env) / "routes.lock")
        calls.append(True)
        record = original(*args)
        if len(calls) == 1:
            assert receivers.bind_receiver(env)["status"] == "bound"
            newer.append(path.read_bytes())
            if failed:
                raise b.w.board.errors.BoardError("injected discovery failure")
        return record

    monkeypatch.setattr(receivers, "_codex", discover)
    with pytest.raises(b.w.board.errors.BoardError):
        receivers.bind_receiver(env)
    assert len(calls) == 2 and path.read_bytes() == newer[0]


def test_failed_discovery_revokes_old_same_generation_route(
    bells: Bells, monkeypatch: pytest.MonkeyPatch
) -> None:
    b = bells
    env, _, _ = b.codex()
    receivers = b.modules.receivers

    def broken(*args: Any) -> Any:
        assert available(registry_lock(b))
        raise b.w.board.errors.BoardError("injected discovery failure")

    monkeypatch.setattr(receivers, "_codex", broken)
    with pytest.raises(b.w.board.errors.BoardError):
        receivers.bind_receiver(env)
    post = b.post(2)
    assert post["recipients"][0]["delivery_reason"] == "route_unavailable"
