"""Actual lifecycle producers, real Git, and the shared fake GitHub boundary."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.support.agents_ops.decision_publication import (
    DecisionEnv,
    navigation,
    publication,
)
from tests.support.agents_ops.worker_logs import journal

capture: Any = importlib.import_module("decision_logs.capture")
sources: Any = importlib.import_module("decision_logs.parent_sources")
lifecycle: Any = importlib.import_module("worker_logs.lifecycle")
continuation: Any = importlib.import_module("parent.log_publication")
store_module: Any = importlib.import_module("parent.state")


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DecisionEnv:
    value = DecisionEnv(tmp_path, monkeypatch)
    original = publication.Publisher
    monkeypatch.setattr(
        publication,
        "Publisher",
        lambda root, log, github=None, **kw: original(
            root, log, github or value.github, **kw
        ),
    )
    monkeypatch.chdir(value.root)
    return value


def canonical_comment(
    env: DecisionEnv,
    event: str,
    *,
    generation: int = 1,
    state: str | None = None,
    number: int,
) -> None:
    op = capture.Operation.begin(
        env.state,
        "o/r",
        1,
        "parent.release" if event != "completion" else "parent.workflow-report",
        "record",
    )
    token = capture.CURRENT.set(op)
    try:
        payload = {"parent": 1, "event": event, "generation": generation}
        if state is not None:
            payload["state"] = state
        sources.comment(
            SimpleNamespace(gh=SimpleNamespace(repo="o/r"), parent=1),
            payload,
            importlib.import_module("parent.policy").marker(payload)
            + "\n\nExplicit canonical parent statement.",
            {"id": number},
        )
        op.finish({"recorded": True})
    finally:
        capture.CURRENT.reset(token)


def end(
    env: DecisionEnv, *, event: str = "ended", generation: int = 1
) -> dict[str, Any]:
    owner = {
        "session": "codex:fixture",
        "generation": generation,
        "status": "active" if event == "handoff_accepted" else event,
        "tasks": {},
        "reviews": [],
    }
    journal.atomic(
        env.state / "registry.json", {"version": 1, "parents": {"o/r#1": owner}}
    )
    args = SimpleNamespace(
        command="release" if event == "ended" else "handoff",
        accept=event == "handoff_accepted",
        ended=True,
        dry_run=False,
    )
    instance = continuation.Continuation(
        args, store_module.Store(env.state), "o/r", 1, owner["session"]
    )
    result = instance.complete(
        args, {"assignment": event, "parent_action_applied": True}
    )
    assert result["parent_action_applied"] is True
    return cast(dict[str, Any], result["publication_exit"])


@pytest.mark.parametrize("event", ["ended", "handoff_pending", "handoff_accepted"])
def test_exit_after_publication_links_acceptance_and_end_without_publication_loop(
    env: DecisionEnv, monkeypatch: pytest.MonkeyPatch, event: str
) -> None:
    canonical_comment(env, "completion", state="completed", number=901)
    assert lifecycle.publish_decisions("o/r", 1, state_dir=env.state, root=env.root)[
        "saved_to_main"
    ]
    before = env.upstream_files()
    generation = 2 if event == "handoff_accepted" else 1
    canonical_comment(
        env,
        "handoff" if event == "handoff_accepted" else event,
        generation=generation,
        number=902,
    )
    order = []
    for name, label in (
        ("continue_workers", "workers"),
        ("capture_exit", "capture"),
        ("publish_decisions", "decisions"),
        ("record_exit", "checkpoint"),
    ):
        original = getattr(continuation, name)

        def wrapped(
            *a: Any, _original: Any = original, _label: str = label, **kw: Any
        ) -> Any:
            order.append(_label)
            if _label == "decisions":
                assert not list((env.state / "publication-exits").rglob("*.json"))
            return _original(*a, **kw)

        monkeypatch.setattr(continuation, name, wrapped)
    checkpoint = end(env, event=event, generation=generation)
    assert order == ["workers", "capture", "decisions", "checkpoint"]
    assert checkpoint["remaining"] == [] and checkpoint["remaining_complete"] is True
    saved = env.upstream_files()
    assert all(saved[name] == raw for name, raw in before.items() if "/event-" in name)
    page = saved[".agents/logs/parents/parent-1/README.md"].decode()
    assert "issuecomment-902" in page
    assert ("最後の親の受入（明示記録時点）: completed" in page) is (generation == 1)
    rows = navigation.records(saved)
    fact = next(
        r["data"]["observation"]
        for r in rows
        if r["kind"] == "result"
        and r["data"].get("observation", {}).get("kind") == "termination"
    )
    assert fact["event"] == event
    assert fact["acceptance"]["state"] == (
        "unverified" if generation == 2 else "completed"
    )
    calls = list(env.github.calls)
    for _ in range(2):
        assert lifecycle.publish_decisions(
            "o/r", 1, state_dir=env.state, root=env.root
        )["saved_to_main"]
    assert env.github.calls == calls


def test_interrupted_exit_reconciliation_keeps_remaining_actor_and_resume(
    env: DecisionEnv,
) -> None:
    canonical_comment(env, "ended", number=903)
    env.github.pause_reads = True
    pending = end(env)
    assert pending["remaining"] and pending["error"]
    env.github.pause_reads = False
    final = end(env)
    assert final["remaining"] == [] and final["error"] is None
    page = env.upstream_files()[".agents/logs/parents/parent-1/README.md"].decode()
    assert "公開の担当: unassigned" in page
    assert "publish-worker-log" in page and "publish-decision-log" in page
    assert "https://github.com/o/r/pull/" in page
    assert "issuecomment-903" in page


def test_supervisor_exit_publishes_typed_checkpoint_and_explicit_acceptance(
    env: DecisionEnv,
) -> None:
    canonical_comment(env, "completion", state="completed", number=904)
    checkpoint = lifecycle.finish_supervised(
        "o/r", 1, 1, state_dir=env.state, root=env.root
    )
    assert checkpoint["remaining"] == [] and checkpoint["error"] is None
    saved = env.upstream_files()
    page = saved[".agents/logs/parents/parent-1/README.md"].decode()
    assert (
        "issuecomment-904" in page
        and "最後の親の受入（明示記録時点）: completed" in page
    )
    fact = next(
        r
        for r in navigation.records(saved)
        if r["kind"] == "result"
        and r["data"].get("observation", {}).get("kind") == "termination"
    )
    assert fact["data"]["previous_result_id"] is not None


@pytest.mark.parametrize("fault", ["unknown_count", "capture_failed"])
def test_exit_keeps_unknown_inventory_or_capture_failure_without_veto(
    env: DecisionEnv, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    if fault == "unknown_count":
        missing = capture.Operation.begin(env.state, "o/r", 1, "parent.note", "record")
        missing.mark_failed(lost_source="fixture.source")
    else:
        bridge = importlib.import_module("decision_logs.lifecycle")

        def unavailable(*a: Any, **kw: Any) -> None:
            raise OSError("fixture capture unavailable")

        monkeypatch.setattr(bridge, "capture_termination", unavailable)
    value = end(env)
    assert value["event"] == "ended" and value["error"] == "input_capture_failed"
    assert (
        store_module.Store(env.state).read_data()["parents"]["o/r#1"]["status"]
        == "ended"
    )
    if fault == "unknown_count":
        page = env.upstream_files()[".agents/logs/parents/parent-1/README.md"].decode()
        assert "未公開記録数: 未確認" in page
