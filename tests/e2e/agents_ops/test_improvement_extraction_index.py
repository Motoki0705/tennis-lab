"""Numeric extraction bindings survive editable body changes and missing owners."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_dispatcher_judgement import Harness, issue
from tests.support.agents_ops.improvements import extraction_receipt


def seed_receipt(h: Harness, *, target: int | None = 1) -> Path:
    h.gh.issues[2] = issue(2)
    h.gh.issues[1]["body"] = (
        "<!-- parent-improvement-extraction:v1 parent=2 proposal=3 candidates=I-2-first,I-2-second -->\n\n"
        + h.gh.issues[1]["body"]
    )
    return extraction_receipt(
        h.store.directory,
        parent=2,
        proposal=3,
        target=target,
        body=h.gh.issues[1]["body"],
        decisions={"first": "a" * 64, "second": "b" * 64},
    )


@pytest.mark.parametrize("edit", ["missing", "candidate", "proposal"])
def test_dispatcher_collection_and_guard_reject_changed_extraction_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edit: str
) -> None:
    h = Harness(tmp_path, monkeypatch)
    seed_receipt(h)
    with h.store.locked():
        snapshot = h.collector.collect(h.gh, h.store, h.work, [1])
    assert snapshot["issues"]["1"]["improvement_sources"] == [2]
    assert set(snapshot["issues"]) == {"1", "2"}
    body = h.gh.issues[1]["body"]
    if edit == "missing":
        body = body.split("\n", 1)[1]
    elif edit == "candidate":
        body = body.replace("I-2-second", "I-2-third")
    else:
        body = body.replace("proposal=3", "proposal=4")
    snapshot["issues"]["1"]["body"] = body
    with pytest.raises(ValueError, match="extraction provenance"):
        h.schema.guard(snapshot, 1)
    h.gh.issues[1]["body"] = body
    with h.store.locked(), pytest.raises(ValueError, match="extraction provenance"):
        h.collector.collect(h.gh, h.store, h.work, [1])
    assert not h.calls.exists() and not h.gh.writes


@pytest.mark.parametrize("fault", ["unreadable", "corrupt", "outcome_unknown"])
def test_extraction_index_failure_never_means_an_unbound_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    h = Harness(tmp_path, monkeypatch)
    path = seed_receipt(h, target=None if fault == "outcome_unknown" else 1)
    h.gh.issues[1]["body"] = h.gh.issues[1]["body"].split("\n", 1)[1]
    if fault == "unreadable":
        original = Path.read_bytes

        def read_bytes(current: Path) -> bytes:
            if current == path:
                raise PermissionError("extraction index unavailable")
            return original(current)

        monkeypatch.setattr(Path, "read_bytes", read_bytes)
    elif fault == "corrupt":
        path.write_text("invalid receipt")
    parent = importlib.import_module("parent.session")
    execution = importlib.import_module("parent.execution")
    improvements = importlib.import_module("parent.improvements")
    with h.store.locked():
        session = parent.ParentSession(h.gh, h.store, 1, None)
        for operation in (
            lambda: improvements.require(session),
            lambda: execution.execution_input(session),
            lambda: h.collector.collect(h.gh, h.store, h.work, [1]),
        ):
            with pytest.raises((ValueError, OSError)):
                operation()
    assert not h.calls.exists() and not h.gh.writes


def test_supervisor_rechecks_numeric_extraction_binding_before_popen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.e2e.agents_ops.test_dispatcher_supervisor import (
        DispatchHarness,
        fixture_runtime,
    )

    d = DispatchHarness(tmp_path, monkeypatch)
    fixture_runtime(d, monkeypatch)
    d.prepare()
    d.control.assign()
    job = d.job()
    assert d.control.claim("o/r#1", job["id"]) is not None
    seed_receipt(d.h)
    d.h.gh.issues[1]["body"] = d.h.gh.issues[1]["body"].split("\n", 1)[1]
    runtime = importlib.import_module("dispatcher.runtime").Runtime(Path(job["runner"]))
    runtime.launch_context = lambda: d.control.launch_admission("o/r#1", job["id"])
    calls: list[Any] = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: calls.append(a))
    with pytest.raises(ValueError, match="extraction provenance"):
        runtime.execute(job, 0)
    assert calls == [] and d.job()["cancelled_before_start"]
    assert d.row()["status"] == "released"
