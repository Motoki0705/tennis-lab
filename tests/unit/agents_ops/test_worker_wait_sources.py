"""Queue observations across real file renames and completion publication."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from src.utils import training_queue

JOB = "100_1_fit.job"
HEAD = "a" * 40


@pytest.fixture
def sources(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    return importlib.import_module("parent.worker_wait_sources")


@pytest.fixture
def monitor_mod(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    return importlib.import_module("parent.worker_monitor")


def make_pr(
    *,
    mergeable: bool | None = True,
    mergeable_state: str | None = "clean",
    head_sha: str = HEAD,
    base_ref: str | None = "main",
) -> dict[str, Any]:
    pr: dict[str, Any] = {
        "number": 12,
        "state": "open",
        "mergeable": mergeable,
        "head": {
            "ref": "codex/worker",
            "sha": head_sha,
            "repo": {"full_name": "o/r"},
        },
    }
    if mergeable_state is not None:
        pr["mergeable_state"] = mergeable_state
    if base_ref is not None:
        pr["base"] = {"sha": "b" * 40, "ref": base_ref, "repo": {"full_name": "o/r"}}
    return pr


def test_queue_requeue_between_active_directory_reads(
    tmp_path: Path, sources: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    queued = tmp_path / "jobs" / JOB
    running = tmp_path / "running" / JOB
    queued.parent.mkdir()
    running.parent.mkdir()
    running.touch()
    original: Callable[[Path], bool] = training_queue.regular

    def regular(path: Path) -> bool:
        found = original(path)
        if path == queued and running.is_file():
            assert not found
            running.rename(queued)
        return found

    monkeypatch.setattr(training_queue, "regular", regular)
    assert (
        sources.queue_result(tmp_path, JOB, now=100.0, missing_completion_since={})
        is None
    )
    assert queued.is_file() and not running.exists()


@pytest.mark.parametrize("published", [True, False])
def test_queue_waits_for_bounded_completion_publication(
    tmp_path: Path, sources: Any, published: bool
) -> None:
    running = tmp_path / "running" / JOB
    terminal = tmp_path / "done" / JOB
    for directory in (running.parent, terminal.parent):
        directory.mkdir()
    running.touch()
    missing: dict[str, float] = {}
    assert (
        sources.queue_result(tmp_path, JOB, now=0.0, missing_completion_since=missing)
        is None
    )
    running.rename(terminal)
    assert (
        sources.queue_result(
            tmp_path, JOB, now=1000.0, missing_completion_since=missing
        )
        is None
    )
    assert (
        sources.queue_result(
            tmp_path, JOB, now=1059.0, missing_completion_since=missing
        )
        is None
    )
    if published:
        completion = training_queue.completion_path(tmp_path, JOB)
        completion.parent.mkdir(parents=True)
        completion.write_text(
            json.dumps(
                {
                    "version": 1,
                    "job": JOB,
                    "state": "done",
                    "time_source": "recorded",
                    "retention_started_at_ns": 100,
                    "finished_at_ns": 100,
                }
            )
        )
        result = sources.queue_result(
            tmp_path, JOB, now=1060.0, missing_completion_since=missing
        )
        assert result["state"] == "done"
    else:
        with pytest.raises(
            ValueError, match="missing training-queue completion record"
        ):
            sources.queue_result(
                tmp_path, JOB, now=1060.0, missing_completion_since=missing
            )


@pytest.mark.parametrize("include_metadata", [True, False])
def test_ci_result_detects_conflict_when_mergeable_is_false_and_python_missing(
    tmp_path: Path, sources: Any, include_metadata: bool
) -> None:
    pr = make_pr(
        mergeable=False,
        mergeable_state="dirty" if include_metadata else None,
        base_ref="main" if include_metadata else None,
    )
    backend = SimpleNamespace(
        repo="o/r",
        pr=Mock(return_value=pr),
        ci=Mock(return_value={"status": "pending", "reasons": ["ci_python_missing"]}),
    )
    result = sources.ci_result(
        backend, {"pr": 12, "head": HEAD}, "codex/worker", root=tmp_path
    )
    expected: dict[str, Any] = {
        "kind": "ci",
        "pr": 12,
        "head": HEAD,
        "status": "conflict",
        "reasons": ["pr_conflict", "ci_python_missing"],
    }
    if include_metadata:
        expected.update(base_ref="main", mergeable_state="dirty")
    assert result == expected
    assert backend.pr.call_count == 2


@pytest.mark.parametrize(
    ("mergeable", "reasons"),
    [(None, ["ci_python_missing"]), (False, ["ci_python_pending"])],
)
def test_ci_result_stays_pending_when_mergeable_is_none_or_ci_is_running(
    tmp_path: Path, sources: Any, mergeable: bool | None, reasons: list[str]
) -> None:
    backend = SimpleNamespace(
        repo="o/r",
        pr=Mock(return_value=make_pr(mergeable=mergeable, mergeable_state="unknown")),
        ci=Mock(return_value={"status": "pending", "reasons": reasons}),
    )
    assert (
        sources.ci_result(
            backend, {"pr": 12, "head": HEAD}, "codex/worker", root=tmp_path
        )
        is None
    )
    backend.pr.assert_called_once_with(12)


@pytest.mark.parametrize("outcome", ["success", "conflict"])
def test_ci_result_transitions_from_null_mergeability_to_terminal_observation(
    tmp_path: Path, sources: Any, outcome: str
) -> None:
    second = make_pr(
        mergeable=outcome == "success",
        mergeable_state="clean" if outcome == "success" else "dirty",
    )
    backend = SimpleNamespace(
        repo="o/r",
        pr=Mock(
            side_effect=[
                make_pr(mergeable=None, mergeable_state="unknown"),
                second,
                second,
            ]
        ),
        ci=Mock(
            side_effect=[
                {"status": "pending", "reasons": ["ci_python_missing"]},
                (
                    {"status": "success", "reasons": []}
                    if outcome == "success"
                    else {"status": "pending", "reasons": ["ci_python_missing"]}
                ),
            ]
        ),
    )
    request = {"pr": 12, "head": HEAD}
    assert sources.ci_result(backend, request, "codex/worker", root=tmp_path) is None
    observed = sources.ci_result(backend, request, "codex/worker", root=tmp_path)
    assert observed is not None
    assert observed["status"] == outcome
    if outcome == "conflict":
        assert observed["reasons"] == ["pr_conflict", "ci_python_missing"]
        assert observed["base_ref"] == "main"
        assert observed["mergeable_state"] == "dirty"


@pytest.mark.parametrize("flipped_to", [True, None, "head_changed"])
def test_ci_result_race_guard_avoids_false_conflict_on_recheck(
    tmp_path: Path, sources: Any, flipped_to: bool | None | str
) -> None:
    if isinstance(flipped_to, str):
        latest = make_pr(mergeable=False, head_sha="b" * 40)
    else:
        latest = make_pr(mergeable=flipped_to, mergeable_state="unknown")
    backend = SimpleNamespace(
        repo="o/r",
        pr=Mock(
            side_effect=[make_pr(mergeable=False, mergeable_state="dirty"), latest]
        ),
        ci=Mock(return_value={"status": "pending", "reasons": ["ci_python_missing"]}),
    )
    request = {"pr": 12, "head": HEAD}
    if flipped_to == "head_changed":
        with pytest.raises(ValueError, match="CI target differs"):
            sources.ci_result(backend, request, "codex/worker", root=tmp_path)
    else:
        assert (
            sources.ci_result(backend, request, "codex/worker", root=tmp_path) is None
        )
    assert backend.pr.call_count == 2


@pytest.mark.parametrize(
    ("mergeable", "observe_status", "expected_status"),
    [
        (None, "pending", None),
        (None, "success", "success"),
        (False, "pending", "conflict"),
    ],
)
def test_ci_result_knowledge_pr_respects_mergeable_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sources: Any,
    mergeable: bool | None,
    observe_status: str,
    expected_status: str | None,
) -> None:
    records = importlib.import_module("lib.ops_review.knowledge_records")
    pr = make_pr(
        mergeable=mergeable,
        mergeable_state="dirty" if mergeable is False else "unknown",
    )
    monkeypatch.setattr(
        records,
        "assess",
        Mock(
            return_value=(
                None,
                SimpleNamespace(head=HEAD, base="b" * 40, merge="c" * 40),
            )
        ),
    )
    monkeypatch.setattr(
        records,
        "observe_ci",
        Mock(
            return_value={
                "workflow": "Knowledge library",
                "status": observe_status,
                "reasons": (
                    []
                    if observe_status == "success"
                    else ["Knowledge library PR run is missing"]
                ),
            }
        ),
    )
    backend = SimpleNamespace(
        repo="o/r",
        pr=Mock(return_value=pr),
        ci=Mock(return_value={"status": "pending", "reasons": ["ci_python_missing"]}),
    )
    observed = sources.ci_result(
        backend, {"pr": 12, "head": HEAD}, "codex/worker", root=tmp_path
    )
    if expected_status is None:
        assert observed is None
    else:
        assert observed is not None and observed["status"] == expected_status
        if expected_status == "conflict":
            assert observed["reasons"][0] == "pr_conflict"


@pytest.mark.parametrize("resume_on_success", [False, True])
def test_resume_prompt_conflict_builds_resolution_prompt_and_rejects_unknown_status(
    monitor_mod: Any, resume_on_success: bool
) -> None:
    wait: dict[str, Any] = {
        "kind": "ci",
        "pr": 12,
        "head": HEAD,
        "resume_on_success": resume_on_success,
        **({"next_step": "ドラフトを解除する。"} if resume_on_success else {}),
    }
    observed = {
        "kind": "ci",
        "pr": 12,
        "head": HEAD,
        "status": "conflict",
        "reasons": ["pr_conflict", "ci_python_missing"],
        "base_ref": "main",
        "mergeable_state": "dirty",
    }
    prompt = monitor_mod.resume_prompt(wait, observed)
    assert "PR #12" in prompt
    assert "merge conflict" in prompt
    assert "最新のmain" in prompt
    assert "PARENT_RESULT" in prompt
    assert '"pr_conflict"' in prompt
    assert "CIが成功" not in prompt
    assert "ドラフトを解除する" not in prompt

    no_base = monitor_mod.resume_prompt(
        wait,
        {
            "kind": "ci",
            "pr": 12,
            "head": HEAD,
            "status": "conflict",
            "reasons": ["pr_conflict"],
        },
    )
    assert "最新のbase branch" in no_base

    with pytest.raises(ValueError, match="unknown CI observation status"):
        monitor_mod.resume_prompt(wait, {**observed, "status": "bogus"})


@pytest.mark.parametrize(
    ("resumptions", "expected_state", "expected_reason"),
    [
        ([], "resuming", None),
        (["same"], "needs_parent", "event_already_consumed"),
        (["a", "b", "c"], "needs_parent", "automatic_resume_limit"),
    ],
)
def test_monitor_accept_resumes_on_conflict_respecting_dedup_and_limit(
    tmp_path: Path,
    monitor_mod: Any,
    resumptions: list[str],
    expected_state: str,
    expected_reason: str | None,
) -> None:
    followup_mod = importlib.import_module("parent.worker_followup")
    store = importlib.import_module("parent.state").Store(tmp_path / "state")
    wait = {"kind": "ci", "pr": 12, "head": HEAD, "resume_on_success": False}
    if resumptions == ["same"]:
        resumptions = [followup_mod.event_key(wait)]
    record = {
        "id": "chain-1",
        "delegation_id": "d" * 32,
        "owner": "claude:p",
        "generation": 1,
        "state": "waiting",
        "wait": wait,
        "resumptions": resumptions,
        "monitor_started": True,
        "run_dir": str(tmp_path / "run"),
        "phase": "implementing",
        "reason": None,
    }
    with store.locked():
        store.parents["o/r#1"] = {
            "session": "claude:p",
            "generation": 1,
            "status": "active",
            "reviews": [],
            "tasks": {
                "10": {
                    "delegation_id": "d" * 32,
                    "manual_followup": record,
                }
            },
        }
        store.save()
    monitor = monitor_mod.Monitor(store, "o/r", 1, 10, "primary", "chain-1")
    observed = {
        "kind": "ci",
        "pr": 12,
        "head": HEAD,
        "status": "conflict",
        "reasons": ["pr_conflict", "ci_python_missing"],
        "base_ref": "main",
    }
    resumed = monitor.accept(observed, json.loads(json.dumps(record)))
    saved = store.read_data()["parents"]["o/r#1"]["tasks"]["10"]["manual_followup"]
    assert saved["state"] == expected_state
    assert saved["reason"] == expected_reason
    assert saved["observation"] == observed
    assert (resumed is not None) is (expected_state == "resuming")
