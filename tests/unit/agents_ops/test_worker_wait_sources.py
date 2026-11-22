"""Queue observations across real file renames and completion publication."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.utils import training_queue

JOB = "100_1_fit.job"


@pytest.fixture
def sources(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    return importlib.import_module("parent.worker_wait_sources")


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
