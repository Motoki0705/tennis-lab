"""Completion evidence must belong to the currently reserved worker."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops"))
handoff: Any = importlib.import_module("parent.review_handoff")
development: Any = importlib.import_module("parent.development")


@pytest.mark.parametrize(
    "change", ["none", "active", "unclean", "old_task", "branch", "worktree"]
)
def test_handoff_accepts_only_completed_current_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    task: dict[str, Any] = {
        "delegated": True,
        "delegation_id": "a" * 32,
        "branch": "codex/worker",
        "worktree": str(tmp_path),
        "worker_runs": [
            {
                "delegation_id": "a" * 32,
                "run_dir": "run",
                "kind": "registered",
                "recorded_at": "2026-10-30T00:00:00+00:00",
            }
        ],
    }
    saved: list[bool] = []
    session = SimpleNamespace(
        row={"tasks": {"10": task}},
        store=SimpleNamespace(directory=tmp_path, save=lambda: saved.append(True)),
    )
    contract = {"task_id": "a" * 32, "branch": "codex/worker", "cwd": str(tmp_path)}
    record = {
        "schema_version": 1,
        "result": "completed",
        "cleanup_complete": True,
        "run_id": "run-1",
        "contract": contract,
    }
    summary = {"runner_alive": False, "agent_alive": False, "agent_group_alive": False}
    if change == "active":
        summary["agent_group_alive"] = True
    elif change == "unclean":
        record["cleanup_complete"] = False
    elif change == "old_task":
        contract["task_id"] = "old"
    elif change == "branch":
        contract["branch"] = "codex/other"
    elif change == "worktree":
        contract["cwd"] = str(tmp_path / "other")
    (tmp_path / "run").mkdir()
    (tmp_path / "run/result.json").write_text(json.dumps(record))
    monkeypatch.setattr(
        development.Runtime,
        "inspect",
        lambda *_: {"record": record, "summary": summary},
    )
    if change != "none":
        with pytest.raises(ValueError, match="current delegation"):
            handoff.runner_ended(session, 10)
        assert "worker_ended" not in task and not saved
    else:
        handoff.runner_ended(session, 10)
        assert task["worker_ended"] == {
            "delegation_id": "a" * 32,
            "evidence": "Common runner confirmed termination and cleanup.",
        }
        handoff.runner_ended(session, 10)
        assert saved == [True]
