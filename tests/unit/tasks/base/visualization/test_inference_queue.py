"""GPU requests must retain the common worktree queue and argument boundaries."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.tasks.base.scripts import inference_worker
from src.tasks.base.visualization import inference_queue
from src.utils.configuration import PathContractError
from src.utils.training_queue import RETENTION_NS


@pytest.mark.parametrize("terminal", ["done", "failed", "cancelled"])
@pytest.mark.parametrize("expired", [False, True])
@pytest.mark.parametrize("result_exists", [False, True])
@pytest.mark.parametrize(
    "job",
    ["123.job", "0-8207631999999999999_2026-10-19T00-00-00-000000000Z_2_inference.job"],
)
def test_queue_uses_shared_root_and_preserves_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal: str,
    expired: bool,
    job: str,
    result_exists: bool,
) -> None:
    root = tmp_path / "main repo"
    root.mkdir()
    monkeypatch.setattr(inference_queue, "shared_repository_root", lambda: root)
    seen: list[list[str]] = []
    request = {"checkpoint": "space `literal` $(literal).ckpt", "device": "cuda"}

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(args)
        assert kwargs["env"]["TRAINING_QUEUE_DIR"] == str(root / ".training_queue")
        if args[2] == "add":
            command = shlex.split(args[3])
            assert command[1:3] == [
                "-m",
                "src.tasks.base.scripts.inference_worker",
            ]
            source = Path(command[3])
            assert json.loads(source.read_text())["request"] == request
            if result_exists:
                (source.parent / "result.bin").write_bytes(b"result")
            if expired:
                completion = (
                    root / ".training_queue/control/completions" / f"{job[:-4]}.json"
                )
                completion.parent.mkdir(parents=True)
                completion.write_text(
                    json.dumps(
                        dict(
                            version=1,
                            job=job,
                            state=terminal,
                            finished_at_ns=1,
                            retention_started_at_ns=1,
                            time_source="recorded",
                            expired_at_ns=RETENTION_NS + 2,
                        )
                    )
                )
            else:
                state = root / ".training_queue" / terminal
                state.mkdir()
                (state / job).touch()
            return subprocess.CompletedProcess(args, 0, f"queued: {job}\n", "")
        return subprocess.CompletedProcess(
            args, 1, "", "worker already running (PID 1)."
        )

    monkeypatch.setattr(inference_queue.subprocess, "run", run)
    if terminal == "done" and result_exists:
        assert (
            inference_queue.run_queued_inference(
                "blcs",
                service={},
                request=request,
            )
            == b"result"
        )
    else:
        with pytest.raises(
            RuntimeError, match="history expired" if expired else terminal
        ):
            inference_queue.run_queued_inference("blcs", service={}, request=request)
    assert seen[0][-2:] == ["--resource", "all"]
    assert seen[1][2] == "start"


def test_unknown_worker_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        inference_queue.execute_request(
            {"task": "unknown", "service": {}, "request": {}}
        )


def test_worker_entry_publishes_result_in_validated_request_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / ".training_queue" / "ui_requests" / "job"
    directory.mkdir(parents=True)
    request = directory / "request.json"
    request.write_text(json.dumps({"task": "test"}))
    monkeypatch.setattr(inference_worker, "shared_repository_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["worker", str(request)])
    monkeypatch.setattr(inference_queue, "execute_request", lambda document: b"result")
    inference_worker.main()
    assert (directory / "result.bin").read_bytes() == b"result"
    assert not (directory / "result.tmp").exists()


def test_worker_entry_rejects_paths_outside_shared_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = tmp_path / "outside.json"
    request.write_text("{}")
    monkeypatch.setattr(inference_worker, "shared_repository_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["worker", str(request)])
    with pytest.raises(PathContractError):
        inference_worker.main()
    assert not (tmp_path / "result.bin").exists()
    assert not (tmp_path / "error.json").exists()


def test_worker_failure_is_recorded_and_reraised(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"task": "unknown", "service": {}, "request": {}}))
    with pytest.raises(ValueError, match="Unknown"):
        inference_queue.process_request(request)
    assert json.loads((tmp_path / "error.json").read_text())["type"] == "ValueError"
    assert not (tmp_path / "result.bin").exists()
