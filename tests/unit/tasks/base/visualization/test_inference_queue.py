"""GPU requests must retain the common worktree queue and argument boundaries."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.tasks.base.visualization import inference_queue


@pytest.mark.parametrize("terminal", ["done", "failed", "cancelled"])
def test_queue_uses_shared_root_and_preserves_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal: str
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
                "src.tasks.base.visualization.inference_queue",
            ]
            source = Path(command[3])
            assert json.loads(source.read_text())["request"] == request
            (source.parent / "result.bin").write_bytes(b"result")
            state = root / ".training_queue" / terminal
            state.mkdir()
            (state / "123.job").touch()
            return subprocess.CompletedProcess(args, 0, "queued: 123.job\n", "")
        return subprocess.CompletedProcess(
            args, 1, "", "worker already running (PID 1)."
        )

    monkeypatch.setattr(inference_queue.subprocess, "run", run)
    if terminal == "done":
        assert (
            inference_queue.run_queued_inference(
                "blcs",
                service={},
                request=request,
            )
            == b"result"
        )
    else:
        with pytest.raises(RuntimeError, match=terminal):
            inference_queue.run_queued_inference("blcs", service={}, request=request)
    assert seen[0][-2:] == ["--resource", "all"]
    assert seen[1][2] == "start"


def test_unknown_worker_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        inference_queue.execute_request(
            {"task": "unknown", "service": {}, "request": {}}
        )
