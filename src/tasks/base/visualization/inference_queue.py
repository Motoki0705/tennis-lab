"""Run web inference in a short-lived worker under the shared GPU queue."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

Task = Literal["blcs", "plcs", "ball_detection", "court_detection"]
SOURCE_ROOT = Path(__file__).resolve().parents[4]


def shared_repository_root(source_root: Path = SOURCE_ROOT) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).parent


def run_queued_inference(
    task: Task, *, service: Mapping[str, Any], request: Mapping[str, Any]
) -> bytes:
    """Wait for the queue's terminal state, then return the worker payload.

    The HTTP server never owns a CUDA model. Request files are retained with
    the queue log so a failed or interrupted run remains diagnosable.
    """
    root = shared_repository_root()
    queue = root / ".training_queue"
    requests = queue / "ui_requests"
    requests.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"{task}-", dir=requests))
    input_path = directory / "request.json"
    input_path.write_text(
        json.dumps({"task": task, "service": dict(service), "request": dict(request)}),
        encoding="utf-8",
    )
    script = root / ".agents/skills/training-queue/scripts/training_queue.sh"
    environment = {**os.environ, "TRAINING_QUEUE_DIR": str(queue)}
    command = shlex.join(
        [
            sys.executable,
            "-m",
            "src.tasks.base.visualization.inference_queue",
            str(input_path),
        ]
    )
    added = subprocess.run(
        [
            "bash",
            str(script),
            "add",
            command,
            "--name",
            f"{task}-web-inference",
            "--provider",
            "codex",
            "--session",
            os.environ.get("CODEX_THREAD_ID", "web-inference"),
            "--resource",
            "all",
        ],
        cwd=SOURCE_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    records = [
        line.removeprefix("queued: ")
        for line in added.stdout.splitlines()
        if line.startswith("queued: ")
    ]
    if len(records) != 1 or Path(records[0]).name != records[0]:
        raise RuntimeError(f"Unexpected queue response: {added.stdout}")
    job = records[0]
    started = subprocess.run(
        ["bash", str(script), "start"],
        cwd=SOURCE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    if started.returncode and "worker already running" not in started.stderr:
        subprocess.run(
            ["bash", str(script), "cancel", job],
            cwd=SOURCE_ROOT,
            env=environment,
            check=True,
            capture_output=True,
        )
        raise RuntimeError(f"GPU queue could not start: {started.stderr.strip()}")
    while True:
        terminal = next(
            (
                state
                for state in ("done", "failed", "cancelled")
                if (queue / state / job).exists()
            ),
            None,
        )
        if terminal is not None:
            error_path = directory / "error.json"
            if error_path.exists():
                error = json.loads(error_path.read_text(encoding="utf-8"))
                if error["type"] == "ValueError":
                    raise ValueError(error["detail"])
                raise RuntimeError(error["detail"])
            output = directory / "result.bin"
            if terminal != "done" or not output.is_file():
                raise RuntimeError(
                    f"GPU inference {terminal}. Queue log: "
                    f"{queue / 'logs' / (job.removesuffix('.job') + '.log')}"
                )
            return output.read_bytes()
        time.sleep(0.25)


def execute_request(document: Mapping[str, Any]) -> bytes:
    """Worker entry: restore only one task's service and release it on exit."""
    config = dict(document["service"])
    for name in (
        "data_root",
        "checkpoint_root",
        "checkpoints_root",
        "outputs_root",
        "project_root",
    ):
        if name in config and config[name] is not None:
            config[name] = Path(config[name])
    request = dict(document["request"])
    if document["task"] == "plcs":
        from src.tasks.plcs.visualization.inference.service import (
            InferenceService,
            PredictionRequest,
        )

        request["cameras"] = tuple(request["cameras"])
        payload: bytes = (
            InferenceService(**config).predict(PredictionRequest(**request)).to_bytes()
        )
        return payload
    if document["task"] == "blcs":
        from src.tasks.blcs.visualization.inference.service import (
            InferenceService as BLCSInferenceService,
        )

        result = BLCSInferenceService(**config).infer(**request)
        return json.dumps(result, allow_nan=False).encode("utf-8")
    if document["task"] in ("ball_detection", "court_detection"):
        from importlib import import_module

        module = import_module(
            f"src.tasks.{document['task']}.visualization.inference.service"
        )
        detection_result = module.DetectionService(**config).infer(**request)
        return json.dumps(detection_result, allow_nan=False).encode("utf-8")
    raise ValueError("Unknown inference task.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    try:
        document = json.loads(args.request.read_text(encoding="utf-8"))
        payload = execute_request(document)
        temporary = args.request.parent / "result.tmp"
        temporary.write_bytes(payload)
        temporary.replace(args.request.parent / "result.bin")
    except Exception as error:
        (args.request.parent / "error.json").write_text(
            json.dumps(
                {
                    "type": "ValueError"
                    if isinstance(error, ValueError)
                    else type(error).__name__,
                    "detail": str(error),
                }
            ),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
