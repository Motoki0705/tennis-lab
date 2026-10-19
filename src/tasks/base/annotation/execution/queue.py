"""Use the shared queue's idempotent registration and its existing process group."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from src.utils.training_queue import expired_completion

from ..artifacts import verify_reference
from ..contracts import ArtifactRef
from ..paths import AnnotationLayout
from ..plan import file_digest
from ..workers.processes import ProcessIdentity, capture_supervisor
from .contracts import QueueConfiguration


def validate_queue(layout: AnnotationLayout, config: QueueConfiguration) -> Path:
    directory, script = Path(config.directory), Path(config.script)
    if directory != layout.repository / ".training_queue":
        raise ValueError("all worktrees must share the main repository training queue")
    if not script.is_absolute() or not script.is_relative_to(layout.repository):
        raise ValueError("queue script must be an absolute path inside the repository")
    if script.resolve() != script or file_digest(script) != config.script_digest:
        raise ValueError("pinned queue script changed")
    if directory.resolve() != directory:
        raise ValueError("shared queue directory must not contain symlinks")
    if "--idempotency-key" not in script.read_text():
        raise ValueError("the pinned training queue lacks idempotent registration")
    return script


def enqueue(
    layout: AnnotationLayout,
    config: QueueConfiguration,
    launch: ArtifactRef,
    code_root: Path,
) -> str:
    script = validate_queue(layout, config)
    path = verify_reference(layout, launch)
    command = [
        str(code_root / ".venv/bin/python"),
        "-m",
        "src.tasks.base.annotation.execution",
        "--launch",
        str(path),
        "--sha256",
        launch.digest,
        "--queued",
    ]
    if layout.mode == "test":
        command += ["--test-root", str(layout.root)]
    environment = os.environ.copy()
    environment["TRAINING_QUEUE_DIR"] = str(config.directory)
    key = "annotation-stage:" + launch.digest
    subprocess.run(
        [
            "bash",
            str(script),
            "add",
            shlex.join(command),
            "--name",
            "annotation-" + launch.digest[:16],
            "--provider",
            config.provider,
            "--session",
            config.session,
            "--issue",
            str(config.issue),
            "--resource",
            "all",
            "--idempotency-key",
            key,
        ],
        cwd=code_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return registration_job(config, launch)


def registration_job(config: QueueConfiguration, launch: ArtifactRef) -> str:
    key = "annotation-stage:" + launch.digest
    registration = (
        Path(config.directory)
        / "control"
        / "registrations"
        / (hashlib.sha256(key.encode()).hexdigest() + ".receipt")
    )
    # The queue's receipt is a request digest and filename, never executable shell.
    fields = registration.read_text().split()
    job = fields[1] if len(fields) == 2 else ""
    if (
        len(fields) != 2
        or not re.fullmatch(r"[0-9a-f]{64}", fields[0])
        or not re.fullmatch(r"[A-Za-z0-9_.-]+\.job", job)
    ):
        raise ValueError("queue did not publish a valid immutable registration")
    return job


def queue_state(config: QueueConfiguration, launch: ArtifactRef) -> dict[str, str]:
    try:
        job = registration_job(config, launch)
    except FileNotFoundError:
        return {
            "state": "pending",
            "reason": "launch has not been registered in the queue",
        }
    directory = Path(config.directory)
    for _ in range(3):
        # Atomic renames can appear in both directories (or neither on requeue).
        # Retry ambiguous observations; the bound limits polling, not job moves.
        states = [
            state
            for state in ("jobs", "running", "done", "failed", "cancelled")
            if (directory / state / job).exists()
        ]
        if len(states) == 1:
            break
    else:
        completion = expired_completion(directory, job)
        if not states and completion is not None:
            state = completion["state"]
            return {
                "state": "cancelled" if state == "cancelled" else "failed",
                "reason": f"queue is {state}; history expired but the stage did not publish completion",
                "history": "expired",
            }
        return {
            "state": "unfinished",
            "reason": "queue registration has no unique lifecycle state during observation; inspect queue if this persists",
        }
    state = states[0]
    if state in {"done", "failed", "cancelled"}:
        return {
            "state": "failed" if state != "cancelled" else state,
            "reason": f"queue is {state} but the stage did not publish completion",
        }
    return {"state": "queued" if state == "jobs" else "running", "queue_job": job}


def queue_owner(
    layout: AnnotationLayout, config: QueueConfiguration
) -> ProcessIdentity:
    validate_queue(layout, config)
    job_id = os.environ.get("TENNIS_RUN_ID", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
        raise ValueError("GPU stage requires a training queue job identity")
    repro = Path(config.directory) / "repro" / job_id
    if (
        os.environ.get("TENNIS_REPRO_DIR") != str(repro)
        or os.environ.get("TENNIS_GPU_RESOURCE") != "all"
    ):
        raise ValueError("GPU stage is not running in the declared queue allocation")
    record = json.loads((repro / "run.json").read_bytes())
    if (
        record.get("provider") != config.provider
        or record.get("session") != config.session
        or record.get("resource") != "all"
    ):
        raise ValueError("queue attribution differs from the pinned launch")
    owner = capture_supervisor(os.getpgrp())
    if os.getsid(0) != owner.session:
        raise ValueError("GPU stage escaped the queue's private session")
    return owner
