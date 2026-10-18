"""Admit before CPU launch; GPU stages acquire resources inside the shared queue."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from src.utils.paths import PROJECT_ROOT

from ..artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_document,
)
from ..contracts import ArtifactRef
from ..paths import AnnotationLayout
from ..plan import load_plan, references
from ..reproducibility import require_frozen_code
from ..resources import BudgetStopped, ResourceManager
from .contracts import StageExit, StageLaunch
from .queue import enqueue


def stage_directory(layout: AnnotationLayout, launch: StageLaunch) -> Path:
    plan = load_plan(verify_reference(layout, launch.resource.plan))
    result: Path = layout.path(
        "campaigns", plan.campaign_id, "jobs", launch.resource.request_id
    )
    return result


def stage_lock(launch: StageLaunch) -> str:
    result: str = "stage-" + launch.sha256
    return result


def validate_launch(layout: AnnotationLayout, launch: StageLaunch) -> None:
    plan = load_plan(verify_reference(layout, launch.resource.plan))
    for reference in references(launch):
        verify_reference(layout, reference)
    gpu = launch.resource.gpu_seconds > 0
    if gpu != (launch.queue is not None):
        raise ValueError("GPU stages must use the shared queue, CPU stages must not")
    if gpu and launch.timeout_seconds + 10 > launch.resource.gpu_seconds:
        raise ValueError("GPU reservation must include ten seconds of teardown margin")
    if Path(launch.code_root) != PROJECT_ROOT:
        raise ValueError("stage must execute the pinned source worktree")
    if layout.mode == "production":
        if launch.test_root is not None:
            raise ValueError("test root is forbidden in production")
        require_frozen_code(
            PROJECT_ROOT,
            plan,
            layout=layout,
            revision=launch.code_revision,
            scope=launch.scope,
        )
    elif launch.test_root != str(layout.root):
        raise ValueError("test stages must explicitly pin their isolated root")


class StageRunner:
    def __init__(self, layout: AnnotationLayout) -> None:
        self.layout = layout
        self.children: dict[str, subprocess.Popen[bytes]] = {}

    def submit(self, launch: StageLaunch) -> dict[str, Any]:
        validate_launch(self.layout, launch)
        manager = ResourceManager(self.layout, launch.scope)
        directory = stage_directory(self.layout, launch)
        if (directory / "exit.json").exists():
            ref = annotation_reference(self.layout, directory / "launch.json")
            if read_document(self.layout, ref, StageLaunch) != launch:
                raise ValueError(
                    "stage request identity already contains a different launch"
                )
            manager.submit(launch.resource, inputs=(ref, *references(launch)))
            return self.status(ref)
        with artifact_lock(self.layout, stage_lock(launch), blocking=False) as lock_fd:
            ref = write_document(self.layout, directory / "launch.json", launch)
            manager.submit(launch.resource, inputs=(ref, *references(launch)))
            if (directory / "exit.json").exists():
                return self.status(ref)
            manager.observe(os.getpid())
            sample = manager.poll()
            if sample.stopped_reason is not None:
                raise BudgetStopped(sample.stopped_reason)
            if launch.queue is not None:
                job = enqueue(self.layout, launch.queue, ref, PROJECT_ROOT)
                return {"state": "queued", "launch": ref.model_dump(), "queue_job": job}
            with manager.lease(launch.resource) as resource_fd:
                command = [
                    str(PROJECT_ROOT / ".venv/bin/python"),
                    "-m",
                    "src.tasks.base.annotation.execution",
                    "--launch",
                    str(self.layout.artifact_path(ref)),
                    "--sha256",
                    ref.digest,
                    "--lock-fd",
                    str(lock_fd),
                    "--resource-fd",
                    str(resource_fd),
                ]
                if self.layout.mode == "test":
                    command += ["--test-root", str(self.layout.root)]
                with (directory / "supervisor.log").open("ab") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=PROJECT_ROOT,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                        pass_fds=(lock_fd, resource_fd),
                    )
                self.children[ref.digest] = process
                return {
                    "state": "running",
                    "launch": ref.model_dump(),
                    "pid": process.pid,
                }

    def _completion(self, ref: ArtifactRef, path: Path) -> dict[str, Any]:
        exit_ref = annotation_reference(self.layout, path)
        result = read_document(self.layout, exit_ref, StageExit)
        if result.launch != ref:
            raise ValueError("stage result belongs to another launch")
        for dependency in references(result):
            verify_reference(self.layout, dependency)
        child = self.children.get(ref.digest)
        if child is not None and child.poll() is not None:
            child.wait()
            self.children.pop(ref.digest)
        return {
            "state": result.status,
            "launch": ref.model_dump(),
            "exit": exit_ref.model_dump(),
            "reason": result.reason,
        }

    def status(self, ref: ArtifactRef) -> dict[str, Any]:
        launch = read_document(self.layout, ref, StageLaunch)
        path = stage_directory(self.layout, launch) / "exit.json"
        try:
            with artifact_lock(self.layout, stage_lock(launch), blocking=False):
                # The supervisor writes its receipt before releasing this lock.
                # Check it here before declaring a dead child/queue job failed.
                if path.exists():
                    return self._completion(ref, path)
                child = self.children.get(ref.digest)
                if child is not None and child.poll() is not None:
                    code = child.wait()
                    self.children.pop(ref.digest)
                    return {
                        "state": "failed",
                        "launch": ref.model_dump(),
                        "reason": f"supervisor exited with code {code} without a completion receipt",
                    }
                if launch.queue is not None:
                    from .queue import queue_state

                    return {
                        "launch": ref.model_dump(),
                        **queue_state(launch.queue, ref),
                    }
                manager = ResourceManager(self.layout, launch.scope)
                with manager.registry.transaction() as db:
                    row = db.execute(
                        "SELECT state,reason FROM resource_requests WHERE id=?",
                        (manager.key(launch.resource),),
                    ).fetchone()
                if row is not None and row["state"] == "pending":
                    return {
                        "state": "pending",
                        "launch": ref.model_dump(),
                        "reason": row["reason"],
                    }
                if child is not None:
                    # Popen and inherited-lock observation can race with process exit.
                    return {"state": "running", "launch": ref.model_dump()}
                return {
                    "state": "interrupted",
                    "launch": ref.model_dump(),
                    "reason": "no owning supervisor or completion receipt; inspect logs before an explicit new attempt",
                }
        except BlockingIOError:
            if path.exists():
                return {
                    **self._completion(ref, path),
                    "state": "draining",
                    "reason": "completion recorded; supervisor still owns its resource lease",
                }
            if launch.queue is not None:
                manager = ResourceManager(self.layout, launch.scope)
                state, reason = manager.request_status(launch.resource)
                if state == "pending":
                    return {
                        "state": "waiting",
                        "launch": ref.model_dump(),
                        "reason": reason or "waiting for queued resource admission",
                    }
            return {"state": "running", "launch": ref.model_dump()}
