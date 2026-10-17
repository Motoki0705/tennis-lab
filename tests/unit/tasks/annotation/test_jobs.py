from __future__ import annotations

import hashlib
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.tasks.annotation.contracts import (
    HistoryManifest,
    HistoryOperation,
    OperationInput,
    PredictOperation,
    Runtime,
    ServiceSessionOperation,
    StageCost,
)
from src.tasks.annotation.jobs import Operations
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef, ModelRef
from src.tasks.base.annotation.execution.contracts import (
    QueueConfiguration,
    StageLaunch,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import (
    CampaignPlan,
    CollectionCounts,
    ProductPlan,
    ResourceBudget,
    StageCapacity,
    TaskPlan,
    file_digest,
    references,
    save_plan,
)
from src.tasks.base.annotation.resources import (
    BudgetLimits,
    ResourceManager,
    ResourceRequest,
)
from src.tasks.base.annotation.resources.contracts import GPUUsage


@pytest.fixture
def operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Operations, ArtifactRef]:
    monkeypatch.setattr(
        "src.tasks.base.annotation.resources.scheduler.gpu_usage",
        lambda _pids: GPUUsage(
            status="unavailable",
            owned_vram_bytes=None,
            device_vram_bytes=None,
            maximum_utilization_percent=None,
            reason="synthetic job tests do not query a GPU",
        ),
    )
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    manifest = write_document(
        layout, layout.path("campaigns", "history.json"), HistoryManifest(entries=())
    )
    plan = CampaignPlan(
        campaign_id="backlog",
        phase="preview",
        annotation_root=str(layout.root),
        counts=CollectionCounts(
            channels=1, videos_per_channel=1, max_clips_per_video=1
        ),
        selection=manifest,
        preparation=manifest,
        code_commit="a" * 40,
        code_diff_digest="b" * 64,
        agent_model="gpt-6.1-sol",
        agent_effort="low",
        resources=ResourceBudget(
            codex_workers=1,
            additional_storage_bytes=64 * 1024**2,
            minimum_free_bytes=1,
            # Backlog recovery must have room for libraries retained by a
            # full-suite pytest worker, in addition to the stage reservation.
            total_pss_bytes=8 * 1024**3,
            gpu_seconds=60,
            stages=(
                StageCapacity(
                    stage="history", kind="io", parallel=1, threads_per_process=1
                ),
                StageCapacity(
                    stage="infer_ball", kind="gpu", parallel=1, threads_per_process=1
                ),
                StageCapacity(
                    stage="render", kind="cpu", parallel=1, threads_per_process=1
                ),
            ),
        ),
        tasks=(
            TaskPlan(
                task="ball",
                model=ModelRef(
                    task="ball",
                    version="fixture",
                    training_weights=manifest,
                    inference_weights=manifest,
                    configuration=manifest,
                ),
                inference=manifest,
                review=None,
                learning=None,
                products=(
                    ProductPlan(
                        product="frames",
                        dataset_task="ball_detection",
                        dataset_version="fixture",
                        component="frames",
                        base_snapshot=None,
                        generation_policy=manifest,
                    ),
                ),
            ),
        ),
    )
    resources = ResourceManager.create(
        layout,
        scope_id="backlog",
        limits=BudgetLimits.from_budget(plan.resources),
        roots=(layout.repository,),
        previously_allocated_bytes=0,
        previously_used_gpu_seconds=0,
        max_pending_per_stage=1,
    )
    script = tmp_path / "queue.sh"
    script.write_text("# synthetic --idempotency-key\n")
    runtime = Runtime(
        plan=annotation_reference(layout, save_plan(layout, plan)),
        scope=resources.reference,
        queue=QueueConfiguration(
            script=str(script),
            script_digest=file_digest(script),
            directory=str(tmp_path / ".training_queue"),
            session="fixture",
            issue=1088,
        ),
        codex_binary="codex",
        codex_home=str(layout.path("cache", "codex")),
        costs=tuple(
            StageCost(
                stage=stage.stage,
                pss_bytes=512 * 1024**2,
                storage_bytes=1024**2,
                timeout_seconds=15,
                gpu_seconds=30 if stage.kind == "gpu" else 0,
            )
            for stage in plan.resources.stages
        ),
    )
    return Operations(layout, runtime, plan, resources), manifest


@pytest.mark.parametrize(
    ("maximum_requests", "remaining_bytes", "expected_state"),
    [
        (2, 11 * 1024**2 // 10, "waiting"),
        (2, 2 * 1024**2 - 1, "waiting"),
        (2, 2 * 1024**2, "running"),
        (2, 2 * 1024**2 + 1, "running"),
        (128, 64 * 1024**2, "waiting"),
    ],
)
def test_session_admission_reserves_every_request_at_storage_boundary(
    operations: tuple[Operations, ArtifactRef],
    monkeypatch: pytest.MonkeyPatch,
    maximum_requests: int,
    remaining_bytes: int,
    expected_state: str,
) -> None:
    jobs, _ = operations
    data = OperationInput(
        plan=jobs.runtime.plan,
        operation=ServiceSessionOperation(
            session=1,
            maximum_requests=maximum_requests,
            maximum_services_per_attempt=512,
            duration_seconds=3,
        ),
    )
    sample = jobs.resources.sample()
    sample = sample.model_copy(
        update={
            "storage": sample.storage.model_copy(
                update={
                    "additional_bytes": (
                        jobs.resources.scope.limits.additional_storage_bytes
                        - remaining_bytes
                    ),
                    "free_bytes": 1024**3,
                }
            )
        }
    )
    monkeypatch.setattr(jobs.resources, "_sample", lambda **_kwargs: sample)
    launches: list[StageLaunch] = []

    def admit(launch: StageLaunch) -> dict[str, Any]:
        launches.append(launch)
        # Exercise real admission without spawning a supervisor or using a GPU.
        with jobs.resources.lease(launch.resource):
            return {"state": "running"}

    monkeypatch.setattr(jobs.stages, "submit", admit)
    state = jobs.operation(data)
    assert state["state"] == expected_state, state
    cost = jobs.runtime.cost("render")
    assert launches[0].resource.storage_bytes == cost.storage_bytes * maximum_requests
    assert launches[0].resource.pss_bytes == cost.pss_bytes
    assert launches[0].timeout_seconds == cost.timeout_seconds
    if expected_state == "waiting":
        assert state["reason"] == "storage reservations do not fit"
        assert jobs.resources._row(launches[0].resource)["state"] == "pending"


@pytest.mark.parametrize(
    ("gpu", "finish_before_return"),
    [(False, False), (False, True), (True, False)],
    ids=["cpu", "cpu-completed-before-return", "gpu"],
)
def test_unaccepted_launch_resubmits_same_attempt_after_backlog_clears(
    operations: tuple[Operations, ArtifactRef],
    monkeypatch: pytest.MonkeyPatch,
    gpu: bool,
    finish_before_return: bool,
) -> None:
    jobs, manifest = operations
    data = OperationInput(
        plan=jobs.runtime.plan,
        operation=PredictOperation(task="ball", clips=(manifest,))
        if gpu
        else HistoryOperation(manifest=manifest),
    )
    cost = jobs.runtime.cost(data.stage)
    blocker = ResourceRequest(
        request_id="blocker",
        plan=data.plan,
        stage=data.stage,
        pss_bytes=cost.pss_bytes,
        storage_bytes=cost.storage_bytes,
        gpu_seconds=cost.gpu_seconds,
    )
    jobs.resources.submit(blocker)
    queued: list[ArtifactRef] = []
    cpu_launches: list[ArtifactRef] = []
    supervisors: list[subprocess.Popen[bytes]] = []
    original_popen = subprocess.Popen

    def assert_admitted(
        layout: AnnotationLayout, reference: ArtifactRef, state: str
    ) -> None:
        launch = read_document(layout, reference, StageLaunch)
        assert jobs.resources._row(launch.resource)["state"] == state
        retained = {
            ref
            for protection in jobs.resources.storage.catalog.snapshot().protections
            for ref in protection.references
        }
        assert {reference, *references(launch)} <= retained

    def start_cpu(command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        assert command[1:3] == ["-m", "src.tasks.base.annotation.execution"]
        reference = annotation_reference(
            jobs.layout, Path(command[command.index("--launch") + 1])
        )
        # Check admission before the child can advance running -> draining.
        assert_admitted(jobs.layout, reference, "running")
        cpu_launches.append(reference)
        process = original_popen(command, **kwargs)
        supervisors.append(process)
        if finish_before_return:
            # Force the valid schedule where the child finishes before submit returns.
            assert process.wait(timeout=20) == 0
            assert jobs.resources._row(launch.resource)["state"] == "draining"
        return process

    def enqueue(
        layout: AnnotationLayout,
        config: QueueConfiguration,
        reference: ArtifactRef,
        code: Path,
    ) -> str:
        assert_admitted(layout, reference, "pending")
        queued.append(reference)
        key = hashlib.sha256(
            ("annotation-stage:" + reference.digest).encode()
        ).hexdigest()
        registration = (
            Path(config.directory) / "control/registrations" / (key + ".receipt")
        )
        registration.parent.mkdir(parents=True)
        registration.write_text("a" * 64 + " synthetic.job\n")
        pending = Path(config.directory) / "jobs/synthetic.job"
        pending.parent.mkdir()
        pending.write_text("synthetic queue entry; never executed")
        return "synthetic.job"

    monkeypatch.setattr("src.tasks.base.annotation.execution.runner.enqueue", enqueue)
    monkeypatch.setattr(
        "src.tasks.base.annotation.execution.runner.subprocess.Popen", start_cpu
    )
    first = jobs.operation(data)
    assert first["state"] == "waiting" and "backlog" in first["reason"]
    directory = jobs.layout.path(
        "campaigns", jobs.plan.campaign_id, "jobs", data.sha256 + "-1"
    )
    launch_ref = annotation_reference(jobs.layout, directory / "launch.json")
    launch = read_document(jobs.layout, launch_ref, StageLaunch)
    with pytest.raises(ValueError, match="missing"):
        jobs.resources._row(launch.resource)
    assert queued == cpu_launches == [] and jobs.stages.children == {}
    assert jobs.operation(data)["state"] == "waiting"
    jobs.resources.cancel_pending(blocker, reason="release synthetic backlog slot")

    # Reconstruct the controller to exercise recovery from persisted launch only.
    jobs = Operations(jobs.layout, jobs.runtime, jobs.plan, jobs.resources)
    try:
        resumed = jobs.operation(data)
        assert resumed["state"] == ("queued" if gpu else "running"), resumed
        assert ArtifactRef.model_validate(resumed["launch"]) == launch_ref
        if gpu:
            assert jobs.resources._row(launch.resource)["state"] == "pending"
        else:
            assert len(supervisors) == 1
            assert supervisors[0].wait(timeout=20) == 0
            resumed = jobs.operation(data)
            assert resumed["state"] == "succeeded", resumed
            assert jobs.receipt(resumed).status == "complete"
        assert jobs.operation(data) == resumed
        assert queued == ([launch_ref] if gpu else [])
        assert cpu_launches == ([] if gpu else [launch_ref])
        assert list(directory.parent.glob(data.sha256 + "-*")) == [directory]
    finally:
        for process in supervisors:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
