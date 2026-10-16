from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pytest

from src.tasks.annotation.contracts import (
    HistoryManifest,
    HistoryOperation,
    OperationInput,
    PredictOperation,
    Runtime,
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


@pytest.fixture
def operations(tmp_path: Path) -> tuple[Operations, ArtifactRef]:
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
            total_pss_bytes=2 * 1024**3,
            gpu_seconds=60,
            stages=(
                StageCapacity(
                    stage="history", kind="io", parallel=1, threads_per_process=1
                ),
                StageCapacity(
                    stage="infer_ball", kind="gpu", parallel=1, threads_per_process=1
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


@pytest.mark.parametrize("gpu", [False, True], ids=["cpu", "gpu"])
def test_unaccepted_launch_resubmits_same_attempt_after_backlog_clears(
    operations: tuple[Operations, ArtifactRef],
    monkeypatch: pytest.MonkeyPatch,
    gpu: bool,
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

    def enqueue(
        layout: AnnotationLayout,
        config: QueueConfiguration,
        reference: ArtifactRef,
        code: Path,
    ) -> str:
        launch = read_document(layout, reference, StageLaunch)
        assert jobs.resources._row(launch.resource)["state"] == "pending"
        retained = {
            ref
            for protection in jobs.resources.storage.catalog.snapshot().protections
            for ref in protection.references
        }
        assert {reference, *references(launch)} <= retained
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
    first = jobs.operation(data)
    assert first["state"] == "waiting" and "backlog" in first["reason"]
    directory = jobs.layout.path(
        "campaigns", jobs.plan.campaign_id, "jobs", data.sha256 + "-1"
    )
    launch_ref = annotation_reference(jobs.layout, directory / "launch.json")
    launch = read_document(jobs.layout, launch_ref, StageLaunch)
    with pytest.raises(ValueError, match="missing"):
        jobs.resources._row(launch.resource)
    assert queued == [] and jobs.stages.children == {}
    assert jobs.operation(data)["state"] == "waiting"
    jobs.resources.cancel_pending(blocker, reason="release synthetic backlog slot")

    # Reconstruct the controller to exercise recovery from persisted launch only.
    jobs = Operations(jobs.layout, jobs.runtime, jobs.plan, jobs.resources)
    resumed = jobs.operation(data)
    assert resumed["state"] == ("queued" if gpu else "running")
    assert ArtifactRef.model_validate(resumed["launch"]) == launch_ref
    assert jobs.resources._row(launch.resource)["state"] == (
        "pending" if gpu else "running"
    )
    deadline = time.monotonic() + 20
    while (
        not gpu
        and resumed["state"] in {"running", "draining"}
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
        resumed = jobs.operation(data)
    assert resumed["state"] == ("queued" if gpu else "succeeded"), resumed
    if not gpu:
        assert jobs.receipt(resumed).status == "complete"
    assert jobs.operation(data) == resumed
    assert queued == ([launch_ref] if gpu else [])
    assert list(directory.parent.glob(data.sha256 + "-*")) == [directory]
