from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from src.tasks.annotation.contracts import Runtime, StageCost
from src.tasks.annotation.controller import Controller
from src.tasks.annotation.reviews import close_reviews, requests, validate_closed
from src.tasks.ball_detection.annotation.inference import BallPredictionAdapter
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
    BallPublicationPolicy,
    SourceSplit,
)
from src.tasks.base.annotation.artifacts import (
    read_document,
    write_document,
)
from src.tasks.base.annotation.collection.contracts import (
    PreparationConfig,
    PreparationReceipt,
)
from src.tasks.base.annotation.contracts import content_digest
from src.tasks.base.annotation.execution.contracts import QueueConfiguration
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, StageCapacity
from src.tasks.base.annotation.predictions import Predictions
from src.tasks.base.annotation.resources import BudgetLimits, ResourceManager
from src.tasks.base.annotation.workers.contracts import WorkRequest
from tests.support.tasks.ball_detection.review_worker import ReviewSession


@pytest.fixture
def session(tmp_path: Path, request: pytest.FixtureRequest) -> ReviewSession:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    preparation = write_document(
        layout,
        layout.path("campaigns", "inputs", "preparation.json"),
        PreparationConfig(
            format_selector="fixture",
            crf=16,
            preset="ultrafast",
            minimum_clip_frames=32,
            maximum_unprepared_sources=1,
        ),
    )

    configured_learning = getattr(request, "param", False)

    def transform(plan: CampaignPlan) -> CampaignPlan:
        from src.tasks.base.annotation.plan import LearningPlan

        task = plan.tasks[0]
        if configured_learning:
            # The orchestration test replaces the learning boundary, not model execution.
            task = task.model_copy(
                update={
                    "learning": LearningPlan(
                        recipe=preparation,
                        validation=preparation,
                        selection_policy=preparation,
                    )
                }
            )
        return plan.model_copy(
            update={
                "tasks": (task,),
                "preparation": preparation,
                "resources": plan.resources.model_copy(
                    update={
                        "codex_workers": 2,
                        "total_pss_bytes": 4 * 1024**3,
                        "stages": tuple(
                            StageCapacity(
                                stage=name, kind=kind, parallel=1, threads_per_process=1
                            )
                            for name, kind in (
                                ("screen", "codex"),
                                ("review", "codex"),
                                ("render", "cpu"),
                                ("publish_ball", "cpu"),
                                ("bootstrap_model", "io"),
                                *(
                                    (name, "gpu")
                                    for name in ("train_ball", "evaluate_ball")
                                    if configured_learning
                                ),
                            )
                        ),
                    }
                ),
            }
        )

    result = ReviewSession(
        layout.root,
        plan_transform=transform,
        publication_policy=BallPublicationPolicy(
            jpeg_quality=95,
            source_splits=(
                SourceSplit(
                    source_id=content_digest(["local", "fixture"]), split="train"
                ),
            ),
        ),
    )
    env = result.env
    receipt = env.save(
        PreparationReceipt(
            source_id=env.manifest.source.canonical_id,
            source_digest=env.manifest.media.digest,
            selection=env.blob,
            configuration=preparation,
            clips=(env.manifest_ref,),
        )
    )
    with result.ledger.registry.transaction() as database:
        database.execute(
            "UPDATE sources SET preparation=? WHERE id=?",
            (receipt.model_dump_json(), env.manifest.source.canonical_id),
        )
        database.execute(
            "INSERT INTO channels VALUES ('local','fixture',?)", (env.plan.campaign_id,)
        )
    return result


def finish(session: ReviewSession, *, deferred: bool = False) -> None:
    image = session.image(1, 2)
    session.confirm(image, 1, 3)
    report = session.evidence(shown=(image,))
    session.finish(report, status="deferred" if deferred else "complete")


def controller_for(session: ReviewSession) -> Controller:
    env = session.env
    scope = ResourceManager.create(
        env.layout,
        scope_id="production",
        limits=BudgetLimits.from_budget(env.plan.resources),
        roots=(env.layout.repository,),
        previously_allocated_bytes=0,
        previously_used_gpu_seconds=0,
        max_pending_per_stage=10,
    )
    script = env.layout.repository / "queue.sh"
    script.write_text("# synthetic queue --idempotency-key\n")
    from src.tasks.base.annotation.plan import file_digest

    runtime = Runtime(
        plan=env.plan_ref,
        scope=scope.reference,
        queue=QueueConfiguration(
            script=str(script),
            script_digest=file_digest(script),
            directory=str(env.layout.repository / ".training_queue"),
            session="fixture",
            issue=1088,
        ),
        codex_binary="codex",
        codex_home=str(env.layout.path("cache", "codex")),
        costs=tuple(
            StageCost(
                stage=stage.stage,
                pss_bytes=512 * 1024**2,
                storage_bytes=10 * 1024**2,
                timeout_seconds=20.0,
                gpu_seconds=30.0 if stage.kind == "gpu" else 0.0,
            )
            for stage in env.plan.resources.stages
        ),
    )
    return Controller(env.layout, runtime)


def test_prediction_query_and_review_request_preserve_canonical_context(
    session: ReviewSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = session.env

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("query must not infer")

    monkeypatch.setattr(BallPredictionAdapter, "produce", forbidden)
    completed = Predictions(env.layout, env.plan).completed(BallPredictionAdapter())
    assert completed == (env.context.prediction,)
    first = requests(env.layout, env.plan, env.plan_ref)
    assert first == requests(env.layout, env.plan, env.plan_ref)
    assert read_document(env.layout, first[0], WorkRequest) == session.request
    assert close_reviews(env.layout, env.plan, env.plan_ref) is None
    finish(session)
    assert (
        Predictions(env.layout, env.plan).completed(BallPredictionAdapter())
        == completed
    )
    closed = close_reviews(env.layout, env.plan, env.plan_ref)
    assert closed is not None
    assert validate_closed(env.layout, env.plan, env.plan_ref, closed) == (
        env.store.current(env.context_ref, env.adapter)[1],
    )


def test_deferred_review_cannot_close_or_publish(session: ReviewSession) -> None:
    finish(session, deferred=True)
    env = session.env
    assert close_reviews(env.layout, env.plan, env.plan_ref) is None
    controller = controller_for(session)
    assert controller.production is not None
    assert controller.production.cycle()["state"] == "blocked"
    assert not env.layout.path(
        "campaigns", env.plan.campaign_id, "workflow", "completion.json"
    ).exists()


def test_missing_prediction_and_source_count_cannot_look_complete(
    session: ReviewSession,
) -> None:
    env = session.env
    finish(session)
    with session.ledger.registry.transaction() as database:
        database.execute("DELETE FROM annotation_predictions")
    with pytest.raises(ValueError, match="no prediction receipt"):
        Predictions(env.layout, env.plan).completed(BallPredictionAdapter())
    missing = env.plan.model_copy(
        update={"counts": env.plan.counts.model_copy(update={"videos_per_channel": 2})}
    )
    # Counts are checked before trying to close a subset of the requested work.
    from src.tasks.annotation.reviews import sources

    assert sources(env.layout, missing) is None


def test_closed_review_rejects_later_preparation_change(session: ReviewSession) -> None:
    env = session.env
    finish(session)
    closed = close_reviews(env.layout, env.plan, env.plan_ref)
    assert closed is not None
    other = env.save(
        PreparationReceipt(
            source_id=env.manifest.source.canonical_id,
            source_digest=env.manifest.media.digest,
            selection=env.blob,
            configuration=env.plan.preparation,
            clips=(),
        )
    )
    with session.ledger.registry.transaction() as database:
        database.execute("UPDATE sources SET preparation=?", (other.model_dump_json(),))
    with pytest.raises(ValueError, match="source/clip set"):
        validate_closed(env.layout, env.plan, env.plan_ref, closed)


def test_publication_uses_all_adoptions_and_resumes_without_reencoding(
    session: ReviewSession,
) -> None:
    finish(session)
    env = session.env
    controller = controller_for(session)
    assert controller.production is not None
    deadline = time.monotonic() + 40
    progress = controller.cycle()["workflow"]
    while progress["state"] == "publishing" and time.monotonic() < deadline:
        time.sleep(0.1)
        progress = controller.cycle()["workflow"]
    assert progress["state"] == "blocked", progress
    assert progress["learning"]["status"] == "not_configured"
    snapshot = (
        env.layout.repository
        / "data/ball_detection"
        / env.plan.tasks[0].products[0].dataset_version
        / "frames/snapshot.json"
    )
    assert snapshot.exists()
    inode = snapshot.stat().st_ino
    # Recreate the entire controller: only immutable receipts and ledgers survive.
    restored = Controller(env.layout, controller.runtime)
    assert restored.cycle()["workflow"] == progress
    assert snapshot.stat().st_ino == inode
    status = restored.status()
    assert status["state"] == "blocked"
    assert len(status["stages"]) == 1
    assert status["stages"][0]["state"] == "succeeded"
    assert controller.stages.children == {}


@pytest.mark.parametrize("session", [True], indirect=True)
def test_orchestration_passes_verified_completion_and_frozen_learning_runtime(
    session: ReviewSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.ball_detection.training.annotation.adapter import BallLearningAdapter
    from src.tasks.base.annotation.learning.contracts import (
        CampaignCompletion,
        LearningOutcome,
        LearningRegistration,
    )

    finish(session)
    env = session.env
    controller = controller_for(session)
    production = controller.production
    assert production is not None
    calls = []
    outcome = "training_pending"

    def register(completion_ref: Any, adapter: Any) -> LearningRegistration:
        completion = read_document(env.layout, completion_ref, CampaignCompletion)
        assert completion.plan == env.plan_ref and completion.complete
        assert completion.tasks[0].products[0].publication is not None
        assert isinstance(adapter, BallLearningAdapter)
        return LearningRegistration(
            task="ball",
            status="ready",
            intent=env.blob,
            reason="fixture learning boundary",
        )

    def tick(intent: Any, adapter: Any, runtime: Any) -> LearningOutcome:
        assert intent == env.blob and isinstance(adapter, BallLearningAdapter)
        assert runtime.scope == controller.runtime.scope
        assert runtime.queue == controller.runtime.queue
        assert runtime.train.gpu_seconds == runtime.evaluate.gpu_seconds == 30.0
        calls.append(intent)
        return LearningOutcome(
            intent=intent,
            state=outcome,
            stage="train",
            reason="fixture learning result",
        )

    monkeypatch.setattr(production.learning, "register", register)
    monkeypatch.setattr(production.learning, "tick", tick)
    deadline = time.monotonic() + 40
    progress = controller.cycle()["workflow"]
    while progress["state"] == "publishing" and time.monotonic() < deadline:
        time.sleep(0.1)
        progress = controller.cycle()["workflow"]
    assert progress["state"] == "learning", progress
    assert controller.status()["state"] == "running"
    outcome = "adopted"
    assert controller.cycle()["workflow"]["state"] == "complete"
    assert controller.status()["state"] == "complete"
    assert controller.cycle()["workflow"]["state"] == "complete"
    assert calls == [env.blob] * 3
    assert (
        len(controller.status()["stages"]) == 1
    )  # No training or GPU process is submitted.


def test_worker_recovery_pins_failed_attempt_and_replays_receipt(
    session: ReviewSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.annotation.contracts import (
        RecoveryInput,
        RecoveryReceipt,
        RetryWorker,
    )
    from src.tasks.annotation.recovery import recover

    finish(session, deferred=True)
    controller = controller_for(session)
    assert controller.production is not None
    request = requests(session.env.layout, session.env.plan, session.env.plan_ref)[0]
    after = session.ledger.task(session.request.work_id)["current_attempt"]
    recovery = RecoveryInput(
        plan=session.env.plan_ref,
        reason="inspect deferred case again",
        operation=RetryWorker(request=request, after_attempt=after),
    )
    launched = []

    def start(work: Any, *, retry: bool) -> str:
        assert work == session.request and retry
        session.open(2)
        attempt = session.ledger.task(work.work_id)["current_attempt"]
        launched.append(attempt)
        return str(attempt)

    monkeypatch.setattr(controller.production.pool, "start", start)
    result = recover(controller, recovery)
    assert (
        read_document(session.env.layout, result, RecoveryReceipt).worker_attempt
        == launched[0]
    )
    assert recover(controller, recovery) == result
    assert len(launched) == 1
    stale = recovery.model_copy(
        update={"reason": "a different request for the old attempt"}
    )
    with pytest.raises(ValueError, match="attempt changed"):
        recover(controller, stale)
    with pytest.raises(ValueError, match="current plan"):
        recover(controller, recovery.model_copy(update={"plan": session.env.blob}))


def test_successful_review_cannot_be_retried(session: ReviewSession) -> None:
    from src.tasks.annotation.contracts import RecoveryInput, RetryWorker
    from src.tasks.annotation.recovery import recover

    finish(session)
    controller = controller_for(session)
    request = requests(session.env.layout, session.env.plan, session.env.plan_ref)[0]
    after = session.ledger.task(session.request.work_id)["current_attempt"]
    with pytest.raises(ValueError, match="stopped or failed"):
        recover(
            controller,
            RecoveryInput(
                plan=session.env.plan_ref,
                reason="must reject",
                operation=RetryWorker(request=request, after_attempt=after),
            ),
        )


def test_source_and_product_guards_precede_publication(session: ReviewSession) -> None:
    from src.tasks.annotation.reviews import require_supported

    env = session.env
    finish(session)
    task = env.plan.tasks[0]
    with pytest.raises(ValueError, match="ball frames only"):
        require_supported(
            env.plan.model_copy(
                update={"tasks": (task.model_copy(update={"task": "player"}),)}
            )
        )
    missing = env.save(
        PreparationReceipt(
            source_id=env.manifest.source.canonical_id,
            source_digest=env.manifest.media.digest,
            selection=env.blob,
            configuration=env.plan.preparation,
            clips=(),
        )
    )
    with session.ledger.registry.transaction() as database:
        database.execute(
            "UPDATE sources SET preparation=?", (missing.model_dump_json(),)
        )
    with pytest.raises(ValueError, match="every prepared clip"):
        close_reviews(env.layout, env.plan, env.plan_ref)


def test_explicit_operation_retry_keeps_history_without_stale_failure(
    session: ReviewSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.annotation.contracts import BootstrapModelOperation, OperationInput
    from src.tasks.base.annotation.contracts import ArtifactRef
    from src.tasks.base.annotation.execution.contracts import StageLaunch

    controller = controller_for(session)
    env = session.env
    operation = OperationInput(
        plan=env.plan_ref,
        operation=BootstrapModelOperation(task="ball", evidence=env.blob),
    )
    states: dict[str, str] = {}

    def status(ref: ArtifactRef) -> dict[str, Any]:
        launch = read_document(env.layout, ref, StageLaunch)
        return {
            "state": states[launch.resource.request_id],
            "launch": ref.model_dump(mode="json"),
            "reason": "synthetic supervisor result",
        }

    def submit(launch: StageLaunch) -> dict[str, Any]:
        directory = env.layout.path(
            "campaigns", env.plan.campaign_id, "jobs", launch.resource.request_id
        )
        ref = write_document(env.layout, directory / "launch.json", launch)
        states[launch.resource.request_id] = "running"
        return status(ref)

    monkeypatch.setattr(controller.stages, "submit", submit)
    monkeypatch.setattr(controller.stages, "status", status)
    first = controller.operation(operation)
    assert first["state"] == "running"
    with pytest.raises(ValueError, match="failed or interrupted"):
        controller.operation(operation, attempt=2)
    states[operation.sha256 + "-1"] = "failed"
    assert controller.operation(operation)["state"] == "failed"
    with pytest.raises(ValueError, match="follow the latest"):
        controller.operation(operation, attempt=3)
    assert controller.operation(operation, attempt=2)["state"] == "running"
    states[operation.sha256 + "-2"] = "succeeded"
    assert controller.operation(operation)["state"] == "succeeded"
    reported = controller.status()
    assert reported["failures"] == []
    assert [(row["state"], row["superseded"]) for row in reported["stages"]] == [
        ("failed", True),
        ("succeeded", False),
    ]
    assert len(states) == 2


def test_bootstrap_is_explicit_and_pins_planned_model(
    session: ReviewSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.annotation.contracts import (
        BootstrapModelOperation,
        OperationInput,
        OperationReceipt,
    )
    from src.tasks.annotation.operations import execute
    from src.tasks.base.annotation.learning.contracts import ModelState
    from src.tasks.base.annotation.learning.models import ModelRegistry

    env = session.env
    controller_for(session)
    models = ModelRegistry(env.layout)
    with pytest.raises(ValueError, match="explicitly adopted"):
        models.current("ball")
    operation = OperationInput(
        plan=env.plan_ref,
        operation=BootstrapModelOperation(task="ball", evidence=env.blob),
    )
    reference = write_document(
        env.layout,
        env.layout.path(
            "campaigns",
            env.plan.campaign_id,
            "jobs",
            operation.sha256 + "-1",
            "operation.json",
        ),
        operation,
    )
    with pytest.raises(ValueError, match="resource-accounted"):
        execute(env.layout, reference)
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", reference.digest)
    receipt_ref = execute(env.layout, reference)
    receipt = read_document(env.layout, receipt_ref, OperationReceipt)
    state = read_document(env.layout, receipt.results[0], ModelState)
    assert state.model == env.plan.tasks[0].model == models.current("ball").model
    assert execute(env.layout, reference) == receipt_ref
    wrong = OperationInput(
        plan=env.plan_ref,
        operation=BootstrapModelOperation(task="player", evidence=env.blob),
    )
    wrong_ref = write_document(
        env.layout,
        env.layout.path(
            "campaigns",
            env.plan.campaign_id,
            "jobs",
            wrong.sha256 + "-1",
            "operation.json",
        ),
        wrong,
    )
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", wrong_ref.digest)
    with pytest.raises(ValueError, match="absent from the frozen plan"):
        execute(env.layout, wrong_ref)


def test_learning_recovery_checks_campaign_and_records_checkpoint(
    session: ReviewSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.annotation.contracts import (
        RecoveryInput,
        RecoveryReceipt,
        RetryLearning,
    )
    from src.tasks.annotation.recovery import recover
    from src.tasks.base.annotation.contracts import ProductOutcome
    from src.tasks.base.annotation.learning.contracts import (
        CampaignCompletion,
        DatasetPartition,
        LearningIntent,
        ModelState,
        TaskProducts,
    )

    env = session.env
    controller = controller_for(session)
    assert controller.production is not None
    completion = CampaignCompletion(
        plan=env.plan_ref,
        tasks=(
            TaskProducts(
                task="ball",
                products=(
                    ProductOutcome(
                        product="frames",
                        status="not_applicable",
                        publication=None,
                        reason="fixture",
                    ),
                ),
            ),
        ),
    )
    completion_ref = env.save(completion)
    intent = LearningIntent(
        completion=completion_ref,
        task="ball",
        publication=env.blob,
        train=DatasetPartition(
            manifest=env.blob,
            split="train",
            sources=(env.manifest.source.canonical_id,),
            clips=(env.manifest.clip_id,),
        ),
        parent=ModelState(generation=1, model=env.plan.tasks[0].model),
        recipe=env.blob,
        fixed_evaluation=env.blob,
        selection_policy=None,
    )
    intent_ref = env.save(intent)
    called = []

    def retry(
        ref: Any, adapter: Any, stage: Any, *, reason: str, resume_checkpoint: Any
    ) -> Any:
        called.append((ref, adapter.task, stage, reason, resume_checkpoint))
        return env.blob

    monkeypatch.setattr(controller.production.learning, "retry", retry)
    recovery = RecoveryInput(
        plan=env.plan_ref,
        reason="resume the same failed intent",
        operation=RetryLearning(
            intent=intent_ref, stage="train", resume_checkpoint=env.blob
        ),
    )
    receipt_ref = recover(controller, recovery)
    assert (
        read_document(env.layout, receipt_ref, RecoveryReceipt).learning_request
        == env.blob
    )
    assert recover(controller, recovery) == receipt_ref
    assert called == [(intent_ref, "ball", "train", recovery.reason, env.blob)]
    wrong_completion = env.save(completion.model_copy(update={"plan": env.blob}))
    wrong_intent = env.save(intent.model_copy(update={"completion": wrong_completion}))
    wrong = recovery.model_copy(
        update={"operation": RetryLearning(intent=wrong_intent, stage="train")}
    )
    with pytest.raises(ValueError, match="another campaign/task"):
        recover(controller, wrong)
    assert len(called) == 1


def test_cli_resumes_only_the_named_worker_campaign(session: ReviewSession) -> None:
    import json
    import subprocess
    import sys

    from src.tasks.annotation.contracts import (
        RecoveryInput,
        RecoveryReceipt,
        ResumeWorkers,
    )
    from src.tasks.base.annotation.contracts import ArtifactRef
    from src.utils.paths import PROJECT_ROOT

    env = session.env
    controller = controller_for(session)
    runtime_ref = env.save(controller.runtime)
    recovery = RecoveryInput(
        plan=env.plan_ref, reason="synthetic quota recovery", operation=ResumeWorkers()
    )
    input_ref = env.save(recovery)
    with session.ledger.registry.transaction() as database:
        database.executemany(
            "INSERT INTO worker_campaign_stops VALUES (?,?)",
            ((env.plan.campaign_id, "fixture stop"), ("unrelated", "keep this stop")),
        )
    command = [
        sys.executable,
        "-m",
        "src.tasks.annotation",
        "recover",
        "--runtime",
        str(env.layout.artifact_path(runtime_ref)),
        "--sha256",
        runtime_ref.digest,
        "--operation",
        str(env.layout.artifact_path(input_ref)),
        "--test-root",
        str(env.layout.root),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    result_ref = ArtifactRef.model_validate(json.loads(completed.stdout))
    result = read_document(env.layout, result_ref, RecoveryReceipt)
    assert read_document(env.layout, result.request, RecoveryInput) == recovery
    with session.ledger.registry.transaction() as database:
        assert [
            tuple(row)
            for row in database.execute(
                "SELECT campaign,reason FROM worker_campaign_stops"
            )
        ] == [("unrelated", "keep this stop")]
    assert controller.resources.reference == controller.runtime.scope


def test_service_fifo_and_child_reply_do_not_need_parent_revisit(
    session: ReviewSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from src.tasks.annotation.contracts import OperationInput, ReviewOperation
    from src.tasks.annotation.operations import execute
    from src.tasks.base.annotation.artifacts import annotation_reference
    from src.tasks.base.annotation.contracts import FrameRange
    from src.tasks.base.annotation.workers.review.contracts import RowsRequest
    from src.tasks.base.annotation.workers.services import ServiceReply

    controller = controller_for(session)
    layout = session.env.layout
    refs = [
        session.request_service(
            "review_rows",
            RowsRequest(
                expected_revision=session.env.head(),
                scope=FrameRange(start=i, stop=i + 1),
            ),
        )
        for i in (1, 2)
    ]
    # Deliberately oppose hash order. FIFO follows submission time, not filenames.
    refs.sort(key=lambda ref: ref.digest, reverse=True)
    for index, ref in enumerate(refs):
        os.utime(layout.artifact_path(ref), ns=(1_000_000 + index, 1_000_000 + index))
    calls = []

    def dispatch(data: OperationInput) -> dict[str, Any]:
        calls.append(data)
        return {"state": "running"}

    controller.broker.operation = dispatch
    assert controller.production is not None
    pool = controller.production.pool
    row = pool.ledger.task(session.request.work_id)
    controller.broker.process(session.request, row, pool)
    assert calls == []
    controller.broker.flush()
    assert [call.operation.service_request for call in calls] == refs[:1]
    op = calls[0]
    assert isinstance(op.operation, ReviewOperation)
    operation_ref = write_document(
        layout,
        layout.path(
            "campaigns",
            session.env.plan.campaign_id,
            "jobs",
            op.sha256 + "-1",
            "operation.json",
        ),
        op,
    )
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", operation_ref.digest)
    # The child writes a durable reply itself. There is no second parent poll.
    execute(layout, operation_ref)
    service_digest = layout.artifact_path(refs[0]).stem
    reply_path = session.directory / "services" / service_digest / "reply.json"
    reply = read_document(
        layout, annotation_reference(layout, reply_path), ServiceReply
    )
    assert reply.status == "succeeded" and reply.result is not None
    controller.broker.process(session.request, row, pool)
    controller.broker.flush()
    assert [call.operation.service_request for call in calls] == refs
    # Ending the worker between collection and dispatch prevents admission.
    controller.broker.process(session.request, row, pool)
    session.state("deferred")
    controller.broker.flush()
    assert len(calls) == 2


def test_child_delivers_failure_and_retry_does_not_inherit_old_service_failure(
    session: ReviewSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.annotation.contracts import OperationInput, ReviewOperation
    from src.tasks.annotation.operations import execute
    from src.tasks.base.annotation.artifacts import annotation_reference
    from src.tasks.base.annotation.contracts import ArtifactRef, FrameRange
    from src.tasks.base.annotation.execution.contracts import StageLaunch
    from src.tasks.base.annotation.workers.review.contracts import RowsRequest
    from src.tasks.base.annotation.workers.services import ServiceReply

    controller = controller_for(session)
    env = session.env
    service_ref = session.request_service(
        "review_rows",
        RowsRequest(expected_revision=env.head(), scope=FrameRange(start=1, stop=2)),
    )
    op = OperationInput(
        plan=env.plan_ref, operation=ReviewOperation(service_request=service_ref)
    )
    directory = env.layout.path(
        "campaigns", env.plan.campaign_id, "jobs", op.sha256 + "-1"
    )

    def submit(launch: StageLaunch) -> dict[str, Any]:
        ref = write_document(env.layout, directory / "launch.json", launch)
        return {"state": "failed", "launch": ref.model_dump()}

    monkeypatch.setattr(controller.stages, "submit", submit)
    state = controller.operation(op)
    launch_ref = ArtifactRef.model_validate(state["launch"])
    monkeypatch.setattr(controller.stages, "status", lambda _ref: state)
    assert controller.status(details=False)["failures"] == [
        dict(state, superseded=False)
    ]
    # A new worker attempt retires only the prior attempt's services.
    session.open(2)
    assert controller.status(details=False)["failures"] == []
    assert controller.status(details=False)["stages"][0]["superseded"]
    operation_ref = annotation_reference(env.layout, directory / "operation.json")
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", operation_ref.digest)
    with pytest.raises(ValueError, match="current running worker attempt"):
        execute(env.layout, operation_ref)
    old_directory = env.layout.artifact_path(service_ref).parents[2]
    reply_ref = annotation_reference(
        env.layout,
        old_directory
        / "services"
        / env.layout.artifact_path(service_ref).stem
        / "reply.json",
    )
    reply = read_document(env.layout, reply_ref, ServiceReply)
    assert reply.status == "failed" and reply.result is None
    assert "current running worker attempt" in reply.reason

    # Live status reuses verified terminal history; details re-verifies it.
    def unexpected(_ref: ArtifactRef) -> dict[str, Any]:
        raise ValueError("synthetic changed historical artifact")

    monkeypatch.setattr(controller.stages, "status", unexpected)
    assert controller.status(details=False)["failures"] == []
    with pytest.raises(ValueError, match="changed historical artifact"):
        controller.status(details=True)
    assert launch_ref == ArtifactRef.model_validate(state["launch"])


def test_bounded_service_session_serves_multiple_requests_under_one_lease(
    session: ReviewSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.annotation.contracts import (
        OperationInput,
        OperationReceipt,
        ServiceSessionOperation,
    )
    from src.tasks.annotation.operations import execute
    from src.tasks.annotation.service_sessions import MemberReceipt, SessionReceipt
    from src.tasks.base.annotation.artifacts import annotation_reference
    from src.tasks.base.annotation.contracts import FrameRange
    from src.tasks.base.annotation.workers.review.contracts import RowsRequest
    from src.tasks.base.annotation.workers.services import ServiceReply

    controller = controller_for(session)
    layout = session.env.layout
    refs = [
        session.request_service(
            "review_rows",
            RowsRequest(
                expected_revision=session.env.head(),
                scope=FrameRange(start=i, stop=i + 1),
            ),
        )
        for i in (1, 2)
    ]
    operation = OperationInput(
        plan=controller.runtime.plan,
        operation=ServiceSessionOperation(
            session=1,
            maximum_requests=2,
            maximum_services_per_attempt=512,
            duration_seconds=3,
        ),
    )
    reference = write_document(
        layout,
        layout.path(
            "campaigns",
            controller.plan.campaign_id,
            "jobs",
            operation.sha256 + "-1",
            "operation.json",
        ),
        operation,
    )
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", reference.digest)
    receipt_ref = execute(layout, reference)
    receipt = read_document(layout, receipt_ref, OperationReceipt)
    report = read_document(layout, receipt.results[0], SessionReceipt)
    assert report.reason == "request limit"
    assert len(report.members) == 2
    for member_ref in report.members:
        member = read_document(layout, member_ref, MemberReceipt)
        assert member.session == reference and member.result is not None
        reply = read_document(layout, member.reply, ServiceReply)
        assert reply.status == "succeeded"
    for service_ref in refs:
        path = layout.artifact_path(service_ref)
        reply_ref = annotation_reference(
            layout, session.directory / "services" / path.stem / "reply.json"
        )
        assert read_document(layout, reply_ref, ServiceReply).result is not None
    with controller.resources.registry.transaction() as db:
        assert db.execute("SELECT count(*) FROM resource_requests").fetchone()[0] == 0
    # Only the enclosing supervisor may execute the nested operations.
    member = read_document(layout, report.members[0], MemberReceipt)
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", member.operation.digest)
    with pytest.raises(ValueError, match="campaign stage"):
        execute(layout, member.operation)


def test_service_session_stage_handles_subsequent_requests_without_parent_poll(
    session: ReviewSession,
) -> None:
    from src.tasks.annotation.contracts import OperationInput, ServiceSessionOperation
    from src.tasks.annotation.service_sessions import SessionReceipt
    from src.tasks.base.annotation.artifacts import annotation_reference
    from src.tasks.base.annotation.contracts import FrameRange
    from src.tasks.base.annotation.workers.review.contracts import RowsRequest
    from src.tasks.base.annotation.workers.services import ServiceReply

    controller = controller_for(session)
    layout = session.env.layout
    operation = OperationInput(
        plan=controller.runtime.plan,
        operation=ServiceSessionOperation(
            session=1,
            maximum_requests=2,
            maximum_services_per_attempt=512,
            duration_seconds=12,
        ),
    )
    state = controller.operation(operation)
    assert state["state"] == "running"
    # The second request does not exist at dispatch. The live, owned session
    # admits it through the same immutable worker binding and returns it directly.
    for index in (1, 2):
        service_ref = session.request_service(
            "review_rows",
            RowsRequest(
                expected_revision=session.env.head(),
                scope=FrameRange(start=index, stop=index + 1),
            ),
        )
        reply_path = (
            session.directory
            / "services"
            / layout.artifact_path(service_ref).stem
            / "reply.json"
        )
        deadline = time.monotonic() + 10
        while not reply_path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        reply = read_document(
            layout, annotation_reference(layout, reply_path), ServiceReply
        )
        assert reply.status == "succeeded"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = controller.operation(operation)
        if state["state"] == "succeeded":
            break
        time.sleep(0.02)
    assert state["state"] == "succeeded", state
    result = controller.receipt(state)
    report = read_document(layout, result.results[0], SessionReceipt)
    assert len(report.members) == 2
    with controller.resources.registry.transaction() as db:
        rows = db.execute("SELECT stage FROM resource_requests").fetchall()
    assert [row[0] for row in rows] == ["render"]
