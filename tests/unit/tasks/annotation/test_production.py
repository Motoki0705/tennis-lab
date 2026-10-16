from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from src.tasks.annotation.contracts import (
    AcquireOperation,
    OperationInput,
    PredictOperation,
    PrepareOperation,
    Runtime,
    StageCost,
)
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
    AcquisitionReceipt,
    PreparationConfig,
    PreparationReceipt,
    SourceCandidate,
)
from src.tasks.base.annotation.contracts import ArtifactRef, content_digest
from src.tasks.base.annotation.execution.contracts import QueueConfiguration
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, StageCapacity
from src.tasks.base.annotation.predictions import Predictions
from src.tasks.base.annotation.resources import BudgetLimits, ResourceManager
from src.tasks.base.annotation.workers.contracts import WorkRequest
from tests.support.tasks.annotation import record_stage
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
        if configured_learning is True:
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
                                ("acquire", "io"),
                                ("prepare", "cpu"),
                                ("infer_ball", "gpu"),
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

    learning_options: dict[str, Any] = {}
    if configured_learning in {"absent", "short"}:
        from src.tasks.ball_detection.data.windows.learning.holdout import (
            freeze_holdout,
        )
        from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
            LegacySourceGroup,
        )
        from src.tasks.ball_detection.generate_dataset.frame_store.publishing.snapshot import (
            freeze_legacy_store,
        )
        from src.tasks.ball_detection.training.annotation.contracts import (
            BallLearningRecipe,
            LearningRuntime,
        )
        from src.tasks.base.annotation.artifacts import write_asset
        from src.tasks.base.annotation.learning.contracts import FixedEvaluation
        from src.tasks.base.annotation.plan import LearningPlan
        from src.utils.paths import PROJECT_ROOT
        from tests.support.tasks.ball_detection.annotation_learning import (
            holdout,
            policy,
        )
        from tests.support.tasks.ball_detection.store import (
            ball,
            frame,
            write_store_clip,
        )

        base = tmp_path / "data/ball_detection/base"
        for split in ("train", "val", "test"):
            write_store_clip(
                base,
                f"{split}/clip",
                [frame(i, ball()) for i in range(32)],
                split=split,
            )
        base_ref = freeze_legacy_store(
            layout,
            base,
            tuple(
                LegacySourceGroup(
                    source="tracknet", group_id=split, source_id=content_digest(split)
                )
                for split in ("train", "val", "test")
            ),
        )
        augmentation = write_asset(
            layout,
            layout.path("models", "augmentation.yaml"),
            (
                PROJECT_ROOT
                / "src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml"
            ).read_bytes(),
        )
        recipe = BallLearningRecipe(
            windows=policy(),
            augmentation=augmentation,
            runtime=LearningRuntime(
                device="cpu",
                precision="fp32",
                num_workers=0,
                pin_memory=False,
                prefetch_factor=1,
                cpu_threads=1,
                compile_mode="off",
                jpeg_decoder="opencv",
                image_prefetch=False,
            ),
            epochs=2,
            freeze_epochs=1,
            windows_per_epoch=1,
            batch_size=1,
            seed=42,
            learning_rate=0.001,
            encoder_lr_ratio=0.1,
            warmup_updates=0,
            log_every=1,
            preview_clips=0,
            stress_evaluation=False,
        )
        fixed = FixedEvaluation(
            task="ball",
            validation=freeze_holdout(layout, holdout(layout, base_ref)),
            test=None,
        )
        learning_options = dict(
            frame_count=34 if configured_learning == "absent" else 32,
            base_snapshot=base_ref,
            learning=LearningPlan(
                recipe=write_document(
                    layout, layout.path("models", "recipe.json"), recipe
                ),
                validation=write_document(
                    layout, layout.path("models", "fixed.json"), fixed
                ),
                selection_policy=None,
            ),
        )
    result = ReviewSession(
        layout.root,
        **learning_options,
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


def record_input_stages(controller: Controller, *, prediction: bool = True) -> None:
    """Give the synthetic ledger the corresponding immutable supervisor receipts."""
    layout = controller.layout
    with controller.collection.ledger.registry.transaction() as database:
        source = database.execute("SELECT * FROM sources").fetchone()
        predicted = database.execute(
            "SELECT receipt FROM annotation_predictions"
        ).fetchone()[0]
    from src.tasks.base.annotation.contracts import ClipManifest
    from src.tasks.base.annotation.predictions import PredictionReceipt

    prediction_ref = ArtifactRef.model_validate_json(predicted)
    context = read_document(layout, prediction_ref, PredictionReceipt).context
    manifest = read_document(layout, context.manifest, ClipManifest)
    acquisition = write_document(
        layout,
        layout.path("campaigns", "fixture-acquisition.json"),
        AcquisitionReceipt(
            candidate_digest=SourceCandidate.model_validate_json(source["candidate"]).sha256,
            configuration=controller.plan.preparation,
            source=manifest.source,
        ),
    )
    with controller.collection.ledger.registry.transaction() as database:
        database.execute(
            "UPDATE sources SET acquisition=? WHERE id=?",
            (acquisition.model_dump_json(), source["id"]),
        )
    operations: list[
        tuple[AcquireOperation | PrepareOperation | PredictOperation, ArtifactRef]
    ] = [
        (AcquireOperation(source_id=source["id"]), acquisition),
        (
            PrepareOperation(source_id=source["id"]),
            ArtifactRef.model_validate_json(source["preparation"]),
        ),
    ]
    if prediction:
        operations.append(
            (PredictOperation(task="ball", clips=(context.manifest,)), prediction_ref)
        )
    for operation, reference in operations:
        record_stage(
            controller,
            OperationInput(plan=controller.runtime.plan, operation=operation),
            results=(reference,),
            exited=True,
        )


@pytest.mark.parametrize(
    "cut", ["operation_receipt", "stage_exit", "failed_exit", "missing_result"]
)
def test_production_requires_prediction_stage_completion_after_ledger_commit(
    session: ReviewSession,
    monkeypatch: pytest.MonkeyPatch,
    cut: str,
) -> None:
    from src.tasks.annotation import operations

    finish(session)
    env = session.env
    controller = controller_for(session)
    record_input_stages(controller, prediction=False)
    data = OperationInput(
        plan=env.plan_ref,
        operation=PredictOperation(task="ball", clips=(env.manifest_ref,)),
    )
    reference = record_stage(controller, data)
    # Re-enter Predictions.run at the ledger commit boundary with the same pinned result.
    with controller.collection.ledger.registry.transaction() as database:
        database.execute("DELETE FROM annotation_predictions")
        database.execute("UPDATE annotation_tasks SET state='pending'")
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", reference.digest)

    def interrupt(layout: AnnotationLayout, path: Path, value: Any) -> ArtifactRef:
        if path.name == "result.json":
            raise RuntimeError("interrupted after prediction ledger commit")
        return write_document(layout, path, value)

    monkeypatch.setattr(
        BallPredictionAdapter,
        "produce",
        lambda *a: pytest.fail("stored prediction must be reused"),
    )
    if cut in {"operation_receipt", "missing_result"}:
        with monkeypatch.context() as faults:
            faults.setattr(operations, "write_document", interrupt)
            with pytest.raises(RuntimeError, match="after prediction ledger commit"):
                operations.execute(env.layout, reference)
    else:
        operations.execute(env.layout, reference)
    if cut in {"failed_exit", "missing_result"}:
        record_stage(controller, data, exited=True, failed=cut == "failed_exit")
    monkeypatch.setattr(
        "src.tasks.base.annotation.execution.queue.queue_state",
        lambda *a: {"state": "interrupted", "reason": "no completion receipt"},
    )
    assert Predictions(env.layout, env.plan).completed(BallPredictionAdapter()) == (
        env.context.prediction,
    )
    assert close_reviews(env.layout, env.plan, env.plan_ref) is not None
    restored = Controller(env.layout, controller.runtime)
    assert restored.production is not None
    publications: list[OperationInput] = []

    def publication(op: OperationInput) -> dict[str, Any]:
        publications.append(op)
        return {"state": "waiting"}

    monkeypatch.setattr(
        restored.production,
        "operation",
        publication,
    )
    monkeypatch.setattr(
        restored.production.learning,
        "register",
        lambda *a: pytest.fail("learning must wait for stage recovery"),
    )
    progress = restored.production.cycle()
    assert progress["state"] == "blocked", progress
    assert publications == []
    assert restored.status()["state"] == "blocked"
    assert not env.layout.path(
        "campaigns", env.plan.campaign_id, "workflow", "completion.json"
    ).exists()
    assert not (
        env.layout.repository
        / "data/ball_detection"
        / env.plan.tasks[0].products[0].dataset_version
    ).exists()
    # An explicit new attempt recovers saved predictions without a model invocation.
    recovery = record_stage(restored, data, attempt=2)
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", recovery.digest)
    operations.execute(env.layout, recovery)
    record_stage(restored, data, attempt=2, exited=True)
    progress = restored.production.cycle()
    assert progress["state"] == "publishing", progress
    assert len(publications) == 1
    assert not restored.status()["failures"]


@pytest.mark.parametrize("action", ["acquire", "prepare"])
@pytest.mark.parametrize("fault", ["missing_exit", "different_result", "latest_failed"])
def test_publication_requires_matching_latest_source_stage_receipts(
    session: ReviewSession,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    fault: str,
) -> None:
    from src.tasks.annotation.contracts import OperationReceipt
    from src.tasks.base.annotation.artifacts import annotation_reference

    finish(session)
    env = session.env
    controller = controller_for(session)
    record_input_stages(controller)
    assert controller.production is not None
    source_id = env.manifest.source.canonical_id
    operation = (
        AcquireOperation(source_id=source_id)
        if action == "acquire"
        else PrepareOperation(source_id=source_id)
    )
    data = OperationInput(plan=env.plan_ref, operation=operation)
    directory = env.layout.path(
        "campaigns", env.plan.campaign_id, "jobs", data.sha256 + "-1"
    )
    if fault == "missing_exit":
        (directory / "exit.json").unlink()
    elif fault == "different_result":
        ref = annotation_reference(env.layout, directory / "result.json")
        receipt = read_document(env.layout, ref, OperationReceipt)
        (directory / "result.json").write_bytes(
            receipt.model_copy(update={"results": (env.blob,)}).canonical_bytes()
        )
    else:
        record_stage(controller, data, attempt=2, exited=True, failed=True)
    monkeypatch.setattr(
        controller.production,
        "operation",
        lambda *a: pytest.fail("must not publish unreconciled source data"),
    )
    progress = controller.production.cycle()
    assert progress["state"] == "blocked" and progress["failures"]


def test_live_input_stage_waits_without_publishing_or_stopping_the_campaign(
    session: ReviewSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.base.annotation.execution.contracts import StageLaunch

    finish(session)
    env = session.env
    controller = controller_for(session)
    record_input_stages(controller)
    assert controller.production is not None
    original_status = controller.stages.status

    def running(ref: ArtifactRef) -> dict[str, Any]:
        launch = read_document(env.layout, ref, StageLaunch)
        if launch.resource.stage == "infer_ball":
            return {"state": "running", "launch": ref.model_dump(), "reason": None}
        state: dict[str, Any] = original_status(ref)
        return state

    monkeypatch.setattr(controller.stages, "status", running)
    monkeypatch.setattr(
        controller.production,
        "operation",
        lambda *a: pytest.fail("input stage is still running"),
    )
    assert controller.production.cycle()["state"] == "reviewing"
    assert controller.status()["state"] == "running"


def test_cached_workflow_completion_does_not_hide_unresolved_stage_failure(
    session: ReviewSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.annotation.contracts import BootstrapModelOperation

    finish(session)
    env = session.env
    controller = controller_for(session)
    record_input_stages(controller)
    assert controller.production is not None
    record_stage(
        controller,
        OperationInput(
            plan=env.plan_ref,
            operation=BootstrapModelOperation(task="ball", evidence=env.blob),
        ),
        exited=True,
        failed=True,
    )
    monkeypatch.setattr(controller.production, "status", lambda: {"state": "complete"})
    state = controller.status()
    assert state["failures"]
    assert state["state"] == "blocked"


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
    record_input_stages(controller)
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
    assert len(status["stages"]) == 4
    assert all(stage["state"] == "succeeded" for stage in status["stages"])
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
    record_input_stages(controller)
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
        len(controller.status()["stages"]) == 4
    )  # No training or GPU process is submitted.


@pytest.mark.parametrize("session", ["absent", "short"], indirect=True)
def test_production_keeps_publication_and_completes_without_coordinate_teachers(
    session: ReviewSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.ball_detection.annotation.review.contracts import (
        BallEdits,
        ConfirmAbsence,
    )
    from src.tasks.ball_detection.data.windows.learning.contracts import TrainingWindows
    from src.tasks.base.annotation.artifacts import annotation_reference
    from src.tasks.base.annotation.contracts import FrameRange
    from src.tasks.base.annotation.learning.contracts import LearningRegistration
    from src.tasks.base.annotation.workers.review.contracts import EditRequest

    env = session.env
    absent = len(env.manifest.frames) == 34
    images = []
    for start in range(1, len(env.manifest.frames) - 1, 8):
        stop = min(start + 8, len(env.manifest.frames) - 1)
        image = session.image(*range(start, stop))
        images.append(image)
        if absent:
            changes = write_document(
                env.layout,
                session.directory / "work" / f"absent-{start}.json",
                BallEdits(
                    images=(image,),
                    operations=(
                        ConfirmAbsence(
                            scope=FrameRange(start=start, stop=stop),
                            out_of_frame_tracks=(),
                            evidence=(image.sha256,),
                        ),
                    ),
                ),
            )
            session.call(
                "review_edit",
                EditRequest(
                    expected_revision=env.head(),
                    changes=changes,
                    reason="confirmed absent",
                ),
            )
        else:
            session.confirm(image, start, stop)
    session.finish(session.evidence(shown=tuple(images)))
    controller = controller_for(session)
    record_input_stages(controller)
    assert controller.production is not None
    initial = controller.production.learning.models.bootstrap(
        env.plan.tasks[0].model, env.blob
    )
    monkeypatch.setattr(
        controller.production.learning,
        "tick",
        lambda *a, **kw: pytest.fail("no learning stage may start"),
    )
    deadline = time.monotonic() + 40
    progress = controller.cycle()["workflow"]
    while progress["state"] == "publishing" and time.monotonic() < deadline:
        time.sleep(0.1)
        progress = controller.cycle()["workflow"]
    assert progress["state"] == "complete", progress
    assert progress["learning"]["status"] == "no_additional_teachers"
    assert progress["completion"] is not None
    root = env.layout.path("campaigns", env.plan.campaign_id, "learning", "ball")
    skipped = read_document(
        env.layout,
        annotation_reference(env.layout, root / "skipped.json"),
        LearningRegistration,
    )
    assert skipped.status == "no_additional_teachers" and skipped.intent is None
    (coverage_path,) = (root / "inputs").glob("*/coverage.json")
    coverage = read_document(
        env.layout, annotation_reference(env.layout, coverage_path), TrainingWindows
    )
    assert coverage.status == "no_additional_coordinate_teachers"
    assert coverage.old_train_clips == 1 and coverage.added_eligible_windows == 0
    assert not (root / "intent.json").exists()
    restored = Controller(env.layout, controller.runtime)
    assert restored.cycle()["workflow"] == progress
    assert restored.status()["state"] == "complete"
    assert len(restored.status()["stages"]) == 4
    assert restored.production is not None
    assert restored.production.learning.models.current("ball") == initial


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
    from src.tasks.annotation.contracts import (
        BootstrapModelOperation,
        OperationInput,
        OperationReceipt,
    )
    from src.tasks.base.annotation.contracts import ArtifactRef
    from src.tasks.base.annotation.execution.contracts import StageLaunch

    # This test exercises stage attempts; no review supervisor was launched.
    session.state("ready")
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
    success = controller.operation(operation)
    assert success["state"] == "succeeded"
    assert any(
        failure["state"] == "invalid_receipt"
        for failure in controller.status()["failures"]
    )
    launch = read_document(
        env.layout, ArtifactRef.model_validate(success["launch"]), StageLaunch
    )
    assert launch.input is not None
    write_document(
        env.layout,
        env.layout.artifact_path(launch.input).parent / "result.json",
        OperationReceipt(
            operation=launch.input,
            status="complete",
            results=(env.blob,),
            reason=None,
        ),
    )
    reported = controller.status()
    assert reported["failures"] == []
    assert [(row["state"], row["superseded"]) for row in reported["stages"]] == [
        ("failed", True),
        ("succeeded", False),
    ]
    assert len(states) == 2


def test_acquisition_retry_reuses_logical_operation_and_supersedes_failure(
    session: ReviewSession,
) -> None:
    from src.tasks.annotation.contracts import AcquireOperation, OperationInput
    from src.tasks.base.annotation.collection.contracts import SourceCandidate

    # Keep the unrelated synthetic review idle while acquisition is retried.
    session.state("ready")
    env = session.env
    controller = controller_for(session)
    source = env.manifest.source
    candidate = SourceCandidate(
        provider="local",
        channel_id=source.channel_id,
        video_id=source.video_id,
        title="synthetic acquisition",
        origin_url=source.origin_url,
        selection_reason="retry regression",
        evidence=env.blob,
        local_media=source.media,
    )
    with controller.collection.ledger.registry.transaction() as db:
        db.execute(
            "UPDATE sources SET state='selected',candidate=? WHERE id=?",
            (candidate.canonical_bytes().decode(), candidate.source_id),
        )
    operation = OperationInput(
        plan=env.plan_ref, operation=AcquireOperation(source_id=candidate.source_id)
    )
    media = env.layout.artifact_path(source.media)
    held = media.with_suffix(".held")
    media.rename(held)

    def settled() -> dict[str, Any]:
        deadline = time.monotonic() + 40
        while True:
            state = controller.operation(operation)
            if state["state"] not in {"running", "draining", "pending"}:
                return state
            assert time.monotonic() < deadline, state
            time.sleep(0.1)

    try:
        assert settled()["state"] == "failed"
    finally:
        held.rename(media)
    assert (
        controller.collection.ledger.source(env.plan, candidate.source_id)["state"]
        == "failed"
    )
    assert controller.operation(operation)["state"] == "failed"
    controller.operation(operation, attempt=2)
    success = settled()
    assert success["state"] == "succeeded", success
    assert controller.receipt(success).status == "complete"
    assert (
        controller.collection.ledger.source(env.plan, candidate.source_id)["state"]
        == "acquired"
    )
    status = controller.status()
    assert status["failures"] == []
    assert [(row["state"], row["superseded"]) for row in status["stages"]] == [
        ("failed", True),
        ("succeeded", False),
    ]
    with pytest.raises(ValueError, match="failed or interrupted"):
        controller.operation(operation, attempt=3)
    # The explicit retry directive must obey the same terminal-failure gate.
    with pytest.raises(ValueError, match="failed or interrupted"):
        controller.operation(OperationInput(
            plan=env.plan_ref,
            operation=AcquireOperation(source_id=candidate.source_id, retry=True),
        ))


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
