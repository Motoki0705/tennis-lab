"""Concrete operations, executed only by the resource-owning stage supervisor."""

from __future__ import annotations

import os
from pathlib import Path

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.collection.ledger import CollectionLedger
from src.tasks.base.annotation.collection.service import Collection, SourceExcluded
from src.tasks.base.annotation.contracts import ArtifactRef, Document
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import load_plan, references
from src.tasks.base.annotation.predictions import PredictionAdapter, Predictions
from src.tasks.base.annotation.screening.contracts import (
    ScreeningPolicy,
    ScreeningRequest,
)
from src.tasks.base.annotation.workers.contracts import Launch
from src.tasks.base.annotation.workers.services import (
    ServiceRequest,
    require_active_attempt,
)

from .contracts import (
    AcquireOperation,
    BootstrapModelOperation,
    CompareStorageOperation,
    HistoryManifest,
    HistoryOperation,
    InspectStorageOperation,
    OperationInput,
    OperationReceipt,
    PredictionPreviewOperation,
    PredictOperation,
    PrepareOperation,
    PreviewOperation,
    PublishOperation,
    RestoreSourceOperation,
    RetireExcludedOperation,
    ReviewOperation,
    ScreenOperation,
    ServiceSessionOperation,
)


class HistoryReceipt(Document):
    manifest: ArtifactRef
    sources: int
    verified_media: int


def preview_input(
    layout: AnnotationLayout, service_ref: ArtifactRef
) -> tuple[ServiceRequest, ScreeningRequest, Path]:
    request = read_document(layout, service_ref, ServiceRequest)
    if request.operation != "screen_preview" or set(request.parameters) != {
        "start",
        "stop",
        "step",
    }:
        raise ValueError("unsupported screening image operation")
    task = read_document(layout, request.task_request, ScreeningRequest)
    directory = layout.artifact_path(request.task_request).parent
    if (
        layout.artifact_path(request.task_request).name != "request.json"
        or directory.relative_to(layout.root).as_posix() != task.attempt_path
        or layout.artifact_path(service_ref)
        != directory / "work/service-requests" / (request.sha256 + ".json")
    ):
        raise ValueError(
            "image request is outside its immutable worker/attempt binding"
        )
    start, stop, step = (request.parameters[key] for key in ("start", "stop", "step"))
    if not all(type(value) in {float, int} for value in (start, stop, step)):
        raise ValueError("preview intervals must be numeric")
    policy = read_document(layout, task.policy, ScreeningPolicy)
    if (
        not 0 <= start < stop <= task.source_duration
        or not policy.boundary_step_seconds <= step <= policy.overview_step_seconds
    ):
        raise ValueError("preview request exceeds source or frozen sampling bounds")
    if (stop - start) / step > 600:
        raise ValueError("a preview request may contain at most 600 sampled frames")
    return request, task, directory


def execute(layout: AnnotationLayout, reference: ArtifactRef) -> ArtifactRef:
    """The supervisor pins this exact input; direct CLI calls cannot bypass ownership."""
    if os.environ.get("TENNIS_ANNOTATION_STAGE_OPERATION") != reference.digest:
        raise ValueError("operation must be launched by its resource-accounted stage")
    data = read_document(layout, reference, OperationInput)
    plan = load_plan(verify_reference(layout, data.plan))
    if layout.artifact_path(reference).parent.parent != layout.path(
        "campaigns", plan.campaign_id, "jobs"
    ):
        raise ValueError("operation input must belong to a campaign stage")
    from .service_completion import execute_and_reply

    return execute_and_reply(layout, reference, lambda: _execute(layout, reference))


def _execute(
    layout: AnnotationLayout,
    reference: ArtifactRef,
    *,
    session: ArtifactRef | None = None,
) -> ArtifactRef:
    data = read_document(layout, reference, OperationInput)
    plan = load_plan(verify_reference(layout, data.plan))
    if plan.phase == "benchmark" or layout.review_namespace is not None:
        raise ValueError("benchmark review requires its dedicated operation boundary")
    directory = layout.artifact_path(reference).parent
    expected = layout.path("campaigns", plan.campaign_id, "jobs")
    if session is None:
        if (
            not directory.is_relative_to(expected)
            or len(directory.relative_to(expected).parts) != 1
        ):
            raise ValueError("operation input must belong to a campaign stage")
    else:
        owner = read_document(layout, session, OperationInput)
        owner_directory = layout.artifact_path(session).parent
        if (
            not isinstance(owner.operation, ServiceSessionOperation)
            or owner.plan != data.plan
            or not isinstance(data.operation, (PreviewOperation, ReviewOperation))
            or os.environ.get("TENNIS_ANNOTATION_STAGE_OPERATION") != session.digest
            or owner_directory.parent != expected
            or directory.parent != owner_directory / "services"
        ):
            raise ValueError("nested service operation differs from its owning session")
    output = directory / "result.json"
    if output.exists():
        ref = annotation_reference(layout, output)
        receipt = read_document(layout, ref, OperationReceipt)
        if receipt.operation != reference:
            raise ValueError("stage operation already contains a different receipt")
        for dependency in references(receipt):
            verify_reference(layout, dependency)
        return ref
    op = data.operation
    result: tuple[ArtifactRef, ...]
    status, reason = "complete", None
    try:
        if isinstance(op, HistoryOperation):
            history = read_document(layout, op.manifest, HistoryManifest)
            ledger = CollectionLedger(layout)
            for entry in history.entries:
                ledger.import_history(entry.source, entry.evidence)
            result = (
                write_document(
                    layout,
                    directory / "history.json",
                    HistoryReceipt(
                        manifest=op.manifest,
                        sources=len(history.entries),
                        verified_media=sum(
                            e.source.media is not None for e in history.entries
                        ),
                    ),
                ),
            )
        elif isinstance(op, AcquireOperation):
            result = (Collection(layout, plan).acquire(op.source_id, retry=op.retry),)
        elif isinstance(op, RestoreSourceOperation):
            from src.tasks.base.annotation.collection.restoration import (
                RestorationReceipt,
            )

            ref = Collection(layout, plan).restore_source(
                op.source_id, attempt=op.attempt
            )
            restoration = read_document(layout, ref, RestorationReceipt)
            result = (ref,)
            if restoration.status != "restored":
                status, reason = "not_applicable", restoration.reason
        elif isinstance(op, ScreenOperation):
            from src.tasks.base.annotation.screening.service import Screening

            result = (Screening(Collection(layout, plan)).prepare(op.source_id),)
        elif isinstance(op, PrepareOperation):
            result = (Collection(layout, plan).prepare(op.source_id),)
        elif isinstance(op, RetireExcludedOperation):
            result = (Collection(layout, plan).retire_excluded(op.source_id),)
        elif isinstance(op, PreviewOperation):
            from src.tasks.base.annotation.screening.images import make_sheets

            service, task, attempt = preview_input(layout, op.service_request)
            if task.plan_digest != plan.sha256:
                raise ValueError("preview request belongs to another campaign")
            actor = annotation_reference(layout, attempt / "launch.json")
            worker = read_document(layout, actor, Launch)
            if (
                Path(worker.directory) != attempt
                or worker.plan != data.plan
                or worker.request.adapter != "source_screening"
                or worker.request.payload != task.prepared_input
            ):
                raise ValueError("preview differs from its immutable worker attempt")
            with artifact_lock(layout, "worker-services-" + worker.work_id):
                require_active_attempt(layout, worker, actor)
                result = (
                    make_sheets(
                        layout,
                        task.source.media,
                        attempt / "previews",
                        origin_pts=task.source_origin_pts,
                        time_base=task.time_base,
                        **service.parameters,
                    ),
                )
        elif isinstance(op, ServiceSessionOperation):
            from .service_sessions import execute_session

            result = (execute_session(layout, reference),)
        elif isinstance(op, ReviewOperation):
            from src.tasks.ball_detection.annotation.review.worker import (
                BallReviewServices,
                BallReviewWorker,
            )
            from src.tasks.base.annotation.workers.review.service import (
                execute_service,
                service_input,
            )

            _service, review = service_input(layout, op.service_request)
            BallReviewWorker.require_work(review.input.work)
            result = (
                execute_service(layout, plan, op.service_request, BallReviewServices()),
            )
        elif isinstance(op, PublishOperation):
            from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
                PublishRequest,
            )
            from src.tasks.ball_detection.generate_dataset.frame_store.publishing.publisher import (
                BallPublisher,
            )

            from .reviews import validate_closed

            adoptions = validate_closed(layout, plan, data.plan, op.review)
            result = (
                BallPublisher(layout).publish(
                    PublishRequest(
                        plan=data.plan, product="frames", adoptions=adoptions
                    )
                ),
            )
        elif isinstance(op, BootstrapModelOperation):
            from src.tasks.base.annotation.learning.models import ModelRegistry

            if plan.phase != "production":
                raise ValueError("model bootstrap requires an explicit production plan")
            model_task = next(
                (task for task in plan.tasks if task.task == op.task), None
            )
            if model_task is None:
                raise ValueError("model bootstrap task is absent from the frozen plan")
            state = ModelRegistry(layout).bootstrap(model_task.model, op.evidence)
            result = (write_document(layout, directory / "model.json", state),)
        elif isinstance(op, InspectStorageOperation):
            from src.tasks.base.annotation.storage.comparison.sources import capture

            from .storage_comparison import open_store

            result = (
                write_document(
                    layout,
                    directory / "capture.json",
                    capture(
                        layout,
                        op.specification,
                        op.limits,
                        open_store,
                    ),
                ),
            )
        elif isinstance(op, CompareStorageOperation):
            from src.tasks.base.annotation.storage.comparison.engine import compare

            from .storage_comparison import open_store

            result = (
                write_document(
                    layout,
                    directory / "comparison.json",
                    compare(
                        layout,
                        op.request,
                        directory / "comparison",
                        open_store,
                    ),
                ),
            )
        elif isinstance(op, PredictionPreviewOperation):
            from .prediction_preview import preview_prediction

            result = (preview_prediction(layout, plan, op),)
        elif isinstance(op, PredictOperation):
            adapter: PredictionAdapter
            if op.task == "ball":
                from src.tasks.ball_detection.annotation.inference import (
                    BallPredictionAdapter,
                )

                adapter = BallPredictionAdapter()
            else:
                from src.tasks.player_detection.annotation.inference import (
                    PlayerPredictionAdapter,
                )

                adapter = PlayerPredictionAdapter()
            predictions = Predictions(layout, plan)
            result = tuple(predictions.run(clip, adapter) for clip in op.clips)
        else:
            raise ValueError("unknown stage operation")
    except SourceExcluded as error:
        result, status, reason = (), "not_applicable", str(error)
    return write_document(
        layout,
        output,
        OperationReceipt.model_validate(
            {
                "operation": reference,
                "status": status,
                "results": result,
                "reason": reason,
            }
        ),
    )
