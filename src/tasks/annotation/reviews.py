"""Campaign-wide review coverage, separate from individual worker completion."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.tasks.ball_detection.annotation.inference import BallPredictionAdapter
from src.tasks.ball_detection.annotation.review.worker import (
    BallReviewServices,
    BallReviewWorker,
)
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.collection.contracts import PreparationReceipt
from src.tasks.base.annotation.collection.ledger import CollectionLedger
from src.tasks.base.annotation.contracts import ArtifactRef, Digest, Document
from src.tasks.base.annotation.editing import ReviewContext, RevisionStore
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, references
from src.tasks.base.annotation.predictions import PredictionReceipt, Predictions
from src.tasks.base.annotation.workers.contracts import (
    AttemptResult,
    Launch,
    WorkRequest,
)
from src.tasks.base.annotation.workers.ledger import WorkerLedger


class SourceCompletion(Document):
    source_id: Digest
    state: Literal["prepared", "excluded"]
    preparation: ArtifactRef | None
    reason: str | None


class ClipReview(Document):
    request: ArtifactRef
    result: ArtifactRef


class CampaignReview(Document):
    schema_version: Literal["annotation.campaign-review.v1"] = (
        "annotation.campaign-review.v1"
    )
    plan: ArtifactRef
    sources: tuple[SourceCompletion, ...] = Field(min_length=1)
    reviews: tuple[ClipReview, ...] = Field(min_length=1)


def require_supported(plan: CampaignPlan) -> None:
    """Enable only products whose complete review/publication path is connected."""
    if plan.phase != "production" or len(plan.tasks) != 1:
        raise ValueError("production workflow currently requires one ball task")
    task = plan.tasks[0]
    if task.task != "ball" or [
        (p.product, p.dataset_task, p.component) for p in task.products
    ] != [("frames", "ball_detection", "frames")]:
        raise ValueError(
            "production review/publication currently supports ball frames only"
        )


def requests(
    layout: AnnotationLayout, plan: CampaignPlan, plan_ref: ArtifactRef
) -> tuple[ArtifactRef, ...]:
    """A prediction maps to exactly one immutable request; no inference is run here."""
    require_supported(plan)
    return prediction_requests(
        layout,
        plan,
        plan_ref,
        Predictions(layout, plan).completed(BallPredictionAdapter()),
    )


def prediction_requests(
    layout: AnnotationLayout,
    plan: CampaignPlan,
    plan_ref: ArtifactRef,
    predictions: tuple[ArtifactRef, ...],
) -> tuple[ArtifactRef, ...]:
    """Bind validated, fixed predictions to the matching review state domain."""
    if read_document(layout, plan_ref, CampaignPlan) != plan:
        raise ValueError("review request plan differs from its immutable reference")
    results = []
    for prediction_ref in predictions:
        prediction = read_document(layout, prediction_ref, PredictionReceipt)
        root = layout.path(
            "campaigns", plan.campaign_id, "review", "ball", prediction.context.clip_id
        )
        context = ReviewContext(
            plan=plan_ref,
            task="ball",
            manifest=prediction.context.manifest,
            prediction=prediction_ref,
        )
        context_ref = RevisionStore(layout).bind_context(context, BallReviewServices())
        request = WorkRequest(
            adapter=BallReviewWorker.name,
            adapter_version=BallReviewWorker.version,
            stage="review",
            subject_id=prediction.context.clip_id,
            payload=context_ref,
            inputs=(context.manifest, context.prediction),
        )
        results.append(write_document(layout, root / "request.json", request))
    return tuple(results)


def sources(
    layout: AnnotationLayout, plan: CampaignPlan
) -> tuple[SourceCompletion, ...] | None:
    ledger = CollectionLedger(layout)
    with ledger.registry.transaction() as database:
        rows = database.execute(
            "SELECT id,state,preparation,reason FROM sources WHERE campaign_id=? ORDER BY id",
            (plan.campaign_id,),
        ).fetchall()
        pending = database.execute(
            "SELECT count(*) FROM annotation_tasks WHERE campaign_id=? AND state NOT IN ('predicted','reviewing','adopted')",
            (plan.campaign_id,),
        ).fetchone()[0]
    if (
        not rows
        or pending
        or any(row["state"] not in {"prepared", "excluded"} for row in rows)
        or sum(row["state"] == "prepared" for row in rows) != plan.counts.videos
    ):
        return None
    result = tuple(
        SourceCompletion(
            source_id=row["id"],
            state=row["state"],
            preparation=ArtifactRef.model_validate_json(row["preparation"])
            if row["preparation"] is not None
            else None,
            reason=row["reason"],
        )
        for row in rows
    )
    for row in result:
        if row.state == "prepared" and row.preparation is None:
            raise ValueError("prepared source has no preparation receipt")
        if row.state == "excluded" and not (row.reason or "").strip():
            raise ValueError("excluded source requires its reason")
        if row.preparation is not None:
            verify_reference(layout, row.preparation)
    return result


def adoption(
    layout: AnnotationLayout,
    request_ref: ArtifactRef,
    result_ref: ArtifactRef,
    plan_ref: ArtifactRef,
) -> ArtifactRef:
    request = read_document(layout, request_ref, WorkRequest)
    result = read_document(layout, result_ref, AttemptResult)
    launch = read_document(layout, result.launch, Launch)
    if (
        launch.request != request
        or launch.plan != plan_ref
        or result.outcome.status != "complete"
        or result.outcome.adoption is None
    ):
        raise ValueError(
            "campaign review requires the matching completed worker adoption"
        )
    for reference in references(result):
        verify_reference(layout, reference)
    return result.outcome.adoption


def close_reviews(
    layout: AnnotationLayout, plan: CampaignPlan, plan_ref: ArtifactRef
) -> ArtifactRef | None:
    source_rows = sources(layout, plan)
    if source_rows is None:
        return None
    expected = requests(layout, plan, plan_ref)
    _coverage(layout, source_rows, expected)
    ledger = WorkerLedger(layout)
    reviews = []
    for request_ref in expected:
        request = read_document(layout, request_ref, WorkRequest)
        row = ledger.task(request.work_id)
        if row["state"] != "complete":
            return None
        result_ref = ArtifactRef.model_validate_json(row["result"])
        adoption(layout, request_ref, result_ref, plan_ref)
        reviews.append(ClipReview(request=request_ref, result=result_ref))
    review = CampaignReview(plan=plan_ref, sources=source_rows, reviews=tuple(reviews))
    result: ArtifactRef = write_document(
        layout,
        layout.path("campaigns", plan.campaign_id, "workflow", "review.json"),
        review,
    )
    return result


def validate_closed(
    layout: AnnotationLayout,
    plan: CampaignPlan,
    plan_ref: ArtifactRef,
    reference: ArtifactRef,
) -> tuple[ArtifactRef, ...]:
    expected_path = layout.path(
        "campaigns", plan.campaign_id, "workflow", "review.json"
    )
    if reference != annotation_reference(layout, expected_path):
        raise ValueError("publication requires the registered campaign review closure")
    review = read_document(layout, reference, CampaignReview)
    expected = requests(layout, plan, plan_ref)
    if (
        review.plan != plan_ref
        or review.sources != sources(layout, plan)
        or tuple(item.request for item in review.reviews) != expected
    ):
        raise ValueError("campaign review does not cover the current source/clip set")
    _coverage(layout, review.sources, expected)
    return tuple(
        sorted(
            (
                adoption(layout, row.request, row.result, plan_ref)
                for row in review.reviews
            ),
            key=lambda item: item.digest,
        )
    )


def _coverage(
    layout: AnnotationLayout,
    source_rows: tuple[SourceCompletion, ...],
    expected: tuple[ArtifactRef, ...],
) -> None:
    prepared = []
    for source in source_rows:
        if source.preparation is None:
            continue
        preparation = read_document(layout, source.preparation, PreparationReceipt)
        if preparation.source_id != source.source_id:
            raise ValueError("source completion names a different preparation")
        if source.state == "prepared":
            prepared.extend(preparation.clips)
    reviewed = []
    for reference in expected:
        work = read_document(layout, reference, WorkRequest)
        context = read_document(layout, work.payload, ReviewContext)
        reviewed.append(context.manifest)
    if (
        len(set(prepared)) != len(prepared)
        or set(prepared) != set(reviewed)
        or len(prepared) != len(reviewed)
    ):
        raise ValueError("campaign review must cover every prepared clip exactly once")
