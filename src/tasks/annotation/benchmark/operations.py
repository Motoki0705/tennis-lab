"""A resource-accounted child may only serve its benchmark review attempt."""

from __future__ import annotations

import os

from src.tasks.ball_detection.annotation.review.worker import BallReviewServices
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, references
from src.tasks.base.annotation.workers.review.service import (
    execute_service,
    service_input,
)

from ..contracts import OperationInput, OperationReceipt, ReviewOperation


def execute(layout: AnnotationLayout, reference: ArtifactRef) -> ArtifactRef:
    if os.environ.get("TENNIS_ANNOTATION_STAGE_OPERATION") != reference.digest:
        raise ValueError("benchmark service requires its resource-accounted stage")
    from ..service_completion import execute_and_reply

    return execute_and_reply(layout, reference, lambda: _execute(layout, reference))


def _execute(layout: AnnotationLayout, reference: ArtifactRef) -> ArtifactRef:
    data = read_document(layout, reference, OperationInput)
    plan = read_document(layout, data.plan, CampaignPlan)
    if plan.phase != "benchmark" or not isinstance(data.operation, ReviewOperation):
        raise ValueError("benchmark stage permits review services only")
    local = layout.for_benchmark(plan.campaign_id)
    directory = local.artifact_path(reference).parent
    expected = local.path("campaigns", plan.campaign_id, "jobs")
    if (
        not directory.is_relative_to(expected)
        or len(directory.relative_to(expected).parts) != 1
    ):
        raise ValueError("benchmark operation must belong to its campaign stage")
    _, attempt = service_input(local, data.operation.service_request)
    if attempt.launch.plan != data.plan or attempt.input.work.adapter != "ball_review":
        raise ValueError("benchmark service belongs to another plan or adapter")
    output = directory / "result.json"
    if output.exists():
        ref = annotation_reference(local, output)
        receipt = read_document(local, ref, OperationReceipt)
        if receipt.operation != reference:
            raise ValueError("benchmark service result changed its operation")
        for dependency in references(receipt):
            verify_reference(local, dependency)
        return ref
    result_ref = execute_service(
        local, plan, data.operation.service_request, BallReviewServices()
    )
    result: ArtifactRef = write_document(
        local,
        output,
        OperationReceipt(
            operation=reference, status="complete", results=(result_ref,), reason=None
        ),
    )
    return result
