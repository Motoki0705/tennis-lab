"""Resource-stage boundary for blank image/edit services; no controller logic."""

from __future__ import annotations

from src.tasks.ball_detection.annotation.video.agent.services import (
    execute_service,
    service_input,
)
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import (
    LocalAnnotationPlan,
    references,
    require_execution_layout,
)

from .contracts import OperationInput, OperationReceipt, ReviewOperation


def execute_video_operation(
    layout: AnnotationLayout,
    reference: ArtifactRef,
    plan: LocalAnnotationPlan,
    *,
    session: ArtifactRef | None = None,
) -> ArtifactRef:
    require_execution_layout(layout, plan)
    data = read_document(layout, reference, OperationInput)
    directory = layout.artifact_path(reference).parent
    expected = layout.path("campaigns", plan.campaign_id, "jobs")
    if (
        session is not None
        or not isinstance(data.operation, ReviewOperation)
        or directory.parent != expected
    ):
        raise ValueError(
            "blank stage permits camera services in its campaign jobs only"
        )
    _, attempt = service_input(layout, data.operation.service_request)
    if attempt.launch.plan != data.plan:
        raise ValueError("blank stage belongs to another plan")
    path = directory / "result.json"
    if path.exists():
        ref = annotation_reference(layout, path)
        receipt = read_document(layout, ref, OperationReceipt)
        if receipt.operation != reference:
            raise ValueError("blank service operation identity changed")
        for dependency in references(receipt):
            verify_reference(layout, dependency)
        return ref
    result = execute_service(layout, plan, data.operation.service_request)
    return write_document(
        layout,
        path,
        OperationReceipt(
            operation=reference, status="complete", results=(result,), reason=None
        ),
    )
