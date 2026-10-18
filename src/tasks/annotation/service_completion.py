"""Publish service replies in the executing child, after its result is durable."""

from __future__ import annotations

from collections.abc import Callable

from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.workers.services import (
    ServiceReply,
    ServiceRequest,
    publish_reply,
)

from .contracts import (
    OperationInput,
    OperationReceipt,
    PreviewOperation,
    ReviewOperation,
)


def execute_and_reply(
    layout: AnnotationLayout,
    reference: ArtifactRef,
    operation: Callable[[], ArtifactRef],
) -> ArtifactRef:
    data = read_document(layout, reference, OperationInput)
    if not isinstance(data.operation, (PreviewOperation, ReviewOperation)):
        return operation()
    service_ref = data.operation.service_request
    service = read_document(layout, service_ref, ServiceRequest)
    try:
        result = operation()
        receipt = read_document(layout, result, OperationReceipt)
        if (
            receipt.operation != reference
            or receipt.status != "complete"
            or len(receipt.results) != 1
        ):
            raise ValueError("service operation returned no exact durable result")
    except (OSError, ValueError, RuntimeError) as error:
        publish_reply(
            layout,
            service_ref,
            ServiceReply(
                request_digest=service.sha256,
                status="failed",
                result=None,
                reason=f"{type(error).__name__}: {error}",
            ),
        )
        raise
    publish_reply(
        layout,
        service_ref,
        ServiceReply(
            request_digest=service.sha256,
            status="succeeded",
            result=receipt.results[0],
            reason=None,
        ),
    )
    return result
