"""Resolve one attempt, check ownership, then mutate or render in the parent stage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ...artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_asset,
    write_document,
)
from ...contracts import ArtifactRef, ClipManifest, Document
from ...editing import EditCommand, ReviewContext, RevisionStore
from ...editing.store import RevisionConflict
from ...handoff import HandoffAdapter, build_handoff
from ...images import read_rendered, render_page
from ...images.contracts import RenderRequest
from ...paths import AnnotationLayout
from ...plan import CampaignPlan
from ...telemetry.contracts import RenderedImage
from ..contracts import Launch, WorkRequest
from ..services import ServiceRequest, require_active_attempt
from .contracts import (
    MAX_EDIT_BYTES,
    MAX_PARAMETERS_BYTES,
    MAX_ROWS_BYTES,
    OPERATIONS,
    EditRequest,
    HandoffRequest,
    ImageRequest,
    ReviewWorkerInput,
    RowsRequest,
)


class ReviewServiceAdapter(HandoffAdapter, Protocol):
    def rows(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        request: RowsRequest,
    ) -> Document: ...

    def render_request(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        request: ImageRequest,
    ) -> RenderRequest: ...


@dataclass(frozen=True)
class ReviewAttempt:
    input: ReviewWorkerInput
    launch: Launch
    actor: ArtifactRef
    directory: Path


def resolve_attempt(
    layout: AnnotationLayout,
    request_ref: ArtifactRef,
    *,
    plan: CampaignPlan | None = None,
    work: WorkRequest | None = None,
) -> ReviewAttempt:
    task = read_document(layout, request_ref, ReviewWorkerInput)
    directory = layout.artifact_path(request_ref).parent
    actor = annotation_reference(layout, directory / "launch.json")
    launch = read_document(layout, actor, Launch)
    if (
        layout.artifact_path(request_ref) != directory / "request.json"
        or task.attempt_path != directory.relative_to(layout.root).as_posix()
        or Path(launch.directory) != directory
        or launch.request != task.work
        or launch.work_id != task.work.work_id
        or (work is not None and task.work != work)
        or (
            plan is not None
            and read_document(layout, launch.plan, CampaignPlan) != plan
        )
    ):
        raise ValueError("review request differs from its immutable worker attempt")
    context = read_document(layout, task.work.payload, ReviewContext)
    manifest = read_document(layout, context.manifest, ClipManifest)
    if (
        launch.plan != context.plan
        or manifest.clip_id != task.work.subject_id
        or task.work.inputs != (context.manifest, context.prediction)
    ):
        raise ValueError("review work differs from its plan/clip/prediction")
    return ReviewAttempt(task, launch, actor, directory)


def service_input(
    layout: AnnotationLayout, reference: ArtifactRef
) -> tuple[ServiceRequest, ReviewAttempt]:
    if reference.size_bytes > MAX_PARAMETERS_BYTES:
        raise ValueError("review service request exceeds its byte limit")
    service = read_document(layout, reference, ServiceRequest)
    if service.operation not in OPERATIONS:
        raise ValueError("unsupported review service")
    attempt = resolve_attempt(layout, service.task_request)
    if layout.artifact_path(reference) != (
        attempt.directory / "work/service-requests" / (service.sha256 + ".json")
    ):
        raise ValueError("service request is outside its own worker inbox")
    return service, attempt


def require_active(layout: AnnotationLayout, attempt: ReviewAttempt) -> None:
    require_active_attempt(layout, attempt.launch, attempt.actor)


def execute_service(
    layout: AnnotationLayout,
    plan: CampaignPlan,
    reference: ArtifactRef,
    adapter: ReviewServiceAdapter,
) -> ArtifactRef:
    service, attempt = service_input(layout, reference)
    context = read_document(layout, attempt.input.work.payload, ReviewContext)
    store = RevisionStore(layout)
    bound_plan, manifest, _ = store.context(context, adapter)
    if bound_plan != plan:
        raise ValueError("review service belongs to another campaign")
    parameters = json.dumps(service.parameters)
    with artifact_lock(layout, "worker-services-" + attempt.launch.work_id):
        require_active(layout, attempt)
        if service.operation == "review_handoff":
            query = HandoffRequest.model_validate_json(parameters)
            return build_handoff(
                store,
                attempt.input.work.payload,
                adapter,
                query.scope,
                previous_revision=query.previous_revision,
                after_row=query.after_row,
                expected_revision=query.expected_revision,
            )
        if service.operation == "review_rows":
            rows_query = RowsRequest.model_validate_json(parameters)
            head, _ = store.current(attempt.input.work.payload, adapter)
            if rows_query.expected_revision != head:
                raise RevisionConflict("row request uses a stale review revision")
            if (
                rows_query.scope.stop > len(manifest.frames)
                or rows_query.scope.stop - rows_query.scope.start > 8
            ):
                raise ValueError("row request must contain at most eight native frames")
            page = adapter.rows(layout, context, manifest, rows_query)
            if len(page.canonical_bytes()) + 1 > MAX_ROWS_BYTES:
                raise ValueError("row page exceeds 64 KiB; request fewer frames")
            return write_document(
                layout, attempt.directory / "rows" / (page.sha256 + ".json"), page
            )
        if service.operation == "review_image":
            image_query = ImageRequest.model_validate_json(parameters)
            head, _ = store.current(attempt.input.work.payload, adapter)
            if image_query.expected_revision != head:
                raise RevisionConflict("image request uses a stale review revision")
            image = render_page(
                layout, adapter.render_request(layout, context, manifest, image_query)
            )
            return write_document(
                layout,
                attempt.directory / "images" / (image.sha256 + ".json"),
                image,
            )
        edit = EditRequest.model_validate_json(parameters)
        if edit.changes.size_bytes > MAX_EDIT_BYTES or not layout.artifact_path(
            edit.changes
        ).is_relative_to(attempt.directory / "work"):
            raise ValueError(
                "review edits must fit the byte limit and belong to attempt/work"
            )
        verify_reference(layout, edit.changes)
        # Freeze mutable worker output before a command/revision may refer to it.
        frozen = write_asset(
            layout,
            attempt.directory / "edits" / (edit.changes.digest + ".json"),
            verify_reference(layout, edit.changes).read_bytes(),
        )
        if frozen.digest != edit.changes.digest:
            raise ValueError("worker edits changed while being frozen")
        return store.apply(
            EditCommand(
                context=attempt.input.work.payload,
                expected_revision=edit.expected_revision,
                changes=frozen,
                actor=attempt.actor,
                reason=edit.reason,
            ),
            adapter,
        )


def image_catalog(
    layout: AnnotationLayout, attempt: ReviewAttempt
) -> tuple[RenderedImage, ...]:
    images = []
    context = read_document(layout, attempt.input.work.payload, ReviewContext)
    for path in sorted((attempt.directory / "images").glob("*.json")):
        image = read_document(layout, annotation_reference(layout, path), RenderedImage)
        request, _ = read_rendered(layout, image)
        if path.name != image.sha256 + ".json" or request.manifest != context.manifest:
            raise ValueError(
                "worker image catalog contains another clip or changed identity"
            )
        images.append(image)
    return tuple(images)
