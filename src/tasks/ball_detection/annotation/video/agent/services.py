"""All input resolution and writes run in the parent, outside the worker boundary."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.editing.contracts import BlankAnnotationContext
from src.tasks.base.annotation.images import read_rendered, render_page
from src.tasks.base.annotation.images.contracts import (
    RenderReceipt,
    RenderRequest,
    Tile,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import LocalAnnotationPlan, references
from src.tasks.base.annotation.telemetry.contracts import RenderedImage
from src.tasks.base.annotation.workers.contracts import Launch
from src.tasks.base.annotation.workers.services import (
    ServiceRequest,
    require_active_attempt,
)

from .contracts import (
    AgentTask,
    AgentView,
    AnnotationRun,
    EditQuery,
    FrameQuery,
    HandoffQuery,
    ImageQuery,
    ServiceEvent,
)
from .store import VideoRevisionStore, progress_summary

OPERATIONS = ("video_handoff", "video_rows", "video_image", "video_edit")


@dataclass(frozen=True)
class VideoAttempt:
    task_ref: ArtifactRef
    task: AgentTask
    launch: Launch
    actor: ArtifactRef
    directory: Path
    store: VideoRevisionStore


def resolve_attempt(layout: AnnotationLayout, task_ref: ArtifactRef) -> VideoAttempt:
    from src.tasks.annotation.contracts import LocalAnnotationRuntime

    task = read_document(layout, task_ref, AgentTask)
    directory = layout.artifact_path(task_ref).parent
    actor = annotation_reference(layout, directory / "launch.json")
    launch = read_document(layout, actor, Launch)
    run = read_document(layout, task.run, AnnotationRun)
    context = read_document(layout, run.context, BlankAnnotationContext)
    runtime = read_document(layout, run.runtime, LocalAnnotationRuntime)
    binary = shutil.which(runtime.codex_binary)
    if (
        layout.artifact_path(task_ref) != directory / "request.json"
        or Path(launch.directory) != directory
        or launch.request != task.work
        or (task.work.adapter, task.work.adapter_version, task.work.stage)
        != ("ball_video_annotation", "3", "review")
        or run.context != task.work.payload
        or run.plan != launch.plan
        or launch.work_id != task.work.work_id
        or task.work.inputs
        != (context.manifest, context.task_input, context.source_media)
        or not launch.prepared.isolate_inputs
        or runtime.plan != run.plan
        or runtime.cost("review").timeout_seconds != run.timeout_seconds
        or launch.timeout_seconds != run.timeout_seconds
        or (
            launch.test_command is None
            and (
                binary is None
                or Path(launch.codex_binary).resolve() != Path(binary).resolve()
            )
        )
        or (
            launch.test_command is None
            and Path(launch.codex_home) != directory / "work/.codex"
        )
    ):
        raise ValueError("blank service differs from its pinned camera/attempt")
    store = VideoRevisionStore(layout, task.run)
    from src.tasks.base.annotation.contracts import ClipManifest

    if (
        read_document(layout, context.manifest, ClipManifest).clip_id
        != task.work.subject_id
    ):
        raise ValueError("blank service subject differs from its camera manifest")
    store.annotation(task.initial_revision)
    return VideoAttempt(task_ref, task, launch, actor, directory, store)


def service_input(
    layout: AnnotationLayout, reference: ArtifactRef
) -> tuple[ServiceRequest, VideoAttempt]:
    if reference.size_bytes > 32768:
        raise ValueError("blank service request exceeds 32 KiB")
    service = read_document(layout, reference, ServiceRequest)
    attempt = resolve_attempt(layout, service.task_request)
    if service.operation not in OPERATIONS or layout.artifact_path(
        reference
    ) != attempt.directory / "work/service-requests" / (service.sha256 + ".json"):
        raise ValueError("unsupported blank service or another attempt inbox")
    return service, attempt


def events(layout: AnnotationLayout, attempt: VideoAttempt) -> tuple[ServiceEvent, ...]:
    result = tuple(
        sorted(
            (
                read_document(layout, annotation_reference(layout, path), ServiceEvent)
                for path in (attempt.directory / "services").glob("*/event.json")
            ),
            key=lambda e: e.sequence,
        )
    )
    for index, event in enumerate(result, 1):
        if event.sequence != index or event.task != attempt.task_ref:
            raise ValueError("blank service event sequence/camera changed")
        service = read_document(layout, event.request, ServiceRequest)
        if service.task_request != event.task or service.operation != event.operation:
            raise ValueError("blank service event differs from request")
        for ref in references(event):
            verify_reference(layout, ref)
    return result


def _retain_event(
    layout: AnnotationLayout,
    attempt: VideoAttempt,
    event_ref: ArtifactRef,
    event: ServiceEvent,
) -> None:
    response = read_document(layout, event.result, AgentView)
    dependencies = [event.result, event.request, *references(response)]
    if response.metadata is not None:
        receipt = read_document(layout, response.metadata, RenderReceipt)
        dependencies.append(receipt.request)
    with attempt.store.registry.transaction() as db:
        attempt.store._protect(db, event_ref, tuple(dependencies))


def execute_service(
    layout: AnnotationLayout, plan: LocalAnnotationPlan, reference: ArtifactRef
) -> ArtifactRef:
    service, attempt = service_input(layout, reference)
    if read_document(layout, attempt.launch.plan, LocalAnnotationPlan) != plan:
        raise ValueError("blank service belongs to a different plan")
    store = attempt.store
    with artifact_lock(layout, "worker-services-" + attempt.launch.work_id):
        require_active_attempt(layout, attempt.launch, attempt.actor)
        event_path = attempt.directory / "services" / service.sha256 / "event.json"
        if event_path.exists():
            event = read_document(
                layout, annotation_reference(layout, event_path), ServiceEvent
            )
            if (
                event.request.digest != reference.digest
                or event.task != service.task_request
            ):
                raise ValueError("saved service event belongs to another request")
            verify_reference(layout, event.result)
            _retain_event(
                layout, attempt, annotation_reference(layout, event_path), event
            )
            return event.result
        query_types: dict[
            str,
            type[HandoffQuery] | type[FrameQuery] | type[ImageQuery] | type[EditQuery],
        ] = {
            "video_handoff": HandoffQuery,
            "video_rows": FrameQuery,
            "video_image": ImageQuery,
            "video_edit": EditQuery,
        }
        query_type = query_types[service.operation]
        query = query_type.model_validate_json(json.dumps(service.parameters))
        camera = store.expected.source.camera_id
        if query.camera_id != camera:
            raise ValueError("service camera differs from the assigned camera")
        head, _ = store.current()
        annotation = store.annotation(head)
        response = AgentView(
            camera_id=camera,
            revision=head,
            summary=progress_summary(annotation),
        )
        directory = attempt.directory / "exchange" / service.sha256
        if isinstance(query, HandoffQuery):
            remaining = [
                f.frame_index
                for f in annotation.frames
                if f.state == "unprocessed"
                and (query.after_frame is None or f.frame_index > query.after_frame)
            ]
            page = remaining[: query.limit]
            response = response.model_copy(
                update={
                    "remaining": tuple(page),
                    "next_after_frame": page[-1]
                    if len(remaining) > len(page)
                    else None,
                }
            )
        elif isinstance(query, FrameQuery):
            if len(set(query.frames)) != len(query.frames) or any(
                not store.expected.frame_range.start
                <= i
                < store.expected.frame_range.stop
                for i in query.frames
            ):
                raise ValueError("row query must name unique assigned source frames")
            rows = {f.frame_index: f for f in annotation.frames}
            response = response.model_copy(
                update={"rows": tuple(rows[i] for i in query.frames)}
            )
        elif isinstance(query, ImageQuery):
            scope = store.expected.frame_range
            if any(not scope.start <= t.frame_index < scope.stop for t in query.tiles):
                raise ValueError("image frame lies outside the assigned camera input")
            render = RenderRequest(
                **query.model_dump(exclude={"camera_id", "tiles"}),
                manifest=store.context.manifest,
                annotation=head,
                annotation_kind="video_revision",
                drawing="plain",
                tiles=tuple(
                    Tile(frame_index=t.frame_index - scope.start, crop=t.crop)
                    for t in query.tiles
                ),
            )
            rendered = render_page(layout, render)
            _, metadata = read_rendered(layout, rendered)
            image = write_asset(
                layout,
                directory / "page.jpg",
                verify_reference(layout, rendered.image).read_bytes(),
            )
            receipt = write_document(
                layout,
                directory / "image.json",
                RenderReceipt(**metadata.model_dump(exclude={"image"}), image=image),
            )
            response = response.model_copy(
                update={
                    "image": image,
                    "metadata": receipt,
                    "image_path": str(layout.artifact_path(image)),
                }
            )
        else:
            assert isinstance(query, EditQuery)
            head = store.apply(query, attempt.actor)
            response = response.model_copy(
                update={
                    "revision": head,
                    "summary": progress_summary(store.annotation(head)),
                }
            )
        result = write_document(layout, directory / "result.json", response)
        history = events(layout, attempt)
        frozen_request = write_asset(
            layout,
            event_path.with_name("request.json"),
            verify_reference(layout, reference).read_bytes(),
        )
        event = ServiceEvent(
            task=service.task_request,
            request=frozen_request,
            operation=service.operation,
            result=result,
            sequence=len(history) + 1,
        )
        event_ref = write_document(layout, event_path, event)
        _retain_event(layout, attempt, event_ref, event)
        return result


def image_catalog(
    layout: AnnotationLayout, attempt: VideoAttempt
) -> tuple[RenderedImage, ...]:
    images = []
    for event in events(layout, attempt):
        if event.operation != "video_image":
            continue
        view = read_document(layout, event.result, AgentView)
        if view.image is None or view.metadata is None:
            raise ValueError("image service omitted its image provenance")
        rendered = RenderedImage(image=view.image, metadata=view.metadata)
        request, _ = read_rendered(layout, rendered)
        if (
            request.manifest != attempt.store.context.manifest
            or request.drawing != "plain"
        ):
            raise ValueError("blank image catalog contains another input or overlay")
        images.append(rendered)
    return tuple(images)
