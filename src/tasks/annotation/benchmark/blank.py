"""Strategy comparisons over identical blank inputs, isolated from publication."""

from __future__ import annotations

from math import hypot
from typing import Literal

from pydantic import Field

from src.tasks.ball_detection.annotation.video.agent.contracts import (
    AgentTask,
    AgentView,
    AnnotationPolicy,
    ServiceEvent,
)
from src.tasks.ball_detection.annotation.video.agent.services import (
    events,
    resolve_attempt,
)
from src.tasks.ball_detection.annotation.video.agent.store import (
    VideoRevisionStore,
    progress_summary,
)
from src.tasks.ball_detection.annotation.video.contracts import VideoBallAnnotation
from src.tasks.ball_detection.annotation.video.io import read_annotation
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.benchmarks.contracts import AttemptSnapshot, UsageSummary
from src.tasks.base.annotation.benchmarks.report import summarize_usage
from src.tasks.base.annotation.collection.local import LocalInputLedger
from src.tasks.base.annotation.contracts import ArtifactRef, Document, Identifier
from src.tasks.base.annotation.editing.contracts import BlankAnnotationContext, Revision
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import LocalAnnotationPlan, save_plan
from src.tasks.base.annotation.telemetry.capture import iter_responses, verify_report
from src.tasks.base.annotation.telemetry.contracts import Usage
from src.tasks.base.annotation.workers.contracts import Launch, WorkRequest
from src.tasks.base.annotation.workers.ledger import WorkerLedger
from src.utils.paths import PROJECT_ROOT

from ..contracts import LocalAnnotationRuntime
from ..reviews import local_video_request


class BlankCondition(Document):
    schema_version: Literal["annotation.blank-condition.v1"] = (
        "annotation.blank-condition.v1"
    )
    source_plan: ArtifactRef
    plan: ArtifactRef
    runtime: ArtifactRef
    policy: AnnotationPolicy
    requests: tuple[ArtifactRef, ...] = Field(min_length=1)


class BlankConditionRegistration(Document):
    source_plan: ArtifactRef
    condition_id: Identifier
    policy: AnnotationPolicy


class BlankComparisonRequest(Document):
    conditions: tuple[ArtifactRef, ...] = Field(min_length=2)
    reference_annotations: tuple[ArtifactRef, ...] = ()


class FrameComparison(Document):
    camera_id: str
    frame_index: int
    pts: int
    state: str
    provenance: str | None
    position_usable: bool
    reference_state: str | None
    error_px: float | None


class ResponseComparison(Document):
    attempt: str
    thread_id: str
    response_id: str
    usage: Usage | None
    revision: ArtifactRef
    summary: dict[str, int | bool]
    mean_reference_error_px: float | None
    reference_pairs: int
    image_requests: int
    image_presentations: int
    revisited_frames: int


class CameraComparison(Document):
    request: ArtifactRef
    camera_id: str
    worker_kind: Literal["not_started", "synthetic", "real", "mixed"]
    run: ArtifactRef | None
    annotation: ArtifactRef | None
    summary: dict[str, int | bool] | None
    image_requests: int
    frames_requested: int
    revisited_frames: int
    usage: UsageSummary
    responses: tuple[ResponseComparison, ...]
    frames: tuple[FrameComparison, ...]


class BlankComparison(Document):
    schema_version: Literal["annotation.blank-comparison.v1"] = (
        "annotation.blank-comparison.v1"
    )
    conditions: tuple[ArtifactRef, ...]
    references: tuple[ArtifactRef, ...]
    results: tuple[tuple[CameraComparison, ...], ...]
    evidence_kind: Literal["operational_measurements"] = "operational_measurements"
    acceptance: Literal["unverified"] = "unverified"


class BlankRunMeasurements(Document):
    """Production telemetry is evidence of execution, not recognition acceptance."""

    plan: ArtifactRef
    results: tuple[CameraComparison, ...]
    acceptance: Literal["unverified"] = "unverified"


def measure_blank_plan(layout: AnnotationLayout, plan_ref: ArtifactRef) -> ArtifactRef:
    plan = read_document(layout, plan_ref, LocalAnnotationPlan)
    if plan.phase != "production" or layout.review_namespace is not None:
        raise ValueError("Use compare_blank_conditions for isolated comparison runs")
    report = BlankRunMeasurements(plan=plan_ref, results=tuple(
        _camera(layout, plan, entry.request, ()) for entry in LocalInputLedger(layout).entries(plan)
    ))
    return write_document(layout, layout.path("campaigns", plan.campaign_id, "reports", "blank-runs", report.sha256 + ".json"), report)


def register_blank_condition(
    layout: AnnotationLayout,
    source_plan: ArtifactRef,
    condition_id: Identifier,
    policy: AnnotationPolicy,
) -> ArtifactRef:
    """Reuse fixed media and native timelines; never copy labels or restart a run."""
    shared = layout.shared()
    original = read_document(shared, source_plan, LocalAnnotationPlan)
    if original.phase != "production" or condition_id == original.campaign_id:
        raise ValueError("blank comparison requires a distinct condition identity")
    local = shared.for_benchmark(condition_id)
    plan = LocalAnnotationPlan.model_validate_json(
        original.model_copy(
            update={
                "campaign_id": condition_id,
                "phase": "benchmark",
                "publish_dataset": False,
            }
        ).canonical_bytes()
    )
    if local.mode == "production":
        from src.tasks.base.annotation.reproducibility import require_frozen_code

        require_frozen_code(PROJECT_ROOT, plan)
    directory = local.path("campaigns", condition_id, "benchmark")
    with artifact_lock(local, "blank-condition-" + condition_id):
        plan_ref = annotation_reference(local, save_plan(local, plan))
        source_runtime = read_document(
            shared,
            annotation_reference(
                shared,
                shared.path(
                    "campaigns", original.campaign_id, "configuration", "runtime.json"
                ),
            ),
            LocalAnnotationRuntime,
        )
        runtime = source_runtime.model_copy(update={"plan": plan_ref})
        runtime_ref = write_document(
            local,
            local.path("campaigns", condition_id, "configuration", "runtime.json"),
            runtime,
        )
        # Policy is pinned before any input registration so retries cannot change it.
        write_document(local, directory / "policy.json", policy)
        registered = []
        for entry in LocalInputLedger(shared).entries(original):
            context = read_document(shared, entry.context, BlankAnnotationContext)
            new = local_video_request(
                local,
                plan,
                plan_ref,
                context.manifest,
                context.task_input,
                context.source_media,
            )
            LocalInputLedger(local).register(plan, new)
            registered.append(new.request)
        if not registered:
            raise ValueError("source plan has no registered blank inputs")
        condition = BlankCondition(
            source_plan=source_plan,
            plan=plan_ref,
            runtime=runtime_ref,
            policy=policy,
            requests=tuple(registered),
        )
        return write_document(local, directory / "blank-condition.json", condition)


def _frames(
    annotation: VideoBallAnnotation, reference: VideoBallAnnotation | None
) -> tuple[FrameComparison, ...]:
    if reference is not None and reference.input_identity != annotation.input_identity:
        raise ValueError("comparison reference has another camera/source/frame/PTS")
    rows = []
    for index, row in enumerate(annotation.frames):
        expected = reference.frames[index] if reference is not None else None
        point = row.position
        target = expected.position if expected is not None else None
        rows.append(
            FrameComparison(
                camera_id=annotation.source.camera_id,
                frame_index=row.frame_index,
                pts=row.pts,
                state=row.state,
                provenance=point.provenance.kind if point else None,
                position_usable=point is not None,
                reference_state=expected.state if expected else None,
                error_px=hypot(point.x - target.x, point.y - target.y)
                if point is not None and target is not None
                else None,
            )
        )
    return tuple(rows)


def _error(rows: tuple[FrameComparison, ...]) -> tuple[float | None, int]:
    values = [r.error_px for r in rows if r.error_px is not None]
    return (sum(values) / len(values) if values else None, len(values))


def _camera(
    layout: AnnotationLayout,
    plan: LocalAnnotationPlan,
    request_ref: ArtifactRef,
    references: tuple[VideoBallAnnotation, ...],
) -> CameraComparison:
    request = read_document(layout, request_ref, WorkRequest)
    context = read_document(layout, request.payload, BlankAnnotationContext)
    from src.tasks.ball_detection.annotation.video import VideoAnnotationInput

    expected = read_document(layout, context.task_input, VideoAnnotationInput)
    reference = next((r for r in references if r.input_identity == expected), None)
    history = WorkerLedger(layout).history(plan, request)
    synthetic = [
        read_document(layout, a.launch, Launch).test_command is not None
        or read_document(layout, a.launch, Launch).test_root is not None
        for a in history
    ]
    snapshots = tuple(
        AttemptSnapshot(
            launch=a.launch,
            state=a.state,
            execution=a.execution,
            telemetry=a.telemetry,
            child=None,
        )
        for a in history
    )
    usage = summarize_usage(layout, snapshots)
    responses = []
    requested: set[int] = set()
    presentations: set[int] = set()
    image_requests = frame_requests = revisits = 0
    run_ref = annotation_ref = summary = None
    frame_rows: tuple[FrameComparison, ...] = ()
    for attempt in history:
        launch = read_document(layout, attempt.launch, Launch)
        task_ref = annotation_reference(
            layout, layout.artifact_path(attempt.launch).parent / "request.json"
        )
        task = read_document(layout, task_ref, AgentTask)
        store = VideoRevisionStore(layout, task.run)
        run_ref = task.run
        head, _ = store.current()
        annotation_ref = read_document(layout, head, Revision).payload
        annotation = store.annotation(head)
        summary = progress_summary(annotation)
        frame_rows = _frames(annotation, reference)
        for event in events(layout, resolve_attempt(layout, task_ref)):
            if event.operation != "video_image":
                continue
            from src.tasks.base.annotation.workers.services import ServiceRequest

            query = read_document(layout, event.request, ServiceRequest).parameters
            image_requests += 1
            for tile in query["tiles"]:
                frame = tile["frame_index"]
                frame_requests += 1
                revisits += frame in requested
                requested.add(frame)
        if attempt.telemetry is None:
            continue
        report = verify_report(layout, attempt.telemetry)
        revision = task.initial_revision
        for response in iter_responses(layout, attempt.telemetry):
            requests_count = shown = repeated = 0
            for output in response.tool_outputs:
                if output.evidence is not None:
                    for artifact in output.evidence.artifacts:
                        if not artifact.path.endswith("/event.json"):
                            continue
                        event = read_document(layout, artifact, ServiceEvent)
                        if event.task != task_ref:
                            raise ValueError(
                                "response evidence belongs to another camera/attempt"
                            )
                        revision = read_document(
                            layout, event.result, AgentView
                        ).revision
                        requests_count += event.operation == "video_image"
                for image in output.images:
                    if image.source is None:
                        continue
                    from src.tasks.base.annotation.images import read_rendered

                    render, receipt = read_rendered(layout, image.source)
                    if render.manifest != context.manifest:
                        raise ValueError("response image belongs to another camera")
                    shown += 1
                    for tile in receipt.tiles:
                        repeated += tile.source_frame_index in presentations
                        presentations.add(tile.source_frame_index)
            current = store.annotation(revision)
            mean, pairs = _error(_frames(current, reference))
            responses.append(
                ResponseComparison(
                    attempt=launch.attempt,
                    thread_id=report.thread_id,
                    response_id=response.response_id,
                    usage=response.usage,
                    revision=revision,
                    summary=progress_summary(current),
                    mean_reference_error_px=mean,
                    reference_pairs=pairs,
                    image_requests=requests_count,
                    image_presentations=shown,
                    revisited_frames=repeated,
                )
            )
    return CameraComparison(
        request=request_ref,
        camera_id=expected.source.camera_id,
        worker_kind="not_started"
        if not history
        else "synthetic"
        if all(synthetic)
        else "real"
        if not any(synthetic)
        else "mixed",
        run=run_ref,
        annotation=annotation_ref,
        summary=summary,
        image_requests=image_requests,
        frames_requested=frame_requests,
        revisited_frames=revisits,
        usage=usage,
        responses=tuple(responses),
        frames=frame_rows,
    )


def compare_blank_conditions(
    layout: AnnotationLayout,
    conditions: tuple[ArtifactRef, ...],
    *,
    reference_annotations: tuple[ArtifactRef, ...] = (),
) -> ArtifactRef:
    """References are parent-only scoring inputs; they never enter a worker run."""
    if len(conditions) < 2 or len(set(conditions)) != len(conditions):
        raise ValueError("compare at least two distinct blank conditions")
    fixed = [read_document(layout, r, BlankCondition) for r in conditions]
    if len({c.source_plan for c in fixed}) != 1 or len(
        {c.policy.sha256 for c in fixed}
    ) != len(fixed):
        raise ValueError(
            "blank conditions need identical input and distinct frozen policies"
        )
    references = tuple(
        read_annotation(verify_reference(layout, r)) for r in reference_annotations
    )
    if len({r.input_identity for r in references}) != len(references):
        raise ValueError("comparison references contain duplicate camera/input scopes")
    from src.tasks.ball_detection.annotation.video import VideoAnnotationInput

    expected_inputs = set()
    for request_ref in fixed[0].requests:
        request = read_document(layout, request_ref, WorkRequest)
        context = read_document(layout, request.payload, BlankAnnotationContext)
        expected_inputs.add(
            read_document(layout, context.task_input, VideoAnnotationInput)
        )
    if any(reference.input_identity not in expected_inputs for reference in references):
        raise ValueError("comparison reference belongs to another camera/input scope")
    results = []
    for condition in fixed:
        plan = read_document(layout, condition.plan, LocalAnnotationPlan)
        if plan.phase != "benchmark" or plan.publish_dataset:
            raise ValueError("blank comparison may not use normal publication state")
        local = layout.shared().for_benchmark(plan.campaign_id)
        canonical = annotation_reference(
            local,
            local.path(
                "campaigns", plan.campaign_id, "benchmark", "blank-condition.json"
            ),
        )
        if read_document(local, canonical, BlankCondition) != condition:
            raise ValueError(
                "comparison condition differs from registered inputs/runtime"
            )
        entries = LocalInputLedger(local).entries(plan)
        if tuple(e.request for e in entries) != condition.requests:
            raise ValueError("comparison camera inputs differ from the registered set")
        for entry in entries:
            context = read_document(local, entry.context, BlankAnnotationContext)
            if (
                read_document(local, context.task_input, VideoAnnotationInput)
                not in expected_inputs
            ):
                raise ValueError("comparison conditions do not share identical inputs")
        results.append(
            tuple(
                _camera(local, plan, request, references)
                for request in condition.requests
            )
        )
    report = BlankComparison(
        conditions=conditions, references=reference_annotations, results=tuple(results)
    )
    directory = layout.path(
        "campaigns",
        read_document(layout, fixed[0].plan, LocalAnnotationPlan).campaign_id,
        "reports",
        "blank-comparisons",
    )
    return write_document(layout, directory / (report.sha256 + ".json"), report)
