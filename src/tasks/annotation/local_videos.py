"""Scene-selected local videos to blank annotation plans, media and work requests."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import Field

from src.tasks.ball_detection.annotation.video.agent.contracts import AnnotationPolicy
from src.tasks.ball_detection.annotation.video.inputs import validate_video_input
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.collection.local import (
    LocalInputLedger,
    LocalInputRegistration,
    retain_local_source,
)
from src.tasks.base.annotation.collection.preparation import prepare_frame_range
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    FrameRange,
    MediaFingerprint,
    SourceIdentity,
    content_digest,
)
from src.tasks.base.annotation.editing import BlankAnnotationContext
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import (
    ExecutionPlan,
    LocalAnnotationPlan,
    load_execution_plan,
    references,
    save_plan,
    validate_new_plan_worker,
)
from src.tasks.base.annotation.reproducibility import require_frozen_code
from src.tasks.base.annotation.resources import ResourceManager
from src.utils.paths import PROJECT_ROOT

from .configuration import (
    ConfigurationReceipt,
    ExistingBudget,
    LocalVideoConfiguration,
    _scope,
    validate_runtime,
)
from .contracts import LocalAnnotationRuntime, LocalVideoManifest, WorkerRuntimeSettings
from .reviews import local_video_request


class LocalVideoConfigurationReceipt(ConfigurationReceipt):
    annotation_policy: ArtifactRef
    registrations: tuple[LocalInputRegistration, ...] = Field(min_length=1)


def _document_reference(
    layout: AnnotationLayout, path: Path, payload: bytes
) -> ArtifactRef:
    """Compute a future immutable reference so all validation precedes persistence."""
    return ArtifactRef(
        role="annotation",
        path=path.relative_to(layout.root).as_posix(),
        digest=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def register_local_videos(
    layout: AnnotationLayout,
    data: LocalVideoConfiguration,
    *,
    annotation_policy: AnnotationPolicy,
) -> ArtifactRef:
    # Reject prediction-specific settings before nested base-class serialization
    # could discard them. JSON callers already get extra-field validation.
    ExecutionPlan.model_validate_json(data.campaign.canonical_bytes())
    WorkerRuntimeSettings.model_validate_json(data.runtime.canonical_bytes())
    data = LocalVideoConfiguration.model_validate_json(data.canonical_bytes())
    annotation_policy = AnnotationPolicy.model_validate_json(
        annotation_policy.canonical_bytes()
    )
    if data.campaign.phase != "production" or layout.review_namespace is not None:
        raise ValueError(
            "local blank inputs require a production campaign without benchmark state"
        )
    if data.campaign.annotation_root != str(layout.root):
        raise ValueError("configuration plan belongs to a different annotation root")
    directory = layout.path("campaigns", data.campaign.campaign_id, "configuration")
    manifest = LocalVideoManifest(videos=data.videos)
    manifest_path = directory / "local-videos.json"
    input_reference = _document_reference(
        layout, manifest_path, manifest.canonical_bytes() + b"\n"
    )
    plan = LocalAnnotationPlan(
        **data.campaign.model_dump(),
        task="ball",
        input_manifest=input_reference,
        publish_dataset=data.publish_dataset,
    )
    validate_new_plan_worker(layout, plan)
    if layout.mode == "production":
        require_frozen_code(PROJECT_ROOT, plan)

    scopes: dict[str, list[FrameRange]] = {}
    timelines = []
    for video in data.videos:
        expected = video.annotation
        source = expected.source
        path = Path(video.path)
        if (
            not path.is_absolute()
            or path.resolve() != path
            or path.name != source.file_name
        ):
            raise ValueError(
                "local video path must be absolute, normalized and match the original file_name"
            )
        # A recording group can contain several original files for one camera.
        # Native frame indices are scoped to the full task-owned video identity.
        intervals = scopes.setdefault(source.sha256, [])
        if any(
            r.start < expected.frame_range.stop and expected.frame_range.start < r.stop
            for r in intervals
        ):
            raise ValueError(
                "local input repeats or overlaps an original video/camera frame range"
            )
        intervals.append(expected.frame_range)
        timelines.append(validate_video_input(path, expected))

    scope = _scope(layout, plan, data.budget)
    validate_runtime(layout, data.runtime, plan, scope.roots)
    declared = {stage.stage: stage for stage in plan.resources.stages}
    for name, kind in (("review", "codex"), ("render", "cpu")):
        data.runtime.cost(name)
        if name not in declared or declared[name].kind != kind:
            raise ValueError(
                "local annotation requires Codex review and CPU render stages"
            )
    for cost in data.runtime.costs:
        if (
            cost.pss_bytes > scope.limits.total_pss_bytes
            or cost.storage_bytes > scope.limits.additional_storage_bytes
        ):
            raise ValueError("a stage reservation exceeds the total campaign budget")

    with artifact_lock(layout, "workflow-configuration-" + plan.campaign_id):
        plan_path = layout.path("campaigns", plan.campaign_id, "plan.json")
        if plan_path.exists() and load_execution_plan(plan_path) != plan:
            raise ValueError(
                "configuration differs from the existing immutable plan/input version"
            )
        # Freeze request identity before media preparation. A stop at any later
        # publication/commit boundary can only resume the same configuration.
        request_path = directory / "request.json"
        if (
            request_path.exists()
            and LocalVideoConfiguration.model_validate_json(request_path.read_bytes())
            != data
        ):
            raise ValueError("local configuration differs from its original request")
        policy_path = directory / "annotation-policy.json"
        if (
            policy_path.exists()
            and read_document(
                layout, annotation_reference(layout, policy_path), AnnotationPolicy
            )
            != annotation_policy
        ):
            raise ValueError("annotation policy differs from its original registration")
        # A new campaign must not reserve a budget for work already bound to
        # another plan. Source/camera and native range define the shared subject.
        for video in data.videos:
            expected = video.annotation
            video_id = expected.source.sha256
            clip_id = content_digest(
                [
                    content_digest(["local", video_id]),
                    expected.frame_range.model_dump(),
                ]
            )
            context_path = layout.path("annotations", "ball", clip_id, "context.json")
            if context_path.exists():
                previous = read_document(
                    layout,
                    annotation_reference(layout, context_path),
                    BlankAnnotationContext,
                )
                if read_document(layout, previous.plan, LocalAnnotationPlan) != plan:
                    raise ValueError(
                        "local subject is already registered with a different plan/input version"
                    )
        if isinstance(data.budget, ExistingBudget):
            resources = ResourceManager(layout, data.budget.scope)
        else:
            resources = ResourceManager.create(
                layout,
                scope_id=scope.scope_id,
                limits=scope.limits,
                roots=tuple(Path(root) for root in scope.roots),
                previously_allocated_bytes=scope.previously_allocated_bytes,
                previously_used_gpu_seconds=scope.previously_used_gpu_seconds,
                max_pending_per_stage=scope.max_pending_per_stage,
                free_space_mounts=tuple(Path(path) for path in scope.free_space_mounts),
            )
        resources.bind(plan)
        request_ref = write_document(layout, request_path, data)
        policy_ref = write_document(layout, policy_path, annotation_policy)
        write_document(layout, manifest_path, manifest)
        plan_ref = annotation_reference(layout, save_plan(layout, plan))
        runtime_ref = write_document(
            layout,
            directory / "runtime.json",
            LocalAnnotationRuntime(
                plan=plan_ref, scope=resources.reference, **data.runtime.model_dump()
            ),
        )
        registrations = []
        ledger = LocalInputLedger(layout)
        for video, timeline in zip(data.videos, timelines, strict=True):
            expected = video.annotation
            identity = expected.source
            original = retain_local_source(
                layout,
                Path(video.path),
                MediaFingerprint(
                    digest=identity.media_sha256, size_bytes=identity.size_bytes
                ),
            )
            source = SourceIdentity(
                provider="local",
                channel_id=content_digest(identity.recording_id),
                video_id=identity.sha256,
                media=original,
                origin_url=Path(video.path).as_uri(),
            )
            task_input = write_document(
                layout,
                layout.path(
                    "campaigns",
                    plan.campaign_id,
                    "local-inputs",
                    expected.sha256 + ".json",
                ),
                expected,
            )
            clip = prepare_frame_range(
                layout,
                source,
                timeline,
                source_range=expected.frame_range,
                target_range=FrameRange(start=0, stop=len(expected.frames)),
                preparation_digest=expected.sha256,
                retained_media="jpeg_q90",
            )
            entry = local_video_request(
                layout, plan, plan_ref, clip, task_input, original
            )
            ledger.register(plan, entry)
            registrations.append(entry)
        receipt = LocalVideoConfigurationReceipt(
            request=request_ref,
            plan=plan_ref,
            scope=resources.reference,
            runtime=runtime_ref,
            annotation_policy=policy_ref,
            registrations=tuple(registrations),
        )
        for reference in references(receipt):
            verify_reference(layout, reference)
        return write_document(layout, directory / "result.json", receipt)
