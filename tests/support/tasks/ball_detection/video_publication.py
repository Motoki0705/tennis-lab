"""Completed original v3 inputs using the real local registration and publisher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.tasks.annotation.configuration import (
    LocalVideoConfiguration,
    configure_local_videos,
)
from src.tasks.annotation.contracts import LocalVideoInput
from src.tasks.annotation.local_videos import LocalVideoConfigurationReceipt
from src.tasks.annotation.reviews import read_local_context
from src.tasks.ball_detection.annotation.video import (
    FrameIdentity,
    PixelPosition,
    PointProvenance,
    VideoBallAnnotation,
    VideoBallFrame,
    write_annotation,
)
from src.tasks.ball_detection.annotation.video.contracts import (
    BallState,
    Occlusion,
    PointKind,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
    BallPublicationPolicy,
    SavedVideoAnnotation,
    SourceSplit,
    Split,
    VideoPublishRequest,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.publisher import (
    BallPublisher,
)
from src.tasks.base.annotation.artifacts import read_document, write_document
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    ClipManifest,
    FrameRange,
    PublicationReceipt,
    content_digest,
)
from src.tasks.base.annotation.editing.contracts import BlankAnnotationContext
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import LocalAnnotationPlan, ProductPlan
from tests.support.tasks.local_annotation import local_configuration


@dataclass(frozen=True)
class VideoEnvironment:
    layout: AnnotationLayout
    plan: LocalAnnotationPlan
    plan_ref: ArtifactRef
    manifests: tuple[ClipManifest, ...]

    @property
    def manifest(self) -> ClipManifest:
        assert len(self.manifests) == 1
        return self.manifests[0]


def scoped(video: LocalVideoInput, start: int, stop: int) -> LocalVideoInput:
    # Independently known VFR timeline of tests.support.tasks.local_annotation.
    pts = (0, 640, 1920, 2560, 4480)
    expected = video.annotation.model_copy(
        update=dict(
            frame_range=FrameRange(start=start, stop=stop),
            frames=tuple(
                FrameIdentity(frame_index=i, pts=pts[i]) for i in range(start, stop)
            ),
        )
    )
    return video.model_copy(update={"annotation": expected})


def prepare(
    layout: AnnotationLayout,
    data: LocalVideoConfiguration,
    *,
    name: str,
    version: str,
    videos: tuple[LocalVideoInput, ...],
    base: ArtifactRef | None = None,
    split: Split = "val",
) -> tuple[VideoEnvironment, VideoPublishRequest]:
    configuration = data.model_copy(
        update=dict(
            campaign=data.campaign.model_copy(update={"campaign_id": name}),
            budget=data.budget.model_copy(update={"scope_id": name + "-scope"}),
            videos=videos,
            publish_dataset=True,
        )
    )
    registration = read_document(
        layout,
        configure_local_videos(layout, configuration),
        LocalVideoConfigurationReceipt,
    )
    saved, manifests = [], []
    for entry in registration.registrations:
        expected = read_local_context(layout, entry.context, registration.plan)
        context = read_document(layout, entry.context, BlankAnnotationContext)
        manifests.append(read_document(layout, context.manifest, ClipManifest))
        frames = []
        decisions: dict[
            int,
            tuple[BallState, Occlusion, PointKind | None, tuple[float, float] | None],
        ]
        if expected.source.camera_id == "cam0":
            decisions = {
                1: ("in_frame", "not_occluded", "observed", (9.0, 13.0)),
                2: ("in_frame", "unknown", "estimated", (22.5, 7.25)),
                3: ("unknown", "unknown", None, None),
                4: ("no_active_ball", "not_applicable", None, None),
            }
        else:
            decisions = {
                0: ("in_frame", "not_occluded", "observed", (61.5, 19.25)),
                1: ("in_frame", "unknown", "interpolated", (49.25, 16.5)),
                2: ("in_frame", "occluded", "occlusion_estimated", (31.75, 8.0)),
                3: ("in_frame", "occluded", None, None),
                4: ("out_of_frame", "not_applicable", None, None),
            }
        for frame in expected.frames:
            state, occlusion, kind, xy = decisions[frame.frame_index]
            support = (
                (1, 3)
                if kind == "estimated"
                else (0, 2)
                if kind == "interpolated"
                else (0, 4)
                if kind == "occlusion_estimated"
                else ()
            )
            position = None
            if xy is not None:
                assert kind is not None
                position = PixelPosition(
                    x=xy[0],
                    y=xy[1],
                    provenance=PointProvenance(kind=kind, support_frames=support),
                )
            frames.append(
                VideoBallFrame(
                    frame_index=frame.frame_index,
                    pts=frame.pts,
                    state=state,
                    occlusion=occlusion,
                    position=position,
                    break_before=frame.frame_index == expected.frame_range.start,
                    notes="original camera context; no triangulation or smoothing",
                )
            )
        annotation = VideoBallAnnotation(
            schema_version="video_ball_annotation.v3",
            source=expected.source,
            target=expected.target,
            frame_range=expected.frame_range,
            coordinate_system="native_pixels_top_left_endpoint",
            frames=tuple(frames),
        )
        reference = write_annotation(
            layout,
            layout.path("annotations", "v3-fixtures", name, expected.sha256 + ".json"),
            annotation,
            expected_input=expected,
            require_complete=True,
        )
        saved.append(SavedVideoAnnotation(context=entry.context, annotation=reference))
    policy = BallPublicationPolicy(
        jpeg_quality=90,
        source_splits=tuple(
            SourceSplit(source_id=content_digest(recording), split=split)
            for recording in sorted({v.annotation.source.recording_id for v in videos})
        ),
        target_semantics="active_ball",
    )
    policy_ref = write_document(
        layout, layout.path("campaigns", name, "publication-policy.json"), policy
    )
    request = VideoPublishRequest(
        plan=registration.plan,
        output=ProductPlan(
            product="frames",
            dataset_task="ball_detection",
            dataset_version=version,
            component="frames",
            base_snapshot=base,
            generation_policy=policy_ref,
        ),
        annotations=tuple(sorted(saved, key=lambda s: s.context.digest)),
    )
    return VideoEnvironment(
        layout,
        read_document(layout, registration.plan, LocalAnnotationPlan),
        registration.plan,
        tuple(manifests),
    ), request


def incremental_case(
    tmp_path: Path, *, include_related: bool = False
) -> tuple[VideoEnvironment, VideoPublishRequest, Path]:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    data = local_configuration(layout)
    first, second = data.videos
    _, initial = prepare(
        layout, data, name="initial", version="baseline", videos=(scoped(first, 1, 3),)
    )
    base = read_document(
        layout, BallPublisher(layout).publish(initial), PublicationReceipt
    ).snapshot
    videos = (
        (scoped(first, 3, 5), scoped(second, 0, 5))
        if include_related
        else (scoped(second, 0, 5),)
    )
    env, request = prepare(
        layout, data, name="next", version="incremental", videos=videos, base=base
    )
    return env, request, layout.artifact_path(base).parent
