"""Compile original completed v3 judgments against independently pinned inputs."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np

from src.tasks.annotation.contracts import LocalVideoManifest
from src.tasks.annotation.reviews import read_local_context
from src.tasks.ball_detection.annotation.video import (
    VideoAnnotationInput,
    read_annotation,
    usage_masks,
)
from src.tasks.ball_detection.generate_dataset.frame_store.clip import (
    BallInstance,
    ClipLabels,
    ClipSpec,
    SourceFrame,
)
from src.tasks.base.annotation.artifacts import read_document, verify_reference
from src.tasks.base.annotation.collection.local import LocalInputLedger
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    ClipManifest,
    content_digest,
)
from src.tasks.base.annotation.editing.contracts import BlankAnnotationContext
from src.tasks.base.annotation.jpeg_media import JpegArchive
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import LocalAnnotationPlan

from .adopted import PublicationClip
from .contracts import VideoPublishRequest


def recording_source_id(recording_id: str) -> str:
    """Use the same recording identity as the local registration's channel ID."""
    result: str = content_digest(recording_id)
    return result


def load_video_annotations(
    layout: AnnotationLayout, request: VideoPublishRequest, plan: LocalAnnotationPlan
) -> tuple[PublicationClip, ...]:
    if plan.task != "ball" or not plan.publish_dataset:
        raise ValueError("local plan must explicitly select ball dataset publication")
    entries = LocalInputLedger(layout).entries(plan)
    if {e.context for e in entries} != {a.context for a in request.annotations}:
        raise ValueError(
            "publication requires every registered camera/clip exactly once"
        )
    inputs = read_document(layout, plan.input_manifest, LocalVideoManifest)
    expected_inputs = []
    results = []
    for saved in request.annotations:
        expected = read_local_context(layout, saved.context, request.plan)
        expected_inputs.append(expected)
        context = read_document(layout, saved.context, BlankAnnotationContext)
        results.append(compile_video_annotation(
            layout, saved.annotation, expected, context.manifest,
            provenance=dict(
                context=saved.context.model_dump(mode="json"),
                input=context.task_input.model_dump(mode="json"),
                source_media=context.source_media.model_dump(mode="json"),
                manifest=context.manifest.model_dump(mode="json"),
            ),
        ))
    if len(expected_inputs) != len(inputs.videos) or {
        e.sha256 for e in expected_inputs
    } != {v.annotation.sha256 for v in inputs.videos}:
        raise ValueError("publication does not cover the frozen local input manifest")
    return tuple(sorted(results, key=lambda a: a.spec.clip_id))


def compile_video_annotation(
    layout: AnnotationLayout, annotation_ref: ArtifactRef,
    expected: VideoAnnotationInput, manifest_ref: ArtifactRef,
    *, provenance: dict[str, Any],
) -> PublicationClip:
    """Compile a judgment after its caller verifies the independently pinned images."""
    annotation = read_annotation(
        verify_reference(layout, annotation_ref),
        expected_input=expected,
        require_complete=True,
    )
    manifest = read_document(layout, manifest_ref, ClipManifest)
    if manifest.jpeg is None:
        raise ValueError("local v3 publication requires prepared native JPEGs")
    masks = usage_masks(annotation)
    frames = []
    for row in annotation.frames:
        position = row.position
        # Preserve general estimates as their own point kind. The exact
        # state/occlusion/support/notes remain in the snapshot's v3 payload.
        balls = (
            ()
            if row.state == "no_active_ball"
            else (
                BallInstance(
                    annotation.target.track_id,
                    position.provenance.kind
                    if position
                    else "out_of_frame"
                    if row.state == "out_of_frame"
                    else "unresolved",
                    (position.x, position.y) if position else None,
                    row.occlusion == "occluded",
                ),
            )
        )
        frames.append(
            SourceFrame(
                pts=row.pts,
                annotated=True,
                is_target=True,
                segment_break=row.break_before,
                event="unlabeled",
                balls=balls,
            )
        )
    source = annotation.source
    source_id = recording_source_id(source.recording_id)
    spec = ClipSpec(
        clip_id=manifest.clip_id,
        source="chat_annotation",
        group_id=source_id,
        camera_id=source.camera_id,
        width=source.width,
        height=source.height,
        time_base=Fraction(source.time_base),
        fps=Fraction(source.nominal_fps),
        has_events=False,
        annotation_path=layout.artifact_path(annotation_ref),
        annotation_sha256=annotation_ref.digest,
        media_sha256=manifest.media.digest,
        media=JpegArchive(
            verify_reference(layout, manifest.media),
            manifest.jpeg,
            source.width,
            source.height,
        ),
        labels=ClipLabels.from_frames(
            frames, width=source.width, height=source.height
        ),
        provenance=dict(
            target_semantics="active_ball",
            recording_id=source.recording_id,
            annotation_schema=annotation.schema_version,
            annotation_revision=annotation_ref.model_dump(mode="json"),
            **provenance,
            source_id=source_id,
        ),
    )
    positive = np.asarray(masks.position_usable, np.bool_)
    negative = np.asarray(masks.active_ball_negative_candidate, np.bool_)
    return PublicationClip(
        annotation_ref, source_id, spec,
        (2 * positive + negative).astype(np.uint8), positive.astype(np.uint8), annotation,
    )
