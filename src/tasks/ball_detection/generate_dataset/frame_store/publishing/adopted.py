"""Compile a sealed review into store labels, retaining the original evidence."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.annotation.review.adapter import BallReviewAdapter
from src.tasks.ball_detection.annotation.review.contracts import (
    BallLabels,
    BallTeacherMask,
)
from src.tasks.ball_detection.generate_dataset.frame_store.clip import (
    BallInstance,
    ClipLabels,
    ClipSpec,
    SourceFrame,
    VideoFrames,
)
from src.tasks.base.annotation.artifacts import read_document, verify_reference
from src.tasks.base.annotation.contracts import (
    AdoptionReceipt,
    ArtifactRef,
    ClipManifest,
)
from src.tasks.base.annotation.editing import ReviewContext, Revision, RevisionStore
from src.tasks.base.annotation.paths import AnnotationLayout


@dataclass(frozen=True)
class AdoptedClip:
    receipt: ArtifactRef
    source_id: str
    spec: ClipSpec
    heatmap: NDArray[np.uint8]
    coordinate: NDArray[np.uint8]


def load_adopted(
    layout: AnnotationLayout, reference: ArtifactRef, plan: ArtifactRef
) -> AdoptedClip:
    receipt = read_document(layout, reference, AdoptionReceipt)
    if receipt.task != "ball":
        raise ValueError("ball publication requires ball adoptions")
    revision = read_document(layout, receipt.revision, Revision)
    context = read_document(layout, revision.context, ReviewContext)
    if context.plan != plan:
        raise ValueError("adoption belongs to another campaign plan")
    adapter = BallReviewAdapter()
    if RevisionStore(layout).current(revision.context, adapter) != (
        receipt.revision,
        reference,
    ):
        raise ValueError("publication requires the exact sealed adoption")
    manifest = read_document(layout, context.manifest, ClipManifest)
    adapter.validate_adoption(layout, context, manifest, receipt.revision, receipt)
    labels = read_document(layout, revision.payload, BallLabels)
    mask = read_document(layout, receipt.teacher_mask, BallTeacherMask)
    media_path = verify_reference(layout, manifest.media)
    frames = []
    for row in labels.frames:
        balls = []
        for observation in row.decision.observations:
            kind = {
                "visible": "observed",
                "interpolated": "interpolated",
                "occluded": "occlusion_estimated"
                if observation.center is not None
                else "unresolved",
                "unresolved": "unresolved",
                "out_of_frame": "out_of_frame",
            }[observation.state]
            balls.append(
                BallInstance(
                    observation.track_id,
                    kind,
                    observation.center,
                    observation.state == "occluded",
                )
            )
        # The store has one event column. Multiple event kinds remain in the
        # immutable revision; never choose one arbitrarily or claim "none".
        events = set(row.decision.boundaries).intersection({"hit", "bounce"})
        event = next(iter(events)) if len(events) == 1 else "unlabeled"
        frames.append(
            SourceFrame(
                pts=row.frame.clip_pts,
                annotated=row.decision.reviewed,
                is_target=row.target,
                segment_break=bool(row.decision.boundaries),
                event=event,
                balls=tuple(balls),
            )
        )
    packed = ClipLabels.from_frames(
        frames, width=manifest.width, height=manifest.height
    )
    spec = ClipSpec(
        clip_id=manifest.clip_id,
        source="chat_annotation",
        group_id=manifest.source.canonical_id,
        camera_id=None,
        width=manifest.width,
        height=manifest.height,
        time_base=Fraction(manifest.time_base),
        fps=Fraction(manifest.nominal_fps),
        has_events=any(f.event != "unlabeled" for f in frames),
        annotation_path=layout.artifact_path(revision.payload),
        annotation_sha256=revision.payload.digest,
        media_sha256=manifest.media.digest,
        media=VideoFrames(
            media_path,
            tuple(f.clip_pts for f in manifest.frames),
            Fraction(manifest.time_base),
        ),
        labels=packed,
        provenance={
            "adoption": reference.model_dump(mode="json"),
            "manifest": context.manifest.model_dump(mode="json"),
            "revision": receipt.revision.model_dump(mode="json"),
            "labels": revision.payload.model_dump(mode="json"),
            "teacher_mask": receipt.teacher_mask.model_dump(mode="json"),
            "source_id": manifest.source.canonical_id,
        },
    )
    return AdoptedClip(
        reference,
        manifest.source.canonical_id,
        spec,
        np.asarray(
            [{"ignore": 0, "negative": 1, "positive": 2}[r.heatmap] for r in mask.rows],
            np.uint8,
        ),
        np.asarray([int(r.coordinate_track is not None) for r in mask.rows], np.uint8),
    )
