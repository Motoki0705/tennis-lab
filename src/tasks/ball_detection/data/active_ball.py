"""Validate explicit active-ball teachers without applying legacy review gates."""

from __future__ import annotations

import json
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from src.tasks.ball_detection.annotation.video import (
    VideoBallAnnotation,
    summarize_annotation,
    usage_masks,
)
from src.tasks.base.annotation.contracts import content_digest

if TYPE_CHECKING:
    from .store import BallFrameStore
    from .teacher_masks import FrameTeacherMasks

ACTIVE_BALL_SCHEMA = "ball_detection_frames.v4"
POSITION_WEIGHTS = dict(
    observed=1.0, interpolated=1.0, estimated=1.0, occlusion_estimated=1.0
)


def validate_active_annotations(
    store: BallFrameStore, raw: Any
) -> dict[str, VideoBallAnnotation]:
    """Validate embedded original judgments and their row-aligned projection.

    Embedding the task-owned v3 document makes the snapshot self-contained even
    when the annotation workspace is unavailable. Artifact references still pin
    the saved revision, registered input and exact image archive used to publish.
    """
    from .store import EVENT_CODES, POINT_KIND_CODES

    if (
        store.metadata.get("target_semantics") != "active_ball"
        or store.metadata.get("position_weights") != POSITION_WEIGHTS
    ):
        raise ValueError(
            "active_ball snapshot requires explicit target semantics and uniform position weights"
        )
    if not isinstance(raw, dict) or set(raw) != {c.clip_id for c in store.clips}:
        raise ValueError("active_ball annotations must cover every snapshot clip")
    annotations = {}
    splits: dict[str, str] = {}
    for clip in store.clips:
        annotation = VideoBallAnnotation.model_validate_json(
            json.dumps(raw[clip.clip_id])
        )
        source = annotation.source
        if not summarize_annotation(annotation).complete:
            raise ValueError("active_ball snapshot cannot contain unprocessed frames")
        if (
            clip.group_id != content_digest(source.recording_id)
            or clip.camera_id != source.camera_id
            or (clip.width, clip.height, clip.source_width, clip.source_height)
            != (source.width, source.height, source.width, source.height)
            or clip.frame_count != len(annotation.frames)
            or Fraction(clip.time_base) != Fraction(source.time_base)
            or Fraction(clip.fps) != Fraction(source.nominal_fps)
        ):
            raise ValueError("active_ball clip differs from its original v3 identity")
        if splits.setdefault(clip.group_id, clip.split) != clip.split:
            raise ValueError("active_ball recording spans different splits")
        rows = store.clip_rows(clip)
        if (
            not store.frames["annotated"][rows].all()
            or not store.frames["is_target"][rows].all()
            or store.frames["pts"][rows].tolist() != [f.pts for f in annotation.frames]
            or store.frames["segment_break"][rows].tolist()
            != [f.break_before for f in annotation.frames]
            or (store.frames["event"][rows] != EVENT_CODES["unlabeled"]).any()
        ):
            raise ValueError(
                "active_ball frame mapping differs from original v3 annotations"
            )
        for row_index, frame in zip(rows, annotation.frames, strict=True):
            instances = store.instances_of(int(row_index))
            if len(instances.point_kind) != (
                0 if frame.state == "no_active_ball" else 1
            ):
                raise ValueError(
                    "active_ball instance count differs from original v3 state"
                )
            if not len(instances.point_kind):
                continue
            position = frame.position
            kind = (
                position.provenance.kind
                if position
                else ("out_of_frame" if frame.state == "out_of_frame" else "unresolved")
            )
            xy = (position.x, position.y) if position else (np.nan, np.nan)
            if (
                instances.point_kind[0] != POINT_KIND_CODES[kind]
                or bool(instances.occluded[0]) != (frame.occlusion == "occluded")
                or not np.array_equal(
                    instances.xy[0], np.asarray(xy, np.float32), equal_nan=True
                )
                or clip.track_ids[int(instances.track_index[0])]
                != annotation.target.track_id
            ):
                raise ValueError(
                    "active_ball teacher differs from original v3 judgment"
                )
        annotations[clip.clip_id] = annotation
    return annotations


def validate_active_masks(store: BallFrameStore, teachers: FrameTeacherMasks) -> None:
    for clip in store.clips:
        masks = usage_masks(store.video_annotations[clip.clip_id])
        rows = store.clip_rows(clip)
        positive = np.asarray(masks.position_usable, np.uint8)
        expected = 2 * positive + np.asarray(
            masks.active_ball_negative_candidate, np.uint8
        )
        if not np.array_equal(teachers.heatmap[rows], expected) or not np.array_equal(
            teachers.coordinate[rows], positive
        ):
            raise ValueError(
                "active_ball teacher masks differ from the v3 usage decisions"
            )
