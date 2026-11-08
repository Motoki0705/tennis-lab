"""Shared ``video_ball_annotation.v3`` schema, persistence and usage decisions."""

from .contracts import (
    BallTarget,
    FrameIdentity,
    PixelPosition,
    PointProvenance,
    VideoAnnotationInput,
    VideoBallAnnotation,
    VideoBallFrame,
    VideoIdentity,
)
from .io import read_annotation, write_annotation
from .usage import (
    AnnotationSummary,
    UsageMasks,
    is_active_ball_negative_candidate,
    is_direct_observation,
    is_estimated,
    is_final_unknown,
    is_position_unavailable,
    is_position_usable,
    is_processed,
    is_unprocessed,
    summarize_annotation,
    usage_masks,
)

__all__ = [
    "AnnotationSummary",
    "BallTarget",
    "FrameIdentity",
    "PixelPosition",
    "PointProvenance",
    "UsageMasks",
    "VideoAnnotationInput",
    "VideoBallAnnotation",
    "VideoBallFrame",
    "VideoIdentity",
    "is_active_ball_negative_candidate",
    "is_direct_observation",
    "is_estimated",
    "is_final_unknown",
    "is_position_unavailable",
    "is_position_usable",
    "is_processed",
    "is_unprocessed",
    "read_annotation",
    "summarize_annotation",
    "usage_masks",
    "write_annotation",
]
