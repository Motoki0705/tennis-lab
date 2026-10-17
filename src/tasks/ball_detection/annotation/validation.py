"""Validate saved proposals without loading a model or importing torch."""

from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.contracts import ArtifactRef, ClipManifest
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.predictions import PredictionContext

from .contracts import BallPredictions, MDDInferenceConfig
from .windows import window_owners


def validate_predictions(
    layout: AnnotationLayout,
    context: PredictionContext,
    manifest: ClipManifest,
    result: ArtifactRef,
) -> BallPredictions:
    document: BallPredictions = read_document(layout, result, BallPredictions)
    config = read_document(layout, context.configuration, MDDInferenceConfig)
    config.validate_media(manifest)
    owners = window_owners(len(manifest.frames), config.stride)
    if document.context_digest != context.sha256 or len(document.rows) != len(
        manifest.frames
    ):
        raise ValueError(
            "ball prediction source/config/frame count differs from its request"
        )
    for row, frame, owner in zip(document.rows, manifest.frames, owners, strict=True):
        if (
            row.frame_index,
            row.source_frame_index,
            row.source_pts,
            row.clip_pts,
            row.window_start,
        ) != (
            frame.frame_index,
            frame.source_frame_index,
            frame.source_pts,
            frame.clip_pts,
            owner,
        ):
            raise ValueError(
                "ball prediction frame/PTS/window identity differs from the manifest"
            )
        if row.x > manifest.width - 1 or row.y > manifest.height - 1:
            raise ValueError("ball proposal is outside native pixel coordinates")
    return document
