"""Validate saved proposals without loading a model or importing torch."""

import math
import struct

from src.tasks.base.annotation.contracts import ArtifactRef, ClipManifest
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.predictions import PredictionContext

from .contracts import (
    HybridInferenceConfig,
    HybridPredictions,
    HybridProposal,
    PredictionDocument,
    read_inference,
    read_predictions,
)
from .windows import window_owners


def validate_predictions(
    layout: AnnotationLayout,
    context: PredictionContext,
    manifest: ClipManifest,
    result: ArtifactRef,
) -> PredictionDocument:
    document = read_predictions(layout, result)
    config = read_inference(layout, context.configuration)
    config.validate_media(manifest)
    if isinstance(document, HybridPredictions) != isinstance(config, HybridInferenceConfig):
        raise ValueError("ball prediction family differs from the frozen inference configuration")
    if isinstance(document, HybridPredictions) and document.heatmap_size != (
        (manifest.height + 3) // 4, (manifest.width + 3) // 4
    ):
        raise ValueError("hybrid heatmap grid differs from native /4 dimensions")
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
        if row.center is not None and (
            row.center[0] > manifest.width - 1 or row.center[1] > manifest.height - 1
        ):
            raise ValueError("ball proposal is outside native pixel coordinates")
        if isinstance(row, HybridProposal) and isinstance(config, HybridInferenceConfig):
            # The model I/O compares sigmoid FP32 tensors with an FP32 threshold.
            threshold = struct.unpack("f", struct.pack("f", config.presence_threshold))[0]
            decay = math.exp(-abs(row.presence_logit))
            expected = 1 / (1 + decay) if row.presence_logit >= 0 else decay / (1 + decay)
            if not math.isclose(row.presence_score, expected, rel_tol=1e-6, abs_tol=6e-8):
                raise ValueError("hybrid presence score differs from its logit")
            if (row.presence == "present") != (row.presence_score >= threshold):
                raise ValueError("hybrid presence decision differs from the frozen threshold")
    return document
