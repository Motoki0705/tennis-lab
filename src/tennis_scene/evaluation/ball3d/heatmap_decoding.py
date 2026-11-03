"""Single-point decoding experiments through the shared baseline reconstruction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from src.tasks.ball_detection.inference.heatmap_decoding import (
    HeatmapDecodingConfig,
    decode_native_sequence,
)
from src.tasks.ball_detection.inference.trajectory_gate import apply_trajectory_gate
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.ball_points import (
    BallPointsInput,
    BallPointsModule,
)
from src.utils.geometry.keypoints import denormalize_grid_keypoints

from .baseline import PipelineBaseline
from .contracts import Ball3DResult, MethodInput, ViewInput


def redecode_view(view: ViewInput, decoding: HeatmapDecodingConfig) -> tuple[BallDetectionOutput, np.ndarray]:
    """Keep source evidence, peak scores, threshold and production trajectory gate."""
    evidence = view.detections.evidence
    if evidence is None:
        raise ValueError("Re-decoding requires saved native heatmaps")
    uv, confidence, status = decode_native_sequence(
        evidence.heatmaps, decoding, subpixel_refine=view.detector_config.subpixel_refine,
    )
    uv_px = denormalize_grid_keypoints(uv, *evidence.source_size_wh)
    observed = np.isfinite(uv_px).all(-1) & np.isfinite(confidence) & (confidence >= view.detector_config.score_threshold)
    gate = view.detector_config.trajectory_gate
    if gate.enabled:
        observed, _ = apply_trajectory_gate(
            np.where(observed[:, None], uv_px, 0).astype(np.float32), observed,
            np.where(observed, confidence, 0).astype(np.float32),
            **{k: v for k, v in asdict(gate).items() if k != "enabled"},
        )
    return BallDetectionOutput(
        view.camera.camera_id, view.detections.frame_indices,
        np.where(observed[:, None], uv_px, 0).astype(np.float32),
        np.where(observed, confidence, 0).astype(np.float32), observed,
        observed.astype(np.uint8), "model_score", evidence,
    ), status


@dataclass(frozen=True)
class HeatmapDecodingMethod:
    """Runner adapter; annotations and held-out observations are never accepted."""

    name: str
    decoding: HeatmapDecodingConfig
    baseline: PipelineBaseline

    @property
    def config(self) -> Mapping[str, Any]:
        return {"decoding": asdict(self.decoding), "downstream": dict(self.baseline.config)}

    def __call__(self, inputs: MethodInput) -> Ball3DResult:
        views, statuses = [], []
        for view in inputs.views:
            detection, status = redecode_view(view, self.decoding)
            # ViewInput's source-mask contract requires matching replayed points.
            # Reuse the real component. PipelineBaseline deliberately replays it
            # again; this adapter's timing includes that unoptimized validation.
            points = BallPointsModule(view.smoothing_config).process(
                BallPointsInput(detection, inputs.source_size_wh, inputs.fps)
            )
            views.append(replace(view, detections=detection, points=points))
            statuses.append(status)
        result = self.baseline(replace(inputs, views=tuple(views)))
        return replace(result, diagnostics={
            **(result.diagnostics or {}), "decoding_fit_status": np.stack(statuses),
        })
