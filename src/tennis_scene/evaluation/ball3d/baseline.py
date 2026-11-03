"""CPU replay of the production detector observations and ball reconstruction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from src.tasks.ball_detection.inference.trajectory_gate import apply_trajectory_gate
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.ball_points import (
    BallPointsInput,
    BallPointsModule,
)
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    reconstruct_ball,
    single_ball_observations,
)
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.utils.data.heatmaps import heatmaps_to_argmax, refine_peaks_log_parabolic
from src.utils.geometry.keypoints import denormalize_grid_keypoints

from .bundle import FixedInputBundle
from .contracts import Ball3DResult, MethodInput, ViewInput


def replay_detector(view: ViewInput) -> BallDetectionOutput:
    """Decode the saved native maps; no predictor, checkpoint, RGB or GPU is used."""
    evidence = view.detections.evidence
    if evidence is None:
        raise ValueError("Detector replay requires native heatmaps")
    coords, scores = [], []
    for start in range(0, len(evidence.heatmaps), 64):
        maps = torch.from_numpy(np.array(evidence.heatmaps[start:start + 64], copy=True))[None]
        uv, score = heatmaps_to_argmax(maps)
        if view.detector_config.subpixel_refine:
            uv = refine_peaks_log_parabolic(maps, uv)
        coords.append(uv[0].numpy())
        scores.append(score[0].numpy())
    uv_px = denormalize_grid_keypoints(np.concatenate(coords), *evidence.source_size_wh)
    confidence = np.concatenate(scores).astype(np.float32)
    observed = np.isfinite(uv_px).all(-1) & np.isfinite(confidence) & (confidence >= view.detector_config.score_threshold)
    gate = view.detector_config.trajectory_gate
    if gate.enabled:
        observed, _ = apply_trajectory_gate(
            np.where(observed[:, None], uv_px, 0).astype(np.float32), observed,
            np.where(observed, confidence, 0).astype(np.float32),
            **{key: value for key, value in asdict(gate).items() if key != "enabled"})
    return BallDetectionOutput(view.camera.camera_id, view.detections.frame_indices,
        np.where(observed[:, None], uv_px, 0).astype(np.float32),
        np.where(observed, confidence, 0).astype(np.float32), observed, observed.astype(np.uint8),
        "model_score", evidence)


@dataclass(frozen=True)
class PipelineBaseline:
    """Reuse pinned decoded/gated detector points and re-run consensus/LOESS/3D.

    ``without_loess`` retains the same consensus acceptance mask and raw
    accepted detector coordinates, isolating only the LOESS coordinate change.
    """

    reprojection_px: float = 20.0
    min_frames: int = 5
    ball_threshold: float = 0.0
    without_loess: bool = False

    @property
    def name(self) -> str:
        return "without_loess" if self.without_loess else "pipeline_baseline"

    @property
    def config(self) -> Mapping[str, Any]:
        return {**asdict(self), "detector_input": "pinned_post_decode_threshold_gate",
                "without_loess_policy": "retain_consensus_mask_use_raw_accepted_coordinates"}

    def __call__(self, inputs: MethodInput) -> Ball3DResult:
        rows = [BallPointsModule(view.smoothing_config).process(
            BallPointsInput(view.detections, inputs.source_size_wh, inputs.fps)) for view in inputs.views]
        uv = np.stack([np.where(row.observed[:, None], view.detections.uv_px, 0)
                       if self.without_loess else row.uv_px
                       for row, view in zip(rows, inputs.views, strict=True)]).astype(np.float32)
        confidence = np.stack([row.confidence for row in rows])
        observed = np.stack([row.observed for row in rows])
        raw = ObjectObservations(inputs.camera_ids, inputs.source_size_wh, inputs.fps,
            uv[:, :, None, None], confidence[:, :, None, None], observed[:, :, None],
            np.zeros((len(rows), 1), np.int64))
        grouped = single_ball_observations(raw, threshold=self.ball_threshold)
        scale = np.hypot(*inputs.source_size_wh) / np.hypot(1920, 1080)
        ball = reconstruct_ball(grouped, tuple(view.camera for view in inputs.views), fps=inputs.fps,
                                reprojection_px=self.reprojection_px * scale, min_frames=self.min_frames)
        trajectory = ball.trajectory
        return Ball3DResult(inputs.camera_ids, inputs.frame_indices, trajectory.positions, trajectory.valid,
                            trajectory.inliers, diagnostics={"rejection_codes": trajectory.reasons,
                            "input_observed": observed, "accepted_uv_px": uv, "status": ball.status})


def baseline_from_bundle(bundle: FixedInputBundle, *, without_loess: bool = False) -> PipelineBaseline:
    settings = bundle.triangulation_settings
    if not settings["config"]["enabled"]:
        raise ValueError("Pinned baseline triangulation is disabled")
    return PipelineBaseline(settings["config"]["reprojection_px"], settings["config"]["min_frames"],
                            settings["threshold"], without_loess)


def verify_baseline(bundle: FixedInputBundle, *, detector_atol_px: float = .002, position_atol_m: float = 2e-6) -> dict[str, Any]:
    """Separate GPU/CPU subpixel rounding from exact replay of pinned observations."""
    counts: dict[str, Any] = {}
    for view in bundle.inputs.views:
        decoded = replay_detector(view)
        np.testing.assert_array_equal(decoded.observed, view.detections.observed)
        np.testing.assert_array_equal(decoded.confidence, view.detections.confidence)
        np.testing.assert_allclose(decoded.uv_px, view.detections.uv_px, atol=detector_atol_px, rtol=0)
        points = BallPointsModule(view.smoothing_config).process(
            BallPointsInput(view.detections, bundle.inputs.source_size_wh, bundle.inputs.fps))
        for field in ("uv_px", "observed", "confidence"):
            np.testing.assert_array_equal(getattr(points, field), getattr(view.points, field))
        counts[view.camera.camera_id] = {"detector_observed": int(decoded.observed.sum()),
            "points_observed": int(points.observed.sum()),
            "max_detector_coordinate_difference_px": float(np.max(np.abs(decoded.uv_px - view.detections.uv_px))),
            "ball_points_exact": True}
    result = baseline_from_bundle(bundle)(bundle.inputs)
    reference = bundle.baseline_reference.ball
    if reference is None:
        raise ValueError("Missing baseline comparison reference")
    np.testing.assert_allclose(result.positions, reference.trajectory.positions, atol=position_atol_m, rtol=0)
    for actual, expected in ((result.valid, reference.trajectory.valid),
                             (result.used_views, reference.trajectory.inliers)):
        np.testing.assert_array_equal(actual, expected)
    return {"source_code_commit": bundle.manifest["source_code_commit"], "bundle_sha256": bundle.manifest_sha256,
            "detector_coordinate_atol_px": detector_atol_px, "views": counts,
            "positions_bitwise_equal": bool(np.array_equal(result.positions, reference.trajectory.positions)),
            "position_atol_m": position_atol_m,
            "max_position_difference_m": float(np.max(np.abs(result.positions - reference.trajectory.positions))),
            "valid_mask_exact": True, "used_view_mask_exact": True,
            "valid_frames": int(result.valid.sum()), "num_frames": bundle.inputs.num_frames,
            "model_inference_rerun": False, "gpu_used": False}
