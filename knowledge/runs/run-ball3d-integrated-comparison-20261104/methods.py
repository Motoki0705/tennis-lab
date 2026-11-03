"""Experiment-only compositions of the existing production reconstruction parts."""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np

from src.tasks.ball_detection.inference.heatmap_decoding import (
    HeatmapDecodingConfig,
    refine_peak_cells,
)
from src.tennis_scene.ball_association.topk import TopKConfig, associate_topk
from src.tennis_scene.ball_calibration.bundle_adjustment import BundleAdjustmentConfig
from src.tennis_scene.ball_gmm.config import CandidateGMMConfig
from src.tennis_scene.ball_likelihood import DenseLikelihoodConfig, reconstruct_dense
from src.tennis_scene.ball_trajectory.config import SegmentPhysicsConfig
from src.tennis_scene.evaluation.ball3d.baseline import baseline_from_bundle
from src.tennis_scene.evaluation.ball3d.candidate_gmm import CandidateGMM
from src.tennis_scene.evaluation.ball3d.contracts import Ball3DResult
from src.tennis_scene.evaluation.ball3d.court_ball_ba import CourtBallBAMethod
from src.tennis_scene.evaluation.ball3d.dense_likelihood import DenseHeatmapMethod
from src.tennis_scene.evaluation.ball3d.heatmap_decoding import (
    HeatmapDecodingMethod,
    redecode_view,
)
from src.tennis_scene.evaluation.ball3d.segment_physics import SegmentPhysicsMethod
from src.tennis_scene.evaluation.ball3d.topk import TopKMultiHypothesis
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
from src.utils.geometry.keypoints import denormalize_grid_keypoints


def typed_config(cls, values):
    """Use only fields of the already frozen configuration; never fill defaults."""

    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, list) else value

    return cls(**{field.name: tuples(values[field.name]) for field in fields(cls)})


def topk_result(inputs, cameras, config, min_frames, overrides=None):
    row = associate_topk(
        tuple(v.detections.evidence for v in inputs.views),
        cameras,
        fps=inputs.fps,
        config=config,
        min_frames=min_frames,
        candidate_uv_px=overrides,
    )
    return Ball3DResult(
        inputs.camera_ids,
        inputs.frame_indices,
        row.positions,
        row.valid,
        row.selected_indices >= 0,
        cameras=cameras,
        diagnostics={
            "selected_candidate_indices": row.selected_indices,
            "selected_candidate_uv_px": row.selected_uv_px,
            "hypothesis_cost": row.hypothesis_cost,
            "hypothesis_score": row.hypothesis_score,
            "hypothesis_count": row.hypothesis_count,
            "temporal_restart": row.temporal_restart,
            "reprojection_px": row.reprojection_px,
        },
    )


def triangulate_points(inputs, cameras, baseline):
    """Existing smoothed 2D points through the existing production triangulator."""
    raw = ObjectObservations(
        inputs.camera_ids,
        inputs.source_size_wh,
        inputs.fps,
        np.stack([v.points.uv_px for v in inputs.views])[:, :, None, None],
        np.stack([v.points.confidence for v in inputs.views])[:, :, None, None],
        np.stack([v.points.observed for v in inputs.views])[:, :, None],
        np.zeros((len(inputs.views), 1), np.int64),
    )
    fitted = reconstruct_ball(
        single_ball_observations(raw, threshold=baseline.ball_threshold),
        cameras,
        fps=inputs.fps,
        reprojection_px=baseline.reprojection_px
        * np.hypot(*inputs.source_size_wh)
        / np.hypot(1920, 1080),
        min_frames=baseline.min_frames,
    )
    t = fitted.trajectory
    return Ball3DResult(
        inputs.camera_ids,
        inputs.frame_indices,
        t.positions,
        t.valid,
        t.inliers,
        cameras=cameras,
        diagnostics={"rejection_codes": t.reasons, "accepted_uv_px": fitted.uv_px},
    )


class FrozenGMM:
    requires_court_lines = False

    def __init__(self, name, config):
        self.name, self.config = name, config

    def __call__(self, inputs):
        cfg = self.config["fold_calibrations"][",".join(inputs.camera_ids)]
        return CandidateGMM(typed_config(CandidateGMMConfig, cfg))(inputs)


class Composition:
    requires_court_lines = False

    def __init__(self, name, config, baseline):
        self.name, self.config, self.baseline = name, config, baseline

    def __call__(self, inputs):
        cfg = self.config
        if cfg["composition"] == "decode_then_ba":
            views = []
            decoder = typed_config(HeatmapDecodingConfig, cfg["decoding"])
            for view in inputs.views:
                detection, _ = redecode_view(view, decoder)
                points = BallPointsModule(view.smoothing_config).process(
                    BallPointsInput(detection, inputs.source_size_wh, inputs.fps)
                )
                views.append(replace(view, detections=detection, points=points))
            return ba_method(cfg["ba"])(replace(inputs, views=tuple(views)))
        cameras = tuple(v.camera for v in inputs.views)
        ba = None
        if cfg["composition"] == "ba_cameras":
            # This call only sees the selected fold, never the all-view result.
            ba = ba_method(cfg["ba"])(inputs)
            cameras = ba.cameras
            if (
                cameras is None
                or tuple(c.camera_id for c in cameras) != inputs.camera_ids
            ):
                raise ValueError("BA did not produce exactly the retained camera axis")
        downstream = cfg["downstream"]
        overrides, codes = None, None
        if cfg["composition"] == "decode_candidates":
            decoder = typed_config(HeatmapDecodingConfig, cfg["decoding"])
            overrides, codes = {}, []
            for view in inputs.views:
                evidence = view.detections.evidence
                coords = np.zeros_like(evidence.candidate_uv_px)
                status = np.zeros(evidence.candidate_valid.shape, np.int64)
                for t in range(inputs.num_frames):
                    uv, status[t] = refine_peak_cells(
                        evidence.heatmaps[t],
                        evidence.candidate_cells[t],
                        evidence.candidate_valid[t],
                        decoder,
                    )
                    coords[t] = denormalize_grid_keypoints(uv, *inputs.source_size_wh)
                overrides[view.camera.camera_id] = coords
                codes.append(status)
        if downstream == "baseline":
            result = triangulate_points(inputs, cameras, self.baseline)
        elif downstream == "dense":
            row = reconstruct_dense(
                tuple(v.detections.evidence for v in inputs.views),
                cameras,
                fps=inputs.fps,
                config=typed_config(DenseLikelihoodConfig, cfg["dense"]),
            )
            result = Ball3DResult(
                inputs.camera_ids,
                inputs.frame_indices,
                row.positions,
                row.valid,
                row.used_views,
                cameras=cameras,
                diagnostics={
                    "log_likelihood": row.log_likelihood,
                    "solver_converged": row.solver_converged,
                    "rejection_codes": row.rejection_codes,
                    "event_boundaries": row.event_boundaries,
                    "saturated_pixels": row.saturated_pixels,
                },
            )
        elif downstream == "topk":
            result = topk_result(
                inputs,
                cameras,
                typed_config(TopKConfig, cfg["topk"]),
                self.baseline.min_frames,
                overrides,
            )
        else:
            raise ValueError(f"Unsupported downstream: {downstream}")
        if cfg["composition"] == "topk_then_loess":
            views = []
            for i, view in enumerate(inputs.views):
                diag = result.diagnostics
                chosen = diag["selected_candidate_indices"][i]
                mask = chosen >= 0
                confidence = np.where(
                    mask,
                    view.detections.evidence.candidate_scores[
                        np.arange(inputs.num_frames), np.maximum(chosen, 0)
                    ],
                    0,
                ).astype(np.float32)
                detection = BallDetectionOutput(
                    view.camera.camera_id,
                    inputs.frame_indices,
                    diag["selected_candidate_uv_px"][i],
                    confidence,
                    mask,
                    mask.astype(np.uint8),
                    "model_score",
                    view.detections.evidence,
                )
                points = BallPointsModule(view.smoothing_config).process(
                    BallPointsInput(detection, inputs.source_size_wh, inputs.fps)
                )
                views.append(replace(view, detections=detection, points=points))
            smoothed = triangulate_points(
                replace(inputs, views=tuple(views)), cameras, self.baseline
            )
            result = replace(
                smoothed,
                diagnostics={
                    **smoothed.diagnostics,
                    "topk_initial_valid": result.valid,
                    "selected_candidate_indices": result.diagnostics[
                        "selected_candidate_indices"
                    ],
                },
            )
        extra = {}
        if ba is not None:
            extra = {
                "ba_fit": ba.diagnostics,
                "ba_estimated_offsets_not_applied_seconds": ba.time_offsets_seconds,
                "time_policy": "camera_only_offsets_zero",
                "ba_retained_camera_ids": inputs.camera_ids,
            }
        if codes is not None:
            extra["candidate_decoding_fit_status"] = np.stack(codes)
        return replace(result, diagnostics={**(result.diagnostics or {}), **extra})


def ba_method(config):
    return CourtBallBAMethod(
        typed_config(BundleAdjustmentConfig, config["adjustment"]),
        config["reprojection_px"],
        config["min_frames"],
        config["ball_threshold"],
    )


def make_method(row, bundle):
    name, kind, config = row["name"], row["kind"], row["config"]
    baseline = baseline_from_bundle(bundle)
    if name == "pipeline_baseline":
        return baseline
    if name == "without_loess":
        return baseline_from_bundle(bundle, without_loess=True)
    if kind == "decode":
        return HeatmapDecodingMethod(
            name, typed_config(HeatmapDecodingConfig, config["decoding"]), baseline
        )
    if kind == "topk":
        return TopKMultiHypothesis(
            typed_config(TopKConfig, config), config["min_frames"]
        )
    if kind == "gmm":
        return FrozenGMM(name, config)
    if kind == "dense":
        return DenseHeatmapMethod(typed_config(DenseLikelihoodConfig, config), name)
    if kind == "physics":
        return SegmentPhysicsMethod(
            typed_config(SegmentPhysicsConfig, config["physics"]), baseline
        )
    if kind == "ba":
        return ba_method(config)
    if kind == "composition":
        return Composition(name, config, baseline)
    raise ValueError(f"Unsupported method: {kind}")
