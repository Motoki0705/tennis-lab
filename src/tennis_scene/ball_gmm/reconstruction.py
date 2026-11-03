"""Method A fusion and point summaries conditional on multi-camera support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.ball_gmm.config import CandidateGMMConfig
from src.tennis_scene.ball_gmm.heatmap import candidate_gaussians, frame_gmm
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
)
from src.utils.geometry.kalman import smooth_positions
from src.utils.geometry.probabilistic_triangulation import (
    GaussianMixture3D,
    GaussianPrior3D,
    LaplaceConfig,
    triangulate_gmm,
)
from src.utils.geometry.triangulation import (
    PinholeCamera,
    PointRejection,
    TriangulatedPoints,
)


@dataclass(frozen=True)
class CandidateGMMTrajectory:
    map_positions: NDArray[np.float64]
    map_covariance: NDArray[np.float64]
    mean_positions: NDArray[np.float64]
    mean_covariance: NDArray[np.float64]
    valid: NDArray[np.bool_]
    map_views: NDArray[np.bool_]
    mean_views: NDArray[np.bool_]
    diagnostics: dict[str, Any]


def reconstruct_candidates(
    detections: tuple[BallDetectionOutput, ...], cameras: tuple[PinholeCamera, ...],
    *, config: CandidateGMMConfig,
) -> CandidateGMMTrajectory:
    if (len(cameras) < 2 or len(cameras) != len(detections)
            or len({c.camera_id for c in cameras}) != len(cameras)):
        raise ValueError("Candidate GMM requires at least two distinct matching cameras")
    frame_indices = detections[0].frame_indices
    sizes = set()
    for camera, detection in zip(cameras, detections, strict=True):
        if camera.camera_id != detection.camera_id or not np.array_equal(detection.frame_indices, frame_indices):
            raise ValueError("Candidate GMM camera/frame axes disagree")
        if detection.evidence is None or detection.score_semantics != "model_score":
            raise ValueError("Candidate GMM requires model heatmap evidence")
        sizes.add(detection.evidence.source_size_wh)
    if len(sizes) != 1:
        raise ValueError("Candidate GMM source image sizes disagree")
    evidence = tuple(candidate_gaussians(d.evidence, config) for d in detections if d.evidence is not None)
    t, v = len(frame_indices), len(cameras)
    map_positions, mean_positions = np.zeros((t, 3)), np.zeros((t, 3))
    map_covariance, mean_covariance = np.zeros((t, 3, 3)), np.zeros((t, 3, 3))
    valid: NDArray[np.bool_] = np.zeros(t, bool)
    map_views: NDArray[np.bool_] = np.zeros((v, t), bool)
    mean_views: NDArray[np.bool_] = np.zeros((v, t), bool)
    mass = np.zeros(t)
    components: NDArray[np.int64] = np.zeros(t, np.int64)
    reasons: NDArray[np.uint8] = np.full(t, int(PointRejection.INSUFFICIENT_VIEWS), np.uint8)
    failures: dict[str, str] = {}
    prior = GaussianPrior3D(np.asarray(config.prior_mean_m, np.float64), np.diag(np.square(config.prior_std_m)))
    laplace = LaplaceConfig(config.max_components, config.max_nfev)
    for frame in range(t):
        obs = frame_gmm(evidence, frame)
        if np.count_nonzero(obs.presence) < 2:
            continue
        try:
            result = triangulate_gmm(obs, cameras, prior=prior, config=laplace)
        except RuntimeError as exc:
            # Explicit whole-frame failure policy. Never discard a failed
            # combination, use a point solver or change the prior as fallback.
            reasons[frame] = int(PointRejection.OPTIMIZATION_FAILED)
            failures[str(frame)] = str(exc)
            continue
        supported = result.camera_subsets.sum(1) >= 2
        posterior = result.distribution
        mass[frame] = posterior.weights[supported].sum()
        components[frame] = int(supported.sum())
        if mass[frame] < config.min_multiview_mass:
            continue
        conditional = GaussianMixture3D(posterior.means[supported], posterior.covariance[supported],
                                       posterior.weights[supported] / mass[frame])
        index = int(np.argmax(conditional.weights))
        map_positions[frame] = conditional.means[index]
        map_covariance[frame] = conditional.covariance[index]
        mean_positions[frame], mean_covariance[frame] = conditional.moments()
        map_views[:, frame] = result.camera_subsets[supported][index]
        mean_views[:, frame] = result.camera_subsets[supported][conditional.weights > 0].any(0)
        valid[frame] = True
        reasons[frame] = int(PointRejection.VALID)
    return CandidateGMMTrajectory(map_positions, map_covariance, mean_positions, mean_covariance,
        valid, map_views, mean_views, {
            "multiview_probability": mass,
            "conditional_component_count": components,
            "rejection_codes": reasons,
            "solver_failures": failures,
            "candidate_indices": np.stack([e.indices for e in evidence]),
            "candidate_means_px": np.stack([e.means_px for e in evidence]),
            "candidate_covariance_px2": np.stack([e.covariance_px2 for e in evidence]),
            "candidate_weights": np.stack([e.weights for e in evidence]),
            "presence": np.stack([e.presence for e in evidence]),
            "bounded_curvature": np.stack([e.bounded for e in evidence]),
            "rejected_curvature_count": np.stack([e.rejected for e in evidence]),
            "distribution": "Method A conditional on at least two active cameras",
            "presence_semantics": "score-calibrated usable top-K support proxy, not physical existence",
            "failure_policy": "reject entire frame with optimizer reason; no component fallback",
        })


def point_estimate(
    trajectory: CandidateGMMTrajectory, *, config: CandidateGMMConfig, fps: float,
) -> tuple[NDArray[np.float32], NDArray[np.float64], NDArray[np.bool_]]:
    if config.estimator == "map":
        positions, covariance, views = trajectory.map_positions, trajectory.map_covariance, trajectory.map_views
    else:
        positions, covariance, views = trajectory.mean_positions, trajectory.mean_covariance, trajectory.mean_views
    if config.temporal != "none":
        positions, covariance = smooth_positions(positions, covariance, trajectory.valid, fps=fps,
            acceleration_std=config.acceleration_std_mps2,
            initial_velocity_std=config.initial_velocity_std_mps, backward=config.temporal == "rts")
    return positions.astype(np.float32), covariance.copy(), views.copy()


def as_ball_result(
    trajectory: CandidateGMMTrajectory, cameras: tuple[PinholeCamera, ...], *,
    config: CandidateGMMConfig, fps: float,
) -> tuple[BallReconstructionResult, NDArray[np.float64]]:
    positions, covariance, views = point_estimate(trajectory, config=config, fps=fps)
    valid = trajectory.valid.copy()
    reasons = trajectory.diagnostics["rejection_codes"].copy()
    uv: NDArray[np.float32] = np.zeros((len(cameras), len(positions), 2), np.float32)
    visibility = np.zeros(uv.shape[:2], bool)
    residuals = np.zeros(uv.shape[:2], np.float32)
    for i, camera in enumerate(cameras):
        projected, front = camera.project(positions)
        visibility[i] = valid & views[i] & front
        for frame in np.flatnonzero(visibility[i]):
            means = trajectory.diagnostics["candidate_means_px"][i, frame]
            available = trajectory.diagnostics["candidate_indices"][i, frame] >= 0
            distance = np.where(available, np.linalg.norm(means-projected[frame], axis=-1), np.inf)
            selected = int(np.argmin(distance))
            uv[i, frame] = means[selected]
            residuals[i, frame] = distance[selected]
    ball = BallReconstructionResult(uv, visibility,
        TriangulatedPoints(positions, valid, reasons, views, residuals),
        "ok" if valid.any() else "ball_insufficient_support")
    return ball, covariance
