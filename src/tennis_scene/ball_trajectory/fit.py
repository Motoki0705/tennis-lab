"""Joint robust image-space fit of segment states, drag and relative clocks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from src.tennis_scene.pipeline.components.ball_evidence import BallHeatmapEvidence
from src.utils.geometry.triangulation import PinholeCamera, TriangulatedPoints

from .config import SegmentPhysicsConfig
from .physics import flight_states
from .segments import detect_segments


@dataclass(frozen=True)
class SegmentPhysicsResult:
    positions: NDArray[np.float32]
    valid: NDArray[np.bool_]
    used_views: NDArray[np.bool_]
    time_offsets_seconds: NDArray[np.float64]
    diagnostics: dict[str, Any]


def _heatmap_residual(
    evidence: BallHeatmapEvidence,
    frames: NDArray[np.int64],
    uv: NDArray[np.float64],
    sigma_px: float,
) -> NDArray[np.float64]:
    """Bilinear native-map likelihood plus a continuous outside-image barrier.

    Read only four cells per observation. No map is copied, resized or stacked.
    Sigmoid scores are relative evidence, not calibrated probabilities.
    """
    h, w = evidence.heatmaps.shape[-2:]
    size = np.asarray(evidence.source_size_wh, np.float64) - 1
    bounded = np.clip(uv, 0, size)
    xy = bounded / size * [w - 1, h - 1]
    ij = np.minimum(np.floor(xy).astype(int), [w - 2, h - 2])
    d = xy - ij
    x, y = ij.T
    a, b = d.T
    maps = evidence.heatmaps
    probability = (
        (1 - a) * (1 - b) * maps[frames, y, x]
        + a * (1 - b) * maps[frames, y, x + 1]
        + (1 - a) * b * maps[frames, y + 1, x]
        + a * b * maps[frames, y + 1, x + 1]
    )
    # The peak provides normalization only; its argmax is not the target.
    peak = np.maximum(evidence.candidate_scores[frames, 0], 1e-8)
    nll = -2 * np.log(np.clip(probability / peak, 1e-8, 1))
    return np.column_stack((np.sqrt(nll + 1e-8), (uv - bounded) / sigma_px))


def _candidate_residual(
    evidence: BallHeatmapEvidence,
    frames: NDArray[np.int64],
    uv: NDArray[np.float64],
    sigma_px: float,
) -> NDArray[np.float64]:
    delta = (uv[:, None] - evidence.candidate_uv_px[frames]) / sigma_px
    penalty = -2 * np.log(np.clip(evidence.candidate_scores[frames], 1e-8, 1))
    energy = np.sum(delta**2, axis=2) + penalty
    energy[~evidence.candidate_valid[frames]] = np.inf
    selected = np.argmin(energy, axis=1)
    rows = np.arange(len(frames))
    return np.column_stack((delta[rows, selected], np.sqrt(penalty[rows, selected])))


def fit_trajectory(
    cameras: tuple[PinholeCamera, ...],
    evidence: tuple[BallHeatmapEvidence, ...],
    initial: TriangulatedPoints,
    *,
    fps: float,
    config: SegmentPhysicsConfig,
) -> SegmentPhysicsResult:
    """Fit only supplied views; preserve initial real-observation output support.

    The lexicographically first supplied camera fixes the world-time gauge.
    There is no held-out-camera or annotation argument, nor an initializer cache.
    """
    v, t = len(cameras), len(initial.valid)
    ids = tuple(c.camera_id for c in cameras)
    if t < 1 or v < 2 or len(set(ids)) != v or len(evidence) != v:
        raise ValueError("Physics requires distinct cameras and matching evidence")
    if (
        initial.positions.shape != (t, 3)
        or initial.inliers.shape != (v, t)
        or initial.valid.dtype != bool
        or initial.inliers.dtype != bool
        or np.any(initial.valid & (initial.inliers.sum(0) < 2))
    ):
        raise ValueError("Initial trajectory/view mask axes disagree")
    if any(
        not isinstance(e, BallHeatmapEvidence) or len(e.heatmaps) != t for e in evidence
    ):
        raise ValueError(
            "Physics requires complete model evidence on the source timeline"
        )
    if len({e.source_size_wh for e in evidence}) != 1:
        raise ValueError("Physics evidence image sizes disagree")
    if any(
        min(e.source_size_wh) < 2 or min(e.heatmaps.shape[-2:]) < 2 for e in evidence
    ):
        raise ValueError("Physics requires at least a two-cell grid and image axes")
    segments, events = detect_segments(initial.positions, initial.valid, fps, config)
    frame_segment: NDArray[np.int64] = np.full(t, -1, np.int64)
    states = []
    for s, segment in enumerate(segments):
        frames = np.arange(segment.start_frame, segment.stop_frame)
        frames = frames[initial.valid[frames]]
        times = (frames - segment.start_frame) / fps
        coefficients = np.polynomial.polynomial.polyfit(
            times, initial.positions[frames], 2
        )
        states.append(np.concatenate((coefficients[0], coefficients[1])))
        frame_segment[segment.start_frame : segment.stop_frame] = s
    anchor = ids.index(min(ids))
    free_views = tuple(i for i in range(v) if i != anchor)
    diagnostics: dict[str, Any] = {
        "status": "insufficient_segments",
        "config": asdict(config),
        "anchor_camera": ids[anchor],
        "camera_ids": list(ids),
        "segments": [asdict(s) for s in segments],
        "events": [asdict(e) for e in events],
        "frame_segment": frame_segment,
        "offset_standard_error_seconds": None,
        "offset_covariance_seconds2": None,
        "uncertainty_note": "Local robust-curvature approximation, conditional on segmentation and candidate assignment; not calibrated confidence",
        "support_policy": "initial valid frames and used-view mask only; no gap filling",
    }
    empty = SegmentPhysicsResult(
        np.zeros((t, 3), np.float32),
        np.zeros(t, bool),
        np.zeros((v, t), bool),
        np.zeros(v, np.float64),
        diagnostics,
    )
    if not states:
        return empty
    s_count = len(states)
    global_start = 6 * s_count
    theta = np.concatenate(
        (np.asarray(states).ravel(), [config.initial_drag_per_m], np.zeros(v - 1))
    )
    lower = np.concatenate(
        (
            np.tile([-40, -40, -0.2, -65, -65, -65], s_count),
            [0],
            np.full(v - 1, -config.offset_bound_frames),
        )
    )
    upper = np.concatenate(
        (
            np.tile([40, 40, 20, 65, 65, 65], s_count),
            [config.max_drag_per_m],
            np.full(v - 1, config.offset_bound_frames),
        )
    )
    theta = np.clip(theta, lower + 1e-7, upper - 1e-7)
    sample_views, sample_frames, sample_segments = [], [], []
    for view, row in enumerate(evidence):
        available = initial.valid & initial.inliers[view] & row.candidate_valid.any(1)
        for s, segment in enumerate(segments):
            # One-frame guards keep all bounded time queries inside a flight,
            # including either side of a bounce. Guards affect fitting only.
            frames = np.arange(segment.start_frame + 1, segment.stop_frame - 1)
            frames = frames[available[frames]]
            sample_views.extend([view] * len(frames))
            sample_frames.extend(frames.tolist())
            sample_segments.extend([s] * len(frames))
    vv: NDArray[np.int64] = np.asarray(sample_views, np.int64)
    ff = np.asarray(sample_frames, np.int64)
    ss: NDArray[np.int64] = np.asarray(sample_segments, np.int64)
    if len(ff) * 2 <= len(theta):
        diagnostics["status"] = "insufficient_image_constraints"
        return empty
    origins = np.asarray([s.start_frame for s in segments])
    base_times = (ff - origins[ss]) / fps
    sparsity = lil_matrix((3 * len(ff), len(theta)), dtype=np.int8)
    for n, (view, segment_index) in enumerate(zip(vv, ss, strict=True)):
        sparsity[3 * n : 3 * n + 3, 6 * segment_index : 6 * segment_index + 6] = 1
        sparsity[3 * n : 3 * n + 3, global_start] = 1
        if view != anchor:
            sparsity[
                3 * n : 3 * n + 3, global_start + 1 + free_views.index(int(view))
            ] = 1

    def offsets(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        result: NDArray[np.float64] = np.zeros(v, np.float64)
        result[list(free_views)] = parameters[global_start + 1 :] / fps
        return result

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        times = base_times + offsets(parameters)[vv]
        state = flight_states(
            parameters[:global_start].reshape(-1, 6)[ss],
            times,
            parameters[global_start],
            gravity_mps2=config.gravity_mps2,
            steps=config.integration_steps,
        )
        result = np.zeros((len(ff), 3))
        for view, camera in enumerate(cameras):
            subset = vv == view
            uv, front = camera.project(state[subset, :3])
            fn = (
                _candidate_residual
                if config.observation == "candidates"
                else _heatmap_residual
            )
            result[subset] = fn(evidence[view], ff[subset], uv, config.sigma_px)
            result[np.flatnonzero(subset)[~front]] = 1e4
        return result.ravel()

    optimized = least_squares(
        residual,
        theta,
        bounds=(lower, upper),
        jac_sparsity=sparsity.tocsr(),
        loss="soft_l1",
        f_scale=config.robust_scale,
        x_scale=np.concatenate(
            (np.tile([5, 10, 2, 20, 20, 10], s_count), [0.02], np.full(v - 1, 0.5))
        ),
        max_nfev=config.max_nfev,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-5,
        tr_options={"atol": 1e-7, "btol": 1e-7},
    )
    parameters = optimized.x
    clock = offsets(parameters)
    diagnostics.update(
        {
            "status": "ok" if optimized.success else "optimization_failed",
            "solver_status": int(optimized.status),
            "solver_message": str(optimized.message),
            "nfev": int(optimized.nfev),
            "cost": float(optimized.cost),
            "optimality": float(optimized.optimality),
            "drag_per_m": float(parameters[global_start]),
            "segment_initial_states": parameters[:global_start].reshape(-1, 6),
            "time_offsets_frames": clock * fps,
            "sample_camera_indices": vv,
            "sample_frames": ff,
            "image_residuals": residual(parameters).reshape(-1, 3),
            "active_bounds": optimized.active_mask,
        }
    )
    if not optimized.success:
        return SegmentPhysicsResult(
            empty.positions, empty.valid, empty.used_views, clock, diagnostics
        )
    jacobian = optimized.jac.toarray()
    information = jacobian.T @ jacobian
    # Rank must not depend on choosing seconds versus frames or metres versus
    # centimetres. Normalize Jacobian columns before the condition/rank check.
    column_scale = np.sqrt(np.maximum(np.diag(information), 1e-30))
    eigenvalues, eigenvectors = np.linalg.eigh(
        information / np.outer(column_scale, column_scale)
    )
    threshold = max(float(eigenvalues[-1]) * 1e-10, 1e-12)
    rank = int((eigenvalues > threshold).sum())
    diagnostics["information_rank"] = rank
    diagnostics["parameter_count"] = len(theta)
    if rank == len(theta):
        variance = 2 * optimized.cost / max(1, len(optimized.fun) - len(theta))
        covariance = (
            (eigenvectors / eigenvalues)
            @ eigenvectors.T
            * variance
            / np.outer(column_scale, column_scale)
        )
        offset_covariance: NDArray[np.float64] = np.zeros((v, v), np.float64)
        offset_covariance[np.ix_(free_views, free_views)] = (
            covariance[global_start + 1 :, global_start + 1 :] / fps**2
        )
        diagnostics["offset_covariance_seconds2"] = offset_covariance
        diagnostics["offset_standard_error_seconds"] = np.sqrt(
            np.maximum(np.diag(offset_covariance), 0)
        )
        diagnostics["drag_standard_error_per_m"] = float(
            np.sqrt(max(covariance[global_start, global_start], 0))
        )
    else:
        diagnostics["uncertainty_unavailable_reason"] = (
            "rank_deficient_local_information"
        )
    diagnostics["uncertainty_at_bound"] = bool(
        np.any(optimized.active_mask[global_start:])
    )
    valid = initial.valid & (frame_segment >= 0)
    frames = np.flatnonzero(valid)
    states_at_frames = flight_states(
        parameters[:global_start].reshape(-1, 6)[frame_segment[frames]],
        (frames - origins[frame_segment[frames]]) / fps,
        parameters[global_start],
        gravity_mps2=config.gravity_mps2,
        steps=config.integration_steps,
    )
    positions: NDArray[np.float32] = np.zeros((t, 3), np.float32)
    positions[frames] = states_at_frames[:, :3]
    # A finite optimizer return is insufficient if the fitted world is impossible.
    valid &= (
        np.isfinite(positions).all(1)
        & (positions[:, 2] >= -0.2)
        & (positions[:, 2] <= 20)
        & (np.abs(positions[:, :2]) <= 40).all(1)
    )
    used = initial.inliers & valid[None]
    positions[~valid] = 0
    # Event positions are continuous only to the extent supported by independent
    # flight fits. Record disagreement rather than imposing an untested bounce law.
    fitted_events = []
    for event in events:
        before = next(
            (i for i, s in enumerate(segments) if s.stop_frame == event.frame), None
        )
        after = next(
            (i for i, s in enumerate(segments) if s.start_frame == event.frame), None
        )
        event_row: dict[str, Any] = asdict(event)
        if before is not None and after is not None:
            state_before = flight_states(
                parameters[6 * before : 6 * before + 6][None],
                np.array([(event.frame - origins[before]) / fps]),
                parameters[global_start],
                gravity_mps2=config.gravity_mps2,
                steps=config.integration_steps,
            )[0]
            state_after = parameters[6 * after : 6 * after + 6]
            event_row["position_discontinuity_m"] = float(
                np.linalg.norm(state_before[:3] - state_after[:3])
            )
            event_row["vertical_restitution"] = (
                float(-state_after[5] / state_before[5])
                if state_before[5] < -1e-6
                else None
            )
        fitted_events.append(event_row)
    diagnostics["events"] = fitted_events
    diagnostics["physical_acceleration_mps2"] = -parameters[
        global_start
    ] * np.linalg.norm(states_at_frames[:, 3:], axis=1)[:, None] * states_at_frames[
        :, 3:
    ] - [0, 0, config.gravity_mps2]
    diagnostics["physical_sample_frames"] = frames
    return SegmentPhysicsResult(positions, valid, used, clock, diagnostics)
