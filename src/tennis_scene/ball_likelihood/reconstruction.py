"""Dense maximum likelihood from candidate seeds or a bounded court volume."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from src.tennis_scene.pipeline.components.ball_evidence import BallHeatmapEvidence
from src.utils.geometry.triangulation import (
    PinholeCamera,
    PointRejection,
    solve_homogeneous_dlt,
)

from .config import DenseLikelihoodConfig
from .likelihood import HeatmapLikelihood, objective


@dataclass(frozen=True)
class DenseTrajectory:
    positions: NDArray[np.float32]
    valid: NDArray[np.bool_]
    used_views: NDArray[np.bool_]
    log_likelihood: NDArray[np.float64]
    event_boundaries: NDArray[np.bool_]
    solver_converged: NDArray[np.bool_]
    saturated_pixels: NDArray[np.int64]
    rejection_codes: NDArray[np.uint8]


def candidate_seeds(evidence: tuple[BallHeatmapEvidence, ...], cameras: tuple[PinholeCamera, ...], frame: int) -> NDArray[np.float64]:
    seeds = []
    for i, j in combinations(range(len(cameras)), 2):
        a = evidence[i].candidate_uv_px[frame, evidence[i].candidate_valid[frame]]
        b = evidence[j].candidate_uv_px[frame, evidence[j].candidate_valid[frame]]
        if not len(a) or not len(b):
            continue
        aa, bb = np.repeat(a, len(b), axis=0), np.tile(b, (len(a), 1))
        pa, pb = cameras[i].matrix, cameras[j].matrix
        design = np.stack((aa[:, 0, None] * pa[2] - pa[0], aa[:, 1, None] * pa[2] - pa[1],
                           bb[:, 0, None] * pb[2] - pb[0], bb[:, 1, None] * pb[2] - pb[1]), axis=1)
        xyz, valid = solve_homogeneous_dlt(design)
        seeds.extend(xyz[valid])
    return np.asarray(seeds, np.float64).reshape(-1, 3)


def _in_bounds(xyz: NDArray[np.float64], config: DenseLikelihoodConfig) -> NDArray[np.bool_]:
    limits = np.array(config.bounds)
    return np.isfinite(xyz).all(-1) & (xyz >= limits[:, 0]).all(-1) & (xyz <= limits[:, 1]).all(-1)


def coarse_seeds(maps: tuple[HeatmapLikelihood, ...], config: DenseLikelihoodConfig) -> NDArray[np.float64]:
    axes = [np.linspace(lo, hi, int(np.ceil((hi - lo) / config.coarse_spacing_m)) + 1) for lo, hi in config.bounds]
    xyz = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    step = config.coarse_spacing_m
    offsets = np.stack(np.meshgrid(*([np.arange(-1, 2)] * 3), indexing="ij"), -1).reshape(-1, 3)
    for level in range(config.coarse_levels):
        costs, _, _ = objective(maps, xyz)
        xyz = xyz[np.argsort(costs, kind="stable")[:config.coarse_keep]]
        if level + 1 < config.coarse_levels:
            step /= 3
            xyz = (xyz[:, None] + offsets[None] * step).reshape(-1, 3)
            xyz = xyz[_in_bounds(xyz, config)]
    return xyz


def _ray_support(point: NDArray[np.float64], support: NDArray[np.bool_], cameras: tuple[PinholeCamera, ...], minimum: float) -> bool:
    active = [i for i in range(len(cameras)) if support[i]]
    for i, j in combinations(active, 2):
        a, b = point - cameras[i].center, point - cameras[j].center
        cosine = abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        if cosine <= np.cos(np.deg2rad(minimum)):
            return True
    return False


def make_frame_maps(evidence: tuple[BallHeatmapEvidence, ...], cameras: tuple[PinholeCamera, ...], frame: int,
                    config: DenseLikelihoodConfig) -> tuple[HeatmapLikelihood, ...]:
    return tuple(HeatmapLikelihood(e.heatmaps[frame], c, e.source_size_wh, config) for e, c in zip(evidence, cameras, strict=True))


def estimate_frame(maps: tuple[HeatmapLikelihood, ...], seeds: NDArray[np.float64], config: DenseLikelihoodConfig) -> tuple[NDArray[np.float64], NDArray[np.bool_], float, bool]:
    seeds = seeds[_in_bounds(seeds, config)]
    if not len(seeds):
        return np.zeros(3), np.zeros(len(maps), bool), 0.0, False
    costs, _, _ = objective(maps, seeds)
    selected: list[NDArray[np.float64]] = []
    for idx in np.argsort(costs, kind="stable"):
        point = seeds[idx]
        if all(np.linalg.norm(point - other) > 0.1 for other in selected):
            selected.append(point)
        if len(selected) >= config.multistart:
            break
    best: tuple[NDArray[np.float64], NDArray[np.bool_], float, bool] = (np.zeros(3), np.zeros(len(maps), bool), float("inf"), False)

    def fun(x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        cost, gradient, _ = objective(maps, x[None])
        return float(cost[0]), gradient[0]

    cameras = tuple(m.camera for m in maps)
    for seed in selected:
        result = minimize(fun, seed, jac=True, method="L-BFGS-B", bounds=config.bounds,
                          options={"maxiter": config.max_iterations, "ftol": 1e-10, "gtol": 1e-6})
        # A failed solver is not silently replaced by its initializer.
        if not result.success or not np.isfinite(result.fun):
            continue
        cost, _, support = objective(maps, result.x[None])
        if cost[0] < best[2] and _ray_support(result.x, support[:, 0], cameras, config.min_ray_angle_deg):
            best = result.x, support[:, 0], float(cost[0]), True
    return best[0], best[1], -best[2] if best[3] else 0.0, best[3]


def event_boundaries(positions: NDArray[np.floating], valid: NDArray[np.bool_], fps: float, threshold: float) -> NDArray[np.bool_]:
    """Input-derived acceleration change points; never consumes annotation events."""
    boundaries: NDArray[np.bool_] = np.zeros(len(valid), bool)
    if len(valid) >= 3:
        acceleration = np.linalg.norm(np.diff(positions.astype(np.float64), n=2, axis=0), axis=-1) * fps ** 2
        boundaries[1:-1] = valid[:-2] & valid[1:-1] & valid[2:] & (acceleration > threshold)
    return boundaries


def temporal_objective(x: NDArray[np.float64], maps: list[tuple[HeatmapLikelihood, ...]], fps: float,
                       config: DenseLikelihoodConfig) -> tuple[float, NDArray[np.float64]]:
    points = x.reshape(-1, 3)
    values = [objective(m, p[None]) for m, p in zip(maps, points, strict=True)]
    cost = float(sum(v[0][0] for v in values))
    gradient = np.stack([v[1][0] for v in values])
    scale = fps ** 2 / config.acceleration_scale_mps2
    acceleration = np.diff(points, n=2, axis=0) * scale
    squared = np.sum(acceleration ** 2, axis=-1)
    # Cauchy prior limits the influence of any undetected impact.
    cost += config.temporal_weight * float(np.log1p(squared).sum())
    deriv = (2 * config.temporal_weight * scale / (1 + squared))[:, None] * acceleration
    gradient[:-2] += deriv
    gradient[1:-1] -= 2 * deriv
    gradient[2:] += deriv
    return cost, gradient.ravel()


def reconstruct_dense(evidence: tuple[BallHeatmapEvidence, ...], cameras: tuple[PinholeCamera, ...], *, fps: float,
                      config: DenseLikelihoodConfig) -> DenseTrajectory:
    if len(evidence) != len(cameras) or len(cameras) < 2 or len({c.camera_id for c in cameras}) != len(cameras):
        raise ValueError("Dense reconstruction requires distinct aligned cameras and evidence")
    frames = len(evidence[0].heatmaps)
    if not np.isfinite(fps) or fps <= 0 or any(len(e.heatmaps) != frames or e.source_size_wh != evidence[0].source_size_wh for e in evidence):
        raise ValueError("Dense evidence frame/image axes or fps disagree")
    if min(evidence[0].source_size_wh) < 2:
        raise ValueError("Dense source image must be at least 2x2")
    positions: NDArray[np.float64] = np.zeros((frames, 3), np.float64)
    valid: NDArray[np.bool_] = np.zeros(frames, bool)
    used: NDArray[np.bool_] = np.zeros((len(cameras), frames), bool)
    likelihood: NDArray[np.float64] = np.zeros(frames, np.float64)
    converged: NDArray[np.bool_] = np.zeros(frames, bool)
    saturation: NDArray[np.int64] = np.zeros((len(cameras), frames), np.int64)
    rejection: NDArray[np.uint8] = np.full(frames, PointRejection.INSUFFICIENT_VIEWS, np.uint8)
    for t in range(frames):
        maps = make_frame_maps(evidence, cameras, t, config)
        saturation[:, t] = [m.saturated for m in maps]
        if sum(m.informative for m in maps) < 2:
            continue
        rejection[t] = PointRejection.OPTIMIZATION_FAILED
        seeds = candidate_seeds(evidence, cameras, t) if config.initialization == "candidates" else coarse_seeds(maps, config)
        point, support, score, success = estimate_frame(maps, seeds, config)
        if success:
            positions[t], used[:, t], likelihood[t], valid[t], converged[t] = point, support, score, True, True
            rejection[t] = PointRejection.VALID
    boundaries = event_boundaries(positions, valid, fps, config.event_acceleration_mps2)
    if config.temporal_weight:
        t = 0
        while t < frames:
            if not valid[t]:
                t += 1
                continue
            stop = t + 1
            while stop < frames and valid[stop] and not boundaries[stop] and stop - t < config.window_size:
                stop += 1
            if stop - t >= 3:
                maps_window = [make_frame_maps(evidence, cameras, f, config) for f in range(t, stop)]
                result = minimize(temporal_objective, positions[t:stop].ravel(), args=(maps_window, fps, config), jac=True,
                                  method="L-BFGS-B", bounds=config.bounds * (stop - t),
                                  options={"maxiter": config.max_iterations, "ftol": 1e-9})
                converged[t:stop] = bool(result.success)
                if not result.success:
                    valid[t:stop] = False
                    positions[t:stop], used[:, t:stop], likelihood[t:stop] = 0, False, 0
                    rejection[t:stop] = PointRejection.OPTIMIZATION_FAILED
                else:
                    for f, point, maps in zip(range(t, stop), result.x.reshape(-1, 3), maps_window, strict=True):
                        cost, _, support = objective(maps, point[None])
                        if _ray_support(point, support[:, 0], cameras, config.min_ray_angle_deg):
                            positions[f], used[:, f], likelihood[f] = point, support[:, 0], -float(cost[0])
                        else:
                            valid[f], positions[f], used[:, f], likelihood[f] = False, 0, False, 0
                            rejection[f] = PointRejection.INSUFFICIENT_VIEWS
            t = stop
    return DenseTrajectory(positions.astype(np.float32), valid, used, likelihood, boundaries, converged, saturation, rejection)
