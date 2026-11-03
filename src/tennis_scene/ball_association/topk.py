"""Bounded multi-hypothesis association of pre-gate ball candidates on CPU."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.pipeline.components.ball_evidence import BallHeatmapEvidence
from src.utils.geometry.triangulation import PinholeCamera, solve_homogeneous_dlt


@dataclass(frozen=True)
class TopKConfig:
    top_k: int = 8
    max_hypotheses: int = 24
    beam_width: int = 96
    max_missing_frames: int = 3
    reprojection_px: float = 20.0
    reprojection_sigma_px: float = 6.0
    score_weight: float = 1.0
    missing_view_cost: float = 1.5
    missing_frame_cost: float = 3.0
    restart_cost: float = 2.0
    max_speed_mps: float = 65.0
    speed_scale_mps: float = 35.0
    speed_weight: float = 0.1
    acceleration_scale_mps2: float = 250.0
    acceleration_weight: float = 1.0
    turn_cost_cap: float = 3.0

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("top_k", 1, 8), ("max_hypotheses", 1, 128),
            ("beam_width", 1, 512), ("max_missing_frames", 0, 30),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower},{upper}]")
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name in {"top_k", "max_hypotheses", "beam_width", "max_missing_frames"}:
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("reprojection_px", "reprojection_sigma_px", "max_speed_mps",
                     "speed_scale_mps", "acceleration_scale_mps2"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class AssociationResult:
    positions: NDArray[np.float32]  # T,3, zero when missing
    valid: NDArray[np.bool_]
    selected_indices: NDArray[np.int64]  # V,T; -1 when unused
    selected_uv_px: NDArray[np.float32]  # V,T,2; zero when unused
    reprojection_px: NDArray[np.float32]  # V,T; zero when unused
    hypothesis_cost: NDArray[np.float64]  # emission cost, zero when missing
    hypothesis_score: NDArray[np.float64]  # exp(-cost), NOT calibrated probability
    hypothesis_count: NDArray[np.int64]  # after geometric gates and pruning
    temporal_restart: NDArray[np.bool_]


@dataclass(frozen=True)
class _Hypotheses:
    xyz: NDArray[np.float64]
    indices: NDArray[np.int64]  # H,V
    cost: NDArray[np.float64]
    error: NDArray[np.float64]  # H,V


def _triangulate(uv: NDArray[np.float64], used: NDArray[np.bool_],
                 matrices: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    # Unit geometric weights: raw sigmoid scores enter the emission separately.
    design = np.stack((uv[..., 0, None] * matrices[:, 2] - matrices[:, 0],
                       uv[..., 1, None] * matrices[:, 2] - matrices[:, 1]), -2)
    xyz, valid = solve_homogeneous_dlt(
        (design * used[..., None, None]).reshape(len(uv), -1, 4))
    return xyz, valid


def frame_hypotheses(uv: NDArray[np.float64], scores: NDArray[np.float64],
                     valid: NDArray[np.bool_], cameras: tuple[PinholeCamera, ...],
                     config: TopKConfig, pixel_scale: float) -> _Hypotheses:
    """Enumerate all valid candidate pairs, plus their best compatible extensions.

    Pair-only proposals explicitly omit other views. Extensions choose a candidate
    by reprojection and score at the seed point, then refit jointly. This is a
    proposal approximation, not an exhaustive V-way Cartesian product.
    """
    v = len(cameras)
    matrices = np.stack([c.matrix for c in cameras])
    proposals = []
    for a, b in combinations(range(v), 2):
        pairs = np.array(list(product(np.flatnonzero(valid[a]), np.flatnonzero(valid[b]))), np.int64)
        if not len(pairs):
            continue
        indices: NDArray[np.int64] = np.full((len(pairs), v), -1, np.int64)
        indices[:, [a, b]] = pairs
        target = uv[np.arange(v), np.maximum(indices, 0)]
        xyz, good = _triangulate(target, indices >= 0, matrices)
        # Reject parallel or effectively collinear rays, including finite far DLT.
        rays_a, rays_b = xyz - cameras[a].center, xyz - cameras[b].center
        denominator = np.linalg.norm(rays_a, axis=-1) * np.linalg.norm(rays_b, axis=-1)
        cosine = np.sum(rays_a * rays_b, axis=-1) / np.maximum(denominator, 1e-12)
        good &= np.abs(cosine) < math.cos(math.radians(1.0))
        indices, xyz = indices[good], xyz[good]
        if not len(indices):
            continue
        proposals.append(indices)
        extended = indices.copy()
        for j, camera in enumerate(cameras):
            if j in (a, b) or not valid[j].any():
                continue
            projected, front = camera.project(xyz)
            error = np.linalg.norm(projected[:, None] - uv[j], axis=-1)
            cost = ((error / (config.reprojection_sigma_px * pixel_scale)) ** 2
                    - config.score_weight * np.log(np.maximum(scores[j], 1e-8)))
            cost[:, ~valid[j]] = np.inf
            best = np.argmin(cost, axis=1)
            accept = front & (error[np.arange(len(best)), best] <= config.reprojection_px * pixel_scale)
            extended[accept, j] = best[accept]
        proposals.append(extended)
    if not proposals:
        return _Hypotheses(np.empty((0, 3)), np.empty((0, v), np.int64),
                           np.empty(0), np.empty((0, v)))
    indices = np.unique(np.concatenate(proposals), axis=0)
    used = indices >= 0
    target = uv[np.arange(v), np.maximum(indices, 0)]
    xyz, good = _triangulate(target, used, matrices)
    errors, front = zip(*(c.project(xyz) for c in cameras), strict=True)
    error = np.linalg.norm(np.stack(errors, axis=1) - target, axis=-1)
    good &= np.all(~used | (np.stack(front, axis=1) & (error <= config.reprojection_px * pixel_scale)), axis=1)
    good &= (xyz[:, 2] >= -0.2) & (xyz[:, 2] <= 20.0) & (np.abs(xyz[:, :2]) <= 40.0).all(1)
    nll = -np.log(np.maximum(scores[np.arange(v), np.maximum(indices, 0)], 1e-8))
    cost = (np.sum(np.where(used, config.score_weight * nll +
                           (error / (config.reprojection_sigma_px * pixel_scale)) ** 2, 0), axis=1) / v
            + config.missing_view_cost * np.sum(~used, axis=1))
    order = np.flatnonzero(good)[np.argsort(cost[good], kind="stable")[:config.max_hypotheses]]
    return _Hypotheses(xyz[order], indices[order], cost[order], np.where(used[order], error[order], 0))


def select_temporal_path(hypotheses: list[_Hypotheses], fps: float,
                         config: TopKConfig) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Beam dynamic programming with second-order motion and explicit null states.

    Deduplicate (last, penultimate) hypothesis identities, retaining the least
    costly history for each pair, then retain at most beam_width pairs. Null
    emissions never invent 3D. After max_missing_frames, discard motion history
    and charge restart_cost. Truncated acceleration lets a bounce/hit pay a
    finite turn cost instead of imposing a quadratic smoothness veto.
    """
    hmax = config.max_hypotheses
    cost = np.zeros(1)
    last: NDArray[np.int64] = np.full(1, -1, np.int64)
    previous = last.copy()
    last_frame = last.copy()
    position = np.zeros((1, 3))
    velocity = position.copy()
    last_dt = np.ones(1) / fps
    has_velocity: NDArray[np.bool_] = np.zeros(1, bool)
    parents, choices, resets = [], [], []
    for t, row in enumerate(hypotheses):
        b, h = len(cost), len(row.cost)
        dt = np.where(last >= 0, (t - last_frame) / fps, 1 / fps)
        new_velocity = (row.xyz[None] - position[:, None]) / dt[:, None, None]
        speed = np.linalg.norm(new_velocity, axis=-1)
        acceleration = np.linalg.norm(new_velocity - velocity[:, None], axis=-1) / ((dt + last_dt)[:, None] / 2)
        motion = (config.speed_weight * (speed / config.speed_scale_mps) ** 2
                  + np.where(has_velocity[:, None], config.acceleration_weight * np.minimum(
                      (acceleration / config.acceleration_scale_mps2) ** 2, config.turn_cost_cap), 0))
        motion = np.where(last[:, None] >= 0, motion, 0)
        joint = cost[:, None] + row.cost + motion
        joint[(last[:, None] >= 0) & (speed > config.max_speed_mps)] = np.inf
        expired = (last >= 0) & ((t - last_frame) > config.max_missing_frames)
        total = np.r_[joint.ravel(), cost + config.missing_frame_cost + expired * config.restart_cost]
        parent = np.r_[np.repeat(np.arange(b), h), np.arange(b)]
        choice = np.r_[np.tile(np.arange(h), b), np.full(b, -1)]
        next_last = np.r_[np.tile(t * hmax + np.arange(h), b), np.where(expired, -1, last)]
        next_previous = np.r_[np.repeat(last, h), np.where(expired, -1, previous)]
        order = np.argsort(total, kind="stable")
        _, first = np.unique(np.column_stack((next_last[order], next_previous[order])), axis=0, return_index=True)
        keep = order[np.sort(first)[:config.beam_width]]
        keep = keep[np.isfinite(total[keep])]
        is_point = choice[keep] >= 0
        p, c = parent[keep], np.maximum(choice[keep], 0)
        restart = is_point & (last[p] < 0) & (t > 0)
        new_position = position[p].copy()
        new_v = velocity[p].copy()
        new_position[is_point] = row.xyz[c[is_point]]
        new_v[is_point] = new_velocity[p[is_point], c[is_point]]
        last_dt = np.where(is_point, dt[p], last_dt[p])
        has_velocity = np.where(is_point, last[p] >= 0, has_velocity[p] & ~expired[p])
        last_frame = np.where(is_point, t, np.where(expired[p], -1, last_frame[p]))
        position, velocity = new_position, new_v
        last, previous = next_last[keep], next_previous[keep]
        cost = total[keep] - np.min(total[keep])
        parents.append(p)
        choices.append(choice[keep])
        resets.append(restart)
    path: NDArray[np.int64] = np.full(len(hypotheses), -1, np.int64)
    restarted: NDArray[np.bool_] = np.zeros(len(hypotheses), bool)
    state = int(np.argmin(cost))
    for t in range(len(hypotheses) - 1, -1, -1):
        path[t], restarted[t] = choices[t][state], resets[t][state]
        state = int(parents[t][state])
    return path, restarted


def associate_topk(evidence: tuple[BallHeatmapEvidence, ...], cameras: tuple[PinholeCamera, ...],
                   *, fps: float, config: TopKConfig, min_frames: int = 1) -> AssociationResult:
    if not math.isfinite(fps) or fps <= 0 or len(evidence) != len(cameras) or len(cameras) < 2:
        raise ValueError("Association requires matching evidence/cameras and positive fps")
    if len({c.camera_id for c in cameras}) != len(cameras):
        raise ValueError("Association camera IDs must be distinct")
    if type(min_frames) is not int or min_frames < 1:
        raise ValueError("Association min_frames must be positive")
    size, frames = evidence[0].source_size_wh, len(evidence[0].candidate_scores)
    if any(e.source_size_wh != size or len(e.candidate_scores) != frames for e in evidence):
        raise ValueError("Candidate source image/frame axes disagree")
    # Only the small candidate arrays are copied; native maps/patches are untouched.
    k = config.top_k
    uv: NDArray[np.float64] = np.zeros((len(evidence), frames, k, 2), np.float64)
    scores: NDArray[np.float64] = np.zeros((len(evidence), frames, k), np.float64)
    valid: NDArray[np.bool_] = np.zeros((len(evidence), frames, k), bool)
    for j, e in enumerate(evidence):
        n = min(k, e.config.max_candidates)
        uv[j, :, :n], scores[j, :, :n], valid[j, :, :n] = e.candidate_uv_px[:, :n], e.candidate_scores[:, :n], e.candidate_valid[:, :n]
    scale = math.hypot(*size) / math.hypot(1920, 1080)
    rows = [frame_hypotheses(uv[:, t], scores[:, t], valid[:, t], cameras, config, scale) for t in range(frames)]
    path, restarted = select_temporal_path(rows, fps, config)
    if int((path >= 0).sum()) < min_frames:
        path[:] = -1
        restarted[:] = False
    positions: NDArray[np.float32] = np.zeros((frames, 3), np.float32)
    indices: NDArray[np.int64] = np.full((len(cameras), frames), -1, np.int64)
    selected: NDArray[np.float32] = np.zeros((len(cameras), frames, 2), np.float32)
    error: NDArray[np.float32] = np.zeros((len(cameras), frames), np.float32)
    costs: NDArray[np.float64] = np.zeros(frames, np.float64)
    for t in np.flatnonzero(path >= 0):
        h = path[t]
        positions[t], indices[:, t], error[:, t], costs[t] = rows[t].xyz[h], rows[t].indices[h], rows[t].error[h], rows[t].cost[h]
        used = indices[:, t] >= 0
        selected[used, t] = uv[used, t, indices[used, t]]
    return AssociationResult(positions, path >= 0, indices, selected, error, costs,
                             np.where(path >= 0, np.exp(-costs), 0),
                             np.array([len(row.cost) for row in rows], np.int64), restarted)
