"""Score camera-local player footpoint sets without cross-camera identities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from .hypothesis import (
    HypothesisScore,
    PlayerCourtSideConfig,
    SideEvidence,
    half_turn_hypotheses,
)


@dataclass(frozen=True)
class PlayerFootpoints:
    xy_m: NDArray[np.float64]  # (T,P,2) in one camera's local court; P may differ by camera
    observed: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if (self.xy_m.ndim != 3 or self.xy_m.shape[-1] != 2
                or self.observed.shape != self.xy_m.shape[:-1] or self.observed.dtype != np.bool_):
            raise ValueError("Player court side requires (T,P,2) footpoints and a (T,P) boolean mask")
        if not np.isfinite(self.xy_m[self.observed]).all():
            raise ValueError("Observed player footpoints must be finite")


def _matched_distances(first: NDArray[np.float64], second: NDArray[np.float64]) -> NDArray[np.float64]:
    distances = np.linalg.norm(first[:, None] - second[None, :], axis=-1)
    rows, cols = linear_sum_assignment(distances)
    return distances[rows, cols]


def _informative_frames(
    views: tuple[PlayerFootpoints, ...], config: PlayerCourtSideConfig,
) -> list[tuple[int, dict[int, NDArray[np.float64]]]]:
    """Keep asymmetric sets and distinct frames with at least two such views."""
    kept: list[tuple[int, dict[int, NDArray[np.float64]]]] = []
    for frame in range(len(views[0].observed)):
        point_sets = {}
        for index, view in enumerate(views):
            points = view.xy_m[frame, view.observed[frame]]
            if len(points) and (_matched_distances(points, -points) > config.symmetry_distance_m).any():
                point_sets[index] = points
        if len(point_sets) < 2:
            continue
        if kept:
            last = kept[-1][1]
            if last.keys() == point_sets.keys() and all(
                len(points) == len(last[index])
                and (_matched_distances(points, last[index]) < config.min_motion_m).all()
                for index, points in point_sets.items()
            ):
                continue
        kept.append((frame, point_sets))
    return kept


def collect_player_side_evidence(
    camera_ids: tuple[str, ...], reference_camera: str, views: tuple[PlayerFootpoints, ...],
    config: PlayerCourtSideConfig,
) -> tuple[SideEvidence, NDArray[np.int64]]:
    """Compare every available view pair within a synchronous frame.

    Each set comparison matches points one to one within that frame. Additional
    points in the larger set have no counterpart and add no cost or support.
    Pairs contribute equally within a frame, and all its pairs must agree for
    the frame to count towards support. Pair scores only depend on relative
    half-turn, so each pair is evaluated just twice.
    """
    if len(set(camera_ids)) != len(camera_ids) or len(views) != len(camera_ids) or reference_camera not in camera_ids:
        raise ValueError("Player court side requires unique camera IDs aligned to observations and including the reference")
    if len({len(v.observed) for v in views}) != 1:
        raise ValueError("Player court side views must share a synchronized timeline")
    assignments = half_turn_hypotheses(len(camera_ids), camera_ids.index(reference_camera))
    turns = np.asarray(assignments, bool)
    relative = {(a, b): (turns[:, a] != turns[:, b]).astype(np.intp) for a, b in combinations(range(len(camera_ids)), 2)}
    kept = _informative_frames(views, config)
    pair_frames: NDArray[np.int64] = np.zeros((len(camera_ids), len(camera_ids)), np.int64)
    cost: NDArray[np.float64] = np.zeros(len(turns), np.float64)
    support: NDArray[np.int64] = np.zeros(len(turns), np.int64)
    for _, points in kept:
        frame_cost = np.zeros_like(cost)
        frame_support: NDArray[np.bool_] = np.ones(len(turns), bool)
        for a, b in combinations(points, 2):
            pair_frames[a, b] += 1
            pair_frames[b, a] += 1
            distances = [_matched_distances(points[a], candidate) for candidate in (points[b], -points[b])]
            pair_cost = np.array([np.minimum((d / config.match_distance_m) ** 2, 1).mean() for d in distances])
            pair_support = np.array([(d <= config.match_distance_m).all() for d in distances])
            frame_cost += pair_cost[relative[a, b]]
            frame_support &= pair_support[relative[a, b]]
        cost += frame_cost / (len(points) * (len(points) - 1) // 2)
        support += frame_support
    frames = len(kept)
    hypotheses = tuple(sorted(
        (HypothesisScore(turn, float(c / frames), float(s / frames), frames)
         for turn, c, s in zip(assignments, cost, support, strict=True)),
        key=lambda h: (h.cost, h.view_half_turns),
    )) if frames else ()
    return (SideEvidence(camera_ids, reference_camera, frames, pair_frames, hypotheses),
            np.asarray([frame for frame, _ in kept], dtype=np.int64))
