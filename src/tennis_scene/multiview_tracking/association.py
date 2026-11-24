"""Fixed-lag path cover with 3D pose and bidirectional motion costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, eye, hstack
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from src.tennis_scene.multiview_pose.schema import FramePoseCandidates
from src.tennis_scene.multiview_tracking.schema import TrackingConfig
from src.utils.schema.player import COCO17_TORSO_TWIST_JOINTS

_TORSO = list(COCO17_TORSO_TWIST_JOINTS)


@dataclass(frozen=True)
class Observations:
    frames: NDArray[np.int64]
    torso: NDArray[np.float64]
    valid: NDArray[np.bool_]
    centres: NDArray[np.float64]
    cameras: NDArray[np.bool_]
    penalty: NDArray[np.float64]

    @classmethod
    def from_candidates(cls, candidates: FramePoseCandidates) -> Observations:
        torso = candidates.positions_m[:, _TORSO].astype(np.float64)
        valid = candidates.joint_valid[:, _TORSO]
        support = valid.sum(axis=1)
        if (support < 2).any():
            raise ValueError(
                "Each candidate needs at least two reconstructed torso joints"
            )
        cameras = candidates.source_rows >= 0
        views = cameras.sum(axis=1)
        score_per_pair = candidates.scores / (views * (views - 1) / 2)
        gap = np.minimum(candidates.relative_score_gap / 0.20, 1)
        penalty = (
            0.08 * (1 - np.clip(score_per_pair, 0, 1))
            + 0.10 * candidates.competing_assignment * (1 - gap)
            + 0.05 * candidates.ambiguous
        )
        return cls(
            np.repeat(
                np.arange(len(candidates.frame_offsets) - 1),
                np.diff(candidates.frame_offsets),
            ),
            torso,
            valid,
            torso.sum(axis=1) / support[:, None],
            cameras,
            penalty,
        )

    def slice(self, start: int, stop: int) -> Observations:
        section = slice(start, stop)
        return Observations(
            self.frames[section],
            self.torso[section],
            self.valid[section],
            self.centres[section],
            self.cameras[section],
            self.penalty[section],
        )

    def displacement(
        self, earlier: NDArray[np.int64], later: NDArray[np.int64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        shared = self.valid[earlier] & self.valid[later]
        support = shared.sum(axis=1)
        differences = self.torso[later] - self.torso[earlier]
        divisor = np.maximum(1, support)
        displacement = (differences * shared[..., None]).sum(axis=1) / divisor[:, None]
        disjoint = support == 0
        displacement[disjoint] = (
            self.centres[later[disjoint]] - self.centres[earlier[disjoint]]
        )
        # Compare the shape/orientation after removing translation on the same joints.
        residual = np.square(differences - displacement[:, None]).sum(axis=2)
        pose_error = np.sqrt((residual * shared).sum(axis=1) / divisor)
        return displacement, pose_error, support


@dataclass(frozen=True)
class Motion:
    velocity: NDArray[np.float64]
    valid: NDArray[np.bool_]


@dataclass(frozen=True)
class Association:
    predecessor: NDArray[np.int64]
    margin: NDArray[np.float64]
    max_link_delta: int
    motion_span: int


@dataclass(frozen=True)
class Edges:
    earlier: NDArray[np.int64]
    later: NDArray[np.int64]
    dt: NDArray[np.float64]
    displacement: NDArray[np.float64]
    position_slack: NDArray[np.float64]
    reward: NDArray[np.float64]


def _edges(
    observations: Observations,
    eligible: NDArray[np.bool_],
    start: int,
    fps: float,
    max_link_delta: int,
    config: TrackingConfig,
) -> Edges:
    frames = observations.frames
    sources = np.flatnonzero(eligible)
    starts = np.searchsorted(frames, np.maximum(start, frames[sources] + 1))
    ends = np.searchsorted(frames, frames[sources] + max_link_delta, side="right")
    counts = ends - starts
    earlier: NDArray[np.int64] = np.repeat(sources, counts)
    later = np.repeat(starts - (np.cumsum(counts) - counts), counts) + np.arange(
        counts.sum()
    )
    delta_frames = frames[later] - frames[earlier]
    dt = delta_frames / fps
    displacement, pose_error, shared = observations.displacement(earlier, later)
    distance = np.linalg.norm(displacement, axis=1)
    before, after = observations.cameras[earlier], observations.cameras[later]
    camera_change = (before != after).any(axis=1)
    two_view = np.minimum(before.sum(axis=1), after.sum(axis=1)) == 2
    # A change in a weak triangulation can shift the apparent 3D position.
    position_slack = (
        0.25 + 0.30 * camera_change + two_view * (0.35 + 0.40 * camera_change)
    )
    allowed = distance <= position_slack + config.max_speed_mps * dt
    # One retained link saves a birth/death pair. Missing frames, position
    # residual, torso deformation and weak frame evidence spend that reward.
    reward = (
        1.0
        - 0.40 * np.square(distance / (position_slack + 4.0 * dt))
        - 0.26
        * np.square(
            pose_error / (0.28 + 0.16 * two_view + 0.15 * camera_change + 0.7 * dt)
        )
        - 0.24 * (delta_frames > 1)
        - 0.38 * (delta_frames - 1) / max(1, max_link_delta - 1)
        - np.maximum(observations.penalty[earlier], observations.penalty[later])
        - 0.05 * (shared < 2)
    )
    legal = allowed & (reward > 0)
    return Edges(
        earlier[legal],
        later[legal],
        dt[legal],
        displacement[legal],
        position_slack[legal],
        reward[legal],
    )


def _successors(predecessor: NDArray[np.int64]) -> NDArray[np.int64]:
    successor: NDArray[np.int64] = np.full(len(predecessor), -1, np.int64)
    linked = np.flatnonzero(predecessor >= 0)
    successor[predecessor[linked]] = linked
    return successor


def _motion(
    observations: Observations,
    neighbours: NDArray[np.int64],
    *,
    fps: float,
    span: int,
    max_speed_mps: float,
) -> Motion:
    frames = observations.frames
    origin: NDArray[np.int64] = np.arange(len(frames), dtype=np.int64)
    endpoint = origin.copy()
    comparable = origin.copy()
    while True:
        following = neighbours[endpoint]
        advance = (following >= 0) & (np.abs(frames[following] - frames) <= span)
        if not advance.any():
            break
        endpoint[advance] = following[advance]
        same_views = (observations.cameras[endpoint] == observations.cameras).all(
            axis=1
        )
        update = advance & same_views
        comparable[update] = endpoint[update]
    dt_frames = frames[comparable] - frames
    valid = np.abs(dt_frames) >= max(1, span // 3)
    velocity: NDArray[np.float64] = np.zeros((len(frames), 3), np.float64)
    rows = np.flatnonzero(valid)
    displacement, _, _ = observations.displacement(rows, comparable[rows])
    velocity[valid] = displacement * (fps / dt_frames[valid, None])
    valid &= np.linalg.norm(velocity, axis=1) <= max_speed_mps
    velocity[~valid] = 0
    return Motion(velocity, valid)


def _motion_reward(edges: Edges, past: Motion, future: Motion) -> NDArray[np.float64]:
    dt = edges.dt
    before, after = past.velocity[edges.earlier], future.velocity[edges.later]
    before_valid, after_valid = past.valid[edges.earlier], future.valid[edges.later]
    scale = 0.60 * edges.position_slack + 1.0 * dt + 4.0 * np.square(dt)
    before_error = np.square(edges.displacement - before * dt[:, None]).sum(axis=1)
    after_error = np.square(edges.displacement - after * dt[:, None]).sum(axis=1)
    residual = (before_valid * before_error + after_valid * after_error) / (
        np.maximum(1, before_valid.astype(int) + after_valid) * np.square(scale)
    )
    speed_difference = np.square(before - after).sum(axis=1) / np.square(3 + 4 * dt)
    return (
        edges.reward
        - 0.33 * np.minimum(residual, 1.3)
        - (0.12 * (before_valid & after_valid) * np.minimum(speed_difference, 1.5))
    )


def _match(
    size: int, edges: Edges, reward: NDArray[np.float64]
) -> tuple[NDArray[np.int64], csr_matrix]:
    keep = reward > 0
    source, target, value = edges.earlier[keep], edges.later[keep], reward[keep]
    gain = csr_matrix((value, (source, target)), shape=(size, size))
    costs = csr_matrix((2 - value, (source, target)), shape=gain.shape)
    # Each candidate has an incoming and an outgoing copy. A private dummy
    # successor at cost 2 represents taking no link from that candidate.
    rows, cols = min_weight_full_bipartite_matching(
        hstack((costs, 2 * eye(size, format="csr")), format="csr")
    )
    predecessor: NDArray[np.int64] = np.full(size, -1, np.int64)
    real = cols < size
    predecessor[cols[real]] = rows[real]
    return predecessor, gain


def _alternative_margins(
    predecessor: NDArray[np.int64], gain: csr_matrix
) -> NDArray[np.float64]:
    """Find the lowest loss among feasible drops and one/two-link exchanges."""
    margin = np.full(len(predecessor), np.inf)
    chosen = np.flatnonzero(predecessor >= 0)
    incoming = np.zeros(len(predecessor))
    outgoing = incoming.copy()
    if len(chosen):
        chosen_weight = np.asarray(gain[predecessor[chosen], chosen]).ravel()
        margin[chosen] = incoming[chosen] = outgoing[predecessor[chosen]] = (
            chosen_weight
        )

    edges = gain.tocoo()
    unused = predecessor[edges.col] != edges.row
    src, dst = edges.row[unused], edges.col[unused]
    if not len(src):
        return np.where(np.isfinite(margin), margin, -1)
    old_target = _successors(predecessor)[src]
    old_source = predecessor[dst]
    loss = outgoing[src] + incoming[dst] - edges.data[unused]
    swap = (old_target >= 0) & (old_source >= 0)
    if swap.any():
        loss[swap] -= np.asarray(gain[old_source[swap], old_target[swap]]).ravel()
    # Clamp floating-point noise at a tie.
    if (loss < -1e-8).any():
        raise RuntimeError("An alternative improved the window matching")
    loss = np.maximum(loss, 0)
    np.minimum.at(margin, dst, loss)
    replaced = old_target >= 0
    np.minimum.at(margin, old_target[replaced], loss[replaced])
    return np.where(np.isfinite(margin), margin, -1)


def associate(
    candidates: FramePoseCandidates,
    observations: Observations,
    fps: float,
    config: TrackingConfig,
) -> Association:
    offsets = candidates.frame_offsets
    frames = observations.frames
    step = max(1, round(config.commit_seconds * fps))
    max_link_delta = round(config.max_occlusion_seconds * fps) + 1
    motion_span = max(1, round(config.motion_span_seconds * fps))
    window = max(
        round(config.window_seconds * fps), step + max_link_delta + motion_span
    )
    num_frames = len(offsets) - 1
    predecessor: NDArray[np.int64] = np.full(len(frames), -1, np.int64)
    margins = np.full(len(frames), -1.0)
    committed_successor: NDArray[np.bool_] = np.zeros(len(frames), bool)
    for start in range(0, num_frames, step):
        end = min(num_frames, start + window)
        commit = min(num_frames, start + step)
        if offsets[start] == offsets[commit]:
            continue
        first = offsets[max(0, start - max_link_delta - motion_span)]
        current = offsets[start] - first
        local = observations.slice(int(first), int(offsets[end]))
        size = len(local.frames)
        eligible = (local.frames >= start - max_link_delta) & ~committed_successor[
            first : offsets[end]
        ]
        edges = _edges(local, eligible, start, fps, max_link_delta, config)
        seed, _ = _match(size, edges, edges.reward)
        history = predecessor[first : offsets[start]] - first
        seed[:current] = np.maximum(history, -1)
        past = _motion(
            local, seed, fps=fps, span=motion_span, max_speed_mps=config.max_speed_mps
        )
        future = _motion(
            local,
            _successors(seed),
            fps=fps,
            span=motion_span,
            max_speed_mps=config.max_speed_mps,
        )
        links, gain = _match(size, edges, _motion_reward(edges, past, future))
        margin = _alternative_margins(links, gain)
        committed = slice(current, offsets[commit] - first)
        previous = links[committed]
        predecessor[offsets[start] : offsets[commit]] = np.where(
            previous >= 0, previous + first, -1
        )
        committed_successor[previous[previous >= 0] + first] = True
        margins[offsets[start] : offsets[commit]] = margin[committed]
    return Association(predecessor, margins, max_link_delta, motion_span)


def trajectory_motion(
    observations: Observations, links: Association, fps: float, config: TrackingConfig
) -> Motion:
    past = _motion(
        observations,
        links.predecessor,
        fps=fps,
        span=links.motion_span,
        max_speed_mps=config.max_speed_mps,
    )
    future = _motion(
        observations,
        _successors(links.predecessor),
        fps=fps,
        span=links.motion_span,
        max_speed_mps=config.max_speed_mps,
    )
    count = past.valid.astype(int) + future.valid
    return Motion(
        (past.velocity + future.velocity) / np.maximum(1, count[:, None]), count > 0
    )
