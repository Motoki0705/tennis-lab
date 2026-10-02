"""Geometry-only pair windows for the frozen #964 run-12 protocol.

The protocol is in knowledge/runs/run-i964-recalibration-r12-20260930/protocol.md.
Inputs are full real-observation timelines. No appearance score or predicted
identity determines a pseudo label. Rejected pairs remain in the audit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from src.tasks.person_tracking.court_linking import (
    LinkingConfig,
    select_linked_candidates,
)
from src.tasks.person_tracking.linked_timeline import linked_timeline
from src.tasks.player_association.appearance.affinity import segment_embedding
from src.tasks.player_association.association.associate import (
    AssociationConfig,
    CameraTracks,
)
from src.tasks.player_association.geometry.footpoints import ground_footpoints
from src.tasks.player_association.geometry.switches import switch_candidates


@dataclass(frozen=True)
class CalibrationClip:
    key: str
    fps: float
    cameras: tuple[CameraTracks, ...]
    handoffs: tuple[np.ndarray, ...]  # (D,T) source-track transitions/overlaps

    def __post_init__(self) -> None:
        if len(self.key.split('/')) != 2 or not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('Calibration clip requires video/clip and positive finite FPS')
        if len(self.cameras) != 3 or len({c.camera.camera_id for c in self.cameras}) != 3 \
                or len({c.observed.shape[1] for c in self.cameras}) != 1 or len(self.handoffs) != 3:
            raise ValueError('Calibration requires all three full camera timelines')
        for camera, handoffs in zip(self.cameras, self.handoffs, strict=True):
            if handoffs.shape != camera.observed.shape or handoffs.dtype != bool or camera.appearance is None:
                raise ValueError('Explicit handoff mask and sampled appearance are required')


@dataclass(frozen=True)
class PairWindow:
    video: str
    clip: str
    cameras: tuple[str, str]
    rows: tuple[int, int]
    sides: tuple[int, int]
    start: int
    end: int
    frames: tuple[int, ...]  # shared real observations, never GSI
    distance_m: float
    cosine: float | None
    positive: bool

    def __post_init__(self) -> None:
        if not self.clip.startswith(self.video + '/') or len(set(self.cameras)) != 2 \
                or not 0 <= self.start < self.end or not self.frames \
                or tuple(sorted(set(self.frames))) != self.frames \
                or self.frames[0] < self.start or self.frames[-1] >= self.end:
            raise ValueError('Invalid pair-window identity/timeline')
        if not np.isfinite(self.distance_m) or self.distance_m < 0 \
                or (self.positive and self.distance_m >= 4) or (not self.positive and self.distance_m <= 10):
            raise ValueError('Pair-window is not an unambiguous geometry pseudo label')
        if self.cosine is not None and (not np.isfinite(self.cosine) or not -1 <= self.cosine <= 1):
            raise ValueError('Invalid pair-window cosine')
        if any(s not in (-1, 0, 1) for s in self.sides) or min(self.rows) < 0:
            raise ValueError('Invalid side/track row')


def prepare_clip(key: str, fps: float, raw: tuple[CameraTracks, ...],
                 config: AssociationConfig) -> tuple[CalibrationClip, list[dict[str, Any]]]:
    """Use the unchanged production court selection and group adapter."""
    cameras, handoffs, audits = [], [], []
    for tracks in raw:
        _, audit = select_linked_candidates(tracks, fps, LinkingConfig(), config.footpoints)
        grouped, origins = linked_timeline(tracks, audit)
        events = np.zeros_like(grouped.observed)
        groups = [g for g in audit['groups'] if g['selected']]
        for row, group in enumerate(groups):
            at = np.flatnonzero(grouped.observed[row])
            transitions = at[1:][np.diff(origins[row, at]) != 0]
            events[row, transitions] = True
            multiplicity = np.zeros(grouped.observed.shape[1], np.int64)
            for index in group['fragments']:
                fragment = audit['fragments'][index]
                span = slice(fragment['start'], fragment['end'])
                multiplicity[span] += tracks.observed[fragment['row'], span]
            events[row] |= multiplicity > 1
        cameras.append(grouped)
        handoffs.append(events)
        audits.append(audit)
    return CalibrationClip(key, fps, tuple(cameras), tuple(handoffs)), audits


def pair_windows(clip: CalibrationClip, config: AssociationConfig) -> tuple[list[PairWindow], list[dict[str, Any]]]:
    points, valid, events = [], [], []
    for camera, handoffs in zip(clip.cameras, clip.handoffs, strict=True):
        p, v = ground_footpoints(camera.boxes_xyxy, camera.observed, camera.camera,
                                 camera.image_size[1], config.footpoints)
        event = handoffs.copy()
        for track_row in range(len(camera.track_ids)):
            event[track_row, switch_candidates(p[track_row], v[track_row], clip.fps, config.switches)] = True
        points.append(p)
        valid.append(v)
        events.append(event)
    frames = clip.cameras[0].observed.shape[1]
    windows = math.ceil(frames / (4 * clip.fps))
    pairs, audit = [], []
    for window in range(windows):
        start, end = math.ceil(window * 4 * clip.fps), min(frames, math.ceil((window + 1) * 4 * clip.fps))
        for a, b in combinations(range(3), 2):
            ca, cb = clip.cameras[a], clip.cameras[b]
            eligible: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
            for i in range(len(ca.track_ids)):
                for j in range(len(cb.track_ids)):
                    shared = np.flatnonzero(ca.observed[i, start:end] & cb.observed[j, start:end]) + start
                    row: dict[str, Any] = {'clip': clip.key, 'cameras': [ca.camera.camera_id, cb.camera.camera_id],
                        'rows': [i, j], 'start': start, 'end': end, 'shared_frames': len(shared)}
                    if events[a][i, start:end].any() or events[b][j, start:end].any():
                        reason = 'switch_or_handoff'
                    elif len(shared) < 2 * clip.fps:
                        reason = 'shared_less_than_2s'
                    elif not (valid[a][i, shared] & valid[b][j, shared]).all():
                        # Do not silently remove invalid real observations from the denominator.
                        reason = 'invalid_footpoint'
                    else:
                        distance = float(np.median(np.linalg.norm(points[a][i, shared] - points[b][j, shared], axis=1)))
                        eligible[i, j] = shared, distance
                        continue
                    audit.append({**row, 'reason': reason})
            for (i, j), (shared, distance) in eligible.items():
                alternatives = [d for (x, y), (_, d) in eligible.items() if (x == i or y == j) and (x, y) != (i, j)]
                positive = distance < 4 and all(d > 10 for d in alternatives)
                sides = (int(np.sign(np.median(points[a][i, shared, 1]))),
                         int(np.sign(np.median(points[b][j, shared, 1]))))
                near_far = ['unknown' if side == 0 or camera.camera.center[1] == 0 else
                            'near' if side * camera.camera.center[1] > 0 else 'far'
                            for side, camera in zip(sides, (ca, cb), strict=True)]
                row = {'clip': clip.key, 'cameras': [ca.camera.camera_id, cb.camera.camera_id],
                       'rows': [i, j], 'start': start, 'end': end, 'shared_frames': len(shared), 'distance_m': distance,
                       'sides': sides, 'near_far': near_far}
                if not positive and distance <= 10:
                    audit.append({**row, 'reason': 'ambiguous_geometry'})
                    continue
                assert ca.appearance is not None and cb.appearance is not None
                aa, ab = ca.appearance[i], cb.appearance[j]
                if aa.parts is not None or ab.parts is not None:
                    raise ValueError('The fixed protocol uses CLIP, not native part appearance')
                ea = segment_embedding(aa.frames, aa.embeddings, start, end)
                eb = segment_embedding(ab.frames, ab.embeddings, start, end)
                cosine = None if ea is None or eb is None else float(np.clip(ea @ eb, -1, 1))
                pairs.append(PairWindow(clip.key.split('/')[0], clip.key, (ca.camera.camera_id, cb.camera.camera_id),
                    (i, j), sides, start, end, tuple(int(f) for f in shared), distance, cosine, positive))
                audit.append({**row, 'reason': 'positive' if positive else 'negative',
                              'sides': sides, 'appearance_missing': cosine is None})
    return pairs, audit


def hierarchical_weights(pairs: list[PairWindow]) -> np.ndarray:
    """Equal video -> clip -> camera-pair/side-pair -> window, then divide pairs."""
    if not pairs:
        raise ValueError('Cannot weight an empty pair set')
    identifiers = [(p.clip, p.cameras, p.rows, p.start, p.end) for p in pairs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate pair-window would inflate support')
    keys = [(p.video, p.clip, (p.cameras, p.sides), (p.start, p.end)) for p in pairs]
    weights: np.ndarray = np.ones(len(pairs), np.float64)

    def divide(indices: list[int], depth: int) -> None:
        if depth == 4:
            weights[indices] /= len(indices)
            return
        groups: dict[Any, list[int]] = {}
        for index in indices:
            groups.setdefault(keys[index][depth], []).append(index)
        weights[indices] /= len(groups)
        for group in groups.values():
            divide(group, depth + 1)

    divide(list(range(len(pairs))), 0)
    return weights
