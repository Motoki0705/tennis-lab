"""Conservative, auditable links between camera-local BoT-SORT tracklets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.pipeline.errors import ReconstructionUnavailable


@dataclass(frozen=True)
class TrackletLinkPolicy:
    """Thresholds that a BoT-SORT tracklet split must satisfy to be joined.

    The values are part of the ``person_tracking`` artifact identity.
    """

    max_gap_frames: int = 60
    max_overlap_span_frames: int = 3
    min_duplicate_containment: float = .95
    max_center_distance_diagonals: float = 1.0
    max_size_ratio: float = 2.0
    max_duplicate_size_ratio: float = 2.5
    max_appearance_lab_distance: float = 20.0
    endpoint_observations: int = 10


@dataclass(frozen=True)
class TrackletLink:
    earlier_id: int
    later_id: int
    missing_frames: int
    overlap_span_frames: int
    shared_observation_frames: int
    duplicate_containment: float | None
    center_distance_diagonals: float
    size_ratio: float
    appearance_lab_distance: float


@dataclass(frozen=True)
class LinkedTracklets:
    history: list[list[dict[str, Any]]]
    source_ids: dict[int, tuple[int, ...]]
    links: tuple[TrackletLink, ...]


@dataclass(frozen=True)
class _Tracklet:
    track_id: int
    frames: tuple[int, ...]
    boxes: NDArray[np.float64]
    appearance_lab: NDArray[np.float64]


def torso_appearance_lab(frame_bgr: NDArray[np.uint8], box_xyxy: NDArray[np.float32]) -> NDArray[np.float64]:
    """Measure the inner upper body, excluding most of the box background."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
        raise ValueError("Tracking appearance requires a BGR uint8 frame")
    box = np.asarray(box_xyxy, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= box[:2]).any():
        raise ValueError("Tracking box must be a finite positive xyxy rectangle")
    x1, y1, x2, y2 = box
    left = max(0, int(round(x1 + .2 * (x2 - x1))))
    right = min(frame_bgr.shape[1], int(round(x1 + .8 * (x2 - x1))))
    top = max(0, int(round(y1 + .15 * (y2 - y1))))
    bottom = min(frame_bgr.shape[0], int(round(y1 + .55 * (y2 - y1))))
    if right <= left or bottom <= top:
        raise ValueError("Tracking appearance crop falls outside the source frame")
    lab = cv2.cvtColor(frame_bgr[top:bottom, left:right], cv2.COLOR_BGR2LAB)
    return np.median(lab.reshape(-1, 3), axis=0).astype(np.float64)


def _tracklets(history: list[list[dict[str, Any]]]) -> dict[int, _Tracklet]:
    grouped: dict[int, list[tuple[int, NDArray[np.float64], NDArray[np.float64]]]] = defaultdict(list)
    for frame_index, observations in enumerate(history):
        seen: set[int] = set()
        for observation in observations:
            track_id = int(observation["id"])
            if track_id in seen:
                raise ValueError("Duplicate tracker ID in one frame")
            seen.add(track_id)
            box = np.asarray(observation["bbx_xyxy"], dtype=np.float64)
            color = np.asarray(observation["appearance_lab"], dtype=np.float64)
            if box.shape != (4,) or color.shape != (3,) or not np.isfinite(box).all() or not np.isfinite(color).all() or (box[2:] <= box[:2]).any():
                raise ValueError("Invalid box or appearance in tracking observation")
            grouped[track_id].append((frame_index, box, color))
    return {
        track_id: _Tracklet(track_id, tuple(row[0] for row in rows),
            np.stack([row[1] for row in rows]), np.stack([row[2] for row in rows]))
        for track_id, rows in grouped.items()
    }


def _candidate(earlier: _Tracklet, later: _Tracklet, policy: TrackletLinkPolicy) -> TrackletLink | None:
    gap = later.frames[0] - earlier.frames[-1] - 1
    if gap > policy.max_gap_frames:
        return None
    overlap_span = max(0, -gap)
    containment: float | None = None
    shared = set(earlier.frames) & set(later.frames)
    if overlap_span:
        if overlap_span > policy.max_overlap_span_frames or later.frames[-1] <= earlier.frames[-1] or len(shared) != 1:
            return None
        frame = next(iter(shared))
        first = earlier.boxes[earlier.frames.index(frame)]
        second = later.boxes[later.frames.index(frame)]
        shared_size = np.maximum(np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2]), 0)
        intersection = float(np.prod(shared_size))
        containment = intersection / min(float(np.prod(first[2:] - first[:2])), float(np.prod(second[2:] - second[:2])))
        if containment < policy.min_duplicate_containment:
            return None
    before, after = earlier.boxes[-1], later.boxes[0]
    center_distance = float(np.linalg.norm((before[:2] + before[2:] - after[:2] - after[2:]) / 2))
    mean_diagonal = float(np.linalg.norm(((before[2:] - before[:2]) + (after[2:] - after[:2])) / 2))
    normalized_distance = center_distance / mean_diagonal
    size_ratio = float(np.max(np.maximum((before[2:] - before[:2]) / (after[2:] - after[:2]),
                                          (after[2:] - after[:2]) / (before[2:] - before[:2]))))
    before_color = np.median(earlier.appearance_lab[-policy.endpoint_observations:], axis=0)
    after_color = np.median(later.appearance_lab[:policy.endpoint_observations], axis=0)
    appearance_distance = float(np.linalg.norm(before_color - after_color))
    allowed_ratio = policy.max_duplicate_size_ratio if overlap_span else policy.max_size_ratio
    if (normalized_distance > policy.max_center_distance_diagonals or size_ratio > allowed_ratio
            or appearance_distance > policy.max_appearance_lab_distance):
        return None
    return TrackletLink(earlier.track_id, later.track_id, max(0, gap), overlap_span, len(shared), containment,
                        normalized_distance, size_ratio, appearance_distance)


def link_tracklets(history: list[list[dict[str, Any]]], policy: TrackletLinkPolicy) -> LinkedTracklets:
    """Join only unique, short-gap matches in time, position, size and clothing.

    A plausible competing link is an identity error, so processing stops for review.
    The source IDs and measured evidence remain in the component artifact.
    """
    tracklets = _tracklets(history)
    ordered = sorted(tracklets.values(), key=lambda row: (row.frames[0], row.track_id))
    candidates = [link for earlier in ordered for later in ordered
                  if earlier.track_id != later.track_id
                  if (link := _candidate(earlier, later, policy)) is not None]
    outgoing: dict[int, list[TrackletLink]] = defaultdict(list)
    incoming: dict[int, list[TrackletLink]] = defaultdict(list)
    for link in candidates:
        outgoing[link.earlier_id].append(link)
        incoming[link.later_id].append(link)
    if any(len(options) > 1 for options in (*outgoing.values(), *incoming.values())):
        raise ReconstructionUnavailable("person_identity_ambiguous", "Multiple plausible camera-local tracklet links require identity review")
    predecessor = {link.later_id: link.earlier_id for link in candidates}

    def root(track_id: int) -> int:
        while track_id in predecessor:
            track_id = predecessor[track_id]
        return track_id

    members: dict[int, list[int]] = defaultdict(list)
    for tracklet in ordered:
        members[root(tracklet.track_id)].append(tracklet.track_id)
    linked: list[list[dict[str, Any]]] = []
    for frame in history:
        retained: dict[int, dict[str, Any]] = {}
        for observation in sorted(frame, key=lambda row: (tracklets[int(row["id"])].frames[0], int(row["id"]))):
            track_id = int(observation["id"])
            stable_id = root(track_id)
            if stable_id in retained:
                # A shared frame is accepted only after the duplicate-box
                # containment check above. Preserve the older source tracklet.
                previous = int(retained[stable_id]["source_track_id"])
                if (previous, track_id) not in {(link.earlier_id, link.later_id) for link in candidates}:
                    raise ReconstructionUnavailable("person_identity_overlap", "Linked tracks overlap without direct duplicate evidence")
                continue
            retained[stable_id] = {**observation, "id": stable_id, "source_track_id": track_id}
        linked.append(list(retained.values()))
    return LinkedTracklets(linked, {track_id: tuple(ids) for track_id, ids in members.items()}, tuple(candidates))
