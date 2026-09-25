"""Conservative, auditable links between camera-local BoT-SORT tracklets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.pipeline.errors import ReconstructionUnavailable

MAX_GAP_FRAMES = 60
MAX_CENTER_DISTANCE_DIAGONALS = 1.0
MAX_SIZE_RATIO = 2.0
MAX_APPEARANCE_LAB_DISTANCE = 20.0
ENDPOINT_OBSERVATIONS = 10


@dataclass(frozen=True)
class TrackletLink:
    earlier_id: int
    later_id: int
    missing_frames: int
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


def _candidate(earlier: _Tracklet, later: _Tracklet) -> TrackletLink | None:
    gap = later.frames[0] - earlier.frames[-1] - 1
    if gap < 0 or gap > MAX_GAP_FRAMES:
        return None
    before, after = earlier.boxes[-1], later.boxes[0]
    center_distance = float(np.linalg.norm((before[:2] + before[2:] - after[:2] - after[2:]) / 2))
    mean_diagonal = float(np.linalg.norm(((before[2:] - before[:2]) + (after[2:] - after[:2])) / 2))
    normalized_distance = center_distance / mean_diagonal
    size_ratio = float(np.max(np.maximum((before[2:] - before[:2]) / (after[2:] - after[:2]),
                                          (after[2:] - after[:2]) / (before[2:] - before[:2]))))
    before_color = np.median(earlier.appearance_lab[-ENDPOINT_OBSERVATIONS:], axis=0)
    after_color = np.median(later.appearance_lab[:ENDPOINT_OBSERVATIONS], axis=0)
    appearance_distance = float(np.linalg.norm(before_color - after_color))
    if (normalized_distance > MAX_CENTER_DISTANCE_DIAGONALS or size_ratio > MAX_SIZE_RATIO
            or appearance_distance > MAX_APPEARANCE_LAB_DISTANCE):
        return None
    return TrackletLink(earlier.track_id, later.track_id, gap, normalized_distance, size_ratio, appearance_distance)


def link_tracklets(history: list[list[dict[str, Any]]]) -> LinkedTracklets:
    """Join only unique, short-gap matches in time, position, size and clothing.

    A plausible competing link is an identity error, so processing stops for review.
    The source IDs and measured evidence remain in the component artifact.
    """
    tracklets = _tracklets(history)
    ordered = sorted(tracklets.values(), key=lambda row: (row.frames[0], row.track_id))
    candidates = [link for earlier in ordered for later in ordered
                  if earlier.track_id != later.track_id
                  if (link := _candidate(earlier, later)) is not None]
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
    linked = [[{**observation, "id": root(int(observation["id"]))} for observation in frame]
              for frame in history]
    if any(len({int(row["id"]) for row in frame}) != len(frame) for frame in linked):
        raise ReconstructionUnavailable("person_identity_overlap", "Linked tracks overlap in a source frame")
    return LinkedTracklets(linked, {track_id: tuple(ids) for track_id, ids in members.items()}, tuple(candidates))
