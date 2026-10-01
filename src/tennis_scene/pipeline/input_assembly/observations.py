"""ID/time-axis joins of declared artifacts; no detection or identity inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.identity import PlayerIdentitiesOutput
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.observation_types import (
    GroupedObservations,
    ObjectObservations,
    group_observations,
)


def gather_people(source: ClipSource, artifacts: Mapping[str, Any]) -> ObjectObservations:
    rows: list[ObjectObservations] = [artifacts[f"pose_{c}"] for c in source.camera_ids]
    carriers = max((r.uv_px.shape[2] for r in rows), default=0)
    shape = (len(rows), source.num_frames, carriers, 17)
    uv = np.zeros((*shape, 2), np.float32)
    confidence = np.zeros(shape, np.float32)
    observed = np.zeros(shape[:3], bool)
    ids = np.full((len(rows), carriers), -1, np.int64)
    boxes = np.zeros((*shape[:3], 3), np.float32)
    for v, (camera_id, row) in enumerate(zip(source.camera_ids, rows, strict=True)):
        if row.camera_ids != (camera_id,) or row.num_frames != source.num_frames or row.size != source.size or abs(row.fps - source.fps) > 1e-6:
            raise ValueError("Person artifact camera/timeline mismatch")
        p = row.uv_px.shape[2]
        uv[v, :, :p], confidence[v, :, :p], observed[v, :, :p] = row.uv_px[0], row.confidence[0], row.observed[0]
        ids[v, :p] = row.local_track_ids[0]
        if row.boxes_xys is None:
            raise ValueError("Pose artifacts must retain source boxes")
        boxes[v, :, :p] = row.boxes_xys[0]
    return ObjectObservations(source.camera_ids, source.size, source.fps, uv, confidence, observed, ids, boxes)


def gather_balls(source: ClipSource, artifacts: Mapping[str, Any]) -> ObjectObservations:
    rows: list[BallDetectionOutput] = [artifacts[f"ball_{c}"] for c in source.camera_ids]
    for camera_id, row in zip(source.camera_ids, rows, strict=True):
        if row.camera_id != camera_id or len(row.frame_indices) != source.num_frames:
            raise ValueError("Ball artifact camera/timeline mismatch")
    return ObjectObservations(source.camera_ids, source.size, source.fps,
        np.stack([r.uv_px for r in rows])[:, :, None, None],
        np.stack([r.confidence for r in rows])[:, :, None, None],
        np.stack([r.observed for r in rows])[:, :, None], np.zeros((len(rows), 1), np.int64))


def identified_people(raw: ObjectObservations, identities: PlayerIdentitiesOutput, threshold: float) -> GroupedObservations:
    """Group pose carriers by clip-global player ID; unseen frames carry no ID."""
    if raw.camera_ids != identities.camera_ids:
        raise ValueError("Player identity cameras differ from the calibrated pose observations")
    if not np.array_equal(identities.local_track_ids, raw.local_track_ids):
        raise ValueError("Player identities were built on a different pose track order")
    ids = np.broadcast_to(identities.player_ids[:, None], raw.observed.shape).copy()
    ids[~raw.visibility(threshold).any(-1)] = -1
    return group_observations(raw, ids, threshold=threshold)
