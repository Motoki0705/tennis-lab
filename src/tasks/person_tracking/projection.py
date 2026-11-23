"""Project COCO17 tracks into real camera detections for tracker comparison."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tasks.person_tracking.comparison import TrackingPrediction
from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.player_association.association.associate import CameraTracks
from src.utils.geometry.triangulation import PinholeCamera


def _project_boxes(
    camera: PinholeCamera, image_size: tuple[int, int], detections: Sequence[DetectionFeatures],
    ids: np.ndarray, joints: np.ndarray, valid: np.ndarray,
) -> CameraTracks:
    width, height = image_size
    uv, front = camera.project(joints)
    visible = valid & front & (uv >= 0).all(-1) & (uv < (width, height)).all(-1)
    boxes = np.zeros((*valid.shape[:2], 4), dtype=np.float32)
    observed = np.zeros(valid.shape[:2], dtype=bool)
    max_rms = 40.0 * math.hypot(width, height) / math.hypot(1920, 1080)
    for frame, detection in enumerate(detections):
        pose = detection.require_poses()
        if detection.frame != frame:
            raise ValueError(f"{camera.camera_id}: detections must cover every source frame in order")
        order = np.argsort(detection.rows)
        bbox, pose = detection.boxes[order], pose[order]
        actual = pose[..., :2]
        projected = uv[:, frame]
        # Compare only jointly available keypoints on the same source grid.
        common = (visible[:, frame, None] & (pose[None, ..., 2] >= .3)
                  & (actual[None] >= 0).all(-1) & (actual[None] < (width, height)).all(-1))
        inside = ((projected[:, None] >= bbox[None, :, None, :2])
                  & (projected[:, None] < bbox[None, :, None, 2:])).all(-1)
        count = common.sum(-1)
        distance2 = ((projected[:, None] - actual[None]) ** 2).sum(-1)
        rms = np.sqrt(np.where(common, distance2, 0).sum(-1) / np.maximum(count, 1))
        admissible = (count >= 4) & ((common & inside).sum(-1) * 2 > count) & (rms <= max_rms)
        capacity = min(count.shape)
        precision = 1_000_000
        quality = np.rint((1. - rms / max_rms) * precision)
        # Sub-unit tie weight keeps lower track IDs when identical projections
        # compete and cannot outweigh a quantized residual or an extra match.
        priority = (len(ids) - np.arange(len(ids)))[:, None] / ((len(ids) + 1) * (capacity + 1))
        reward = np.where(admissible, (capacity + 1) * (precision + 1) + quality + priority, 0.)
        track, row = linear_sum_assignment(-reward)
        keep = admissible[track, row]
        boxes[track[keep], frame] = bbox[row[keep]]
        observed[track[keep], frame] = True
    return CameraTracks(camera, image_size, ids, boxes, observed)


def project_3d_prediction(
    cameras: Mapping[str, PinholeCamera], image_size: tuple[int, int],
    detections: Mapping[str, Sequence[DetectionFeatures]], track_ids: np.ndarray,
    joints_xyz: np.ndarray, valid: np.ndarray, selected: np.ndarray, *,
    stop_reason: str | None = None,
) -> TrackingPrediction:
    """Assign projected 3D IDs to source boxes, independently of reference labels.

    World joints and validity are (P,T,17,3) and (P,T,17) on the synchronized
    source grid. Selection is the 3D route's own player mask (P,T). IDs are
    clip-global and serve as both raw and group IDs on matched source boxes.
    """
    if (not cameras or cameras.keys() != detections.keys()
            or any(name != camera.camera_id for name, camera in cameras.items()) or min(image_size) <= 0):
        raise ValueError("Projection requires the same camera IDs for calibration and real detections")
    if (joints_xyz.ndim != 4 or joints_xyz.shape[-2:] != (17, 3)
            or joints_xyz.shape[1] < 1 or joints_xyz.shape[:-1] != valid.shape
            or valid.dtype != np.bool_ or selected.shape != valid.shape[:2] or selected.dtype != np.bool_
            or track_ids.shape != (valid.shape[0],) or track_ids.dtype != np.int64
            or len(np.unique(track_ids)) != len(track_ids) or (track_ids < 0).any()
            or not np.isfinite(joints_xyz[valid]).all()
            or any(len(frames) != valid.shape[1] for frames in detections.values())):
        raise ValueError("3D IDs, joints, validity and selection must share the source timeline")
    order = np.argsort(track_ids)
    ids, joints, mask, selected = track_ids[order], joints_xyz[order], valid[order], selected[order]
    raw = {name: _project_boxes(camera, image_size, detections[name], ids, joints, mask)
           for name, camera in cameras.items()}
    chosen = {name: tracks.observed & selected for name, tracks in raw.items()}
    groups = {name: CameraTracks(tracks.camera, image_size, ids, tracks.boxes_xyxy, chosen[name])
              for name, tracks in raw.items()}
    player_ids = {name: np.where(keep, ids[:, None], -1) for name, keep in chosen.items()}
    valid_3d = {int(identity): mask[row].any(-1) & selected[row] for row, identity in enumerate(ids)}
    return TrackingPrediction(raw, chosen, groups, player_ids, valid_3d, stop_reason)
