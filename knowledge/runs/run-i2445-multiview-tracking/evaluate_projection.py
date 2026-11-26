"""Evaluate projected 3D identities using the existing Meiji box-label units."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from src.tasks.person_tracking.evaluation import score_units, tracking_units
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tennis_scene.multiview_pose.inputs import PoseInputs
from src.tennis_scene.multiview_tracking.schema import PoseTracks
from src.utils.schema.player import COCO17_TORSO_TWIST_JOINTS

TORSO = list(COCO17_TORSO_TWIST_JOINTS)
MIN_2D_SCORE = 0.25
PROJECTION_LIMIT_PX_AT_1080P = 24


def match_projected_torso(
    projected_uv: NDArray[np.float64],
    projected_valid: NDArray[np.bool_],
    keypoints_2d: NDArray[np.float32],
    image_size: tuple[int, int],
    limit_px: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    uv = keypoints_2d[..., :2]
    visible = (keypoints_2d[..., 2] >= MIN_2D_SCORE) & (
        (uv >= 0) & (uv < image_size)
    ).all(axis=-1)
    common = projected_valid[:, None] & visible[None]
    count = common.sum(axis=2)
    distance = np.linalg.norm(projected_uv[:, None] - uv[None], axis=-1)
    ordered = np.sort(np.where(common, distance, np.inf), axis=-1)
    middle_a = np.take_along_axis(
        ordered, np.maximum(0, count - 1)[..., None] // 2, axis=2
    )[..., 0]
    middle_b = np.take_along_axis(ordered, count[..., None] // 2, axis=2)[..., 0]
    error = (middle_a + middle_b) / 2
    admissible = (count >= 2) & (error <= limit_px)
    # Prioritize the number of usable matches, then minimize the median error.
    reward = np.where(admissible, min(admissible.shape) + 1 - error / limit_px, 0)
    rows, cols = linear_sum_assignment(-reward)
    keep = admissible[rows, cols]
    return rows[keep], cols[keep]


def camera_tracks(
    inputs: PoseInputs, tracks: PoseTracks, *, geometric: bool
) -> tuple[CameraTracks, ...]:
    source = inputs.source
    if (
        tracks.camera_ids != source.camera_ids
        or len(tracks.frame_offsets) != source.num_frames + 1
    ):
        raise ValueError("Tracking and projection need the same source axes")
    outputs = []
    count = len(tracks.end_frames)
    track_ids = np.arange(count, dtype=np.int64)
    for view, (camera, dino, pose) in enumerate(
        zip(inputs.cameras, inputs.detections, inputs.poses, strict=True)
    ):
        boxes = np.zeros((count, source.num_frames, 4), np.float32)
        visible = np.zeros((count, source.num_frames), bool)
        if geometric:
            uv, front = camera.project(tracks.positions_m[:, TORSO])
            valid = (
                tracks.measured_joints[:, TORSO]
                & front
                & ((uv >= 0) & (uv < source.size)).all(axis=-1)
            )
            keypoints = pose.keypoints[:, TORSO]
        for frame, (start, end) in enumerate(
            zip(tracks.frame_offsets[:-1], tracks.frame_offsets[1:], strict=True)
        ):
            measured = np.flatnonzero(tracks.observed[start:end]) + start
            first, last = dino.frame_offsets[frame : frame + 2]
            if not len(measured) or first == last:
                continue
            if geometric:
                sample, row = match_projected_torso(
                    uv[measured],
                    valid[measured],
                    keypoints[first:last],
                    source.size,
                    PROJECTION_LIMIT_PX_AT_1080P * source.pixel_threshold_scale,
                )
                ids, original = tracks.track_ids[measured[sample]], first + row
            else:
                own = tracks.source_rows[measured, view]
                supported = own >= 0
                ids, original = tracks.track_ids[measured[supported]], own[supported]
            if ((original < first) | (original >= last)).any():
                raise ValueError(
                    "A projected identity refers to a row from a different frame"
                )
            boxes[ids, frame] = dino.boxes_xyxy[original]
            visible[ids, frame] = True
        outputs.append(CameraTracks(camera, source.size, track_ids, boxes, visible))
    return tuple(outputs)


def evaluate_clip(
    inputs: PoseInputs, labels: ClipLabels, tracks: PoseTracks
) -> dict[str, dict[str, Any]]:
    if labels.clip_id != inputs.source.clip_id or set(labels.cameras) != set(
        tracks.camera_ids
    ):
        raise ValueError("Mismatched label clip or cameras")
    methods = {}
    for method, geometric in (("projected", True), ("supporting_rows", False)):
        views = {}
        for view in camera_tracks(inputs, tracks, geometric=geometric):
            selected = np.isin(view.track_ids, tracks.selection.selected_track_ids)
            units, unknown = tracking_units(
                view, view.observed & selected[:, None], labels
            )
            # The human reference describes player identities. Unselected
            # nonplayers are still tracked and have no identity score here.
            all_units, _ = tracking_units(view, view.observed, labels)
            views[view.camera.camera_id] = {
                "units": units,
                "all_player_units": [
                    unit for unit in all_units if unit["role"] == "player"
                ],
                "unknown": unknown,
                "selected_camera_frames": int(view.observed[selected].sum()),
            }
        methods[method] = views
    return methods


def _compact(units: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = score_units(units)
    return {
        k: v
        for k, v in metrics.items()
        if k not in {"mapping", "switch_events", "fragment_events"}
    }


def summarize_views(views: list[dict[str, Any]]) -> dict[str, Any]:
    selected = _compact([unit for view in views for unit in view["units"]])
    hit, missed = selected["player_units_kept"], selected["player_units"]
    false_selection = selected["nonplayer_units_kept"]
    return {
        "selected_id": selected,
        "all_player_id": _compact(
            [unit for view in views for unit in view["all_player_units"]]
        ),
        "selection": {
            "tp": hit,
            "fp": false_selection,
            "fn": missed - hit,
            "precision": hit / (hit + false_selection)
            if hit + false_selection
            else None,
            "recall": hit / missed if missed else None,
            "unlabelled_predictions": sum(view["unknown"] for view in views),
            "selected_camera_frames": sum(
                view["selected_camera_frames"] for view in views
            ),
        },
    }
