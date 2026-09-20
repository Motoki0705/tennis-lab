"""Read already-associated COCO-17 detections from an exported real clip."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.base.triangulation_residual.contracts import RealResidualScene
from src.tasks.base.triangulation_residual.real_clip import load_clip_calibration
from src.tennis_scene.pipeline.components.player_association import (
    PlayerAssociationResult,
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _validate_association(
    association: dict[str, Any],
    metadata: dict[str, Any],
    *,
    camera_ids: list[str],
    frames: int,
) -> list[int]:
    if association != metadata.get("player_association"):
        raise ValueError("Archive and player_association_result.json disagree")
    if association.get("camera_ids") != camera_ids:
        raise ValueError("Player association and calibration camera order disagree")
    player_ids = association.get("canonical_player_ids")
    if (
        not isinstance(player_ids, list)
        or not player_ids
        or any(type(value) is not int or value < 0 for value in player_ids)
        or len(set(player_ids)) != len(player_ids)
    ):
        raise ValueError(
            "canonical_player_ids must contain unique nonnegative integers"
        )
    if metadata.get("track_ids") != player_ids:
        raise ValueError("Archive track_ids must match canonical player order")
    local_ids = metadata.get("track_ids_by_camera")
    if not isinstance(local_ids, list) or len(local_ids) != len(camera_ids):
        raise ValueError("track_ids_by_camera must match the camera count")
    for ids in local_ids:
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(value) is not int or value < 0 for value in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("Each camera must have unique nonnegative local track IDs")
    segments = association.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Player association must have temporal assignment segments")
    for segment in segments:
        if (
            not isinstance(segment, dict)
            or type(segment.get("start_frame")) is not int
            or type(segment.get("end_frame")) is not int
        ):
            raise ValueError("Association frame bounds must be integers")
        assignments = np.asarray(segment.get("assignments"))
        if assignments.dtype.kind not in "iu" or assignments.shape != (
            len(player_ids),
            len(camera_ids),
        ):
            raise ValueError("Association assignments must be an integer (P,V) array")
    parsed = PlayerAssociationResult.from_dict(association)
    valid, errors = parsed.validate(
        num_frames=frames, local_player_counts=[len(ids) for ids in local_ids]
    )
    if not valid:
        raise ValueError("Invalid player association: " + "; ".join(errors))
    return player_ids


def load_real_clip(clip_dir: Path) -> list[RealResidualScene]:
    """Return one scene per canonical player, preserving view/time/joint order.

    The archive has already applied the persisted cross-camera assignments.
    Its normalized UV is converted using each camera's width/height exactly
    once. Association and camera half-turns are never applied again. Model
    predictions and SMPL arrays in the archive are not ground truth and are
    not loaded.
    """
    rig, court_px, court_scores, fps, info = load_clip_calibration(clip_dir)
    frames = info["num_frames"]
    if type(frames) is not int or frames <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Real clip requires positive num_frames/fps")
    archive_dir = clip_dir / "annotations" / "tennis_scene"
    metadata = _read_object(archive_dir / "scene.metadata.json")
    association_path = clip_dir / "annotations" / "player_association_result.json"
    association = _read_object(association_path)
    player_ids = _validate_association(
        association, metadata, camera_ids=info["camera_ids"], frames=frames
    )
    archive_path = archive_dir / "scene.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        uv = archive["human_kp_2d"]
        scores = archive["human_kp_vis"]
        track_ids = archive["player_track_ids"]
    expected_shape = (len(player_ids), len(rig.K), frames, 17, 2)
    if uv.shape != expected_shape or not np.issubdtype(uv.dtype, np.floating):
        raise ValueError(f"human_kp_2d must be a floating {expected_shape} array")
    if scores.shape != expected_shape[:-1] or not np.issubdtype(
        scores.dtype, np.floating
    ):
        raise ValueError(f"human_kp_vis must be a floating {expected_shape[:-1]} array")
    if (
        track_ids.shape != (len(player_ids),)
        or track_ids.dtype.kind not in "iu"
        or not np.array_equal(track_ids, player_ids)
    ):
        raise ValueError("player_track_ids must match the canonical association order")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("human_kp_vis must contain finite nonnegative detector scores")
    if np.isinf(uv).any() or ((~np.isfinite(uv).all(axis=-1)) & (scores > 0)).any():
        raise ValueError("Nonfinite human keypoints require zero detector score")
    pixels = uv.astype(np.float64) * rig.image_size[None, :, None, None, :]
    raw_scores = scores.astype(np.float64)
    return [
        RealResidualScene(
            observations_px=pixels[index],
            scores=raw_scores[index],
            court_px=court_px,
            court_scores=court_scores,
            rig=rig,
            fps=fps,
            metadata={
                **info,
                "player_id": player_id,
                "player_index": index,
                "player_association": association,
                "association_source": str(association_path.resolve()),
                "association_already_applied": True,
                "observation_source": {
                    "archive": str(archive_path.resolve()),
                    "coordinates": "human_kp_2d",
                    "scores": "human_kp_vis",
                    "input_units": "normalized_image_uv",
                    "joint_convention": "coco17",
                },
                "independent_3d_ground_truth": False,
            },
        )
        for index, player_id in enumerate(player_ids)
    ]
