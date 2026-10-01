"""Read the common, static court calibration of an exported real clip."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.base.generate_dataset import (
    IDENTITY_ROTATION_3D,
    CourtKeypointContractMetadata,
    CourtReferenceFrameProvenance,
    CourtViewRecord,
    extract_court_view_records,
    validate_reference_frame_provenance,
)
from src.tasks.plcs.data.residual_types import CameraRig, RealResidualScene
from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.schema import SceneResult


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _frame_index(value: object, name: str, frames: int) -> int:
    if type(value) is not int or not 0 <= value < frames:
        raise ValueError(f"{name} must be an integer in [0, {frames})")
    return value


def _positive_fps(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_reference_metadata(
    metadata: dict[str, Any], camera_ids: list[str], frames: int
) -> tuple[
    dict[str, Any],
    CourtKeypointContractMetadata,
    CourtReferenceFrameProvenance,
    tuple[CourtViewRecord, ...],
]:
    """Prove that the already-aligned archive uses the physical court gauge."""
    reference = metadata.get("court_reference")
    if not isinstance(reference, dict):
        raise ValueError("court_reference must contain explicit calibration metadata")
    if camera_ids != metadata.get("camera_ids") or camera_ids != reference.get(
        "camera_ids"
    ):
        raise ValueError("Clip, archive and calibration camera order disagree")
    if _positive_integer(metadata.get("num_cameras"), "num_cameras") != len(camera_ids):
        raise ValueError("num_cameras disagrees with camera_ids")
    contract = CourtKeypointContractMetadata.from_mapping(
        metadata.get("court_keypoints"), location="scene.metadata.court_keypoints"
    )
    nested_contract = CourtKeypointContractMetadata.from_mapping(
        reference.get("court_keypoints"), location="court_reference.court_keypoints"
    )
    if contract != nested_contract:
        raise ValueError("Root and nested court_keypoints metadata disagree")
    provenance = CourtReferenceFrameProvenance.from_mapping(
        metadata.get("court_reference_provenance"),
        location="scene.metadata.court_reference_provenance",
    )
    nested_provenance = CourtReferenceFrameProvenance.from_mapping(
        reference.get("court_reference_provenance"),
        location="court_reference.court_reference_provenance",
    )
    if provenance != nested_provenance:
        raise ValueError("Root and nested court_reference_provenance metadata disagree")
    if provenance.contract != contract.contract:
        raise ValueError(
            "Court reference provenance and court_keypoints contract disagree"
        )
    if (
        provenance.reference_from_physical != IDENTITY_ROTATION_3D
        or provenance.physical_from_reference != IDENTITY_ROTATION_3D
    ):
        raise ValueError(
            "Physical-court residual inference requires identity reference transforms"
        )
    views = extract_court_view_records(
        reference, contract=contract.contract, location="court_reference"
    )
    if views is None or [view.camera_id for view in views] != camera_ids:
        raise ValueError("court_keypoint_views must preserve calibration camera order")
    validate_reference_frame_provenance(provenance, views)
    if (
        "reference_camera" not in reference
        or reference["reference_camera"] != provenance.reference_camera_id
    ):
        raise ValueError("reference_camera disagrees with court reference provenance")
    half_turns = reference.get("view_half_turns")
    if (
        not isinstance(half_turns, list)
        or len(half_turns) != len(views)
        or any(type(value) is not bool for value in half_turns)
        or half_turns
        != [view.canonical_from_physical != IDENTITY_ROTATION_3D for view in views]
    ):
        raise ValueError("view_half_turns must agree with ordered court_keypoint_views")
    fits = reference.get("camera_fits")
    if not isinstance(fits, list) or len(fits) != len(camera_ids) or any(not isinstance(fit, dict) for fit in fits):
        raise ValueError("Missing camera fit")
    indices = {_frame_index(fit.get("calibration_frame_index"), "calibration_frame_index", frames) for fit in fits}
    if len(indices) != 1:
        raise ValueError("Every camera must be calibrated on the same frame")
    reference = {**reference, "calibration_frame_index": indices.pop()}
    return reference, contract, provenance, views


def _published_scene(annotation_dir: Path) -> tuple[Path, SceneResult]:
    """The SceneResult v2 the clip's ``annotation.json`` publishes; never a guessed file."""
    from src.tennis_scene.pipeline.storage.scene_index import annotation_scene_path

    path: Path = annotation_scene_path(annotation_dir, _read_object(annotation_dir / "annotation.json"))
    scene = load_scene_result(path)
    if scene.schema_version != 2:
        raise ValueError(f"Residual inference reads only SceneResult v2, got v{scene.schema_version}")
    return path, scene


def _load_calibration(
    clip_dir: Path,
) -> tuple[CameraRig, np.ndarray, np.ndarray, float, dict[str, Any], Path, SceneResult]:
    manifest = _read_object(clip_dir / "clip.json")
    scene_path, scene = _published_scene(clip_dir / "annotations/tennis_scene")
    metadata = scene.metadata
    ids = manifest.get("camera_ids")
    if (
        not isinstance(ids, list)
        or len(ids) < 2
        or any(not isinstance(value, str) or not value.strip() for value in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("camera_ids must contain at least two unique nonempty strings")
    width = _positive_integer(manifest.get("width"), "clip.width")
    height = _positive_integer(manifest.get("height"), "clip.height")
    frames = _positive_integer(manifest.get("num_frames"), "clip.num_frames")
    fps = _positive_fps(manifest.get("fps"), "clip.fps")
    if metadata.get("sync_assumption") != "preprocessed":
        raise ValueError("Residual inference requires pre-synchronized cameras")
    reference, contract, provenance, views = _validate_reference_metadata(
        metadata, ids, frames
    )
    fits = reference["camera_fits"]
    if (scene.width, scene.height, scene.num_frames) != (width, height, frames):
        raise ValueError("Clip dimensions/timeline disagree with SceneResult")
    if abs(_positive_fps(scene.fps, "SceneResult fps") - fps) > 1e-5:
        raise ValueError("Clip fps disagrees with SceneResult")
    court = scene.court_kp.astype(np.float64)
    court_scores = scene.court_vis.astype(np.float64)
    if court.shape != (len(ids), frames, 14, 2) or court_scores.shape != court.shape[:-1]:
        raise ValueError("Expected SceneResult physical CourtKP14")
    if (court_scores < 0).any():
        raise ValueError("Court calibration scores must be nonnegative")
    if not np.allclose(court, court[:, :1], atol=1e-7) or not np.allclose(
        court_scores, court_scores[:, :1]
    ):
        raise ValueError(
            "Time-varying calibration is not supported by this fixed-camera profile"
        )
    rig = CameraRig(
        np.array([f["K"] for f in fits], np.float64),
        np.array([f["R"] for f in fits], np.float64),
        np.array([f["t"] for f in fits], np.float64),
        np.tile(np.array([width, height], np.int64), (len(ids), 1)),
    )
    for i, (fit, view) in enumerate(zip(fits, views, strict=True)):
        center = np.asarray(fit.get("camera_center_court_m"), dtype=np.float64)
        if center.shape != (3,) or not np.array_equal(
            center, view.camera_center_court_m
        ):
            raise ValueError(
                "Camera fit centers disagree with ordered court_keypoint_views"
            )
        if not np.allclose(rig.centers[i], center, rtol=0, atol=1e-5):
            raise ValueError("Stored camera center and R/t disagree")
    info = {
        "clip_id": manifest["clip_id"],
        "camera_ids": ids,
        "num_frames": frames,
        "width": width,
        "height": height,
        "calibration_source": str(scene_path.with_suffix(".metadata.json")),
        "calibration_frame_index": reference["calibration_frame_index"],
        "court_coordinate_frame": "physical_court",
        "court_keypoints": contract.to_dict(),
        "court_reference_provenance": provenance.to_dict(),
        "independent_3d_ground_truth": False,
        "calibration": [f["calibration"] for f in fits],
    }
    return rig, court[:, 0] * [width, height], court_scores[:, 0], fps, info, scene_path, scene


def load_clip_calibration(
    clip_dir: Path,
) -> tuple[CameraRig, np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Read a fixed calibration with explicit identity physical-frame provenance.

    The calibration is the published SceneResult v2's ``court_reference``.
    CourtKP14 in SceneResult has already been aligned to its recorded reference.
    Camera-local half-turns are validated as provenance, never reapplied here.
    """
    rig, court_px, court_scores, fps, info, _, _ = _load_calibration(clip_dir)
    return rig, court_px, court_scores, fps, info


def load_real_clip(clip_dir: Path) -> list[RealResidualScene]:
    """Return one scene per associated player, preserving view/time/joint order.

    The SceneResult v2 player axis is already the cross-camera association of
    the pipeline's ``player_association`` component; it is never re-applied.
    Normalized UV is converted using each camera's width/height exactly once.
    Model predictions and SMPL arrays in the scene are not ground truth and are
    not used.
    """
    rig, court_px, court_scores, fps, info, scene_path, scene = _load_calibration(clip_dir)
    frames = info["num_frames"]
    if scene.player_track_ids is None or scene.human_kp_2d is None or scene.human_kp_vis is None:
        raise ValueError("SceneResult v2 lacks the associated player observations")
    player_ids = scene.player_track_ids.tolist()
    if not player_ids or any(value < 0 for value in player_ids) or len(set(player_ids)) != len(player_ids):
        raise ValueError("player_track_ids must contain unique nonnegative player IDs")
    if scene.metadata.get("track_ids") != player_ids:
        raise ValueError("Scene track_ids must match the player axis")
    uv, scores = scene.human_kp_2d, scene.human_kp_vis
    expected_shape = (len(player_ids), len(rig.K), frames, 17, 2)
    if uv.shape != expected_shape or scores.shape != expected_shape[:-1]:
        raise ValueError(f"human_kp_2d/human_kp_vis must have shapes {expected_shape} and {expected_shape[:-1]}")
    if (scores < 0).any():
        raise ValueError("human_kp_vis must contain nonnegative detector scores")
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
                "association_source": {"scene": str(scene_path.resolve()), "axis": "player_track_ids"},
                "association_already_applied": True,
                "observation_source": {
                    "archive": str(scene_path.resolve()),
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
