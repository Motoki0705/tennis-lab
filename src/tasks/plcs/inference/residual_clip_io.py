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
from src.tasks.plcs.data.manual_association import (
    PlayerAssociationResult,
)
from src.tasks.plcs.data.residual_types import CameraRig, RealResidualScene


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
    calibration_index = _frame_index(
        reference.get("calibration_frame_index"), "calibration_frame_index", frames
    )
    if (
        _frame_index(metadata.get("frame_index"), "frame_index", frames)
        != calibration_index
    ):
        raise ValueError("frame_index and calibration_frame_index disagree")
    indices = metadata.get("court_kp_frame_indices")
    if (
        not isinstance(indices, list)
        or any(type(value) is not int for value in indices)
        or indices != list(range(frames))
    ):
        raise ValueError(
            "court_kp_frame_indices must match the complete ordered clip timeline"
        )
    return reference, contract, provenance, views


def load_clip_calibration(
    clip_dir: Path,
) -> tuple[CameraRig, np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Read a fixed calibration with explicit identity physical-frame provenance.

    CourtKP14 in SceneResult has already been aligned to its recorded reference.
    Camera-local half-turns are validated as provenance, never reapplied here.
    """
    manifest = _read_object(clip_dir / "clip.json")
    archive_dir = clip_dir / "annotations/tennis_scene"
    from src.tennis_scene.pipeline.storage.scene_index import indexed_scene_path
    scene_path = indexed_scene_path(archive_dir / "scene.json") if (archive_dir / "scene.json").is_file() else archive_dir / "scene.npz"
    metadata = _read_object(scene_path.with_suffix(".metadata.json"))
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
    fits = reference.get("camera_fits")
    if (
        not isinstance(fits, list)
        or len(fits) != len(ids)
        or any(not isinstance(fit, dict) for fit in fits)
    ):
        raise ValueError("Missing camera fit")
    with np.load(scene_path, allow_pickle=False) as archive:
        for name, expected in (
            ("width", width),
            ("height", height),
            ("num_frames", frames),
        ):
            value = archive[name]
            if (
                value.shape != ()
                or value.dtype.kind not in "iu"
                or int(value) != expected
            ):
                raise ValueError("Clip dimensions/timeline disagree with SceneResult")
        fps_array = archive["fps"]
        if fps_array.shape != ():
            raise ValueError("SceneResult fps must be scalar")
        archive_fps = _positive_fps(fps_array.item(), "SceneResult fps")
        if abs(archive_fps - fps) > 1e-5:
            raise ValueError("Clip fps disagrees with SceneResult")
        court = archive["court_kp"].astype(np.float64)
        court_scores = archive["court_vis"].astype(np.float64)
        if (
            court.shape != (len(ids), frames, 14, 2)
            or court_scores.shape != court.shape[:-1]
        ):
            raise ValueError("Expected SceneResult physical CourtKP14")
        if (
            not np.isfinite(court).all()
            or not np.isfinite(court_scores).all()
            or (court_scores < 0).any()
        ):
            raise ValueError(
                "Court calibration observations/scores must be finite and scores nonnegative"
            )
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
    return rig, court[:, 0] * [width, height], court_scores[:, 0], fps, info


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
    from src.tennis_scene.pipeline.storage.scene_index import indexed_scene_path
    scene_path = indexed_scene_path(archive_dir / "scene.json") if (archive_dir / "scene.json").is_file() else archive_dir / "scene.npz"
    metadata = _read_object(scene_path.with_suffix(".metadata.json"))
    association_path = clip_dir / "annotations" / "player_association_result.json"
    association = _read_object(association_path)
    player_ids = _validate_association(
        association, metadata, camera_ids=info["camera_ids"], frames=frames
    )
    archive_path = scene_path
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
