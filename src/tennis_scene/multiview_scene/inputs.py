"""Bind 3D time IDs to their original detector rows and the adopted court gauge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from src.tasks.person_tracking.features import FeatureConfig, pose_boxes
from src.tennis_scene.multiview_pose.schema import NODE as POSE_NODE
from src.tennis_scene.multiview_pose.schema import FramePoseCandidates
from src.tennis_scene.multiview_tracking.schema import NODE as TRACK_NODE
from src.tennis_scene.multiview_tracking.schema import PoseTracks
from src.tennis_scene.multiview_tracking.storage import (
    read_candidate_store,
    read_tracks,
)
from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_features import (
    PersonPoseFeaturesOutput,
    require_alignment,
)
from src.tennis_scene.pipeline.components.player_reconstruction import PlayerSkeleton
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.observation_types import (
    GroupedObservations,
    ObjectObservations,
    group_observations,
)
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, restore_type
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PointRejection

if TYPE_CHECKING:
    from pathlib import Path

    from src.tennis_scene.pipeline.clip_inputs import DatasetClipInputs

RECIPE = "multiview_3d_tracking"
INPUT_NODE = "multiview_scene_inputs"
_T = TypeVar("_T")


@dataclass(frozen=True)
class PlayerInputs:
    raw: ObjectObservations
    grouped: GroupedObservations
    skeleton: PlayerSkeleton
    source_rows: NDArray[np.int64]  # P,V,T, with -1 for absent detector rows.
    interpolated_joints: NDArray[np.bool_]  # P,T,17, separate from measured joints.
    frame_ambiguous: NDArray[np.bool_]  # P,T
    id_ambiguous: NDArray[np.bool_]  # P,T
    sides: tuple[int, ...]


@dataclass(frozen=True)
class FrozenInputs:
    people: PlayerInputs
    alignment: CameraAlignmentOutput
    calibration: CourtCalibrationOutput
    ball: BallReconstructionResult
    pins: dict[str, Any]


def assemble_players(
    source: ClipSource,
    tracks: PoseTracks,
    candidates: FramePoseCandidates,
    detections: tuple[PersonDetectionOutput, ...],
    poses: tuple[PersonPoseFeaturesOutput, ...],
    *,
    players_per_side: int,
    threshold: float,
    crops: FeatureConfig,
) -> PlayerInputs:
    if tracks.selection.players_per_side != players_per_side or any(
        tracks.selection.missing_per_side
    ):
        raise ValueError(
            f"Required {players_per_side} selected IDs per side; missing {tracks.selection.missing_per_side}"
        )
    if (
        tracks.camera_ids != source.camera_ids
        or candidates.camera_ids != source.camera_ids
    ):
        raise ValueError("Tracking, candidates and source camera order disagree")
    views, frames = len(source.camera_ids), source.num_frames
    if (
        len(tracks.frame_offsets) != frames + 1
        or len(candidates.frame_offsets) != frames + 1
    ):
        raise ValueError("Tracking, candidates and source frame axes disagree")
    ids = np.array(tracks.selection.selected_track_ids, dtype=np.int64)
    frame_indices: NDArray[np.int64] = np.repeat(
        np.arange(frames), np.diff(tracks.frame_offsets)
    )
    selected = np.isin(tracks.track_ids, ids)
    observed = selected & tracks.observed
    p = np.searchsorted(ids, tracks.track_ids[observed])
    t = frame_indices[observed]
    rows = tracks.candidate_indices[observed]
    if (
        (rows < candidates.frame_offsets[t]).any()
        or (rows >= candidates.frame_offsets[t + 1]).any()
        or not np.array_equal(
            tracks.source_rows[observed], candidates.source_rows[rows]
        )
        or not np.array_equal(
            tracks.measured_joints[observed], candidates.joint_valid[rows]
        )
        or not np.array_equal(
            tracks.positions_m[observed], candidates.positions_m[rows]
        )
    ):
        raise ValueError(
            "Selected track observations disagree with the pinned 3D candidates"
        )

    shape = (len(ids), frames, 17)
    xyz = np.zeros((*shape, 3), dtype=np.float32)
    measured = np.zeros(shape, dtype=bool)
    reasons = np.full(shape, PointRejection.INSUFFICIENT_VIEWS, dtype=np.uint8)
    inliers = np.zeros((len(ids), views, frames, 17), dtype=bool)
    residuals = np.zeros(inliers.shape, dtype=np.float32)
    xyz[p, t], measured[p, t] = (
        tracks.positions_m[observed],
        tracks.measured_joints[observed],
    )
    reasons[p, t] = candidates.joint_rejection_codes[rows]
    # Move the camera axis before the player-frame advanced indexing.
    inliers.transpose(0, 2, 1, 3)[p, t] = candidates.inlier_mask[rows]
    residuals.transpose(0, 2, 1, 3)[p, t] = candidates.reprojection_px[rows]

    source_rows = np.full((len(ids), frames, views), -1, dtype=np.int64)
    source_rows[p, t] = tracks.source_rows[observed]
    interpolated = np.zeros(shape, dtype=bool)
    frame_ambiguous = np.zeros(shape[:2], dtype=bool)
    id_ambiguous = np.zeros(shape[:2], dtype=bool)
    all_p, all_t = (
        np.searchsorted(ids, tracks.track_ids[selected]),
        frame_indices[selected],
    )
    interpolated[all_p, all_t] = tracks.predicted_joints[selected]
    frame_ambiguous[all_p, all_t] = tracks.frame_ambiguous[selected]
    id_ambiguous[all_p, all_t] = tracks.id_ambiguous[selected]

    raw_shape = (views, frames, len(ids))
    uv = np.zeros((*raw_shape, 17, 2), dtype=np.float32)
    confidence = np.zeros((*raw_shape, 17), dtype=np.float32)
    boxes = np.zeros((*raw_shape, 3), dtype=np.float32)
    raw_rows = source_rows.transpose(2, 1, 0)
    seen = raw_rows >= 0
    for v, (video, det, pose) in enumerate(
        zip(source.videos, detections, poses, strict=True)
    ):
        row_ids = require_alignment(video, det, pose)
        if not np.array_equal(det.frame_offsets, candidates.detection_offsets[v]):
            raise ValueError(
                f"Candidate/master detector timeline differs: {video.camera_id}"
            )
        frame, player = np.nonzero(seen[v])
        index = raw_rows[v, frame, player]
        if (
            len(np.unique(index)) != len(index)
            or (index < det.frame_offsets[frame]).any()
            or (index >= det.frame_offsets[frame + 1]).any()
            or not np.array_equal(row_ids[index], index)
        ):
            raise ValueError(
                f"Detector row is reused or outside its source frame: {video.camera_id}"
            )
        assigned = det.boxes_xyxy[index]
        if (assigned[:, 2:] <= assigned[:, :2]).any():
            raise ValueError(
                f"Invalid detector crop for selected row: {video.camera_id}"
            )
        uv[v, frame, player] = pose.keypoints[index, :, :2]
        confidence[v, frame, player] = np.clip(pose.keypoints[index, :, 2], 0, 1)
        boxes[v, frame, player] = pose_boxes(assigned, crops).numpy()
    raw = ObjectObservations(
        source.camera_ids,
        source.size,
        source.fps,
        uv,
        confidence,
        seen,
        np.broadcast_to(ids, (views, len(ids))).copy(),
        boxes,
    )
    grouped = group_observations(raw, np.where(seen, ids, -1), threshold=threshold)
    if not np.array_equal(grouped.identities, ids):
        raise ValueError("A selected 3D ID has no original detector support")
    return PlayerInputs(
        raw,
        grouped,
        PlayerSkeleton(xyz, measured, reasons, inliers, residuals),
        source_rows.transpose(0, 2, 1),
        interpolated,
        frame_ambiguous,
        id_ambiguous,
        tuple(tracks.selection.decisions[i].side for i in ids),
    )


def read_payload(
    store: StoreReader, node: str, schema: str, version: int, output: type[_T]
) -> _T:
    ref = store.references.get(node)
    if ref is None or (ref["schema"], ref["version"]) != (schema, version):
        raise ValueError(f"Expected {node} with schema {schema} v{version}")
    descriptor = store.descriptor(node)
    if (
        descriptor["status"] != "complete"
        or descriptor["execution_key"] != ref["execution_key"]
    ):
        raise ValueError(f"Incomplete saved component: {node}")
    value: _T = ArtifactCodec(output).load(
        descriptor["payload"], (store.root / ref["path"]).parent, descriptor["arrays"]
    )
    return value


def legacy_source(legacy: StoreReader, clip: DatasetClipInputs) -> ClipSource:
    source: ClipSource = restore_type(legacy.source, ClipSource)
    record = clip.record
    if (
        document_digest(source) != legacy.source_sha256
        or (source.clip_id, source.num_frames, *source.size, source.fps)
        != (record.clip_id, record.num_frames, record.width, record.height, record.fps)
        or source.camera_ids != clip.camera_ids
        or {v.camera_id: v.sha256 for v in source.videos} != clip.media_sha256
        or tuple(p.resolve() for p in source.paths)
        != tuple(p.resolve() for p in clip.video_paths)
    ):
        raise ValueError("Adopted scene and dataset source differ")
    return source


def read_inputs(
    clip: DatasetClipInputs,
    legacy: StoreReader,
    tracks: StoreReader,
    candidates: Path,
    *,
    players_per_side: int,
    threshold: float,
    crops: FeatureConfig,
) -> tuple[ClipSource, FrozenInputs]:
    source = legacy_source(legacy, clip)
    track_reference = tracks.references[TRACK_NODE]
    track_doc = tracks.descriptor(TRACK_NODE)
    provenance = track_doc["provenance"]
    if (
        tracks.source_sha256 != legacy.source_sha256
        or provenance.get("fragment_integration") is None
    ):
        raise ValueError(
            "Downstream recipe requires the integrated 3D tracking store for this clip"
        )
    saved_tracks = read_tracks(
        tracks.root, expected_artifact_id=track_reference["artifact_id"]
    )
    candidate_ref = provenance["frame_pose_candidates"]
    candidate_source, reference, poses3d = read_candidate_store(
        candidates, expected_artifact_id=candidate_ref["artifact_id"]
    )
    if candidate_source != source or json_value(reference) != candidate_ref:
        raise ValueError("Tracking does not pin the supplied candidate store")
    candidate_index = StoreReader(candidates / "scene.json")
    if candidate_index.references.get(POSE_NODE) != candidate_ref:
        raise ValueError("Candidate pointer changed during loading")
    candidate_provenance = candidate_index.descriptor(POSE_NODE)["provenance"]
    observed_refs = candidate_provenance["observations"]
    adopted = legacy.references["scene_assembly"]
    if legacy.document["exports"]["scene"]["inputs"] != {"scene_assembly": adopted}:
        raise ValueError("Adopted scene and export pointers disagree")
    assembly = legacy.descriptor("scene_assembly")
    deps = assembly["dependencies"]
    name_for_ref = {v["artifact_id"]: k for k, v in legacy.references.items()}
    required = {
        port: name_for_ref[deps[port]["artifact_id"]]
        for port in ("calibration", "alignment", "ball")
    }
    if any(legacy.references[node] != deps[port] for port, node in required.items()):
        raise ValueError("Adopted scene references superseded ball or camera inputs")
    calibration = read_payload(
        legacy,
        required["calibration"],
        "local_court_calibration",
        1,
        CourtCalibrationOutput,
    )
    alignment = read_payload(
        legacy, required["alignment"], "aligned_cameras", 1, CameraAlignmentOutput
    )
    ball = read_payload(
        legacy,
        required["ball"],
        "external_ball_trajectory",
        1,
        BallReconstructionResult,
    )
    ball_desc = legacy.descriptor(required["ball"])
    if (
        ball_desc["dependencies"]["alignment"] != deps["alignment"]
        or not ball.trajectory.valid.any()
    ):
        raise ValueError(
            "Adopted external ball has missing geometry or a different alignment"
        )
    side = candidate_provenance["court_side"]
    if (
        alignment.geometry is None
        or calibration.calibration.excluded
        or tuple(v.source_index for v in calibration.calibration.views)
        != tuple(range(len(source.videos)))
        or side["evidence"] != "ball"
        or side["reference_camera"] != calibration.reference_camera
        or observed_refs["court_calibration"] != deps["calibration"]
        or alignment.geometry.camera_ids != source.camera_ids
    ):
        raise ValueError(
            "Tracking requires all adopted cameras in the same ball-based court gauge"
        )
    for local, turn, adopted_camera in zip(
        calibration.calibration.views,
        side["view_half_turns"],
        alignment.geometry.cameras,
        strict=True,
    ):
        if not np.allclose(
            local.camera.half_turned(turn).matrix,
            adopted_camera.matrix,
            atol=1e-7,
            rtol=1e-7,
        ):
            raise ValueError(
                f"Tracking and adopted camera matrices differ: {adopted_camera.camera_id}"
            )

    detections, poses = [], []
    for camera in source.camera_ids:
        det, pose = f"person_detection/{camera}", f"person_pose_features/{camera}"
        if any(legacy.references[node] != observed_refs[node] for node in (det, pose)):
            raise ValueError(f"Upstream observations changed: {camera}")
        if legacy.descriptor(pose)["dependencies"] != {
            "detections": observed_refs[det]
        }:
            raise ValueError(f"Pose observations use different detector rows: {camera}")
        detections.append(
            read_payload(legacy, det, "person_detections", 2, PersonDetectionOutput)
        )
        poses.append(
            read_payload(
                legacy, pose, "person_pose_features", 2, PersonPoseFeaturesOutput
            )
        )
    people = assemble_players(
        source,
        saved_tracks,
        poses3d,
        tuple(detections),
        tuple(poses),
        players_per_side=players_per_side,
        threshold=threshold,
        crops=crops,
    )
    index_sha = dual_sha256(legacy.index_path)
    if json.loads(legacy.index_path.read_text()) != legacy.document:
        raise ValueError("Adopted scene changed during preparation")
    # Retain the original references, including descriptor SHA-256s, in the
    # new store and in the final SceneResult sidecar.
    return source, FrozenInputs(
        people,
        alignment,
        calibration,
        ball,
        {
            "recipe": RECIPE,
            "source_sha256": legacy.source_sha256,
            "legacy_index_sha256": index_sha,
            "legacy_scene": adopted,
            "alignment": deps["alignment"],
            "ball": deps["ball"],
            "calibration": deps["calibration"],
            "tracking": track_reference,
            "candidates": candidate_ref,
            "observations": observed_refs,
            "external_ball": assembly["payload"]["metadata"].get("external_ball"),
            "players_per_side": players_per_side,
        },
    )
