"""Bind the original batch cohort to the currently adopted scenes and pinned poses."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.multiview_pose.schema import NODE, FramePoseCandidates
from src.tennis_scene.multiview_pose.storage import read_candidates
from src.tennis_scene.multiview_tracking.schema import NODE as TRACKING_NODE
from src.tennis_scene.multiview_tracking.schema import PoseTracks
from src.tennis_scene.multiview_tracking.storage import read_tracks
from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.components.event_detection import EVENT_DETECTION
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.storage.codec import restore_type, unpack_value
from src.tennis_scene.pipeline.storage.scene_index import resolve_indexed_scene
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.video.reader import probe_video_info

STOP_LABELS = {
    "court_side_no_consistent_hypothesis": (
        "コートの向きを決める幾何支持が不足",
        "Court side: insufficient support",
    ),
    "player_association_ambiguous_players": (
        "選手選別の境界で次点が接近",
        "Player selection: close runner-up",
    ),
    "player_association_player_not_found": (
        "片側に必要人数の選手が見つからない",
        "Player association: too few players",
    ),
    "player_association_ambiguous_association": (
        "長い区間どうしの人物対応が未決",
        "Player association: close alternatives",
    ),
}


@dataclass(frozen=True)
class ClipPlan:
    clip_id: str
    path: Path
    frames: int
    baseline_reason: str | None
    baseline_error: str | None
    node_seconds: float


@dataclass(frozen=True)
class TrackBoxes:
    ids: NDArray[np.int64]
    xyxy: NDArray[np.float32]
    observed: NDArray[np.bool_]
    label: str


@dataclass(frozen=True)
class AdoptedScene:
    cameras: tuple[PinholeCamera, ...]
    player_ids: tuple[int, ...]
    xyz: NDArray[np.float32]  # (P,T,17,3)
    valid: NDArray[np.bool_]  # (P,T,17)
    ball: NDArray[np.float32]  # (T,3)
    ball_valid: NDArray[np.bool_]
    alignment_ref: dict[str, Any]
    ball_ref: dict[str, Any] | None
    schema_version: int = 2
    unused_view_errors: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class ComparisonInputs:
    source: ClipSource
    legacy: AdoptedScene | None
    tracks: tuple[TrackBoxes, ...]
    candidates: FramePoseCandidates
    cameras: tuple[PinholeCamera, ...]
    boxes: tuple[NDArray[np.float32], ...]
    provenance: dict[str, Any]


def select_clips(
    progress: dict[str, Any], seed: int
) -> tuple[list[ClipPlan], dict[str, Any]]:
    outcomes = progress["clips"]
    targets = {
        item["record"]["clip_id"]: item["record"] for item in progress["targets"]
    }
    stopped = sorted(key for key, row in outcomes.items() if row["status"] == "failed")
    completed = sorted(
        key for key, row in outcomes.items() if row["status"] == "generated"
    )
    if set(targets) != outcomes.keys() or len(stopped) + len(completed) != len(targets):
        raise ValueError(
            "Comparison requires a finished batch with every clip classified"
        )
    sampled = sorted(random.Random(seed).sample(completed, 2))
    plans = []
    for clip_id in [*stopped, *sampled]:
        row, target = outcomes[clip_id], targets[clip_id]
        relative = Path(target["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.name != Path(clip_id).name
        ):
            raise ValueError(f"Invalid clip path in progress: {clip_id}")
        seconds = sum(
            float(node["seconds"])
            for node in progress["nodes"][clip_id].values()
            if node["status"] == "executed"
        )
        plans.append(
            ClipPlan(
                clip_id,
                relative,
                target["num_frames"],
                row.get("reason"),
                row.get("error"),
                seconds,
            )
        )
    return plans, {
        "seed": seed,
        "method": "all failed clips, then random.Random(seed).sample(sorted(generated), 2)",
        "stopped": stopped,
        "success_population": len(completed),
        "sampled_successes": sampled,
    }


def bind_sources(legacy: StoreReader, new: StoreReader, plan: ClipPlan) -> ClipSource:
    for store in (legacy, new):
        if document_digest(store.source) != store.source_sha256:
            raise ValueError(f"Source digest mismatch: {store.index_path}")
    source = restore_type(legacy.source, ClipSource)
    other = restore_type(new.source, ClipSource)
    if (
        source.clip_id != other.clip_id
        or source.clip_id != plan.clip_id
        or source.num_frames != plan.frames
        or len(source.videos) != 4
        or source.num_frames != other.num_frames
        or source.fps != other.fps
        or source.size != other.size
        or [(v.camera_id, v.sha256) for v in source.videos]
        != [(v.camera_id, v.sha256) for v in other.videos]
    ):
        raise ValueError(
            f"Unaligned clip, frames, camera order or source media: {plan.clip_id}"
        )
    return source


def check_media(source: ClipSource) -> None:
    for video in source.videos:
        if dual_sha256(video.path) != video.sha256:
            raise ValueError(f"Source video SHA-256 mismatch: {video.path}")
        info = probe_video_info(video.path)
        if (info.width, info.height, info.frame_count) != (
            *source.size,
            source.num_frames,
        ) or not np.isclose(info.fps, source.fps, rtol=0, atol=1e-3):
            raise ValueError(
                f"Video metadata disagrees with the source index: {video.path}"
            )


def check_candidate_pin(store: StoreReader, expected: dict[str, Any]) -> dict[str, Any]:
    reference = store.references[NODE]
    if (
        reference["artifact_id"] != expected["artifact_id"]
        or reference["sha256"] != expected["descriptor_sha256"]
        or store.source_sha256 != expected["source_sha256"]
    ):
        raise ValueError(
            f"Candidate store disagrees with the pinned run record: {store.index_path}"
        )
    descriptor: dict[str, Any] = store.descriptor(NODE)
    return descriptor


def _payload_fields(
    store: StoreReader, node: str, fields: tuple[str, ...]
) -> dict[str, Any]:
    """Read selected fields only. Component .npy arrays are checksum-verified mmap views."""
    desc = store.descriptor(node)
    directory = (store.root / store.references[node]["path"]).parent
    return {
        name: unpack_value(desc["payload"][name], directory, desc["arrays"])
        for name in fields
    }


def _new_cameras(
    old: StoreReader, source: ClipSource, provenance: dict[str, Any]
) -> tuple[PinholeCamera, ...]:
    node = "court_calibration"
    if old.references[node] != provenance["observations"][node]:
        raise ValueError(f"Candidate calibration was superseded: {source.clip_id}")
    cal = _payload_fields(old, node, ("calibration", "reference_camera"))
    side = provenance["court_side"]
    turns = side["view_half_turns"]
    if (
        side["evidence"] != "ball"
        or side["reference_camera"] != cal["reference_camera"]
        or cal["calibration"]["excluded"]
        or any(type(turn) is not bool for turn in turns)
    ):
        raise ValueError(f"Invalid pinned candidate court alignment: {source.clip_id}")
    cameras = tuple(
        PinholeCamera(**view["camera"]).half_turned(turn)
        for view, turn in zip(cal["calibration"]["views"], turns, strict=True)
    )
    if tuple(camera.camera_id for camera in cameras) != source.camera_ids:
        raise ValueError(f"Candidate calibration order mismatch: {source.clip_id}")
    return cameras


def _new_boxes(
    old: StoreReader, candidates: FramePoseCandidates, observations: dict[str, Any]
) -> tuple[NDArray[np.float32], ...]:
    boxes = []
    for camera, offsets in zip(
        candidates.camera_ids, candidates.detection_offsets, strict=True
    ):
        node = f"person_detection/{camera}"
        if old.references[node] != observations[node]:
            raise ValueError(
                f"Candidate detection rows were superseded: {old.source['clip_id']} {camera}"
            )
        fields = _payload_fields(
            old, node, ("camera_id", "frame_offsets", "boxes_xyxy")
        )
        if fields["camera_id"] != camera or not np.array_equal(
            fields["frame_offsets"], offsets
        ):
            raise ValueError(f"Candidate detection offsets mismatch: {camera}")
        if fields["boxes_xyxy"].shape != (int(offsets[-1]), 4):
            raise ValueError(f"Candidate boxes have the wrong row axis: {camera}")
        boxes.append(fields["boxes_xyxy"])
    return tuple(boxes)


def _tracks(store: StoreReader) -> tuple[TrackBoxes, ...]:
    tracks = []
    for camera in store.camera_ids:
        node = f"player_selection/{camera}"
        label = "Selected 2D tracks"
        if node in store.references:
            desc = store.descriptor(node)
            payload = desc["payload"]["tracks"]
        else:
            node = f"person_tracking/{camera}"
            desc = store.descriptor(node)
            payload = desc["payload"]
            label = "Raw 2D tracks"
        directory = (store.root / store.references[node]["path"]).parent
        ids, boxes, observed = (
            unpack_value(payload[key], directory, desc["arrays"])
            for key in ("track_ids", "boxes_xyxy", "observed")
        )
        if (
            payload["camera_id"] != camera
            or boxes.shape != (*observed.shape, 4)
            or observed.shape != (len(ids), store.frame_count)
        ):
            raise ValueError(f"Legacy 2D tracks disagree with the source: {camera}")
        tracks.append(TrackBoxes(ids, boxes, observed, label))
    return tuple(tracks)


def _component_node(
    store: StoreReader, reference: dict[str, Any], schema: str, version: int
) -> str:
    nodes: list[str] = [
        node for node, current in store.references.items() if current == reference
    ]
    if len(nodes) != 1 or (reference["schema"], reference["version"]) != (
        schema,
        version,
    ):
        raise ValueError(
            f"Expected an adopted {schema} v{version} in {store.index_path}"
        )
    return nodes[0]


def _unused_view_errors(
    cameras: tuple[PinholeCamera, ...],
    xyz: NDArray[np.float32],
    valid: NDArray[np.bool_],
    skeleton: dict[str, Any],
    points: NDArray[np.float32],
    visibility: NDArray[np.float32],
    size: tuple[int, int],
) -> NDArray[np.float64]:
    observed_shape = (len(xyz), len(cameras), *valid.shape[1:])
    inliers = skeleton["inliers"]
    if (
        inliers.shape != observed_shape
        or inliers.dtype != np.bool_
        or not np.array_equal(skeleton["valid"], valid)
        or not np.array_equal(skeleton["positions"], xyz)
        or points.shape != (*observed_shape, 2)
        or points.dtype != np.float32
        or visibility.shape != observed_shape
        or visibility.dtype != np.float32
        or not np.isfinite(points).all()
        or not np.isfinite(visibility).all()
    ):
        raise ValueError(
            "Scene joints or observations disagree with the adopted skeleton axes"
        )

    scale = np.asarray(size, dtype=np.float32)
    errors = []
    for view, camera in enumerate(cameras):
        projected, front = camera.project(xyz)
        unused = valid & front & (visibility[:, view] > 0) & ~inliers[:, view]
        errors.append(
            np.linalg.norm(projected[unused] - points[:, view][unused] * scale, axis=-1)
        )
    result: NDArray[np.float64] = np.concatenate(errors)
    return result


def _read_scene(store: StoreReader, source: ClipSource) -> AdoptedScene | None:
    assembly = store.references.get("scene_assembly")
    export = store.document["exports"].get("scene")
    if assembly is None and export is None:
        return None
    if (
        assembly is None
        or export is None
        or (assembly["schema"], assembly["version"]) != ("scene_result", 2)
    ):
        raise ValueError(
            f"The adopted scene and its export disagree: {store.index_path}"
        )
    events = store.references.get(EVENT_DETECTION)
    if export["inputs"] == {"scene_assembly": assembly}:
        version = 2
    elif (
        events is not None
        and (events["schema"], events["version"]) == ("scene_result", 3)
        and export["inputs"] == {EVENT_DETECTION: events}
        and store.descriptor(EVENT_DETECTION)["dependencies"].get("scene") == assembly
    ):
        version = 3
    else:
        raise ValueError(
            f"The adopted scene and its export disagree: {store.index_path}"
        )

    dependencies = store.descriptor("scene_assembly")["dependencies"]
    tracked = dependencies.get("tracked")
    if tracked is None:
        alignment = dependencies["alignment"]
        alignment_node = _component_node(store, alignment, "aligned_cameras", 1)
        ball_ref = dependencies.get("ball")
    else:
        alignment = ball_ref = tracked
        alignment_node = _component_node(store, tracked, "multiview_scene_inputs", 1)
    archive = resolve_indexed_scene(
        store.document, store.root,
        required_components={"scene_assembly": assembly, alignment_node: alignment},
    )
    skeleton = None
    if tracked is None:
        geometry = _payload_fields(store, alignment_node, ("geometry",))["geometry"]
        skeleton_ref = dependencies.get("skeleton")
        if skeleton_ref is not None:
            node = _component_node(store, skeleton_ref, "player_skeletons", 1)
            skeleton = _payload_fields(store, node, ("skeleton",))["skeleton"]
    else:
        frozen = _payload_fields(store, alignment_node, ("alignment", "people"))
        geometry = frozen["alignment"]["geometry"]
        skeleton = frozen["people"]["skeleton"]
    cameras = tuple(PinholeCamera(**entry) for entry in geometry["cameras"])
    metadata = json.loads(
        archive.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    ids = tuple(metadata["track_ids"])
    if (
        tuple(camera.camera_id for camera in cameras) != source.camera_ids
        or tuple(metadata["camera_ids"]) != source.camera_ids
        or metadata["dataset_clip_id"] != source.clip_id
        or metadata["scene_schema_version"] != version
        or len(set(ids)) != len(ids)
        or any(type(value) is not int for value in ids)
    ):
        raise ValueError(f"Legacy scene identity or camera order mismatch: {archive}")
    frames = source.num_frames
    with np.load(archive, allow_pickle=False) as arrays:
        if (
            int(arrays["width"]),
            int(arrays["height"]),
            int(arrays["num_frames"]),
            float(arrays["fps"]),
        ) != (*source.size, frames, source.fps):
            raise ValueError(f"Legacy scene timeline mismatch: {archive}")
        xyz, valid = arrays["player_kp_3d"], arrays["player_kp_3d_vis"]
        ball, ball_valid = arrays["ball_3d"], arrays["ball_3d_valid"]
        if (
            xyz.dtype != np.float32
            or xyz.shape != (len(ids), frames, 17, 3)
            or not np.isfinite(xyz).all()
            or valid.dtype != np.bool_
            or valid.shape != xyz.shape[:-1]
            or ball.dtype != np.float32
            or ball.shape != (frames, 3)
            or not np.isfinite(ball).all()
            or ball_valid.dtype != np.bool_
            or ball_valid.shape != (frames,)
        ):
            raise ValueError(f"Legacy scene has invalid 3D arrays or masks: {archive}")
        errors = None
        if skeleton is not None and valid.any():
            errors = _unused_view_errors(
                cameras,
                xyz,
                valid,
                skeleton,
                arrays["human_kp_2d"],
                arrays["human_kp_vis"],
                source.size,
            )
    return AdoptedScene(
        cameras, ids, xyz, valid, ball, ball_valid, alignment, ball_ref, version, errors
    )


def load_comparison(
    old: StoreReader, new: StoreReader, plan: ClipPlan, record: dict[str, Any]
) -> ComparisonInputs:
    source = bind_sources(old, new, plan)
    expected = record["clips"][source.clip_id]
    desc = check_candidate_pin(new, expected)
    provenance = desc["provenance"]
    if (
        expected["frames"] != source.num_frames
        or provenance["court_side"] != expected["side"]
        or provenance["court_side"]["digest"] != record["side_digest"]
        or provenance["config"] != record["config"]
    ):
        raise ValueError(
            f"Candidate inputs disagree with the run record: {source.clip_id}"
        )
    candidates = read_candidates(new.root)
    cameras = _new_cameras(old, source, provenance)
    boxes = _new_boxes(old, candidates, provenance["observations"])
    scene = _read_scene(old, source)
    if scene is None and plan.baseline_reason is None:
        raise ValueError(
            f"Originally generated scene is no longer adopted: {source.clip_id}"
        )
    if (
        scene is not None
        and scene.ball_valid.any()
        and not all(
            np.allclose(left.matrix, right.matrix, rtol=1e-7, atol=1e-7)
            for left, right in zip(scene.cameras, cameras, strict=True)
        )
    ):
        raise ValueError(
            f"Cannot share the adopted 3D ball across different court gauges: {source.clip_id}"
        )
    tracks = _tracks(old) if scene is None or not scene.valid.any() else ()
    check_media(source)
    for store in (old, new):
        store.assert_unchanged()
    return ComparisonInputs(
        source,
        scene,
        tracks,
        candidates,
        cameras,
        boxes,
        {
            "legacy_scene_assembly": old.references.get("scene_assembly"),
            "legacy_scene_export": old.document["exports"].get("scene"),
            "legacy_scene_schema_version": scene.schema_version
            if scene is not None
            else None,
            "legacy_camera_alignment": scene.alignment_ref
            if scene is not None
            else None,
            "legacy_ball": scene.ball_ref if scene is not None else None,
            "legacy_2d": {
                camera: {
                    "label": track.label,
                    "reference": old.references.get(f"player_selection/{camera}")
                    or old.references[f"person_tracking/{camera}"],
                }
                for camera, track in zip(old.camera_ids, tracks, strict=False)
            },
            "candidate_artifact": new.references[NODE],
            "candidate_implementation": desc["identity"]["implementation"],
            "candidate_court_side": provenance["court_side"],
            "candidate_observations": provenance["observations"],
            "source_sha256": new.source_sha256,
            "videos": {v.camera_id: v.sha256 for v in source.videos},
        },
    )


def load_tracking(
    store: StoreReader, data: ComparisonInputs, record: dict[str, Any]
) -> tuple[PoseTracks, dict[str, Any]]:
    clip_id = data.source.clip_id
    expected = record["clips"][clip_id]
    reference = store.references[TRACKING_NODE]
    candidate = data.provenance["candidate_artifact"]
    if (
        store.source_sha256 != data.provenance["source_sha256"]
        or expected["source_frames"] != data.source.num_frames
        or reference["artifact_id"] != expected["tracking_artifact_id"]
        or reference["sha256"] != expected["tracking_descriptor_sha256"]
        or candidate["artifact_id"] != expected["candidate_artifact_id"]
        or candidate["sha256"] != expected["candidate_descriptor_sha256"]
    ):
        raise ValueError(
            f"Tracking store disagrees with the pinned run record: {clip_id}"
        )
    descriptor = store.descriptor(TRACKING_NODE)
    if (
        descriptor["provenance"]["frame_pose_candidates"] != candidate
        or descriptor["provenance"]["config"] != record["config"]
    ):
        raise ValueError(f"Tracking inputs or settings disagree for {clip_id}")
    tracks = read_tracks(store.root, expected_artifact_id=reference["artifact_id"])
    store.assert_unchanged()
    return tracks, {
        **data.provenance,
        "tracking_artifact": reference,
        "tracking_implementation": descriptor["identity"]["implementation"],
        "tracking_config": descriptor["provenance"]["config"],
    }
