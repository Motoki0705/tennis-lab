"""Read the fixed Meiji inputs and verify the saved detection-row provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.tasks.person_tracking.sequence import TrackEvidence
from src.tasks.player_association.evaluation.dataset_labels import label_path
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    load_dataset_manifest,
)
from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.storage.codec import restore_type, unpack_value
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PinholeCamera

DEV = (
    "video_000/clip_000",
    "video_000/clip_007",
    "video_001/clip_001",
    "video_002/clip_013",
)
RESERVED = ("video_000/clip_002", "video_001/clip_003", "video_002/clip_003")
REPAIRED_CLIP = "video_001/clip_003"

CLIP_ENCODER = "clipreid_vitb16_market1501"
DEV_FEATURES = "person_tracking/evaluate/dev_features/i964-features-r7-20260930"
DEV_TRACKS = "person_tracking/evaluate/tracker_matrix/i964-default-merge-r11-20260930"
DEV_VARIANT = f"strongsort_pp_pose_merge_off__{CLIP_ENCODER}"
DEV_COURTS = "player_association/evaluate/meiji_clips/i933-observe-v1-20260927"
DEV_SIDES = (
    "court_side/evaluate/meiji_clips/i932-detector-v1-20260927/decisions_v2.json"
)
RESERVED_TRACKS = "person_tracking/evaluate/i964-unseen-r16-20261001"
RECOVERY_RUN = (
    Path(__file__).resolve().parents[1] / "run-multiview-pose-link-recall-20261124"
)


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_bytes()))


class Receipt(dict[str, str]):
    """Hash files by their role so the receipt can be checked after relocation."""

    def record(self, key: str, path: Path, expected: str | None = None) -> str:
        digest: str = dual_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f"{path}: expected SHA-256 {expected}, got {digest}")
        self[key] = digest
        return digest

    def read(self, key: str, path: Path, expected: str | None = None) -> dict[str, Any]:
        self.record(key, path, expected)
        return read_json(path)


def read_component(
    store: StoreReader, node: str, schema: str, version: int, *fields: str
) -> dict[str, Any]:
    reference = store.references[node]
    if (reference["schema"], reference["version"]) != (schema, version):
        raise ValueError(f"{store.index_path}: wrong schema/version for {node}")
    descriptor = store.descriptor(node)
    directory = (store.root / reference["path"]).parent
    payload = descriptor["payload"]
    if not fields:
        return cast(
            dict[str, Any], unpack_value(payload, directory, descriptor["arrays"])
        )
    result = {}
    for field in fields:
        value = payload
        for name in field.split("."):
            value = value[name]
        result[field] = unpack_value(value, directory, descriptor["arrays"])
    return result


def source_from_index(store: StoreReader, clip: str) -> ClipSource:
    if (
        store.source["clip_id"] != clip
        or document_digest(store.source) != store.source_sha256
    ):
        raise ValueError(f"{clip}: unexpected saved source identity")
    return cast(ClipSource, restore_type(store.source, ClipSource))


def _turned_cameras(
    source: ClipSource, local: tuple[PinholeCamera, ...], turns: list[bool]
) -> tuple[PinholeCamera, ...]:
    if tuple(camera.camera_id for camera in local) != source.camera_ids or any(
        type(turn) is not bool for turn in turns
    ):
        raise ValueError(f"{source.clip_id}: calibration and side camera axes differ")
    return tuple(
        camera.half_turned(turn) for camera, turn in zip(local, turns, strict=True)
    )


def _archive_camera(raw: dict[str, Any]) -> PinholeCamera:
    return PinholeCamera(
        raw["camera_id"],
        np.array(raw["intrinsic"], np.float64),
        np.array(raw["rotation"], np.float64),
        np.array(raw["translation"], np.float64),
    )


def _dev_track(
    clip: str,
    camera: str,
    num_frames: int,
    archive: Path,
    pins: Receipt,
    track_identity: str,
    feature_record: dict[str, Any],
    clip_model_sha: str,
    source_sha: str,
) -> PersonTrackingOutput:
    track = archive / DEV_TRACKS / "tracks" / DEV_VARIANT / clip / f"{camera}.json"
    record = pins.read(f"{camera}/track_receipt", track)
    if (
        record["status"] != "ok"
        or record["identity_sha256"] != track_identity
        or record["method"] != DEV_VARIANT
        or not record["source_detection_rows"]
    ):
        raise ValueError(f"{track}: expected the frozen run 11 merge-off raw tracks")
    path = track.with_suffix(".npz")
    pins.record(f"{camera}/raw_tracks", path, record["arrays"]["sha256"])
    with np.load(path, allow_pickle=False) as tracks:
        ids, boxes, observed, origins = (
            tracks[k] for k in ("track_ids", "boxes", "observed", "origins")
        )
    if (
        observed.shape != (len(ids), num_frames)
        or origins.dtype != np.int64
        or not np.array_equal(origins >= 0, observed)
    ):
        raise ValueError(f"{track}: raw tracks differ from the source grid")

    feature = archive / DEV_FEATURES / CLIP_ENCODER / clip / f"{camera}.features.npz"
    pins.record(f"{camera}/dino_pose_clip_features", feature, feature_record["sha256"])
    with np.load(feature, allow_pickle=False) as features:
        metadata = json.loads(features["metadata"].item())
        if (
            metadata["schema"] != "person_detection_features_v2"
            or feature_record["frames"] != num_frames
        ):
            raise ValueError(f"{feature}: unexpected dev feature schema or grid")
        rows, offsets = features["rows"], features["offsets"]
        if (
            offsets.shape != (num_frames + 1,)
            or offsets[0] != 0
            or offsets[-1] != len(rows)
            or (np.diff(offsets) < 0).any()
            or len(np.unique(rows)) != len(rows)
            or metadata["provenance"]["source"]["sha256"] != source_sha
            or metadata["provenance"]["encoder"] != CLIP_ENCODER
            or metadata["provenance"]["models"]["appearance"]["sha256"]
            != clip_model_sha
            or metadata["provenance"]["plan_sha256"] != pins["r7_feature_plan"]
        ):
            raise ValueError(f"{feature}: invalid video, feature row IDs or frame grid")
        raw_rows, frames = np.nonzero(observed)
        order = np.argsort(rows)
        indices = np.searchsorted(rows, origins[observed], sorter=order)
        if (indices >= len(rows)).any():
            raise ValueError(f"{track}: unknown feature row")
        indices = order[indices]
        if (
            not np.array_equal(rows[indices], origins[observed])
            or ((indices < offsets[frames]) | (indices >= offsets[frames + 1])).any()
            or not np.array_equal(features["boxes"][indices], boxes[observed])
        ):
            raise ValueError(
                f"{track}: track observations differ from their source detection rows"
            )
        poses = np.zeros((*observed.shape, 17, 3), np.float32)
        all_embeddings = features["embeddings"]
        embeddings = np.zeros((*observed.shape, all_embeddings.shape[-1]), np.float32)
        valid = np.zeros(observed.shape, bool)
        poses[raw_rows, frames] = features["poses"][indices]
        embeddings[raw_rows, frames] = all_embeddings[indices]
        valid[raw_rows, frames] = features["appearance_valid"][indices]

    evidence = TrackEvidence(
        origins, poses, embeddings, valid, CLIP_ENCODER, clip_model_sha
    )
    return PersonTrackingOutput(
        camera,
        ids,
        boxes,
        observed,
        tuple(tuple(g) for g in record["source_track_ids"]),
        (),
        evidence,
    )


@dataclass(frozen=True)
class LabelledInputs:
    source: ClipSource
    cameras: tuple[PinholeCamera, ...]
    raw: dict[str, PersonTrackingOutput]
    labels: ClipLabels
    pins: Receipt


def _dev_inputs(
    archive: Path, clip: str, pins: Receipt
) -> tuple[ClipSource, tuple[PinholeCamera, ...], dict[str, PersonTrackingOutput]]:
    store = StoreReader(archive / DEV_COURTS / "stores" / clip / "scene.json")
    pins.record("calibration_scene_index", store.index_path)
    source = source_from_index(store, clip)
    calibration = cast(
        CourtCalibrationOutput,
        restore_type(
            read_component(store, "court_calibration", "local_court_calibration", 1),
            CourtCalibrationOutput,
        ),
    )
    sides = pins.read("reviewed_ball_sides", archive / DEV_SIDES)
    (side,) = (entry for entry in sides["clips"] if entry["clip_id"] == clip)
    if (
        not side["annotation"]["decided"]
        or tuple(side["camera_ids"]) != source.camera_ids
        or side["reference_camera"] != calibration.reference_camera
    ):
        raise ValueError(f"{clip}: missing checked dev camera orientations")
    cameras = _turned_cameras(
        source,
        tuple(view.camera for view in calibration.calibration.views),
        side["annotation"]["view_half_turns"],
    )

    identity_path = archive / DEV_TRACKS / "identity.json"
    identity = pins.read("r11_identity", identity_path)
    features_path = archive / DEV_FEATURES
    plan = pins.read(
        "r7_feature_plan", features_path / "plan.json", identity["plan"]["sha256"]
    )
    features = pins.read(
        "r7_feature_receipt",
        features_path / "features.json",
        identity["features"]["sha256"],
    )
    if (
        plan["source_name"] != "coco_fullframe_0.30"
        or features["status"] != "ok"
        or features["plan_sha256"] != pins["r7_feature_plan"]
    ):
        raise ValueError("Expected the frozen dev full-frame DINO and CLIP features")
    raw = {
        video.camera_id: _dev_track(
            clip,
            video.camera_id,
            source.num_frames,
            archive,
            pins,
            pins["r11_identity"],
            features["records"][CLIP_ENCODER][f"{clip}/{video.camera_id}"],
            plan["models"][CLIP_ENCODER]["sha256"],
            video.sha256,
        )
        for video in source.videos
    }
    return source, cameras, raw


def _reserved_inputs(
    archive: Path, clip: str, pins: Receipt, labels: ClipLabels
) -> tuple[ClipSource, tuple[PinholeCamera, ...], dict[str, PersonTrackingOutput]]:
    attestation = pins.read(
        "holdout_input_attestation", RECOVERY_RUN / "holdout-inputs.json"
    )
    entry = attestation["clips"][clip]
    root = archive / RESERVED_TRACKS
    pins.record("r18_plan", root / "plan.json", attestation["r18_plan_sha256"])
    if (
        not labels.provenance["blind_to_association"]
        or labels.provenance["plan"]["sha256"] != pins["r18_plan"]
    ):
        raise ValueError(f"{clip}: reserved labels refer to a different plan")
    clip_root = root / clip
    person = pins.read(
        "r18_person_receipt",
        clip_root / "person-execute.json",
        labels.provenance["raw_receipt"]["sha256"],
    )
    court = pins.read(
        "r18_court_receipt",
        clip_root / "court-execute.json",
        entry["court_receipt_sha256"],
    )
    store = StoreReader(clip_root / "store/scene.json")
    pins.record("person_scene_index", store.index_path)
    source = source_from_index(store, clip)
    if (
        store.source_sha256 != entry["source_sha256"]
        or source.num_frames != entry["frames"]
    ):
        raise ValueError(f"{clip}: reserved source differs from the attestation")
    if clip == REPAIRED_CLIP:
        recovery = pins.read(
            "r18_cam0_recovery", RECOVERY_RUN / "holdout-camera-recovery.json"
        )
        side = recovery["ball_decision"]
        if (
            recovery["clip"] != clip
            or recovery["source_sha256"] != store.source_sha256
            or recovery["prior_calibration_receipt_sha256"] != pins["r18_court_receipt"]
            or recovery["court_fit"]["rmse_px"] > recovery["court_fit"]["limit_px"]
            or side["status"] != "decided"
            or tuple(side["camera_ids"]) != source.camera_ids
            or court["cam0"]["status"] != "stopped"
        ):
            raise ValueError(f"{clip}: invalid fixed cam0 recovery")
        local = tuple(_archive_camera(item) for item in recovery["cameras"])
        turns = side["view_half_turns"]
    else:
        local = tuple(
            _archive_camera(court[camera]["calibration"]["views"][0]["camera"])
            for camera in source.camera_ids
            if court[camera]["status"] == "ok"
        )
        turns = entry["reviewed_ball_turns"]
    cameras = _turned_cameras(source, local, turns)
    raw = {}
    for camera in source.camera_ids:
        node = f"person_tracking/{camera}"
        if person["references"][node] != store.references[node]:
            raise ValueError(
                f"{clip}/{camera}: track reference differs from the labelled receipt"
            )
        data = read_component(store, node, "person_tracks", 5)
        # v5 predates encoder_identity. The exact features are pinned by the raw receipt.
        data["evidence"]["encoder_identity"] = None
        raw[camera] = cast(
            PersonTrackingOutput, restore_type(data, PersonTrackingOutput)
        )
    return source, cameras, raw


def labelled_inputs(dataset: Path, archive: Path, clip: str) -> LabelledInputs:
    pins = Receipt()
    if clip not in (*DEV, *RESERVED):
        raise ValueError(f"Unknown labelled evaluation clip {clip}")
    manifest = load_dataset_manifest(dataset)
    metadata = ClipManifest.load(dataset / manifest.clips[clip].path)
    path = label_path(dataset, clip)
    pins.record("dataset_manifest", dataset / "dataset.json")
    pins.record("clip_manifest", dataset / manifest.clips[clip].path / "clip.json")
    pins.record("labels", path)
    labels = ClipLabels.load(path)
    pins.record(
        "label_review",
        path.with_name("review.yaml"),
        labels.provenance["review"]["sha256"],
    )
    source, cameras, raw = (
        _dev_inputs(archive, clip, pins)
        if clip in DEV
        else _reserved_inputs(archive, clip, pins, labels)
    )
    if (
        manifest.dataset_id != "meiji_3cam"
        or labels.clip_id != clip
        or not labels.num_frames == source.num_frames == metadata.num_frames
        or source.size != (metadata.width, metadata.height)
        or source.camera_ids != metadata.camera_ids
        or set(labels.cameras) != set(raw)
        or abs(source.fps - metadata.fps) > 1e-5
    ):
        raise ValueError(f"{clip}: source, manifest and labels differ")
    return LabelledInputs(source, cameras, raw, labels, pins)
