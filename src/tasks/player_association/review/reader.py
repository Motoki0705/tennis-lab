"""Explicit adapters for the two saved Meiji observation layouts, without writers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_association.evaluation.metrics import (
    CameraPrediction,
    match_to_labels,
)
from src.tasks.player_association.geometry.footpoints import (
    FootpointConfig,
    ground_footpoints,
)
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.pipeline.storage.codec import unpack_value
from src.tennis_scene.pipeline.storage.scene_index import read_component_descriptor
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PinholeCamera


def within(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Review input escapes its declared root: {path}")
    return resolved


@dataclass(frozen=True)
class RawTracks:
    ids: NDArray[np.int64]
    boxes: NDArray[np.float64]
    observed: NDArray[np.bool_]
    source_ids: tuple[tuple[int, ...], ...]
    version: int

    @classmethod
    def read(cls, payload: dict[str, Any], version: int, frames: int) -> RawTracks:
        ids = np.asarray(payload["track_ids"])
        boxes = np.asarray(payload["boxes_xyxy"], np.float64)
        observed = np.asarray(payload["observed"])
        if ids.dtype != np.int64 or observed.dtype != np.bool_:
            raise ValueError("Raw track IDs/mask must be int64/bool")
        count = len(ids)
        if ids.shape != (count,) or len(set(ids.tolist())) != count:
            raise ValueError("Raw track IDs must be unique")
        if boxes.shape != (count, frames, 4) or observed.shape != (count, frames):
            raise ValueError("Raw tracks disagree with the clip timeline")
        if (
            not np.isfinite(boxes[observed]).all()
            or (boxes[observed, 2:] <= boxes[observed, :2]).any()
        ):
            raise ValueError("Observed raw boxes must be finite positive rectangles")
        source_ids = tuple(
            tuple(int(x) for x in group) for group in payload["source_track_ids"]
        )
        if len(source_ids) != count:
            raise ValueError("Source track IDs disagree with raw track rows")
        return cls(ids, boxes, observed, source_ids, version)


@dataclass
class ReviewClip:
    manifest: ClipManifest
    labels: ClipLabels
    tracks: dict[str, RawTracks]
    matched: dict[str, NDArray[np.int64]]
    cameras: dict[str, PinholeCamera]
    points: dict[str, NDArray[np.float64]]
    valid: dict[str, NDArray[np.bool_]]
    provenance: dict[str, Any]
    geometry_reason: str | None


def _component(
    root: Path,
    index: dict[str, Any],
    node: str,
    schema: str,
    versions: tuple[int, ...],
    fields: tuple[str, ...],
) -> tuple[dict[str, Any], int]:
    reference = index["artifacts"][node]
    if reference["schema"] != schema or reference["version"] not in versions:
        raise ValueError(
            f"Unsupported review component {node}: {reference['schema']} v{reference['version']}"
        )
    descriptor = read_component_descriptor(
        root, reference, node=node, source_sha256=index["source_sha256"]
    )
    # Decode only the declared review fields, avoiding unused pose/CLIP/GSI arrays.
    # The canonical codec checks checksums, shapes and dtypes. Also reject symlinks
    # out of the descriptor's directory before handing any array to that codec.
    directory = within(root, root / reference["path"]).parent
    for name in descriptor["arrays"]:
        within(directory, directory / name)
    payload = {key: descriptor["payload"][key] for key in fields}
    value: dict[str, Any] = unpack_value(payload, directory, descriptor["arrays"])
    return value, int(reference["version"])


def _camera(value: dict[str, Any]) -> PinholeCamera:
    return PinholeCamera(
        str(value["camera_id"]),
        np.asarray(value["intrinsic"], np.float64),
        np.asarray(value["rotation"], np.float64),
        np.asarray(value["translation"], np.float64),
    )


def observation_location(
    clip: ClipManifest, artifact_root: Path
) -> tuple[Path | None, str]:
    review_path = within(
        clip.clip_dir, clip.clip_dir / "annotations/player_association/review.yaml"
    )
    if not review_path.is_file():
        return None, "review.yamlなし: raw観測の出自が不明"
    review = yaml.safe_load(review_path.read_text())
    relative = Path(review["observe_run"])
    if relative.is_absolute() or relative.parts[:1] != ("outputs",):
        raise ValueError("observe_run must be an outputs-relative path")
    run = within(artifact_root, artifact_root.joinpath(*relative.parts[1:]))
    if (run / "observe.json").is_file():
        observation = json.loads((run / "observe.json").read_text())
        if observation.get("schema") != "player_association_observe_v1":
            raise ValueError("Unsupported observation manifest")
        return within(
            artifact_root, run / "stores" / clip.clip_id
        ), "historical_observe_v1"
    receipt = run / clip.clip_id / "person-execute.json"
    if receipt.is_file():
        return within(artifact_root, receipt.parent / "store"), "raw_reference_v5"
    return None, "宣言された観測runのmanifest/receiptなし"


def _geometry(
    artifact_root: Path,
    store: Path,
    index: dict[str, Any],
    clip: ClipManifest,
    layout: str,
    sides: Path | None,
) -> tuple[dict[str, PinholeCamera], str | None, dict[str, Any]]:
    provenance: dict[str, Any] = {}
    local: dict[str, PinholeCamera] = {}
    turns: dict[str, bool] = {}
    if layout == "historical_observe_v1":
        if "court_calibration" not in index["artifacts"]:
            return {}, "保存済みcourt calibrationなし", provenance
        payload, _ = _component(
            store,
            index,
            "court_calibration",
            "local_court_calibration",
            (1,),
            ("calibration",),
        )
        local = {
            v["camera"]["camera_id"]: _camera(v["camera"])
            for v in payload["calibration"]["views"]
        }
        if sides is None:
            return {}, "side解決済みの保存report未指定", provenance
        document = json.loads(within(artifact_root, sides).read_text())
        records = [r for r in document["clips"] if r["clip_id"] == clip.clip_id]
        if len(records) != 1 or not records[0].get("annotation", {}).get("decided"):
            return {}, "保存reportに確定sideなし", provenance
        record = records[0]
        turns = dict(
            zip(
                record["camera_ids"],
                record["annotation"]["view_half_turns"],
                strict=True,
            )
        )
        provenance["side_report"] = str(sides)
    elif layout == "raw_reference_v5":
        court_path = within(artifact_root, store.parent / "court-execute.json")
        prediction_path = within(artifact_root, store.parent / "prediction.json")
        if not court_path.is_file() or not prediction_path.is_file():
            return {}, "保存済みcourt/side receiptなし", provenance
        court = json.loads(court_path.read_text())
        prediction = json.loads(prediction_path.read_text())
        if prediction["clip"] != clip.clip_id:
            raise ValueError("Saved side prediction clip mismatch")
        half_turns = prediction.get("view_half_turns")
        if half_turns is None:
            return {}, "保存predictionにside未確定", provenance
        turns = dict(zip(clip.camera_ids, half_turns, strict=True))
        for camera_id in clip.camera_ids:
            record = court[camera_id]
            if record["status"] != "ok":
                continue
            for view in record["calibration"]["views"]:
                local[view["camera"]["camera_id"]] = _camera(view["camera"])
        provenance.update(
            court_receipt=str(court_path), side_report=str(prediction_path)
        )
    if set(turns) != set(clip.camera_ids) or any(
        type(x) is not bool for x in turns.values()
    ):
        raise ValueError("Saved sides must align with the source cameras")
    cameras = {key: camera.half_turned(turns[key]) for key, camera in local.items()}
    provenance["half_turns"] = turns
    return (
        cameras,
        None if len(cameras) == len(clip.camera_ids) else "一部cameraに校正なし",
        provenance,
    )


def load_review_clip(
    clip: ClipManifest, labels: ClipLabels, artifact_root: Path, sides: Path | None
) -> ReviewClip:
    if (
        labels.clip_id != clip.clip_id
        or labels.num_frames != clip.num_frames
        or set(labels.cameras) != set(clip.camera_ids)
    ):
        raise ValueError("Labels disagree with the canonical clip manifest")
    store, layout = observation_location(clip, artifact_root)
    provenance: dict[str, Any] = {
        "layout": layout,
        "labels_provenance": labels.provenance,
    }
    tracks: dict[str, RawTracks] = {}
    cameras: dict[str, PinholeCamera] = {}
    geometry_reason: str | None = "raw観測storeなし"
    if store is not None and (store / "scene.json").is_file():
        index_path = within(store, store / "scene.json")
        index = json.loads(index_path.read_text())
        if (
            index.get("schema") != "tennis_scene_index_v1"
            or index["source"]["clip_id"] != clip.clip_id
        ):
            raise ValueError("Scene index source/schema mismatch")
        videos = index["source"]["videos"]
        if [v["camera_id"] for v in videos] != list(clip.camera_ids):
            raise ValueError("Observation camera order differs from the manifest")
        for video in videos:
            camera_id = video["camera_id"]
            if (video["width"], video["height"], video["num_frames"]) != (
                clip.width,
                clip.height,
                clip.num_frames,
            ) or abs(video["fps"] - clip.fps) > 1e-5:
                raise ValueError(
                    "Observation source size/time grid differs from the manifest"
                )
            if Path(video["path"]).resolve() != clip.media_path(camera_id).resolve():
                raise ValueError("Observation media differs from the manifest")
            if (
                "sha256" in video
                and dual_sha256(clip.media_path(camera_id)) != video["sha256"]
            ):
                raise ValueError("Source RGB checksum differs from the observation")
            node = f"person_tracking/{camera_id}"
            if node not in index["artifacts"]:
                continue
            value, version = _component(
                store,
                index,
                node,
                "person_tracks",
                (3, 5),
                (
                    "camera_id",
                    "track_ids",
                    "boxes_xyxy",
                    "observed",
                    "source_track_ids",
                ),
            )
            if value["camera_id"] != camera_id:
                raise ValueError("Raw tracks camera mismatch")
            tracks[camera_id] = RawTracks.read(value, version, clip.num_frames)
        receipt = labels.provenance.get("raw_receipt")
        if receipt is not None:
            path = store.parent / "person-execute.json"
            if dual_sha256(path) != receipt["sha256"]:
                raise ValueError("Label observation receipt checksum mismatch")
        provenance.update(
            store=str(store),
            index_sha256=dual_sha256(index_path),
            track_versions={key: value.version for key, value in tracks.items()},
        )
        cameras, geometry_reason, geometry_provenance = _geometry(
            artifact_root, store, index, clip, layout, sides
        )
        provenance.update(geometry_provenance)
    else:
        provenance["raw_missing"] = layout
    # The canonical label matcher requires all camera slots. Empty slots preserve
    # missing raw observations explicitly, while still showing saved label boxes.
    predictions = {}
    for camera_id in clip.camera_ids:
        track = tracks.get(camera_id)
        predictions[camera_id] = CameraPrediction(
            np.empty(0, np.int64) if track is None else track.ids,
            np.empty((0, clip.num_frames, 4)) if track is None else track.boxes,
            np.empty((0, clip.num_frames), np.bool_)
            if track is None
            else track.observed,
            np.full(
                (0 if track is None else len(track.ids), clip.num_frames), -1, np.int64
            ),
        )
    matched = match_to_labels(labels, predictions, 0.5)
    points, valid = {}, {}
    for camera_id, track in tracks.items():
        if camera_id in cameras:
            points[camera_id], valid[camera_id] = ground_footpoints(
                track.boxes,
                track.observed,
                cameras[camera_id],
                clip.height,
                FootpointConfig(),
            )
    return ReviewClip(
        clip,
        labels,
        tracks,
        matched,
        cameras,
        points,
        valid,
        provenance,
        geometry_reason,
    )


def identity_spans(
    person: NDArray[np.int64], observed: NDArray[np.bool_]
) -> list[dict[str, int]]:
    """Half-open spans; an unobserved gap never becomes labelled presence."""
    spans: list[dict[str, int]] = []
    for frame in np.flatnonzero(observed):
        value = int(person[frame])
        if (
            spans
            and spans[-1]["end"] == int(frame)
            and spans[-1]["person_index"] == value
        ):
            spans[-1]["end"] += 1
        else:
            spans.append(
                {"start": int(frame), "end": int(frame) + 1, "person_index": value}
            )
    return spans
