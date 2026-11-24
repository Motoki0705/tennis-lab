"""Decode the adopted production artifacts of the pinned Meiji benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.evaluation.player_pose.bundle import pinned_file
from src.tennis_scene.evaluation.player_pose.reference import scene_cameras
from src.tennis_scene.pipeline.storage.scene_index import indexed_scene_path
from src.tennis_scene.schema import SceneResult
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.geometry.keypoints import denormalize_keypoints
from src.utils.geometry.triangulation import PinholeCamera

SOURCE_BENCHMARK = "meiji_4cam/multiview-v1"
SOURCE_SHA256 = "b9a33f87ac980c55ebc285c9a5a6f5edd657687be920aea2e14857c1ee845a45"
CAMERA_IDS = ("cam0", "cam1", "cam2", "cam3")
CLIPS = {
    "clip_003": "Reference clip with four valid placed meshes throughout",
    "clip_057": "Largest fraction of frames with observed player overlap",
    "clip_004": "Frequent player overlap and missing body placement",
    "clip_013": "Frequent missing body placement",
    "clip_040": "Small people and quick movement",
    "clip_028": "Long rally for temporal stability",
}


def source_records(root: Path) -> list[dict[str, Any]]:
    path = pinned_file(root, {"path": "manifest.json", "sha256": SOURCE_SHA256})
    doc = json.loads(path.read_bytes())
    if doc["benchmark_id"] != SOURCE_BENCHMARK:
        raise ValueError("Unexpected source benchmark ID")
    by_id = {row["clip_id"]: row for row in doc["clips"]}
    return [by_id[f"video_000/{clip}"] for clip in CLIPS]


@dataclass(frozen=True)
class ClipInputs:
    record: dict[str, Any]
    scene: SceneResult
    cameras: tuple[PinholeCamera, ...]
    media: tuple[Path, ...]
    start_seconds: float
    request_data: dict[str, np.ndarray]
    source_components: dict[str, Any]

    @property
    def clip_id(self) -> str:
        return self.record["clip_id"]

    @property
    def size(self) -> tuple[int, int]:
        return self.scene.width, self.scene.height

    @property
    def player_ids(self) -> np.ndarray:
        assert self.scene.player_track_ids is not None
        return self.scene.player_track_ids

    @property
    def inferred(self) -> np.ndarray:
        return self.request_data["segment_index"] >= 0

    @property
    def evaluation_frames(self) -> np.ndarray:
        return np.flatnonzero(self.inferred.any(0))

    @property
    def observed_keypoints(self) -> np.ndarray:
        assert (
            self.scene.human_kp_2d is not None and self.scene.human_kp_vis is not None
        )
        xy = denormalize_keypoints(self.scene.human_kp_2d, *self.size)
        return np.concatenate((xy, self.scene.human_kp_vis[..., None]), axis=-1)


def _by_player(
    rows: list[dict[str, Any]], players: np.ndarray
) -> dict[int, dict[str, Any]]:
    by_id = {row["person_id"]: row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != set(players.tolist()):
        raise ValueError("Recorded player IDs do not match the scene")
    return by_id


def _fixed_requests(
    scene: SceneResult,
    cameras: tuple[PinholeCamera, ...],
    selections: list[dict[str, Any]],
    bodies: list[dict[str, Any]],
    people: dict[str, Any],
) -> dict[str, np.ndarray]:
    players = scene.player_track_ids
    assert players is not None
    by_selection = _by_player(selections, players)
    by_body = _by_player(bodies, players)
    group, raw, skeleton = (people[key] for key in ("grouped", "raw", "skeleton"))
    if (
        not np.array_equal(group["identities"], players)
        or not np.array_equal(skeleton["positions"], scene.player_kp_3d)
        or not np.array_equal(skeleton["valid"], scene.player_kp_3d_vis)
        or not np.array_equal(skeleton["reasons"], scene.player_kp_3d_rejection_code)
    ):
        raise ValueError(
            "Tracked skeleton or player order differs from the adopted scene"
        )

    shape = (len(players), scene.num_frames)
    data = {
        "selected_view_index": np.full(len(players), -1, np.int16),
        "segment_index": np.full(shape, -1, np.int16),
        "boxes_xys": np.full(shape + (3,), np.nan, np.float32),
        "keypoints_xyc": np.full(shape + (17, 3), np.nan, np.float32),
        "selected_view_observed": np.zeros(shape, bool),
        "selected_intrinsic": np.full((len(players), 3, 3), np.nan),
        "triangulated_joints_m": skeleton["positions"],
        "triangulated_valid": skeleton["valid"],
        "triangulated_rejection_code": skeleton["reasons"],
        "triangulated_inliers": skeleton["inliers"],
        "triangulated_reprojection_px": skeleton["reprojection_px"],
        "player_frame_ambiguous": people["frame_ambiguous"],
        "player_id_ambiguous": people["id_ambiguous"],
    }
    camera_ids = tuple(camera.camera_id for camera in cameras)
    for p, player in enumerate(players.tolist()):
        selection, body = by_selection[player], by_body[player]
        if body["camera_id"] != selection["camera_id"]:
            raise ValueError(f"Selected and GVHMR camera disagree for player {player}")
        view = camera_ids.index(body["camera_id"])
        data["selected_view_index"][p] = view
        data["selected_intrinsic"][p] = cameras[view].intrinsic
        for segment_index, (request, output, count) in enumerate(
            zip(
                selection["requests"],
                body["segments"],
                selection["observed_samples"],
                strict=True,
            )
        ):
            frames = output["source_frames"]
            boxes, keypoints = request["boxes_xys"], request["keypoints"]
            if (
                frames.dtype != np.int64
                or frames.ndim != 1
                or len(frames) < 2
                or (np.diff(frames) <= 0).any()
                or frames[0] < 0
                or frames[-1] >= shape[1]
                or not np.array_equal(frames, request["source_frames"])
                or (data["segment_index"][p, frames] >= 0).any()
                or tuple(request["size"]) != (scene.width, scene.height)
                or not np.array_equal(request["intrinsic"], cameras[view].intrinsic)
                or boxes.shape != (len(frames), 3)
                or (boxes[:, 2] <= 0).any()
                or keypoints.shape != (len(frames), 17, 3)
                or not np.isfinite(keypoints).all()
                or not np.isfinite(boxes).all()
            ):
                raise ValueError(
                    f"Invalid saved segment axes or crop for player {player}"
                )
            for name, width in (
                ("body_pose", 63),
                ("betas", 10),
                ("global_orient", 3),
                ("transl", 3),
            ):
                values = output["parameters"][name]
                if (
                    values.shape != (len(frames), width)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(f"Invalid GVHMR {name} rows for player {player}")

            indices = group["raw_indices"][p, view, frames]
            present = indices >= 0
            if (
                int(present.sum()) != count
                or count != output["observed_samples"]
                or not raw["observed"][view, frames[present], indices[present]].all()
                or not np.array_equal(
                    boxes[present],
                    raw["boxes_xys"][view, frames[present], indices[present]],
                )
            ):
                raise ValueError(
                    f"Selected crop observation support differs for player {player}"
                )
            data["segment_index"][p, frames] = segment_index
            data["boxes_xys"][p, frames] = boxes
            data["keypoints_xyc"][p, frames] = keypoints
            data["selected_view_observed"][p, frames] = present
    return data


def load_inputs(root: Path, record: dict[str, Any]) -> ClipInputs:
    index = pinned_file(root, record["scene"])
    clip = json.loads(pinned_file(root, record["clip_json"]).read_bytes())
    store = StoreReader(index)
    component_versions = {
        "body_view_selection": ("body_view_selection", 1),
        "gvhmr": ("body_parameters", 2),
        "multiview_scene_inputs": ("multiview_scene_inputs", 1),
    }
    payloads, component_refs = {}, {}
    for name, expected in component_versions.items():
        payload, descriptor = store.payload(name)
        reference = store.references[name]
        if (descriptor["output_schema"], descriptor["output_version"]) != expected:
            raise ValueError(f"Unexpected saved schema for {name}")
        path = (store.root / reference["path"]).relative_to(root)
        payloads[name] = payload
        component_refs[name] = {
            **reference,
            "path": path.as_posix(),
            "arrays": {
                (path.parent / filename).as_posix(): entry["sha256"]
                for filename, entry in descriptor["arrays"].items()
            },
        }
    archive = indexed_scene_path(
        index,
        required_components={name: store.references[name] for name in component_refs},
    )
    for key, path in (
        ("scene/scene", archive),
        ("scene/metadata", archive.with_suffix(".metadata.json")),
    ):
        pin = record["exports"][key]
        exported = store.document["exports"]["scene"]["files"][key.rsplit("/", 1)[1]]
        if root / pin["path"] != path or exported["sha256"] != pin["sha256"]:
            raise ValueError(f"Adopted scene file differs from benchmark pin: {path}")
    scene = load_scene_result(archive)
    cameras = scene_cameras(scene)
    camera_records = record["cameras"]
    source_media = store.source["videos"]
    camera_ids = tuple(camera.camera_id for camera in cameras)
    if (
        clip["clip_id"] != record["clip_id"]
        or store.source["clip_id"] != record["clip_id"]
        or scene.metadata["dataset_clip_id"] != record["clip_id"]
        or record["scene"]["source_sha256"] != store.source_sha256
        or scene.metadata["scene_schema_version"] != 2
        or scene.metadata["normalization"] != "image_width_height"
        or len(scene.player_track_ids) != 4
        or set(scene.player_track_ids.tolist())
        != set(
            record["players"]["selected_ids_negative_y"]
            + record["players"]["selected_ids_positive_y"]
        )
        or not camera_ids == store.camera_ids == tuple(clip["camera_ids"]) == CAMERA_IDS
        or (scene.width, scene.height) != (clip["width"], clip["height"])
        or not scene.num_frames == clip["num_frames"] == record["frames"]
        or not scene.fps == clip["fps"] == record["fps"]
    ):
        raise ValueError(
            f"Scene, source benchmark and clip axes disagree for {record['clip_id']}"
        )
    media = []
    for camera, pin, video in zip(cameras, camera_records, source_media, strict=True):
        if (
            not camera.camera_id == pin["camera_id"] == video["camera_id"]
            or pin["sha256"] != video["sha256"]
            or not pin["num_frames"] == video["num_frames"] == scene.num_frames
            or not pin["fps"] == video["fps"] == scene.fps
            or (video["width"], video["height"]) != (scene.width, scene.height)
        ):
            raise ValueError(
                f"Source media pin disagrees with scene for {camera.camera_id}"
            )
        media.append(pinned_file(root, pin))

    data = _fixed_requests(
        scene,
        cameras,
        payloads["body_view_selection"]["selections"],
        payloads["gvhmr"]["bodies"],
        payloads["multiview_scene_inputs"]["people"],
    )
    return ClipInputs(
        record,
        scene,
        cameras,
        tuple(media),
        clip["global_start_sec"],
        data,
        component_refs,
    )
