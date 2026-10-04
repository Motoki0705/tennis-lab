"""Describe only the artifacts adopted by the supplied scene index."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from src.tennis_scene.motion_alignment.temporal import PlacementRejection
from src.tennis_scene.pipeline.storage.scene_index import indexed_scene_path
from src.tennis_scene.review.trajectory import mask_intervals
from src.tennis_scene.schema import SceneResult, validate_scene_result_arrays
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PointRejection
from src.utils.schema.player import COCO17_SKELETON
from src.utils.video import probe_video_info

if TYPE_CHECKING:
    from src.tennis_scene.scripts.visualize_component_store import Review


def scene_snapshot(value: dict[str, Any]) -> dict[str, Any]:
    """Serialize observations/masks, excluding large meshes and model parameters."""
    scene = SceneResult(**value)
    validate_scene_result_arrays(scene)
    if scene.schema_version != 2:
        raise ValueError("Frame inspection requires SceneResult v2 validity masks")
    fields = (
        "player_position", "player_yaw", "player_observed", "player_valid",
        "player_heading_valid", "player_smpl_valid", "player_rejection_code",
        "player_kp_3d", "player_kp_3d_vis", "player_kp_3d_rejection_code",
        "human_kp_2d", "human_kp_vis", "ball_3d", "ball_3d_valid",
        "ball_rejection_code", "ball_uv", "ball_vis", "court_kp", "court_vis",
    )
    arrays = {name: np.asarray(value[name]).tolist() for name in fields}
    player_ids = np.asarray(value["player_track_ids"]).tolist() if value["player_track_ids"] is not None else list(range(len(scene.player_position)))
    rejection_labels = {str(int(item)): item.name for item in PointRejection}
    rejection_labels.update({str(int(item)): item.name for item in PlacementRejection})
    rejection_labels["1"] = "INSUFFICIENT_VIEWS"  # point domain; placement code 1 has its own label below
    gaps: list[dict[str, Any]] = []
    for row, name in [(np.asarray(value["ball_3d_valid"]), "ball 3D"),
                      *((mask, f"player {player_ids[i]} root") for i, mask in enumerate(np.asarray(value["player_valid"])) )]:
        gaps.extend({"name": name, "start": a, "end": b} for a, b in mask_intervals(~row))
    return {"arrays": arrays, "player_ids": player_ids, "status": scene.metadata["status"],
            "camera_ids": scene.metadata["camera_ids"], "reference": scene.metadata["court_reference"],
            "rejection_labels": rejection_labels, "gaps": gaps,
            "skeleton": COCO17_SKELETON}


def build_snapshot(review: Review, components: dict[str, Any], *, online: bool) -> dict[str, Any]:
    """Verify media, then publish a source/lineage overview and frame evidence."""
    sources: list[dict[str, Any]] = []
    for video in review.source["videos"]:
        path = review.videos[video["camera_id"]]
        available = path.is_file()
        if available and dual_sha256(path) != video["sha256"]:
            raise ValueError(f"Source video checksum mismatch: {path}")
        if available:
            info = probe_video_info(path)
            if (info.width, info.height) != (video["width"], video["height"]) or info.frame_count < video["num_frames"] or abs(info.fps - video["fps"]) > 1e-3:
                raise ValueError(f"Source video dimensions/frame count/FPS mismatch: {path}")
        sources.append({**video, "available": available, "checksum_verified": available})

    scene = None
    scene_reason = "scene_assembly 未生成。3D・品質mask・棄却理由は未保存です。"
    if "scene_assembly" in components:
        entry = components["scene_assembly"]
        if entry["status"] == "rendered":
            value, _ = review.payload("scene_assembly")
            if value["num_frames"] != review.frame_count or (value["width"], value["height"]) != review.size or abs(value["fps"] - sources[0]["fps"]) > 1e-3:
                raise ValueError("Scene timeline/image metadata disagrees with its source")
            scene = scene_snapshot(value)
            if set(scene["camera_ids"]) != set(review.camera_ids) or len(scene["camera_ids"]) != len(value["court_kp"]):
                raise ValueError("Scene camera axis disagrees with its source")
            scene_reason = "保存scene componentの推定結果。独立した3D GTではありません。"
        elif entry["status"] != "missing":
            scene_reason = f"scene_assembly: {entry['status']}。旧依存/未対応schemaの3Dを現版として表示しません。"

    document = review.document
    if "scene" in document["exports"]:
        try:
            export_path = indexed_scene_path(review.index_path)
            export = {"status": "verified", "path": str(export_path), "message": "保存indexの依存鎖とexport checksumを検証済み"}
        except ValueError as error:
            export = {"status": "unavailable", "message": str(error)}
    else:
        export = {"status": "missing", "message": "完成scene exportなし。未生成と失敗の区別はindexだけでは不明。"}

    frames = set(review.samples())
    if scene is not None:
        # Each rejection domain has a real representative; include the largest root gap.
        arrays = scene["arrays"]
        for codes in [np.asarray(arrays["ball_rejection_code"]), *np.asarray(arrays["player_rejection_code"])]:
            for code in np.unique(codes):
                if code:
                    frames.add(int(np.flatnonzero(codes == code)[0]))
        for gap in sorted(scene["gaps"], key=lambda item: item["end"] - item["start"], reverse=True)[:3]:
            frames.update((max(0, gap["start"] - 1), gap["start"], min(review.frame_count - 1, gap["end"])))
    samples: dict[str, dict[str, str]] = {}
    for video in sources:
        camera = video["camera_id"]
        samples[camera] = {}
        if not video["available"]:
            continue
        for frame in sorted(frames):
            name = f"source_{review.camera_ids.index(camera)}_{frame}.jpg"
            path = review.output / "images" / name
            image = cv2.resize(review.frame(camera, frame), (960, 540), interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(path), image):
                raise OSError(f"Could not write review snapshot: {path}")
            samples[camera][str(frame)] = f"images/{name}"
    observations: dict[str, dict[str, Any]] = {}
    for video in sources:
        camera = video["camera_id"]
        view: dict[str, Any] = {}
        node = f"pose_estimation/{camera}"
        if node in components and components[node]["status"] in ("rendered", "unavailable"):
            pose, _ = review.payload(node)
            view["pose"] = {name: np.asarray(pose[name])[0].tolist() for name in ("uv_px", "confidence", "observed", "local_track_ids")}
        node = f"ball_detection/{camera}"
        if node in components and components[node]["status"] in ("rendered", "unavailable"):
            ball, _ = review.payload(node)
            view["ball"] = {name: np.asarray(ball[name]).tolist() for name in ("uv_px", "point_kind", "observed")}
        observations[camera] = view
    nodes: list[dict[str, Any]] = []
    for node, entry in components.items():
        if node not in review.references:
            nodes.append({"node": node, "status": "missing", "dependencies": []})
            continue
        descriptor = review.descriptor(node)
        dependencies = []
        for port, dependency in descriptor["dependencies"].items():
            dependencies.append({"port": port, "artifact_id": dependency["artifact_id"],
                                 "producer": review._active_by_artifact.get(dependency["artifact_id"]),
                                 "path": dependency["path"], "sha256": dependency["sha256"]})
        nodes.append({"node": node, "status": entry["status"], "reasons": entry.get("reasons", []),
                      "artifact_id": descriptor["artifact_id"], "schema": descriptor["output_schema"],
                      "version": descriptor["output_version"], "origin": descriptor["provenance"],
                      "path": review.references[node]["path"], "sha256": review.references[node]["sha256"],
                      "dependencies": dependencies})
    return {"clip_id": review.source["clip_id"], "index_path": str(review.index_path),
            "index_sha256": dual_sha256(review.index_path), "source_sha256": review.source_sha256,
            "frames": review.frame_count, "fps": sources[0]["fps"], "sources": sources,
            "nodes": nodes, "export": export, "scene": scene, "scene_reason": scene_reason,
            "samples": samples, "sample_frames": sorted(frames), "online": online,
            "observations": observations, "skeleton": COCO17_SKELETON}
