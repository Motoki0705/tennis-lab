"""Checksum-checked inspection of explicitly adopted component snapshots.

Historical identity v2 and current v3 are distinguished explicitly. No current
pipeline definition, model load, inference, annotation write or nearest-frame
substitution is used by this reader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.submodules.visualization.review.geometry import placed_vertices
from src.tennis_scene.pipeline.storage.codec import unpack_value
from src.tennis_scene.pipeline.storage.scene_index import (
    assert_current_component_lineage,
    read_component_descriptor,
)
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PointRejection
from src.utils.schema.player import COCO17_SKELETON, COCO_KP_NAMES

SCHEMAS = {
    "gvhmr": ("body_parameters", (1,)),
    "body_view_selection": ("body_view_selection", (1,)),
    "body_placement": ("placed_bodies", (1,)),
    "pose_estimation": ("person_poses", (1,)),
    "player_association": ("person_identities", (2, 3)),
    "player_triangulation": ("player_skeletons", (1,)),
}


def exact_sample(frames: np.ndarray, frame: int) -> int | None:
    """Find an exact source frame, never a nearby sampled frame."""
    if frames.ndim != 1 or frames.dtype.kind not in "iu" or (np.diff(frames) <= 0).any():
        raise ValueError("Source frames must be a strictly increasing integer vector")
    index = int(np.searchsorted(frames, frame))
    return index if index < len(frames) and int(frames[index]) == frame else None


class PoseReviewStore:
    def __init__(self, root: Path, *, topology: Path | None = None) -> None:
        self.root = root.resolve()
        index = self.root / "scene.json"
        self.index_hash = dual_sha256(index)
        self.document: dict[str, Any] = json.loads(index.read_text())
        if self.document["schema"] != "tennis_scene_index_v1":
            raise ValueError("Unsupported scene index schema")
        source = self.document["source"]
        self.videos = {video["camera_id"]: video for video in source["videos"]}
        if not self.videos or len(self.videos) != len(source["videos"]):
            raise ValueError("Scene source requires unique cameras")
        first = source["videos"][0]
        self.frame_count, self.fps = int(first["num_frames"]), float(first["fps"])
        if self.frame_count < 1 or not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Source requires a positive frame count and fps")
        self.files: dict[Path, tuple[int, int, int]] = {}
        self._remember(index)
        self.media_status: dict[str, str] = {}
        for camera, video in self.videos.items():
            if video["num_frames"] != self.frame_count or video["fps"] != self.fps:
                raise ValueError("Review requires synchronized source frame axes")
            path = Path(video["path"])
            if not path.is_file():
                self.media_status[camera] = "missing"
            else:
                if dual_sha256(path) != video["sha256"]:
                    raise ValueError(f"Source video checksum mismatch: {camera}")
                self._remember(path)
                self.media_status[camera] = "available"
        self.payloads: dict[str, dict[str, Any]] = {}
        self.artifacts: dict[str, dict[str, Any]] = {}
        relevant = {node: reference for node, reference in self.document["artifacts"].items() if node.split("/")[0] in SCHEMAS}
        if relevant:
            # Include ancestors outside the displayed families (e.g. calibration),
            # so a current immediate input cannot hide an overwritten ancestor.
            assert_current_component_lineage(self.document, self.root, relevant)
        for node, reference in self.document["artifacts"].items():
            family = node.split("/")[0]
            if family not in SCHEMAS:
                continue
            schema, versions = SCHEMAS[family]
            if reference["schema"] != schema or reference["version"] not in versions:
                raise ValueError(f"Unsupported {node}: {reference['schema']} v{reference['version']}")
            descriptor = read_component_descriptor(self.root, reference, node=node, source_sha256=self.document["source_sha256"])
            directory = (self.root / reference["path"]).parent
            payload = unpack_value(descriptor["payload"], directory, descriptor["arrays"])
            if not isinstance(payload, dict):
                raise ValueError(f"Expected structured {node}")
            self.payloads[node] = payload
            self.artifacts[node] = {"schema": schema, "version": reference["version"], "artifact_id": reference["artifact_id"], "origin": descriptor["provenance"]["origin"]}
            self._remember(directory / Path(reference["path"]).name)
            for name in descriptor["arrays"]:
                self._remember(directory / name)
        self.faces: list[list[int]] | None = None
        if topology is not None:
            with np.load(topology, allow_pickle=False) as archive:
                faces = np.asarray(archive["f"])
            if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu" or (faces < 0).any() or (faces >= 6890).any():
                raise ValueError("Topology must contain SMPL 6890-vertex triangle indices")
            self.faces = faces.tolist()
            self._remember(topology)
        identities = self.payloads.get("player_association")
        self.people = sorted(int(value) for value in np.unique(identities["player_ids"]) if value >= 0) if identities is not None else []
        recovered = self.payloads.get("gvhmr")
        if recovered is not None:
            self.people = sorted(set(self.people) | {int(body["person_id"]) for body in recovered["bodies"]})
        placement = self.payloads.get("body_placement")
        self.players = placement["players"] if placement is not None else None
        if self.players is not None and (identities is None or len(self.people) != len(self.players["position"])):
            raise ValueError("Cannot map saved placement rows to identified people")
        self._validate_axes()

    def _remember(self, path: Path) -> None:
        stat = path.stat()
        self.files[path] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def assert_unchanged(self) -> None:
        for path, expected in self.files.items():
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != expected:
                raise ValueError(f"Review snapshot changed; restart explicitly: {path.name}")

    def _validate_axes(self) -> None:
        identities = self.payloads.get("player_association")
        if identities is not None:
            version = self.artifacts["player_association"]["version"]
            tracks = np.asarray(identities["local_track_ids"])
            expected = tracks.shape if version == 2 else (*tracks.shape, self.frame_count)
            if np.asarray(identities["player_ids"]).shape != expected:
                raise ValueError("Identity schema disagrees with source frame axis")
        for camera, video in self.videos.items():
            pose = self.payloads.get(f"pose_estimation/{camera}")
            if pose is not None:
                count = len(pose["local_track_ids"][0])
                if pose["camera_ids"] != [camera] or pose["uv_px"].shape != (1, self.frame_count, count, 17, 2) or pose["observed"].shape != (1, self.frame_count, count) or pose["confidence"].shape != (1, self.frame_count, count, 17):
                    raise ValueError(f"Pose frame/camera axes disagree: {camera}")
                if pose["size"] != [video["width"], video["height"]] or pose["fps"] != self.fps:
                    raise ValueError(f"Pose source size/fps disagree: {camera}")
        for family, key, segments in (("gvhmr", "bodies", "segments"), ("body_view_selection", "selections", "requests")):
            payload = self.payloads.get(family)
            if payload is None:
                continue
            for body in payload[key]:
                if body["camera_id"] not in self.videos:
                    raise ValueError("Body camera is absent from source")
                seen: set[int] = set()
                for segment in body[segments]:
                    frames = np.asarray(segment["source_frames"])
                    exact_sample(frames, -1)
                    if len(frames) and (frames[0] < 0 or frames[-1] >= self.frame_count or seen.intersection(frames.tolist())):
                        raise ValueError("Body segments overlap or escape the source frame axis")
                    seen.update(frames.tolist())
                    if family == "body_view_selection":
                        if segment["keypoints"].shape != (len(frames), 17, 3) or segment["boxes_xys"].shape != (len(frames), 3):
                            raise ValueError("GVHMR input shape disagrees with source frames")
                    else:
                        for name, width in (("body_pose", 63), ("global_orient", 3), ("betas", 10), ("transl", 3)):
                            if segment["parameters"][name].shape != (len(frames), width):
                                raise ValueError("GVHMR parameter shape disagrees with source frames")
        if self.players is not None:
            expected = (len(self.people), self.frame_count)
            for key in ("root_valid", "heading_valid", "smpl_valid", "root_reasons", "yaw", "mesh_reprojection_px"):
                if self.players[key].shape != expected:
                    raise ValueError(f"Placement frame axis mismatch: {key}")
            for key in ("root_valid", "heading_valid", "smpl_valid"):
                if self.players[key].dtype != bool:
                    raise ValueError(f"Placement requires boolean {key}")
            if np.any(self.players["smpl_valid"] & ~self.players["root_valid"]):
                raise ValueError("Valid mesh requires valid root")
            for key in ("position", "global_orient"):
                if self.players[key].shape != (*expected, 3) or not np.isfinite(self.players[key]).all():
                    raise ValueError(f"Invalid placement coordinates: {key}")
            vertices = self.players["vertices_local"]
            if vertices is not None and vertices.shape != (*expected, 6890, 3):
                raise ValueError("Saved SMPL vertices must match person/source frame axes")
        skeleton = self.payloads.get("player_triangulation")
        if skeleton is not None and skeleton["skeleton"] is not None:
            value = skeleton["skeleton"]
            if value["positions"].shape != (len(self.people), self.frame_count, 17, 3) or value["valid"].shape != (len(self.people), self.frame_count, 17) or value["reasons"].shape != value["valid"].shape or value["valid"].dtype != bool:
                raise ValueError("Triangulated joints disagree with person/source frame axes")
        recovered = self.payloads.get("gvhmr")
        if recovered is not None and "body_view_selection" in self.payloads:
            for body in recovered["bodies"]:
                for segment in body["segments"]:
                    for frame in segment["source_frames"]:
                        request, index, camera = self._sample("body_view_selection", int(body["person_id"]), int(frame))
                        if request is None or index is None or camera != body["camera_id"]:
                            raise ValueError("GVHMR sample has no matching saved input frame/view")

    def _person_row(self, person: int) -> int:
        if person not in self.people:
            raise ValueError("Unknown person")
        return self.people.index(person)

    def observation(self, person: int, camera: str, frame: int) -> dict[str, Any]:
        pose = self.payloads.get(f"pose_estimation/{camera}")
        identities = self.payloads.get("player_association")
        if pose is None or identities is None:
            return {"state": "not_saved", "joints": None, "box": None, "track_id": None}
        if camera not in identities["camera_ids"]:
            return {"state": "identity_unknown", "joints": None, "box": None, "track_id": None}
        view = identities["camera_ids"].index(camera)
        labels = np.asarray(identities["player_ids"])[view]
        if self.artifacts["player_association"]["version"] == 3:
            labels = labels[:, frame]
        candidates = identities["local_track_ids"][view][labels == person]
        matches = np.flatnonzero(np.isin(pose["local_track_ids"][0], candidates) & pose["observed"][0, frame])
        if len(matches) > 1:
            raise ValueError("Ambiguous person observations in one camera/frame")
        if not len(matches):
            return {"state": "unobserved", "joints": None, "box": None, "track_id": None}
        row = int(matches[0])
        joints = np.column_stack((pose["uv_px"][0, frame, row], pose["confidence"][0, frame, row]))
        boxes = pose["boxes_xys"]
        return {"state": "observed", "joints": joints.tolist(), "box": boxes[0, frame, row].tolist() if boxes is not None else None, "track_id": int(pose["local_track_ids"][0, row])}

    def _sample(self, family: str, person: int, frame: int) -> tuple[dict[str, Any] | None, int | None, str | None]:
        value = self.payloads.get(family)
        if value is None:
            return None, None, None
        key, segments = ("bodies", "segments") if family == "gvhmr" else ("selections", "requests")
        for body in value[key]:
            if body["person_id"] != person:
                continue
            for segment in body[segments]:
                index = exact_sample(segment["source_frames"], frame)
                if index is not None:
                    return segment, index, str(body["camera_id"])
            return None, None, str(body["camera_id"])
        return None, None, None

    def frame(self, person: int, camera: str, frame: int) -> dict[str, Any]:
        self.assert_unchanged()
        row = self._person_row(person)
        if camera not in self.videos or not 0 <= frame < self.frame_count:
            raise ValueError("Unknown camera or source frame out of range")
        request, request_index, input_camera = self._sample("body_view_selection", person, frame)
        sample, sample_index, recovered_camera = self._sample("gvhmr", person, frame)
        request_value = None
        if request is not None and request_index is not None:
            request_value = {"joints": request["keypoints"][request_index].tolist(), "box": request["boxes_xys"][request_index].tolist()}
        parameters = None
        if sample is not None and sample_index is not None:
            parameters = {key: value[sample_index].tolist() for key, value in sample["parameters"].items()}
        placement = None
        vertices = None
        if self.players is not None:
            p = self.players
            reason = int(p["root_reasons"][row, frame])
            codes = p["diagnostics"]["rejection_codes"]
            names = [key for key, value in codes.items() if int(value) == reason]
            root, heading, mesh = (bool(p[key][row, frame]) for key in ("root_valid", "heading_valid", "smpl_valid"))
            placement = {"root_valid": root, "heading_valid": heading, "smpl_valid": mesh, "reason_code": reason, "reason": "valid" if reason == 0 else "/".join(names) if names else f"unknown_code_{reason}", "position": p["position"][row, frame].tolist() if root else None, "yaw": float(p["yaw"][row, frame]) if heading else None, "reprojection_px": float(p["mesh_reprojection_px"][row, frame]) if mesh else None}
            if mesh and p["vertices_local"] is not None:
                vertices = placed_vertices(p["vertices_local"][row, frame], p["global_orient"][row, frame], float(p["yaw"][row, frame]), p["position"][row, frame]).tolist()
        skeleton = self.payloads.get("player_triangulation")
        joints_3d = None
        reasons_3d = None
        if skeleton is not None and skeleton["skeleton"] is not None:
            s = skeleton["skeleton"]
            joints_3d = [s["positions"][row, frame, j].tolist() if s["valid"][row, frame, j] else None for j in range(17)]
            reasons_3d = s["reasons"][row, frame].tolist()
        observations = {view: self.observation(person, view, frame) for view in self.videos}
        return {"person": person, "camera": camera, "frame": frame, "seconds": frame / self.fps, "observation": observations[camera], "observations_by_camera": observations, "request_camera": input_camera, "request": request_value, "recovered_camera": recovered_camera, "parameters": parameters, "placement": placement, "vertices": vertices, "joints_3d": joints_3d, "joint_reasons_3d": reasons_3d}

    def metadata(self) -> dict[str, Any]:
        return {"clip_id": self.document["source"]["clip_id"], "store": str(self.root), "index_sha256": self.index_hash, "frame_count": self.frame_count, "fps": self.fps, "people": self.people, "cameras": [{"id": camera, "width": video["width"], "height": video["height"], "media": self.media_status[camera]} for camera, video in self.videos.items()], "artifacts": self.artifacts, "faces": self.faces, "joint_names": COCO_KP_NAMES, "edges": COCO17_SKELETON, "triangulation_rejection_codes": {str(int(reason)): reason.name for reason in PointRejection}, "identity_policy": "historical_static_v2" if "player_association" in self.artifacts and self.artifacts["player_association"]["version"] == 2 else "per_frame_v3" if "player_association" in self.artifacts else "not_saved"}

    def timeline(self, person: int, camera: str) -> dict[str, Any]:
        row = self._person_row(person)
        observed = [self.observation(person, camera, frame)["state"] == "observed" for frame in range(self.frame_count)] if f"pose_estimation/{camera}" in self.payloads and "player_association" in self.payloads else None
        sampled = [self._sample("gvhmr", person, frame)[1] is not None for frame in range(self.frame_count)] if "gvhmr" in self.payloads else None
        return {"observed": observed, "sampled": sampled, "root": self.players["root_valid"][row].tolist() if self.players is not None else None, "mesh": self.players["smpl_valid"][row].tolist() if self.players is not None else None}

    def image(self, camera: str, frame: int) -> bytes:
        self.assert_unchanged()
        if camera not in self.videos or not 0 <= frame < self.frame_count:
            raise ValueError("Unknown camera or source frame out of range")
        if self.media_status[camera] != "available":
            raise FileNotFoundError("Source RGB is not available")
        video = self.videos[camera]
        capture = cv2.VideoCapture(video["path"])
        try:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame):
                raise OSError("Cannot seek source frame")
            okay, image = capture.read()
            if not okay or image.shape[:2] != (video["height"], video["width"]):
                raise OSError("Cannot decode source frame at declared dimensions")
            okay, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not okay:
                raise OSError("Cannot encode review frame")
            return encoded.tobytes()
        finally:
            capture.release()
