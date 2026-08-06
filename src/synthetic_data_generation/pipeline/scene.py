"""Semantic reader for the NHT standard scene file boundary."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@dataclass(frozen=True, slots=True)
class StandardScene:
    root: Path
    payload: dict[str, Any]
    cameras: tuple[dict[str, Any], ...]
    points: NDArray[np.float32]

    @classmethod
    def load(cls, scene_path: Path) -> StandardScene:
        root = scene_path.resolve().parent
        payload = json.loads(scene_path.read_text())
        if payload.get("schema") != "nht_standard_scene_v1":
            raise ValueError("Unsupported NHT standard scene schema")
        if not isinstance(payload.get("scene_id"), str) or not payload["scene_id"]:
            raise ValueError("NHT standard scene must declare a scene_id")
        conventions = (
            "camera_coordinate_convention",
            "scene_coordinate_convention",
            "pixel_coordinate_convention",
            "image_resolution_semantics",
        )
        if any(
            not isinstance(payload.get(field), str) or not payload[field].strip()
            for field in conventions
        ):
            raise ValueError("NHT scene coordinate conventions must be explicit")
        camera_path = (root / payload["cameras"]).resolve()
        point_path = (root / payload["point_cloud"]["path"]).resolve()
        for path in (camera_path, point_path):
            if not path.is_relative_to(root):
                raise ValueError("Scene reference escapes export root")
        camera_payload = json.loads(camera_path.read_text())
        if camera_payload.get("schema") != "nht_standard_cameras_v1":
            raise ValueError("Unsupported NHT standard camera schema")
        if (
            camera_payload.get("camera_coordinate_convention")
            != payload["camera_coordinate_convention"]
        ):
            raise ValueError("NHT camera coordinate conventions disagree")
        cameras = tuple(camera_payload["cameras"])
        points = np.load(point_path, allow_pickle=False)
        if (
            points.ndim != 2
            or points.shape[1] != 6
            or points.dtype != np.float32
            or not np.isfinite(points).all()
        ):
            raise ValueError("NHT point cloud must be finite float32 Nx6")
        if payload["point_cloud"].get("shape") != list(points.shape):
            raise ValueError("NHT declared point cloud shape is inconsistent")
        if len(points) and (
            float(points[:, 3:].min()) < 0.0 or float(points[:, 3:].max()) > 1.0
        ):
            raise ValueError("NHT point colors must lie in [0,1]")
        if not cameras or payload["camera_count"] != len(cameras):
            raise ValueError("NHT scene camera count is inconsistent")
        identifiers: set[str] = set()
        image_paths: set[str] = set()
        for camera in cameras:
            identifier = str(camera["camera_id"])
            if identifier in identifiers:
                raise ValueError(f"Duplicate NHT camera ID: {identifier}")
            identifiers.add(identifier)
            pose = np.asarray(camera["camera_to_scene"], dtype=np.float64)
            intrinsics = np.asarray(camera["intrinsics"]["matrix"], dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"Invalid NHT camera pose: {identifier}")
            if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
                raise ValueError(f"Invalid NHT intrinsics: {identifier}")
            if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
                raise ValueError(f"Non-positive NHT focal length: {identifier}")
            parameters = np.asarray(
                camera["intrinsics"].get("params", []), dtype=np.float64
            )
            if not len(parameters) or not np.isfinite(parameters).all():
                raise ValueError(f"Invalid NHT camera parameters: {identifier}")
            if camera["intrinsics"].get("model") != "PINHOLE":
                raise ValueError(f"Consumer camera is not PINHOLE: {identifier}")
            if camera["intrinsics"].get("distortion_model") != "NONE":
                raise ValueError(f"Consumer camera retains distortion: {identifier}")
            if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
                raise ValueError(f"Non-homogeneous NHT camera pose: {identifier}")
            rotation = pose[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
                raise ValueError(f"Non-orthonormal NHT camera pose: {identifier}")
            determinant = float(np.linalg.det(rotation))
            if not math.isclose(determinant, 1.0, abs_tol=1.0e-5):
                raise ValueError(f"Improper NHT camera pose: {identifier}")
            if camera["width"] < 1 or camera["height"] < 1:
                raise ValueError(f"Invalid NHT image resolution: {identifier}")
            image_relative = str(camera["image"])
            if image_relative in image_paths:
                raise ValueError(f"Duplicate NHT camera image: {image_relative}")
            image_paths.add(image_relative)
            image_path = (root / camera["image"]).resolve()
            if not image_path.is_relative_to(root) or not image_path.is_file():
                raise FileNotFoundError(f"NHT camera image is absent: {identifier}")
            with Image.open(image_path) as image:
                if image.size != (camera["width"], camera["height"]):
                    raise ValueError(f"NHT image resolution mismatch: {identifier}")
        scene_from_sfm = np.asarray(payload["scene_from_sfm"], dtype=np.float64)
        sfm_from_scene = np.asarray(payload["sfm_from_scene"], dtype=np.float64)
        if (
            scene_from_sfm.shape != (4, 4)
            or sfm_from_scene.shape != (4, 4)
            or not np.isfinite(scene_from_sfm).all()
            or not np.isfinite(sfm_from_scene).all()
            or not np.allclose(scene_from_sfm @ sfm_from_scene, np.eye(4), atol=1.0e-5)
        ):
            raise ValueError("NHT scene transforms are invalid or not inverse")
        model_root = (root / payload["model_root"]).resolve()
        if not model_root.is_relative_to(root) or not model_root.is_dir():
            raise FileNotFoundError("NHT model root is absent or escapes export")
        renderer = payload.get("renderer", {})
        if renderer.get("command") != "nht-render":
            raise ValueError("NHT scene does not declare nht-render boundary")
        if "nht_rendering_model" not in payload.get("capabilities", []):
            raise ValueError("NHT scene lacks the rendering capability")
        for field in ("checkpoint", "runtime_config"):
            artifact = (root / renderer[field]).resolve()
            if not artifact.is_relative_to(root) or not artifact.is_file():
                raise FileNotFoundError(f"NHT renderer {field} is absent")
        return cls(root, payload, cameras, points)

    def camera_centers(self) -> NDArray[np.float64]:
        return np.asarray(
            [
                np.asarray(camera["camera_to_scene"], dtype=np.float64)[:3, 3]
                for camera in self.cameras
            ],
            dtype=np.float64,
        )
