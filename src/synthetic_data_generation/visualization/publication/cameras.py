"""Exact-schema camera adapters for publication geometry."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.reconstruction.scene_export import (
    NHT_CAMERAS_SCHEMA,
    validate_standard_scene_export,
)
from src.synthetic_data_generation.scene_contract import SceneCamera

METRIC_CAMERA_COORDINATE_CONVENTION = (
    "OpenCV camera axes (+x right,+y down,+z forward) mapped by camera_to_scene "
    "into right-handed metric scene metres"
)


@dataclass(frozen=True, slots=True)
class PublicationCameraCollection:
    """One explicit camera inventory normalized into metric scene coordinates."""

    owner: str
    schema: str
    scene_id: str
    logical_scene_id: str | None
    camera_ids: tuple[str, ...]
    cameras: tuple[SceneCamera, ...]
    camera_to_metric_scene: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.owner not in {"reconstruction"}:
            raise ValueError("Camera owner must be reconstruction.")
        if not self.schema or not self.scene_id:
            raise ValueError("Camera schema and scene_id must be non-empty.")
        cameras = tuple(self.cameras)
        camera_ids = tuple(self.camera_ids)
        if not cameras or camera_ids != tuple(item.camera_id for item in cameras):
            raise ValueError(
                "Camera collection identity/order differs from its records."
            )
        transforms = np.asarray(self.camera_to_metric_scene, dtype=np.float64)
        if (
            transforms.shape != (len(cameras), 4, 4)
            or not np.isfinite(transforms).all()
        ):
            raise ValueError("camera_to_metric_scene must be finite (N, 4, 4).")
        expected = np.stack([item.camera_to_scene.matrix() for item in cameras])
        if self.owner != "reconstruction" and not np.array_equal(transforms, expected):
            raise ValueError(
                "Generated-dataset camera poses must already be metric scene poses."
            )
        transforms = transforms.copy()
        transforms.setflags(write=False)
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(self, "camera_ids", camera_ids)
        object.__setattr__(self, "camera_to_metric_scene", transforms)

    @property
    def intrinsics(self) -> NDArray[np.float64]:
        """Return intrinsics in exact camera order."""
        return np.stack(
            [
                np.asarray(item.intrinsics, dtype=np.float64).reshape(3, 3)
                for item in self.cameras
            ]
        )

    @property
    def image_sizes(self) -> NDArray[np.int64]:
        """Return explicit ``(width, height)`` pairs in exact camera order."""
        return np.asarray(
            [(item.width, item.height) for item in self.cameras], dtype=np.int64
        )


def load_captured_cameras(
    scene_json: Path,
    *,
    scene_id: str,
    camera_ids: tuple[str, ...],
    metric_adapter: MetricSceneAdapter,
) -> PublicationCameraCollection:
    """Validate the standard export and convert every declared camera via the adapter."""
    export = validate_standard_scene_export(scene_json)
    if export.scene_id != scene_id:
        raise ValueError("Reconstruction scene_id differs from the publication scene.")
    if tuple(camera_ids) != export.camera_ids:
        raise ValueError(
            "captured_camera_ids must equal the complete reconstruction camera order."
        )
    transforms = np.stack(
        [
            metric_adapter.metric_from_nht_camera(camera.camera_to_scene).matrix()
            for camera in export.cameras
        ]
    )
    return PublicationCameraCollection(
        owner="reconstruction",
        schema=NHT_CAMERAS_SCHEMA,
        scene_id=scene_id,
        logical_scene_id=None,
        camera_ids=tuple(camera_ids),
        cameras=export.cameras,
        camera_to_metric_scene=transforms,
    )


def _nested_scene_cameras(value: object, *, name: str) -> tuple[SceneCamera, ...]:
    records = tuple(
        _exact(
            item,
            name=f"{name} record",
            keys={
                "slot_id",
                "court_local_center_m",
                "court_local_look_at_m",
                "hfov_degrees",
                "camera",
            },
        )
        for item in _sequence(value, name=name)
    )
    cameras = tuple(SceneCamera.from_dict(item["camera"]) for item in records)
    if not cameras or len(cameras) != len({item.camera_id for item in cameras}):
        raise ValueError(f"{name} must contain a non-empty unique camera inventory.")
    return cameras


def _contained_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError("Owner-relative paths must be normalized portable paths.")
    path = root.joinpath(*pure.parts)
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise FileNotFoundError(
            f"Required owner file is missing or escapes its owner: {relative}"
        )
    return path


def _load_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Required publication JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed JSON object.")
    return cast(Mapping[str, object], value)


def _exact(value: object, *, name: str, keys: set[str]) -> Mapping[str, object]:
    result = _mapping(value, name=name)
    if set(result) != keys:
        raise ValueError(
            f"{name} keys differ; missing={sorted(keys - set(result))}, "
            f"unknown={sorted(set(result) - keys)}."
        )
    return result


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a JSON array.")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


__all__ = [
    "METRIC_CAMERA_COORDINATE_CONVENTION",
    "PublicationCameraCollection",
    "load_captured_cameras",
]
