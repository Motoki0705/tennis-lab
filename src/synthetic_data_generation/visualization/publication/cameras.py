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
from src.synthetic_data_generation.dataset.blcs.assembler import (
    validate_blcs_dataset_envelope,
)
from src.synthetic_data_generation.dataset.blcs.contracts import BLCS_DATASET_SCHEMA
from src.synthetic_data_generation.dataset.plcs.assembler import PLCS_DATASET_SCHEMA
from src.synthetic_data_generation.dataset.plcs.validation import validate_plcs_dataset
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
        if self.owner not in {"reconstruction", "blcs", "plcs"}:
            raise ValueError("Camera owner must be reconstruction, blcs, or plcs.")
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


def load_blcs_cameras(
    root: Path,
    *,
    scene_id: str,
    logical_scene_id: str,
    camera_ids: tuple[str, ...],
) -> PublicationCameraCollection:
    """Load the canonical nested BLCS trajectory-plan camera schema exactly."""
    validate_blcs_dataset_envelope(root)
    manifest = _exact(
        _load_json(root / "dataset.json"),
        name="BLCS dataset",
        keys={
            "schema",
            "scene_id",
            "domain",
            "frame_inventory",
            "target_courts",
            "metadata",
            "diagnostics",
            "performance",
            "trajectories",
            "samples",
        },
    )
    if (
        manifest["schema"] != BLCS_DATASET_SCHEMA
        or manifest["domain"] != "blcs"
        or manifest["scene_id"] != scene_id
    ):
        raise ValueError("BLCS owner schema/domain/scene identity is inconsistent.")
    trajectories = tuple(
        _exact(
            value,
            name="BLCS trajectory",
            keys={
                "trajectory_id",
                "split",
                "source_frame_count",
                "global_frame_offset",
                "frame_inventory",
                "target_court",
                "candidate_id",
                "transform",
                "camera_profile",
                "camera_seed",
                "camera_ids",
                "attempt_token",
                "chunk_count",
                "chunk_directories",
                "background_store",
                "plan_json",
                "plan_npz",
            },
        )
        for value in _sequence(manifest["trajectories"], name="BLCS trajectories")
    )
    matching = tuple(
        value for value in trajectories if value["trajectory_id"] == logical_scene_id
    )
    if len(matching) != 1:
        raise KeyError(f"Unknown BLCS logical scene: {logical_scene_id!r}.")
    trajectory = matching[0]
    declared_ids = tuple(
        _text(value, name="BLCS camera_id")
        for value in _sequence(trajectory["camera_ids"], name="BLCS camera_ids")
    )
    if declared_ids != tuple(camera_ids):
        raise ValueError(
            "blcs_camera_ids differ from the complete canonical camera order."
        )
    plan_path = _contained_file(
        root, _text(trajectory["plan_json"], name="BLCS plan_json")
    )
    plan = _exact(
        _load_json(plan_path),
        name="BLCS plan",
        keys={
            "trajectory_id",
            "split",
            "fps",
            "source_frame_count",
            "global_frame_offset",
            "global_frame_indices",
            "tracks",
            "target_court",
            "camera_profile",
            "camera_seed",
            "cameras",
            "chunks",
            "composition",
            "source_metadata",
        },
    )
    if plan["trajectory_id"] != logical_scene_id:
        raise ValueError("BLCS plan identity differs from the requested logical scene.")
    cameras = _nested_scene_cameras(plan["cameras"], name="BLCS plan cameras")
    if tuple(item.camera_id for item in cameras) != declared_ids:
        raise ValueError("BLCS plan and owner camera order differs.")
    return PublicationCameraCollection(
        owner="blcs",
        schema=BLCS_DATASET_SCHEMA,
        scene_id=scene_id,
        logical_scene_id=logical_scene_id,
        camera_ids=declared_ids,
        cameras=cameras,
        camera_to_metric_scene=np.stack(
            [item.camera_to_scene.matrix() for item in cameras]
        ),
    )


def load_plcs_cameras(
    root: Path,
    *,
    scene_id: str,
    logical_scene_id: str,
    camera_ids: tuple[str, ...],
) -> PublicationCameraCollection:
    """Load the canonical nested PLCS logical-scene camera schema exactly."""
    validate_plcs_dataset(root)
    manifest = _exact(
        _load_json(root / "dataset.json"),
        name="PLCS dataset",
        keys={
            "schema",
            "scene_id",
            "domain",
            "frame_inventory",
            "target_courts",
            "metadata",
            "diagnostics",
            "storage",
        },
    )
    if (
        manifest["schema"] != PLCS_DATASET_SCHEMA
        or manifest["domain"] != "plcs"
        or manifest["scene_id"] != scene_id
    ):
        raise ValueError("PLCS owner schema/domain/scene identity is inconsistent.")
    metadata = _exact(
        manifest["metadata"],
        name="PLCS metadata",
        keys={
            "coordinate_contract",
            "court_coordinate_normalization",
            "seed",
            "logical_scene_count",
            "aggregate_global_frame_count",
            "aggregate_source_frame_count",
            "required_motion_categories",
            "accepted_court_instance_ids",
            "logical_scenes",
        },
    )
    logical_scenes = tuple(
        _exact(
            value,
            name="PLCS logical scene",
            keys={
                "scene_id",
                "split",
                "aggregate_frame_offset",
                "frame_inventory",
                "mode",
                "target_court",
                "camera_profile",
                "cameras",
                "motion_sources",
                "tracks",
                "continuity",
            },
        )
        for value in _sequence(metadata["logical_scenes"], name="PLCS logical_scenes")
    )
    matching = tuple(
        value for value in logical_scenes if value["scene_id"] == logical_scene_id
    )
    if len(matching) != 1:
        raise KeyError(f"Unknown PLCS logical scene: {logical_scene_id!r}.")
    cameras = _nested_scene_cameras(matching[0]["cameras"], name="PLCS cameras")
    declared_ids = tuple(item.camera_id for item in cameras)
    if declared_ids != tuple(camera_ids):
        raise ValueError(
            "plcs_camera_ids differ from the complete canonical camera order."
        )
    return PublicationCameraCollection(
        owner="plcs",
        schema=PLCS_DATASET_SCHEMA,
        scene_id=scene_id,
        logical_scene_id=logical_scene_id,
        camera_ids=declared_ids,
        cameras=cameras,
        camera_to_metric_scene=np.stack(
            [item.camera_to_scene.matrix() for item in cameras]
        ),
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
    "load_blcs_cameras",
    "load_captured_cameras",
    "load_plcs_cameras",
]
