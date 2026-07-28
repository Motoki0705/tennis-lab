"""Versioned BLCS Gaussian scene plans derived from physical trajectories."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, Self, cast

import numpy as np
import torch
from numpy.typing import NDArray

from src.synthetic_data_generation.composition.contracts import (
    GaussianAsset,
    GaussianInstance,
)
from src.synthetic_data_generation.dataset.blcs.artifacts.asset_registry import (
    BallAssetEntry,
    BallAssetRegistry,
    BallAssetSelection,
    select_ball_asset,
    verify_local_ball_asset_registry,
)
from src.synthetic_data_generation.scene_contract import (
    CAMERA_AXES_OPENCV,
    COURT_AXES_METRES,
    PIXEL_COORDINATES,
    SceneCamera,
    SimilarityTransform,
)

BLCS_GAUSSIAN_PLAN_SCHEMA = "tennis_blcs_gaussian_scene_plan_v2"
_SCENE_FROM_ASSET_SEMANTICS = (
    "metric asset origin follows ball centre; scale and rotation are inherited "
    "from scene_from_court; spin is not inferred"
)
_GEOMETRIC_VISIBILITY_SEMANTICS = (
    "camera_geometric_visible is present, positive-depth, and in-frame; it does "
    "not claim Gaussian occlusion visibility or rendered RGB"
)
_ARRAY_FILENAMES = {
    "positions_court_m": "positions_court_m.npy",
    "velocities_court_mps": "velocities_court_mps.npy",
    "present": "present.npy",
    "positions_scene": "positions_scene.npy",
    "scene_from_asset": "scene_from_asset.npy",
    "camera_uv": "camera_uv.npy",
    "camera_depth": "camera_depth.npy",
    "camera_geometric_visible": "camera_geometric_visible.npy",
    "instance_ids": "instance_ids.npy",
}


class BLCSSceneLike(Protocol):
    """Physical trajectory fields consumed from the existing BLCS generator."""

    scene_id: str
    ball_pos_world: torch.Tensor
    ball_vel_world: torch.Tensor
    ball_present: torch.Tensor | None
    num_balls: int
    fps_out: int


@dataclass(frozen=True)
class BallAssetAssignment:
    """One asset selection fixed to one persistent lifecycle column."""

    instance_id: int
    selection: BallAssetSelection

    def __post_init__(self) -> None:
        if isinstance(self.instance_id, bool) or self.instance_id <= 0:
            raise ValueError("instance_id must be a positive integer.")

    def to_dict(self) -> dict[str, object]:
        """Return the complete auditable selection record."""
        return {
            "instance_id": self.instance_id,
            "variant_id": self.selection.entry.variant_id,
            "entry_index": self.selection.entry_index,
            "selection_sha256": self.selection.selection_sha256,
            "nominal_diameter_m": self.selection.entry.nominal_diameter_m,
            "asset": self.selection.entry.asset.to_dict(),
        }


@dataclass(frozen=True)
class BLCSGaussianScenePlan:
    """Immutable per-frame placement and projection plan before native rendering."""

    schema: str
    plan_fingerprint: str
    scene_id: str
    seed: int
    fps: float
    registry: BallAssetRegistry
    scene_from_court: SimilarityTransform
    assignments: tuple[BallAssetAssignment, ...]
    cameras: tuple[SceneCamera, ...]
    positions_court_m: NDArray[np.float64]
    velocities_court_mps: NDArray[np.float64]
    present: NDArray[np.bool_]
    positions_scene: NDArray[np.float64]
    scene_from_asset: NDArray[np.float64]
    camera_uv: NDArray[np.float64]
    camera_depth: NDArray[np.float64]
    camera_geometric_visible: NDArray[np.bool_]
    instance_ids: NDArray[np.int64]

    def __post_init__(self) -> None:
        if self.schema != BLCS_GAUSSIAN_PLAN_SCHEMA:
            raise ValueError(
                f"Unsupported BLCS plan schema {self.schema!r}; "
                f"expected {BLCS_GAUSSIAN_PLAN_SCHEMA!r}."
            )
        verify_local_ball_asset_registry(self.registry)
        if not self.scene_id.strip():
            raise ValueError("scene_id must not be empty.")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer.")
        fps = _finite_float(self.fps, name="fps")
        if fps <= 0.0:
            raise ValueError("fps must be positive.")
        assignments = tuple(self.assignments)
        cameras = tuple(self.cameras)
        if not assignments:
            raise ValueError("BLCS plan requires at least one object assignment.")
        if not cameras:
            raise ValueError("BLCS plan requires at least one camera.")
        registry_fingerprint = self.registry.registry_fingerprint
        _require_unique(
            [assignment.instance_id for assignment in assignments],
            name="assignment instance ids",
        )
        _require_unique(
            [camera.camera_id for camera in cameras],
            name="camera ids",
        )

        arrays = self._validated_arrays()
        frame_count, object_count, _ = arrays["positions_court_m"].shape
        camera_count = len(cameras)
        expected_shapes = {
            "velocities_court_mps": (frame_count, object_count, 3),
            "present": (frame_count, object_count),
            "positions_scene": (frame_count, object_count, 3),
            "scene_from_asset": (frame_count, object_count, 4, 4),
            "camera_uv": (camera_count, frame_count, object_count, 2),
            "camera_depth": (camera_count, frame_count, object_count),
            "camera_geometric_visible": (
                camera_count,
                frame_count,
                object_count,
            ),
            "instance_ids": (object_count,),
        }
        for name, expected_shape in expected_shapes.items():
            if arrays[name].shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, "
                    f"got {arrays[name].shape}."
                )
        if len(assignments) != object_count:
            raise ValueError(
                "Assignment count must equal trajectory object columns: "
                f"{len(assignments)} != {object_count}."
            )
        for object_index, assignment in enumerate(assignments):
            expected_selection = select_ball_asset(
                self.registry,
                seed=self.seed,
                selection_key=f"{self.scene_id}:object:{object_index}",
            )
            if assignment.instance_id != object_index + 1:
                raise ValueError(
                    "BLCS assignment instance IDs must equal one-based object columns."
                )
            if assignment.selection != expected_selection:
                raise ValueError(
                    "BLCS asset assignment does not match deterministic registry "
                    f"selection for object column {object_index}."
                )
        if not arrays["present"].any(axis=0).all():
            raise ValueError(
                "Every object column must be present in at least one frame."
            )
        expected_ids = np.asarray(
            [assignment.instance_id for assignment in assignments],
            dtype=np.int64,
        )
        if not np.array_equal(arrays["instance_ids"], expected_ids):
            raise ValueError("instance_ids do not match assignment order.")
        present = cast(NDArray[np.bool_], arrays["present"])
        geometric_visible = cast(
            NDArray[np.bool_],
            arrays["camera_geometric_visible"],
        )
        camera_depth = cast(NDArray[np.float64], arrays["camera_depth"])
        if np.any(geometric_visible & ~present[None]):
            raise ValueError("An absent object cannot be geometrically visible.")
        if np.any(geometric_visible & (camera_depth <= 0.0)):
            raise ValueError("A visible object must have positive camera depth.")

        fingerprint = _sha256(self.plan_fingerprint, name="plan_fingerprint")
        expected_fingerprint = compute_blcs_gaussian_plan_fingerprint(
            scene_id=self.scene_id,
            seed=self.seed,
            fps=fps,
            registry_fingerprint=registry_fingerprint,
            scene_from_court=self.scene_from_court,
            assignments=assignments,
            cameras=cameras,
            arrays=arrays,
        )
        if fingerprint != expected_fingerprint:
            raise ValueError(
                "BLCS plan fingerprint mismatch: "
                f"declared {fingerprint}, computed {expected_fingerprint}."
            )

        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(self, "plan_fingerprint", fingerprint)
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @classmethod
    def create(
        cls,
        *,
        scene_id: str,
        seed: int,
        fps: float,
        registry: BallAssetRegistry,
        scene_from_court: SimilarityTransform,
        assignments: Sequence[BallAssetAssignment],
        cameras: Sequence[SceneCamera],
        positions_court_m: NDArray[np.float64],
        velocities_court_mps: NDArray[np.float64],
        present: NDArray[np.bool_],
        positions_scene: NDArray[np.float64],
        scene_from_asset: NDArray[np.float64],
        camera_uv: NDArray[np.float64],
        camera_depth: NDArray[np.float64],
        camera_geometric_visible: NDArray[np.bool_],
        instance_ids: NDArray[np.int64],
    ) -> Self:
        """Create a fully validated plan with a canonical content fingerprint."""
        assignment_tuple = tuple(assignments)
        camera_tuple = tuple(cameras)
        arrays: dict[str, NDArray[np.generic]] = {
            "positions_court_m": positions_court_m,
            "velocities_court_mps": velocities_court_mps,
            "present": present,
            "positions_scene": positions_scene,
            "scene_from_asset": scene_from_asset,
            "camera_uv": camera_uv,
            "camera_depth": camera_depth,
            "camera_geometric_visible": camera_geometric_visible,
            "instance_ids": instance_ids,
        }
        return cls(
            schema=BLCS_GAUSSIAN_PLAN_SCHEMA,
            plan_fingerprint=compute_blcs_gaussian_plan_fingerprint(
                scene_id=scene_id,
                seed=seed,
                fps=fps,
                registry_fingerprint=registry.registry_fingerprint,
                scene_from_court=scene_from_court,
                assignments=assignment_tuple,
                cameras=camera_tuple,
                arrays=arrays,
            ),
            scene_id=scene_id,
            seed=seed,
            fps=fps,
            registry=registry,
            scene_from_court=scene_from_court,
            assignments=assignment_tuple,
            cameras=camera_tuple,
            positions_court_m=positions_court_m,
            velocities_court_mps=velocities_court_mps,
            present=present,
            positions_scene=positions_scene,
            scene_from_asset=scene_from_asset,
            camera_uv=camera_uv,
            camera_depth=camera_depth,
            camera_geometric_visible=camera_geometric_visible,
            instance_ids=instance_ids,
        )

    @property
    def num_frames(self) -> int:
        """Number of frames on the shared global timeline."""
        return int(self.positions_court_m.shape[0])

    @property
    def num_objects(self) -> int:
        """Number of stable lifecycle columns."""
        return int(self.positions_court_m.shape[1])

    @property
    def registry_fingerprint(self) -> str:
        """Content identity of the explicit asset inventory."""
        return self.registry.registry_fingerprint

    def instances_at(self, frame_index: int) -> tuple[GaussianInstance, ...]:
        """Return only active Gaussian instances for one frame."""
        if not 0 <= frame_index < self.num_frames:
            raise IndexError(
                f"frame_index {frame_index} outside [0, {self.num_frames})."
            )
        result = []
        rotation = self.scene_from_court.rotation
        for object_index, assignment in enumerate(self.assignments):
            if not bool(self.present[frame_index, object_index]):
                continue
            result.append(
                GaussianInstance(
                    instance_id=assignment.instance_id,
                    asset=assignment.selection.entry.asset,
                    scene_from_asset=SimilarityTransform(
                        scale=self.scene_from_court.scale,
                        rotation=rotation,
                        translation=tuple(
                            float(value)
                            for value in self.positions_scene[frame_index, object_index]
                        ),
                    ),
                )
            )
        return tuple(result)

    def _validated_arrays(self) -> dict[str, NDArray[np.generic]]:
        arrays: dict[str, NDArray[np.generic]] = {}
        float_names = {
            "positions_court_m",
            "velocities_court_mps",
            "positions_scene",
            "scene_from_asset",
            "camera_uv",
            "camera_depth",
        }
        bool_names = {"present", "camera_geometric_visible"}
        for name in _ARRAY_FILENAMES:
            value = np.asarray(getattr(self, name))
            if name in float_names:
                if not np.issubdtype(value.dtype, np.floating):
                    raise TypeError(f"{name} must be floating point.")
                value = np.array(value, dtype=np.float64, order="C", copy=True)
                if not np.isfinite(value).all():
                    raise ValueError(f"{name} contains non-finite values.")
            elif name in bool_names:
                if value.dtype != np.bool_:
                    raise TypeError(f"{name} must have bool dtype.")
                value = np.array(value, dtype=np.bool_, order="C", copy=True)
            else:
                if value.dtype != np.int64:
                    raise TypeError("instance_ids must have int64 dtype.")
                value = np.array(value, dtype=np.int64, order="C", copy=True)
            arrays[name] = value
        positions = arrays["positions_court_m"]
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError(
                "positions_court_m must have shape [frames, objects, 3], "
                f"got {positions.shape}."
            )
        return arrays


def build_blcs_gaussian_plan_from_scene(
    scene: BLCSSceneLike,
    *,
    registry: BallAssetRegistry,
    seed: int,
    scene_from_court: SimilarityTransform,
    cameras: Sequence[SceneCamera],
) -> BLCSGaussianScenePlan:
    """Build stable placements and OpenCV projections from physical BLCS output."""
    positions = _tensor_to_float64(scene.ball_pos_world, name="ball_pos_world")
    velocities = _tensor_to_float64(scene.ball_vel_world, name="ball_vel_world")
    if velocities.shape != positions.shape:
        raise ValueError(
            f"Velocity shape {velocities.shape} must match positions {positions.shape}."
        )
    if positions.ndim == 2:
        if positions.shape[1:] != (3,):
            raise ValueError(
                f"Single-ball positions must have shape [T, 3], got {positions.shape}."
            )
        positions = positions[:, None, :]
        velocities = velocities[:, None, :]
        if scene.ball_present is not None:
            raise ValueError("Single-ball scenes must not provide ball_present.")
        if (
            isinstance(scene.num_balls, bool)
            or not isinstance(scene.num_balls, int)
            or scene.num_balls != 1
        ):
            raise ValueError("Single-ball scenes must declare num_balls=1.")
        present = np.ones(positions.shape[:2], dtype=np.bool_)
    elif positions.ndim == 3:
        if positions.shape[-1] != 3:
            raise ValueError(
                f"Multi-ball positions must have shape [T, O, 3], got {positions.shape}."
            )
        if scene.ball_present is None:
            raise ValueError("Multi-ball scenes require explicit ball_present.")
        present = _tensor_to_bool(scene.ball_present, name="ball_present")
        if present.shape != positions.shape[:2]:
            raise ValueError(
                f"Presence shape {present.shape} must match {positions.shape[:2]}."
            )
        if (
            isinstance(scene.num_balls, bool)
            or not isinstance(scene.num_balls, int)
            or scene.num_balls <= 0
            or scene.num_balls > positions.shape[1]
        ):
            raise ValueError(
                "num_balls must select a positive number of available object columns."
            )
        positions = positions[:, : scene.num_balls]
        velocities = velocities[:, : scene.num_balls]
        present = present[:, : scene.num_balls]
    else:
        raise ValueError(
            "ball_pos_world must be a single [T, 3] or multi [T, O, 3] tensor."
        )
    fps = _finite_float(scene.fps_out, name="fps_out")
    if fps <= 0.0:
        raise ValueError("fps_out must be positive.")
    camera_tuple = tuple(cameras)
    if not camera_tuple:
        raise ValueError("At least one renderer camera is required.")

    object_count = positions.shape[1]
    assignments = tuple(
        BallAssetAssignment(
            instance_id=object_index + 1,
            selection=select_ball_asset(
                registry,
                seed=seed,
                selection_key=f"{scene.scene_id}:object:{object_index}",
            ),
        )
        for object_index in range(object_count)
    )
    positions_scene = scene_from_court.apply(positions)
    scene_from_asset = _placement_matrices(
        positions_scene=positions_scene,
        scene_from_court=scene_from_court,
    )
    camera_uv, camera_depth, geometric_visible = _project_scene_points(
        positions_scene=positions_scene,
        present=present,
        cameras=camera_tuple,
    )
    return BLCSGaussianScenePlan.create(
        scene_id=scene.scene_id,
        seed=seed,
        fps=fps,
        registry=registry,
        scene_from_court=scene_from_court,
        assignments=assignments,
        cameras=camera_tuple,
        positions_court_m=positions,
        velocities_court_mps=velocities,
        present=present,
        positions_scene=positions_scene,
        scene_from_asset=scene_from_asset,
        camera_uv=camera_uv,
        camera_depth=camera_depth,
        camera_geometric_visible=geometric_visible,
        instance_ids=np.arange(1, object_count + 1, dtype=np.int64),
    )


def compute_blcs_gaussian_plan_fingerprint(
    *,
    scene_id: str,
    seed: int,
    fps: float,
    registry_fingerprint: str,
    scene_from_court: SimilarityTransform,
    assignments: Sequence[BallAssetAssignment],
    cameras: Sequence[SceneCamera],
    arrays: Mapping[str, NDArray[np.generic]],
) -> str:
    """Hash trajectory, assignments, cameras, transforms, and all labels."""
    payload = {
        "schema": BLCS_GAUSSIAN_PLAN_SCHEMA,
        "scene_id": scene_id,
        "seed": seed,
        "fps": fps,
        "registry_fingerprint": registry_fingerprint,
        "coordinate_systems": {
            "court": COURT_AXES_METRES,
            "camera": CAMERA_AXES_OPENCV,
            "pixels": PIXEL_COORDINATES,
        },
        "scene_from_court": scene_from_court.to_dict(),
        "assignments": [assignment.to_dict() for assignment in assignments],
        "cameras": [camera.to_dict() for camera in cameras],
        "arrays": {
            name: _array_content_identity(arrays[name])
            for name in sorted(_ARRAY_FILENAMES)
        },
        "visibility_semantics": (
            "geometric_in_frame_only; native-render occlusion is a separate "
            "required render-stage label"
        ),
    }
    return _canonical_sha256(payload)


def write_blcs_gaussian_plan(
    output_dir: Path,
    plan: BLCSGaussianScenePlan,
) -> Path:
    """Atomically publish deterministic arrays and manifest without overwrite."""
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite BLCS plan: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    try:
        array_records: dict[str, dict[str, object]] = {}
        for name, filename in _ARRAY_FILENAMES.items():
            path = temporary / filename
            np.save(path, getattr(plan, name), allow_pickle=False)
            _fsync_file(path)
            array_records[name] = _published_array_record(
                path=path,
                relative_path=filename,
            )
        manifest = {
            "schema": plan.schema,
            "plan_fingerprint": plan.plan_fingerprint,
            "scene_id": plan.scene_id,
            "seed": plan.seed,
            "fps": plan.fps,
            "num_frames": plan.num_frames,
            "num_objects": plan.num_objects,
            "registry": plan.registry.to_dict(),
            "coordinate_systems": {
                "court": COURT_AXES_METRES,
                "camera": CAMERA_AXES_OPENCV,
                "pixels": PIXEL_COORDINATES,
            },
            "scene_from_court": plan.scene_from_court.to_dict(),
            "scene_from_asset_semantics": _SCENE_FROM_ASSET_SEMANTICS,
            "assignments": [assignment.to_dict() for assignment in plan.assignments],
            "cameras": [camera.to_dict() for camera in plan.cameras],
            "arrays": array_records,
            "visibility_semantics": _GEOMETRIC_VISIBILITY_SEMANTICS,
            "render_stage_complete": False,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        _fsync_file(manifest_path)
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_blcs_gaussian_plan_output(destination)
    return destination / "manifest.json"


def load_blcs_gaussian_plan(output_dir: Path) -> BLCSGaussianScenePlan:
    """Strictly load one published pre-render plan and verify every input byte."""
    root = output_dir.resolve()
    manifest_path = root / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    raw = _strict_mapping(
        value,
        name="BLCS plan manifest",
        keys={
            "schema",
            "plan_fingerprint",
            "scene_id",
            "seed",
            "fps",
            "num_frames",
            "num_objects",
            "registry",
            "coordinate_systems",
            "scene_from_court",
            "scene_from_asset_semantics",
            "assignments",
            "cameras",
            "arrays",
            "visibility_semantics",
            "render_stage_complete",
        },
    )
    if raw["schema"] != BLCS_GAUSSIAN_PLAN_SCHEMA:
        raise ValueError(f"Unsupported BLCS plan schema: {raw['schema']!r}.")
    if raw["render_stage_complete"] is not False:
        raise ValueError("Pre-render BLCS plan must not claim render completion.")
    if raw["scene_from_asset_semantics"] != _SCENE_FROM_ASSET_SEMANTICS:
        raise ValueError("BLCS placement semantics differ from the v2 contract.")
    if raw["visibility_semantics"] != _GEOMETRIC_VISIBILITY_SEMANTICS:
        raise ValueError("BLCS visibility semantics differ from the v2 contract.")
    arrays = raw["arrays"]
    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must be a JSON object.")
    if set(arrays) != set(_ARRAY_FILENAMES):
        raise ValueError("BLCS plan array names differ from the v1 schema.")
    loaded: dict[str, NDArray[np.generic]] = {}
    for name, record_value in arrays.items():
        record = _strict_mapping(
            record_value,
            name=f"array record {name}",
            keys={"relative_path", "sha256", "size_bytes", "dtype", "shape"},
        )
        relative_path = _relative_path(record["relative_path"])
        path = (root / relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Invalid or missing BLCS label array: {relative_path}")
        if path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"BLCS label array size mismatch: {relative_path}")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"BLCS label array hash mismatch: {relative_path}")
        array = np.load(path, allow_pickle=False)
        if str(array.dtype) != record["dtype"]:
            raise ValueError(f"BLCS label array dtype mismatch: {relative_path}")
        if list(array.shape) != record["shape"]:
            raise ValueError(f"BLCS label array shape mismatch: {relative_path}")
        loaded[str(name)] = array
    num_frames = _integer(raw["num_frames"], name="num_frames")
    num_objects = _integer(raw["num_objects"], name="num_objects")
    if list(loaded["positions_court_m"].shape[:2]) != [num_frames, num_objects]:
        raise ValueError("BLCS plan counts do not match trajectory arrays.")
    scene_id = _string(raw["scene_id"], name="scene_id")
    seed = _integer(raw["seed"], name="seed")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    fps = _finite_float(raw["fps"], name="fps")
    registry = BallAssetRegistry.from_dict(raw["registry"])
    verify_local_ball_asset_registry(registry)
    coordinate_systems = _strict_mapping(
        raw["coordinate_systems"],
        name="coordinate_systems",
        keys={"court", "camera", "pixels"},
    )
    expected_coordinate_systems = {
        "court": COURT_AXES_METRES,
        "camera": CAMERA_AXES_OPENCV,
        "pixels": PIXEL_COORDINATES,
    }
    if coordinate_systems != expected_coordinate_systems:
        raise ValueError("BLCS plan coordinate systems differ from the v1 contract.")
    assignments_value = raw["assignments"]
    if not isinstance(assignments_value, Sequence) or isinstance(
        assignments_value,
        (str, bytes),
    ):
        raise TypeError("assignments must be a JSON array.")
    assignments = tuple(_assignment_from_dict(value) for value in assignments_value)
    cameras_value = raw["cameras"]
    if not isinstance(cameras_value, Sequence) or isinstance(
        cameras_value,
        (str, bytes),
    ):
        raise TypeError("cameras must be a JSON array.")
    cameras = tuple(SceneCamera.from_dict(value) for value in cameras_value)
    if len(assignments) != num_objects:
        raise ValueError("Assignment count does not match num_objects.")
    for object_index, assignment in enumerate(assignments):
        expected_selection = select_ball_asset(
            registry,
            seed=seed,
            selection_key=f"{scene_id}:object:{object_index}",
        )
        if assignment.instance_id != object_index + 1:
            raise ValueError("Assignment instance IDs must match object columns.")
        if assignment.selection != expected_selection:
            raise ValueError(
                "Published asset selection does not match its explicit registry."
            )
    declared_fingerprint = _sha256(
        _string(raw["plan_fingerprint"], name="plan_fingerprint"),
        name="plan_fingerprint",
    )
    computed_fingerprint = compute_blcs_gaussian_plan_fingerprint(
        scene_id=scene_id,
        seed=seed,
        fps=fps,
        registry_fingerprint=registry.registry_fingerprint,
        scene_from_court=SimilarityTransform.from_dict(raw["scene_from_court"]),
        assignments=assignments,
        cameras=cameras,
        arrays=loaded,
    )
    if declared_fingerprint != computed_fingerprint:
        raise ValueError(
            "BLCS plan fingerprint mismatch: "
            f"declared {declared_fingerprint}, computed {computed_fingerprint}."
        )
    verified_plan = BLCSGaussianScenePlan(
        schema=BLCS_GAUSSIAN_PLAN_SCHEMA,
        plan_fingerprint=declared_fingerprint,
        scene_id=scene_id,
        seed=seed,
        fps=fps,
        registry=registry,
        scene_from_court=SimilarityTransform.from_dict(raw["scene_from_court"]),
        assignments=assignments,
        cameras=cameras,
        positions_court_m=cast(
            NDArray[np.float64],
            loaded["positions_court_m"],
        ),
        velocities_court_mps=cast(
            NDArray[np.float64],
            loaded["velocities_court_mps"],
        ),
        present=cast(NDArray[np.bool_], loaded["present"]),
        positions_scene=cast(NDArray[np.float64], loaded["positions_scene"]),
        scene_from_asset=cast(
            NDArray[np.float64],
            loaded["scene_from_asset"],
        ),
        camera_uv=cast(NDArray[np.float64], loaded["camera_uv"]),
        camera_depth=cast(NDArray[np.float64], loaded["camera_depth"]),
        camera_geometric_visible=cast(
            NDArray[np.bool_],
            loaded["camera_geometric_visible"],
        ),
        instance_ids=cast(NDArray[np.int64], loaded["instance_ids"]),
    )
    return verified_plan


def verify_blcs_gaussian_plan_output(output_dir: Path) -> dict[str, object]:
    """Verify one published plan and return a compact evidence summary."""
    plan = load_blcs_gaussian_plan(output_dir)
    return {
        "plan_fingerprint": plan.plan_fingerprint,
        "num_frames": plan.num_frames,
        "num_objects": plan.num_objects,
        "num_cameras": len(plan.cameras),
        "geometric_visible_count": int(plan.camera_geometric_visible.sum()),
        "render_stage_complete": False,
    }


def _placement_matrices(
    *,
    positions_scene: NDArray[np.float64],
    scene_from_court: SimilarityTransform,
) -> NDArray[np.float64]:
    frame_count, object_count, _ = positions_scene.shape
    result = np.broadcast_to(
        np.eye(4, dtype=np.float64),
        (frame_count, object_count, 4, 4),
    ).copy()
    rotation = np.asarray(scene_from_court.rotation, dtype=np.float64).reshape(3, 3)
    result[..., :3, :3] = scene_from_court.scale * rotation
    result[..., :3, 3] = positions_scene
    return result


def _project_scene_points(
    *,
    positions_scene: NDArray[np.float64],
    present: NDArray[np.bool_],
    cameras: Sequence[SceneCamera],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    frame_count, object_count, _ = positions_scene.shape
    camera_count = len(cameras)
    uv = np.zeros((camera_count, frame_count, object_count, 2), dtype=np.float64)
    depth = np.zeros((camera_count, frame_count, object_count), dtype=np.float64)
    visible = np.zeros((camera_count, frame_count, object_count), dtype=np.bool_)
    homogeneous = np.concatenate(
        [
            positions_scene,
            np.ones((frame_count, object_count, 1), dtype=np.float64),
        ],
        axis=-1,
    )
    for camera_index, camera in enumerate(cameras):
        camera_to_scene = np.asarray(
            camera.camera_to_scene,
            dtype=np.float64,
        ).reshape(4, 4)
        scene_to_camera = np.linalg.inv(camera_to_scene)
        points_camera = homogeneous @ scene_to_camera.T
        z = points_camera[..., 2]
        positive_depth = z > 0.0
        safe_z = np.where(positive_depth, z, 1.0)
        intrinsics = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
        projected_homogeneous = points_camera[..., :3] @ intrinsics.T
        projected_u = projected_homogeneous[..., 0] / safe_z
        projected_v = projected_homogeneous[..., 1] / safe_z
        projected = np.stack([projected_u, projected_v], axis=-1)
        in_frame = (
            (projected[..., 0] >= 0.0)
            & (projected[..., 0] < camera.width)
            & (projected[..., 1] >= 0.0)
            & (projected[..., 1] < camera.height)
        )
        uv[camera_index] = projected
        depth[camera_index] = z
        visible[camera_index] = present & positive_depth & in_frame
    return uv, depth, visible


def _tensor_to_float64(
    value: torch.Tensor,
    *,
    name: str,
) -> NDArray[np.float64]:
    array = value.detach().cpu().numpy()
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must be floating point.")
    result = np.ascontiguousarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values.")
    return result


def _tensor_to_bool(
    value: torch.Tensor,
    *,
    name: str,
) -> NDArray[np.bool_]:
    array = value.detach().cpu().numpy()
    if array.dtype != np.bool_:
        raise TypeError(f"{name} must have bool dtype.")
    return np.ascontiguousarray(array, dtype=np.bool_)


def _array_content_identity(array: NDArray[np.generic]) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    return {
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
    }


def _published_array_record(
    *,
    path: Path,
    relative_path: str,
) -> dict[str, object]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "relative_path": relative_path,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }


def _assignment_from_dict(value: object) -> BallAssetAssignment:
    raw = _strict_mapping(
        value,
        name="ball assignment",
        keys={
            "instance_id",
            "variant_id",
            "entry_index",
            "selection_sha256",
            "nominal_diameter_m",
            "asset",
        },
    )
    entry = BallAssetEntry(
        variant_id=_string(raw["variant_id"], name="variant_id"),
        nominal_diameter_m=_finite_float(
            raw["nominal_diameter_m"],
            name="nominal_diameter_m",
        ),
        asset=GaussianAsset.from_dict(raw["asset"]),
    )
    return BallAssetAssignment(
        instance_id=_integer(raw["instance_id"], name="instance_id"),
        selection=BallAssetSelection(
            entry=entry,
            entry_index=_integer(raw["entry_index"], name="entry_index"),
            selection_sha256=_string(
                raw["selection_sha256"],
                name="selection_sha256",
            ),
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str):
        raise TypeError("relative_path must be a string.")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"Invalid relative artifact path: {value!r}.")
    return path


def _strict_mapping(
    value: object,
    *,
    name: str,
    keys: set[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object.")
    raw = {str(key): item for key, item in value.items()}
    missing = keys.difference(raw)
    extra = set(raw).difference(keys)
    if missing or extra:
        raise ValueError(
            f"{name} keys differ; missing={sorted(missing)}, extra={sorted(extra)}."
        )
    return raw


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string.")
    return value


def _sha256(value: str, *, name: str) -> str:
    digest = value.lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a full SHA-256 digest.")
    return digest


def _require_unique(values: Sequence[object], *, name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates.")
