"""Typed full-timeline contracts for the canonical BLCS dataset stage."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from src.synthetic_data_generation.composition import (
    GaussianAsset,
    GaussianDeformationKind,
    GaussianFrame,
    GaussianSceneObject,
)

BLCS_DATASET_SCHEMA = "canonical_blcs_compact_dataset_v4"
BLCS_DATASET_SCHEMA_V3 = "canonical_blcs_compact_dataset_v3"
BLCS_SAMPLE_SCHEMA = "canonical_blcs_compact_sample_v1"
BLCS_BALL_ASSET_SCHEMA = "canonical_blcs_ball_asset_v1"
BLCS_BALL_COMPOSITION_SCHEMA = "canonical_blcs_ball_composition_v1"

_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ParsedT = TypeVar("_ParsedT")


class BLCSSceneLike(Protocol):
    """Physical BLCS scene fields accepted from the existing source generator."""

    scene_id: str
    ball_pos_world: Tensor
    ball_vel_world: Tensor
    ball_present: Tensor | None
    num_balls: int
    fps_out: int
    track_instances: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class BLCSTrack:
    """One stable ball identity and its lossless source-frame mapping."""

    object_id: str
    source_trajectory_id: str
    source_frame_indices: tuple[int | None, ...]

    def __post_init__(self) -> None:
        _identifier(self.object_id, name="object_id")
        _identifier(self.source_trajectory_id, name="source_trajectory_id")
        if not self.source_frame_indices:
            raise ValueError("source_frame_indices must not be empty.")
        active = [value for value in self.source_frame_indices if value is not None]
        if not active:
            raise ValueError(f"BLCS track {self.object_id!r} is never present.")
        if any(isinstance(value, bool) or value < 0 for value in active):
            raise ValueError("Source frame indices must be non-negative integers.")
        if active != list(range(active[0], active[0] + len(active))):
            raise ValueError(
                f"BLCS track {self.object_id!r} source frames are not consecutive."
            )
        active_global = [
            index
            for index, value in enumerate(self.source_frame_indices)
            if value is not None
        ]
        if active_global != list(range(active_global[0], active_global[-1] + 1)):
            raise ValueError(
                f"BLCS track {self.object_id!r} presence is not one continuous interval."
            )

    def to_dict(self) -> dict[str, object]:
        """Return the complete source mapping without an identity digest."""
        return {
            "object_id": self.object_id,
            "source_trajectory_id": self.source_trajectory_id,
            "source_frame_indices": list(self.source_frame_indices),
        }


@dataclass(frozen=True, slots=True)
class BLCSTrajectory:
    """One complete physics trajectory before court placement or rendering."""

    trajectory_id: str
    split: str
    fps: float
    positions_court_m: NDArray[np.float64]
    velocities_court_mps: NDArray[np.float64]
    present: NDArray[np.bool_]
    tracks: tuple[BLCSTrack, ...]
    source_metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        _identifier(self.trajectory_id, name="trajectory_id")
        if not self.split.strip() or self.split != self.split.strip():
            raise ValueError("split must be a non-empty trimmed string.")
        fps = _positive_float(self.fps, name="fps")
        positions = _float64_array(self.positions_court_m, name="positions_court_m")
        velocities = _float64_array(
            self.velocities_court_mps, name="velocities_court_mps"
        )
        present = np.asarray(self.present)
        if present.dtype != np.bool_:
            raise TypeError("present must use bool dtype.")
        present = np.array(present, dtype=np.bool_, order="C", copy=True)
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("positions_court_m must have shape [T, O, 3].")
        if velocities.shape != positions.shape:
            raise ValueError("velocities_court_mps must match positions_court_m.")
        if present.shape != positions.shape[:2]:
            raise ValueError("present must have shape [T, O].")
        if positions.shape[0] <= 0 or positions.shape[1] <= 0:
            raise ValueError("BLCS trajectories require at least one frame and object.")
        tracks = tuple(self.tracks)
        if len(tracks) != positions.shape[1]:
            raise ValueError("BLCS track count must match the object axis.")
        if len({track.object_id for track in tracks}) != len(tracks):
            raise ValueError("BLCS object_id values must be unique.")
        for object_index, track in enumerate(tracks):
            if len(track.source_frame_indices) != positions.shape[0]:
                raise ValueError(
                    "Every BLCS source-frame map must cover all global frames."
                )
            mapped_presence = np.asarray(
                [value is not None for value in track.source_frame_indices],
                dtype=np.bool_,
            )
            if not np.array_equal(mapped_presence, present[:, object_index]):
                raise ValueError(
                    f"Presence disagrees with source mapping for {track.object_id!r}."
                )
        metadata = _json_mapping(self.source_metadata, name="source_metadata")
        for array in (positions, velocities, present):
            array.setflags(write=False)
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "positions_court_m", positions)
        object.__setattr__(self, "velocities_court_mps", velocities)
        object.__setattr__(self, "present", present)
        object.__setattr__(self, "tracks", tracks)
        object.__setattr__(self, "source_metadata", metadata)

    @property
    def frame_count(self) -> int:
        """Return the full source trajectory length."""
        return int(self.positions_court_m.shape[0])

    @property
    def object_count(self) -> int:
        """Return the number of stable ball identities."""
        return int(self.positions_court_m.shape[1])

    @classmethod
    def from_scene(
        cls,
        scene: BLCSSceneLike,
        *,
        split: str,
    ) -> BLCSTrajectory:
        """Adapt a physical BLCS scene without truncating or reordering frames."""
        positions = _source_array(scene.ball_pos_world, name="ball_pos_world")
        velocities = _source_array(scene.ball_vel_world, name="ball_vel_world")
        if positions.ndim == 2:
            if positions.shape[1:] != (3,):
                raise ValueError("Single-ball positions must have shape [T, 3].")
            if scene.ball_present is not None or scene.num_balls != 1:
                raise ValueError(
                    "Single-ball scenes require num_balls=1 and no ball_present array."
                )
            positions = positions[:, None, :]
            velocities = velocities[:, None, :]
            present = np.ones(positions.shape[:2], dtype=np.bool_)
        elif positions.ndim == 3:
            if positions.shape[-1] != 3:
                raise ValueError("Multi-ball positions must have shape [T, O, 3].")
            if (
                isinstance(scene.num_balls, bool)
                or not isinstance(scene.num_balls, int)
                or not 0 < scene.num_balls <= positions.shape[1]
                or scene.ball_present is None
            ):
                raise ValueError(
                    "Multi-ball scenes require a valid num_balls and ball_present array."
                )
            positions = positions[:, : scene.num_balls]
            velocities = velocities[:, : scene.num_balls]
            present_value = _source_array(scene.ball_present, name="ball_present")
            if present_value.dtype != np.bool_:
                raise TypeError("ball_present must use bool dtype.")
            present = np.asarray(present_value[:, : scene.num_balls], dtype=np.bool_)
        else:
            raise ValueError("ball_pos_world must have shape [T, 3] or [T, O, 3].")
        if velocities.shape != positions.shape:
            raise ValueError("ball_vel_world must match ball_pos_world shape.")
        tracks = _tracks_from_scene(
            trajectory_id=scene.scene_id,
            frame_count=int(positions.shape[0]),
            object_count=int(positions.shape[1]),
            present=present,
            placements=scene.track_instances,
        )
        return cls(
            trajectory_id=scene.scene_id,
            split=split,
            fps=float(scene.fps_out),
            positions_court_m=positions,
            velocities_court_mps=velocities,
            present=present,
            tracks=tracks,
            source_metadata={
                "generator": "blcs_physics",
                "source_scene": scene.scene_id,
            },
        )


@dataclass(frozen=True, slots=True)
class BLCSBallGaussianSettings:
    """Physical surface and visibility settings for one tennis ball asset."""

    radius_m: float
    radial_scale_m: float
    tangential_scale_m: float
    opacity: float
    base_color_linear_rgb: tuple[float, float, float]
    seam_color_linear_rgb: tuple[float, float, float]
    seam_width_radians: float
    visibility_threshold: float

    def __post_init__(self) -> None:
        radius = _positive_float(self.radius_m, name="radius_m")
        radial = _positive_float(self.radial_scale_m, name="radial_scale_m")
        tangential = _positive_float(
            self.tangential_scale_m,
            name="tangential_scale_m",
        )
        if radial >= radius or tangential >= radius:
            raise ValueError("Ball Gaussian scales must be smaller than the ball radius.")
        opacity = _unit_interval(self.opacity, name="opacity", open_interval=True)
        base = _linear_rgb(self.base_color_linear_rgb, name="base_color_linear_rgb")
        seam = _linear_rgb(self.seam_color_linear_rgb, name="seam_color_linear_rgb")
        seam_width = _positive_float(
            self.seam_width_radians,
            name="seam_width_radians",
        )
        if seam_width >= math.pi / 2.0:
            raise ValueError("seam_width_radians must be smaller than pi/2.")
        visibility = _unit_interval(
            self.visibility_threshold,
            name="visibility_threshold",
            open_interval=True,
        )
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "radial_scale_m", radial)
        object.__setattr__(self, "tangential_scale_m", tangential)
        object.__setattr__(self, "opacity", opacity)
        object.__setattr__(self, "base_color_linear_rgb", base)
        object.__setattr__(self, "seam_color_linear_rgb", seam)
        object.__setattr__(self, "seam_width_radians", seam_width)
        object.__setattr__(self, "visibility_threshold", visibility)


class BLCSBallRendering(StrEnum):
    """Explicit rendering implementation selected for the physical ball asset."""

    GAUSSIAN = "gaussian"
    MESH = "mesh"


@dataclass(frozen=True, slots=True)
class BLCSBallMeshAsset:
    """One validated data-root-owned GLB mesh source."""

    path: Path
    data_root_relative_path: str
    maximum_file_bytes: int
    maximum_source_vertices: int
    maximum_source_faces: int
    maximum_faces: int

    def __post_init__(self) -> None:
        path = self.path
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("BLCS mesh asset path must be an absolute pathlib.Path.")
        if path.suffix.lower() != ".glb":
            raise ValueError("BLCS mesh asset must use the .glb format.")
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(
                f"BLCS mesh asset must be an ordinary existing .glb file: {path}"
            )
        relative = _relative_file(
            self.data_root_relative_path,
            name="data_root_relative_path",
        )
        if Path(relative).suffix.lower() != ".glb":
            raise ValueError("BLCS mesh asset relative path must end in .glb.")
        if (
            isinstance(self.maximum_file_bytes, bool)
            or not isinstance(self.maximum_file_bytes, int)
            or self.maximum_file_bytes < 1
        ):
            raise ValueError("BLCS mesh maximum_file_bytes must be a positive integer.")
        if (
            isinstance(self.maximum_source_vertices, bool)
            or not isinstance(self.maximum_source_vertices, int)
            or self.maximum_source_vertices < 4
        ):
            raise ValueError(
                "BLCS mesh maximum_source_vertices must be an integer >= 4."
            )
        if (
            isinstance(self.maximum_source_faces, bool)
            or not isinstance(self.maximum_source_faces, int)
            or self.maximum_source_faces < 4
        ):
            raise ValueError("BLCS mesh maximum_source_faces must be an integer >= 4.")
        if (
            isinstance(self.maximum_faces, bool)
            or not isinstance(self.maximum_faces, int)
            or self.maximum_faces < 4
        ):
            raise ValueError("BLCS mesh maximum_faces must be an integer >= 4.")
        if path.stat().st_size > self.maximum_file_bytes:
            raise ValueError(
                "BLCS mesh asset exceeds its configured maximum_file_bytes limit."
            )
        object.__setattr__(self, "path", path.resolve(strict=True))
        object.__setattr__(self, "data_root_relative_path", relative)


@dataclass(frozen=True, slots=True)
class BLCSBallAssetMetadata:
    """Serializable authority for the exact ball geometry used by a plan."""

    rendering: BLCSBallRendering
    asset_id: str
    radius_m: float
    gaussian: GaussianAsset | None = None
    mesh_data_root_relative_path: str | None = None
    mesh_maximum_file_bytes: int | None = None
    mesh_maximum_source_vertices: int | None = None
    mesh_maximum_source_faces: int | None = None
    mesh_maximum_faces: int | None = None
    schema: str = BLCS_BALL_ASSET_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BLCS_BALL_ASSET_SCHEMA:
            raise ValueError(f"Unsupported BLCS ball asset schema: {self.schema!r}.")
        if not isinstance(self.rendering, BLCSBallRendering):
            raise TypeError("rendering must be BLCSBallRendering.")
        _identifier(self.asset_id, name="asset_id")
        radius = _positive_float(self.radius_m, name="radius_m")
        if self.rendering is BLCSBallRendering.GAUSSIAN:
            if not isinstance(self.gaussian, GaussianAsset):
                raise TypeError("Gaussian BLCS metadata requires one GaussianAsset.")
            if self.gaussian.asset_id != self.asset_id:
                raise ValueError("BLCS Gaussian metadata asset IDs disagree.")
            if (
                self.mesh_data_root_relative_path is not None
                or self.mesh_maximum_file_bytes is not None
                or self.mesh_maximum_source_vertices is not None
                or self.mesh_maximum_source_faces is not None
                or self.mesh_maximum_faces is not None
            ):
                raise ValueError("Gaussian BLCS metadata cannot declare a mesh source.")
        else:
            if self.gaussian is not None:
                raise ValueError("Mesh BLCS metadata cannot contain Gaussian metadata.")
            relative = _relative_file(
                self.mesh_data_root_relative_path,
                name="mesh_data_root_relative_path",
            )
            if Path(relative).suffix.lower() != ".glb":
                raise ValueError("BLCS mesh metadata path must end in .glb.")
            if (
                isinstance(self.mesh_maximum_file_bytes, bool)
                or not isinstance(self.mesh_maximum_file_bytes, int)
                or self.mesh_maximum_file_bytes < 1
            ):
                raise ValueError(
                    "BLCS mesh metadata maximum_file_bytes must be positive."
                )
            if (
                isinstance(self.mesh_maximum_source_vertices, bool)
                or not isinstance(self.mesh_maximum_source_vertices, int)
                or self.mesh_maximum_source_vertices < 4
            ):
                raise ValueError(
                    "BLCS mesh metadata maximum_source_vertices must be >= 4."
                )
            if (
                isinstance(self.mesh_maximum_source_faces, bool)
                or not isinstance(self.mesh_maximum_source_faces, int)
                or self.mesh_maximum_source_faces < 4
            ):
                raise ValueError(
                    "BLCS mesh metadata maximum_source_faces must be >= 4."
                )
            if (
                isinstance(self.mesh_maximum_faces, bool)
                or not isinstance(self.mesh_maximum_faces, int)
                or self.mesh_maximum_faces < 4
            ):
                raise ValueError("BLCS mesh metadata maximum_faces must be >= 4.")
            object.__setattr__(self, "mesh_data_root_relative_path", relative)
        object.__setattr__(self, "radius_m", radius)

    def to_dict(self) -> dict[str, object]:
        """Return the strict source record persisted in every trajectory plan."""
        source: dict[str, object]
        if self.rendering is BLCSBallRendering.GAUSSIAN:
            assert self.gaussian is not None
            source = {"gaussian": self.gaussian.to_dict()}
        else:
            assert self.mesh_data_root_relative_path is not None
            assert self.mesh_maximum_file_bytes is not None
            assert self.mesh_maximum_source_vertices is not None
            assert self.mesh_maximum_source_faces is not None
            assert self.mesh_maximum_faces is not None
            source = {
                "format": "glb",
                "appearance_model": "glb_base_color_lambertian_v1",
                "data_root_relative_path": self.mesh_data_root_relative_path,
                "maximum_file_bytes": self.mesh_maximum_file_bytes,
                "maximum_source_vertices": self.mesh_maximum_source_vertices,
                "maximum_source_faces": self.mesh_maximum_source_faces,
                "maximum_faces": self.mesh_maximum_faces,
            }
        return {
            "schema": self.schema,
            "rendering": self.rendering.value,
            "asset_id": self.asset_id,
            "asset_class": "ball",
            "coordinate_space": "right_handed_asset_local_metres",
            "radius_m": self.radius_m,
            "source": source,
        }

    @classmethod
    def from_dict(cls, value: object) -> BLCSBallAssetMetadata:
        """Parse the strict plan metadata without reopening the source asset."""
        raw = _strict_mapping(
            value,
            name="BLCS ball asset metadata",
            keys={
                "schema",
                "rendering",
                "asset_id",
                "asset_class",
                "coordinate_space",
                "radius_m",
                "source",
            },
        )
        if raw["asset_class"] != "ball":
            raise ValueError("BLCS asset metadata requires asset_class='ball'.")
        if raw["coordinate_space"] != "right_handed_asset_local_metres":
            raise ValueError("BLCS asset metadata uses an unknown coordinate space.")
        try:
            rendering = BLCSBallRendering(
                _string_value(raw["rendering"], name="rendering")
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "BLCS asset metadata has an unknown rendering mode."
            ) from error
        source = _strict_mapping(
            raw["source"],
            name="BLCS ball asset source",
            keys={"gaussian"}
            if rendering is BLCSBallRendering.GAUSSIAN
            else {
                "format",
                "appearance_model",
                "data_root_relative_path",
                "maximum_file_bytes",
                "maximum_source_vertices",
                "maximum_source_faces",
                "maximum_faces",
            },
        )
        if rendering is BLCSBallRendering.GAUSSIAN:
            return cls(
                schema=_string_value(raw["schema"], name="schema"),
                rendering=rendering,
                asset_id=_string_value(raw["asset_id"], name="asset_id"),
                radius_m=_positive_float(raw["radius_m"], name="radius_m"),
                gaussian=GaussianAsset.from_dict(source["gaussian"]),
            )
        if source["format"] != "glb":
            raise ValueError("BLCS mesh metadata requires format='glb'.")
        if source["appearance_model"] != "glb_base_color_lambertian_v1":
            raise ValueError("BLCS mesh metadata has an unknown appearance model.")
        return cls(
            schema=_string_value(raw["schema"], name="schema"),
            rendering=rendering,
            asset_id=_string_value(raw["asset_id"], name="asset_id"),
            radius_m=_positive_float(raw["radius_m"], name="radius_m"),
            mesh_data_root_relative_path=_string_value(
                source["data_root_relative_path"], name="data_root_relative_path"
            ),
            mesh_maximum_file_bytes=_positive_integer(
                source["maximum_file_bytes"], name="maximum_file_bytes"
            ),
            mesh_maximum_source_vertices=_positive_integer(
                source["maximum_source_vertices"], name="maximum_source_vertices"
            ),
            mesh_maximum_source_faces=_positive_integer(
                source["maximum_source_faces"], name="maximum_source_faces"
            ),
            mesh_maximum_faces=_positive_integer(
                source["maximum_faces"], name="maximum_faces"
            ),
        )


@dataclass(frozen=True, slots=True)
class BLCSCompositionAssets:
    """The explicit Gaussian or GLB ball asset used by every BLCS render."""

    ball: GaussianAsset
    settings: BLCSBallGaussianSettings
    rendering: BLCSBallRendering = BLCSBallRendering.GAUSSIAN
    mesh: BLCSBallMeshAsset | None = None

    def __post_init__(self) -> None:
        from src.synthetic_data_generation.composition import GaussianAssetRole

        if self.ball.role is not GaussianAssetRole.MOVABLE:
            raise ValueError("BLCS ball asset must have role=movable.")
        if self.ball.asset_class != "ball":
            raise ValueError("BLCS movable asset must declare asset_class='ball'.")
        if self.ball.feature_dim != 3:
            raise ValueError("BLCS ball assets must carry explicit linear RGB features.")
        if self.ball.floating_dtype != "float32":
            raise ValueError("BLCS ball assets must use float32 for the public NHT boundary.")
        if self.ball.appearance_model != "rgb" or self.ball.appearance_space != "linear_rgb":
            raise ValueError("BLCS ball assets must use the rgb/linear_rgb appearance contract.")
        if not isinstance(self.settings, BLCSBallGaussianSettings):
            raise TypeError("settings must be BLCSBallGaussianSettings.")
        if not isinstance(self.rendering, BLCSBallRendering):
            raise TypeError("rendering must be BLCSBallRendering.")
        if self.rendering is BLCSBallRendering.GAUSSIAN:
            if self.mesh is not None:
                raise ValueError("Gaussian BLCS rendering cannot declare a mesh asset.")
        elif not isinstance(self.mesh, BLCSBallMeshAsset):
            raise TypeError("Mesh BLCS rendering requires one BLCSBallMeshAsset.")

    def metadata(self) -> BLCSBallAssetMetadata:
        """Return the serializable render-source contract for trajectory plans."""
        if self.rendering is BLCSBallRendering.GAUSSIAN:
            return BLCSBallAssetMetadata(
                rendering=self.rendering,
                asset_id=self.ball.asset_id,
                radius_m=self.settings.radius_m,
                gaussian=self.ball,
            )
        assert self.mesh is not None
        return BLCSBallAssetMetadata(
            rendering=self.rendering,
            asset_id=self.ball.asset_id,
            radius_m=self.settings.radius_m,
            mesh_data_root_relative_path=self.mesh.data_root_relative_path,
            mesh_maximum_file_bytes=self.mesh.maximum_file_bytes,
            mesh_maximum_source_vertices=self.mesh.maximum_source_vertices,
            mesh_maximum_source_faces=self.mesh.maximum_source_faces,
            mesh_maximum_faces=self.mesh.maximum_faces,
        )


@dataclass(frozen=True, slots=True)
class BLCSBallComposition:
    """Renderer-neutral rigid ball placement timeline persisted by BLCS."""

    scene_id: str
    composition_id: str
    asset: BLCSBallAssetMetadata
    objects: tuple[GaussianSceneObject, ...]
    frames: tuple[GaussianFrame, ...]
    schema: str = BLCS_BALL_COMPOSITION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BLCS_BALL_COMPOSITION_SCHEMA:
            raise ValueError(
                f"Unsupported BLCS ball composition schema: {self.schema!r}."
            )
        _identifier(self.scene_id, name="scene_id")
        _identifier(self.composition_id, name="composition_id")
        if not isinstance(self.asset, BLCSBallAssetMetadata):
            raise TypeError("asset must be BLCSBallAssetMetadata.")
        objects = tuple(self.objects)
        frames = tuple(self.frames)
        if not objects or any(
            not isinstance(item, GaussianSceneObject) for item in objects
        ):
            raise TypeError("BLCS composition objects must be non-empty scene objects.")
        if not frames or any(not isinstance(frame, GaussianFrame) for frame in frames):
            raise TypeError(
                "BLCS composition frames must be non-empty GaussianFrame values."
            )
        if len({item.object_id for item in objects}) != len(objects):
            raise ValueError("BLCS composition object IDs must be unique.")
        if len({item.instance_id for item in objects}) != len(objects):
            raise ValueError("BLCS composition instance IDs must be unique.")
        if any(item.asset_id != self.asset.asset_id for item in objects):
            raise ValueError(
                "BLCS composition objects reference an unknown ball asset."
            )
        if any(
            item.deformation_kind is not GaussianDeformationKind.RIGID
            for item in objects
        ):
            raise ValueError("BLCS ball objects must use rigid deformation.")
        if tuple(frame.frame_index for frame in frames) != tuple(range(len(frames))):
            raise ValueError("BLCS composition frames must exactly equal 0..T-1.")
        object_ids = {item.object_id for item in objects}
        used: set[str] = set()
        for frame in frames:
            for instance in frame.instances:
                if instance.object_id not in object_ids:
                    raise ValueError("BLCS frame references an unknown ball object.")
                used.add(instance.object_id)
        if used != object_ids:
            raise ValueError(
                "Every declared BLCS ball object must appear in the timeline."
            )
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "frames", frames)

    def to_dict(self) -> dict[str, object]:
        """Return the renderer-neutral plan record."""
        return {
            "schema": self.schema,
            "scene_id": self.scene_id,
            "composition_id": self.composition_id,
            "asset": self.asset.to_dict(),
            "objects": [item.to_dict() for item in self.objects],
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @classmethod
    def from_dict(cls, value: object) -> BLCSBallComposition:
        """Parse one strict renderer-neutral BLCS composition."""
        raw = _strict_mapping(
            value,
            name="BLCS ball composition",
            keys={"schema", "scene_id", "composition_id", "asset", "objects", "frames"},
        )
        return cls(
            schema=_string_value(raw["schema"], name="schema"),
            scene_id=_string_value(raw["scene_id"], name="scene_id"),
            composition_id=_string_value(raw["composition_id"], name="composition_id"),
            asset=BLCSBallAssetMetadata.from_dict(raw["asset"]),
            objects=_typed_sequence(
                raw["objects"], GaussianSceneObject.from_dict, name="objects"
            ),
            frames=_typed_sequence(
                raw["frames"], GaussianFrame.from_dict, name="frames"
            ),
        )


@dataclass(frozen=True, slots=True)
class BLCSChunk:
    """One contiguous range written as a compact foreground chunk."""

    chunk_index: int
    frame_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.chunk_index, bool) or self.chunk_index < 0:
            raise ValueError("chunk_index must be a non-negative integer.")
        if not self.frame_indices:
            raise ValueError("BLCS chunks must not be empty.")
        first = self.frame_indices[0]
        if first < 0 or self.frame_indices != tuple(
            range(first, first + len(self.frame_indices))
        ):
            raise ValueError("BLCS chunk frame indices must be contiguous and ordered.")

    def to_dict(self) -> dict[str, object]:
        """Return the chunk's explicit global frame inventory."""
        return {
            "chunk_index": self.chunk_index,
            "frame_indices": list(self.frame_indices),
        }


@dataclass(frozen=True, slots=True)
class BLCSSampleRecord:
    """One logical sample backed by a shared background and compact delta."""

    trajectory_id: str
    split: str
    global_frame_index: int
    source_frame_index: int
    chunk_index: int
    camera_id: str
    background_store: str
    foreground_chunk: str
    chunk_sample_index: int

    def __post_init__(self) -> None:
        _identifier(self.trajectory_id, name="trajectory_id")
        _identifier(self.camera_id, name="camera_id")
        if not self.split.strip():
            raise ValueError("sample split must be non-empty.")
        for name, value in (
            ("global_frame_index", self.global_frame_index),
            ("source_frame_index", self.source_frame_index),
            ("chunk_index", self.chunk_index),
            ("chunk_sample_index", self.chunk_sample_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        _relative_file(self.background_store, name="background_store")
        _relative_file(self.foreground_chunk, name="foreground_chunk")

    def to_dict(self) -> dict[str, object]:
        """Return one strict compact sample record."""
        return {
            "trajectory_id": self.trajectory_id,
            "split": self.split,
            "global_frame_index": self.global_frame_index,
            "source_frame_index": self.source_frame_index,
            "chunk_index": self.chunk_index,
            "camera_id": self.camera_id,
            "background_store": self.background_store,
            "foreground_chunk": self.foreground_chunk,
            "chunk_sample_index": self.chunk_sample_index,
        }


def _tracks_from_scene(
    *,
    trajectory_id: str,
    frame_count: int,
    object_count: int,
    present: NDArray[np.bool_],
    placements: Sequence[Mapping[str, object]],
) -> tuple[BLCSTrack, ...]:
    if placements:
        if len(placements) != object_count:
            raise ValueError("track_instances must contain one record per ball column.")
        by_track: dict[int, Mapping[str, object]] = {}
        for placement in placements:
            required = {
                "track_id",
                "source_scene_id",
                "source_start",
                "source_end",
                "birth_frame",
                "death_frame",
            }
            if set(placement) != required:
                raise ValueError("track_instances contains unknown or missing fields.")
            track_id = placement["track_id"]
            if isinstance(track_id, bool) or not isinstance(track_id, int):
                raise TypeError("track_id must be an integer.")
            if track_id in by_track:
                raise ValueError("track_instances contains duplicate track_id values.")
            by_track[track_id] = placement
        if set(by_track) != set(range(object_count)):
            raise ValueError("track_id values must equal the physical object columns.")
        result: list[BLCSTrack] = []
        for track_id in range(object_count):
            placement = by_track[track_id]
            source_scene = placement["source_scene_id"]
            if not isinstance(source_scene, str):
                raise TypeError("source_scene_id must be a string.")
            source_start = _int_value(placement["source_start"], name="source_start")
            source_end = _int_value(placement["source_end"], name="source_end")
            birth = _int_value(placement["birth_frame"], name="birth_frame")
            death = _int_value(placement["death_frame"], name="death_frame")
            if not (0 <= birth < death <= frame_count):
                raise ValueError(
                    "BLCS track birth/death interval is outside the timeline."
                )
            if source_end - source_start != death - birth:
                raise ValueError(
                    "BLCS source and global track intervals differ in length."
                )
            expected_presence: NDArray[np.bool_] = np.zeros(frame_count, dtype=np.bool_)
            expected_presence[birth:death] = True
            if not np.array_equal(expected_presence, present[:, track_id]):
                raise ValueError("track_instances disagrees with ball_present.")
            source_mapping: list[int | None] = [None] * frame_count
            source_mapping[birth:death] = range(source_start, source_end)
            result.append(
                BLCSTrack(
                    object_id=f"ball-{track_id + 1:03d}",
                    source_trajectory_id=source_scene,
                    source_frame_indices=tuple(source_mapping),
                )
            )
        return tuple(result)

    tracks = []
    for object_index in range(object_count):
        active = np.flatnonzero(present[:, object_index])
        if active.size == 0:
            raise ValueError(
                "Every BLCS ball column must be present in at least one frame."
            )
        if not np.array_equal(active, np.arange(active[0], active[-1] + 1)):
            raise ValueError(
                "BLCS presence requires track_instances for non-contiguous tracks."
            )
        source_mapping = [None] * frame_count
        for source_index, global_index in enumerate(active.tolist()):
            source_mapping[global_index] = source_index
        tracks.append(
            BLCSTrack(
                object_id=f"ball-{object_index + 1:03d}",
                source_trajectory_id=trajectory_id,
                source_frame_indices=tuple(source_mapping),
            )
        )
    return tuple(tracks)


def _source_array(value: object, *, name: str) -> NDArray[Any]:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == np.dtype("O"):
        raise TypeError(f"{name} must be a numeric or boolean array.")
    return array


def _float64_array(value: object, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must use a floating dtype.")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _unit_interval(value: object, *, name: str, open_interval: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    valid = 0.0 < result < 1.0 if open_interval else 0.0 <= result <= 1.0
    if not math.isfinite(result) or not valid:
        interval = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name} must be finite and lie in {interval}.")
    return result


def _linear_rgb(value: object, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a three-value RGB sequence.")
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    channels = tuple(
        _unit_interval(channel, name=f"{name}[{index}]", open_interval=False)
        for index, channel in enumerate(value)
    )
    return channels[0], channels[1], channels[2]


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _PORTABLE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a portable non-empty identifier.")
    return value


def _int_value(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer.")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    result = _int_value(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _string_value(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return value


def _strict_mapping(
    value: object,
    *,
    name: str,
    keys: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping.")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{name} has unknown or missing keys: expected={sorted(keys)}, "
            f"actual={sorted(actual)}."
        )
    return value


def _typed_sequence(
    value: object,
    parser: Callable[[object], _ParsedT],
    *,
    name: str,
) -> tuple[_ParsedT, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an array.")
    return tuple(parser(item) for item in value)


def _relative_file(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or "\\" in value
    ):
        raise ValueError(f"{name} must be a non-empty relative POSIX path.")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{name} must not contain empty or traversal segments.")
    return value


def _json_mapping(value: Mapping[str, object], *, name: str) -> dict[str, object]:
    return {key: _json_value(item, name=f"{name}.{key}") for key, item in value.items()}


def _json_value(value: object, *, name: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number.")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} keys must be strings.")
        return {
            key: _json_value(item, name=f"{name}.{key}") for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item, name=name) for item in value]
    raise TypeError(f"{name} must be JSON-compatible, got {type(value).__name__}.")


__all__ = [
    "BLCS_BALL_ASSET_SCHEMA",
    "BLCS_BALL_COMPOSITION_SCHEMA",
    "BLCS_DATASET_SCHEMA",
    "BLCS_DATASET_SCHEMA_V3",
    "BLCS_SAMPLE_SCHEMA",
    "BLCSBallAssetMetadata",
    "BLCSBallComposition",
    "BLCSChunk",
    "BLCSBallGaussianSettings",
    "BLCSBallMeshAsset",
    "BLCSBallRendering",
    "BLCSCompositionAssets",
    "BLCSSampleRecord",
    "BLCSSceneLike",
    "BLCSTrack",
    "BLCSTrajectory",
]
