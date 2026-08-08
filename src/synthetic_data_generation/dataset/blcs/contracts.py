"""Typed full-timeline contracts for the canonical BLCS dataset stage."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from src.synthetic_data_generation.composition import GaussianAsset

BLCS_DATASET_SCHEMA = "canonical_blcs_compact_dataset_v1"
BLCS_SAMPLE_SCHEMA = "canonical_blcs_compact_sample_v1"

_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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
class BLCSCompositionAssets:
    """Semantic background and ball assets used by every planned frame."""

    background: GaussianAsset
    ball: GaussianAsset
    ball_radius_m: float

    def __post_init__(self) -> None:
        from src.synthetic_data_generation.composition import GaussianAssetRole

        if self.background.role is not GaussianAssetRole.BACKGROUND:
            raise ValueError("BLCS background asset must have role=background.")
        if self.ball.role is not GaussianAssetRole.MOVABLE:
            raise ValueError("BLCS ball asset must have role=movable.")
        if self.ball.asset_class != "ball":
            raise ValueError("BLCS movable asset must declare asset_class='ball'.")
        object.__setattr__(
            self,
            "ball_radius_m",
            _positive_float(self.ball_radius_m, name="ball_radius_m"),
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


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _PORTABLE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a portable non-empty identifier.")
    return value


def _int_value(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer.")
    return value


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
    "BLCS_DATASET_SCHEMA",
    "BLCS_SAMPLE_SCHEMA",
    "BLCSChunk",
    "BLCSCompositionAssets",
    "BLCSSampleRecord",
    "BLCSSceneLike",
    "BLCSTrack",
    "BLCSTrajectory",
]
