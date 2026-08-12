"""Public contracts for canonical synthetic-dataset video visualization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

VISUALIZATION_METADATA_SCHEMA = "canonical_synthetic_dataset_visualization_v1"


class DatasetVisualizationDomain(StrEnum):
    """Canonical generated-dataset domains supported by the visualizer."""

    COURT = "court"
    BLCS = "blcs"
    PLCS = "plcs"


@dataclass(frozen=True, slots=True)
class DatasetVisualizationConfiguration:
    """One fail-closed visualization selection and encoding contract."""

    domain: DatasetVisualizationDomain
    dataset_root: Path
    output_video: Path
    trajectory_id: str | None
    logical_scene_id: str | None
    camera_id: str | None
    fps: float
    crf: int
    history_frames: int

    def __post_init__(self) -> None:
        if not isinstance(self.domain, DatasetVisualizationDomain):
            raise TypeError("domain must be a DatasetVisualizationDomain.")
        dataset_root = _absolute_path(self.dataset_root, name="dataset_root")
        output_video = _absolute_path(self.output_video, name="output_video")
        expected_owner_parts = ("datasets", self.domain.value)
        if dataset_root.parts[-2:] != expected_owner_parts:
            raise ValueError(
                "dataset_root must be the canonical published owner "
                f".../{'/'.join(expected_owner_parts)}."
            )
        if dataset_root.is_symlink() or not dataset_root.is_dir():
            raise ValueError("dataset_root must be an existing ordinary directory.")
        if output_video.suffix.lower() != ".mp4":
            raise ValueError("output_video must use the .mp4 suffix.")
        if output_video.resolve(strict=False).is_relative_to(
            dataset_root.resolve(strict=True)
        ):
            raise ValueError(
                "Visualization output must stay outside the dataset owner."
            )
        metadata_path = output_video.with_suffix(".json")
        if output_video.exists() or output_video.is_symlink():
            raise FileExistsError(f"Visualization video already exists: {output_video}")
        if metadata_path.exists() or metadata_path.is_symlink():
            raise FileExistsError(
                f"Visualization metadata already exists: {metadata_path}"
            )
        trajectory_id = _optional_identifier(self.trajectory_id, name="trajectory_id")
        logical_scene_id = _optional_identifier(
            self.logical_scene_id, name="logical_scene_id"
        )
        camera_id = _optional_identifier(self.camera_id, name="camera_id")
        if self.domain is DatasetVisualizationDomain.COURT:
            if trajectory_id is None:
                raise ValueError("Court visualization requires trajectory_id.")
            if logical_scene_id is not None or camera_id is not None:
                raise ValueError(
                    "Court visualization does not accept logical_scene_id or camera_id."
                )
        elif trajectory_id is not None:
            raise ValueError("BLCS/PLCS visualization does not accept trajectory_id.")
        elif logical_scene_id is None or camera_id is None:
            raise ValueError(
                "BLCS/PLCS visualization requires logical_scene_id and camera_id."
            )
        fps = float(self.fps)
        if not math.isfinite(fps) or not 0.0 < fps <= 240.0:
            raise ValueError("fps must be finite and lie in (0, 240].")
        if isinstance(self.crf, bool) or not isinstance(self.crf, int):
            raise TypeError("crf must be an integer.")
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must lie in [0, 51].")
        if (
            isinstance(self.history_frames, bool)
            or not isinstance(self.history_frames, int)
            or not 0 <= self.history_frames <= 120
        ):
            raise ValueError("history_frames must be an integer in [0, 120].")
        object.__setattr__(self, "dataset_root", dataset_root)
        object.__setattr__(self, "output_video", output_video)
        object.__setattr__(self, "trajectory_id", trajectory_id)
        object.__setattr__(self, "logical_scene_id", logical_scene_id)
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "fps", fps)

    @property
    def metadata_path(self) -> Path:
        """Return the deterministic sidecar path paired with the MP4."""
        return self.output_video.with_suffix(".json")


@dataclass(frozen=True, slots=True)
class DatasetVisualizationResult:
    """Published video and deterministic metadata summary."""

    video_path: Path
    metadata_path: Path
    frame_count: int
    width: int
    height: int


def _absolute_path(value: Path, *, name: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{name} must be an absolute pathlib.Path.")
    return value


def _optional_identifier(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be null or a non-empty trimmed string.")
    return value


# The public request name describes its runtime role. The defining class keeps
# the repository-wide ``*Configuration`` naming contract so source discovery
# can expose its fields without maintaining a second schema.
DatasetVisualizationRequest: TypeAlias = DatasetVisualizationConfiguration


__all__ = [
    "DatasetVisualizationDomain",
    "DatasetVisualizationConfiguration",
    "DatasetVisualizationRequest",
    "DatasetVisualizationResult",
    "VISUALIZATION_METADATA_SCHEMA",
]
