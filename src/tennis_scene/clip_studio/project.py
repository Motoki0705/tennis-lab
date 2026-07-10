"""Project model for the multi-camera clip studio.

A project bundles the unsynchronized source videos (one per camera, each with
a sync offset onto a shared global timeline) and the clips defined on that
global timeline. It round-trips through a JSON file so GUI sessions can be
resumed and the exporter can run headlessly.

Sync convention (used consistently across the clip studio):
``local_time = global_time + offset_sec``. A source therefore covers the
global interval ``[-offset_sec, duration_sec - offset_sec]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.io import load_json, save_json_atomic

PROJECT_SCHEMA_VERSION = 1


@dataclass
class ClipSource:
    """One camera's source video and its sync offset.

    Attributes:
        path: Video file path. Relative paths are resolved against the
            project file's directory on load.
        camera_id: Unique camera identifier (e.g. ``cam0``).
        offset_sec: Sync offset; local source time that corresponds to
            global time zero (``local = global + offset_sec``).
    """

    path: Path
    camera_id: str
    offset_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "camera_id": self.camera_id,
            "offset_sec": float(self.offset_sec),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClipSource:
        return cls(
            path=Path(str(data["path"])),
            camera_id=str(data["camera_id"]),
            offset_sec=float(data["offset_sec"]),
        )


@dataclass
class Clip:
    """A half-open segment ``[start_sec, end_sec)`` on the global timeline."""

    name: str
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def contains(self, global_sec: float) -> bool:
        return self.start_sec <= global_sec < self.end_sec

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_sec": float(self.start_sec),
            "end_sec": float(self.end_sec),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Clip:
        return cls(
            name=str(data["name"]),
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
        )


@dataclass
class ClipStudioProject:
    """Sources and clips of one editing session."""

    sources: list[ClipSource] = field(default_factory=list)
    clips: list[Clip] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Return human-readable consistency errors (empty when valid)."""
        errors: list[str] = []
        if not self.sources:
            errors.append("project must contain at least one source")
        camera_ids = [source.camera_id for source in self.sources]
        if len(set(camera_ids)) != len(camera_ids):
            errors.append(f"camera_ids must be unique, got {camera_ids}")
        for source in self.sources:
            if not source.camera_id:
                errors.append(f"camera_id must be non-empty for source {source.path}")
            if not math.isfinite(source.offset_sec):
                errors.append(
                    f"offset_sec must be finite for {source.camera_id}, "
                    f"got {source.offset_sec}"
                )
        clip_names = [clip.name for clip in self.clips]
        if len(set(clip_names)) != len(clip_names):
            errors.append(f"clip names must be unique, got {clip_names}")
        for clip in self.clips:
            if not clip.name:
                errors.append("clip name must be non-empty")
            if not (math.isfinite(clip.start_sec) and math.isfinite(clip.end_sec)):
                errors.append(f"clip '{clip.name}' has non-finite bounds")
            elif clip.end_sec <= clip.start_sec:
                errors.append(
                    f"clip '{clip.name}' must have end_sec > start_sec, "
                    f"got [{clip.start_sec}, {clip.end_sec})"
                )
        return errors

    def next_clip_name(self) -> str:
        """Return the first unused ``clip_%03d`` name."""
        used = {clip.name for clip in self.clips}
        index = 0
        while f"clip_{index:03d}" in used:
            index += 1
        return f"clip_{index:03d}"

    def clip_index_by_name(self, name: str) -> int:
        for index, clip in enumerate(self.clips):
            if clip.name == name:
                return index
        raise KeyError(f"clip '{name}' not found")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PROJECT_SCHEMA_VERSION,
            "sources": [source.to_dict() for source in self.sources],
            "clips": [clip.to_dict() for clip in self.clips],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClipStudioProject:
        version = data.get("version")
        if version != PROJECT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported project version {version!r}; "
                f"expected {PROJECT_SCHEMA_VERSION}"
            )
        return cls(
            sources=[ClipSource.from_dict(item) for item in data["sources"]],
            clips=[Clip.from_dict(item) for item in data["clips"]],
        )

    def save(self, path: str | Path) -> Path:
        """Validate and atomically write the project JSON."""
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid project: {errors}")
        saved_path: Path = save_json_atomic(self.to_dict(), path)
        return saved_path

    @classmethod
    def load(cls, path: str | Path) -> ClipStudioProject:
        """Load and validate a project JSON.

        Relative source paths are resolved against the project file's parent
        directory so a project directory can be moved as a unit.
        """
        project_path = Path(path)
        project = cls.from_dict(load_json(project_path))
        for source in project.sources:
            if not source.path.is_absolute():
                source.path = (project_path.parent / source.path).resolve()
        errors = project.validate()
        if errors:
            raise ValueError(f"Invalid project at {project_path}: {errors}")
        return project


__all__ = ["Clip", "ClipSource", "ClipStudioProject", "PROJECT_SCHEMA_VERSION"]
