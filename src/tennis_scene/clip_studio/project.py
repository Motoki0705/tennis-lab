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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.configuration import PathResolver, PathRole
from src.utils.io import load_json, save_json_atomic

PROJECT_SCHEMA_VERSION = 2
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _require_exact_keys(data: dict[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(data)
    if actual != expected:
        raise ValueError(
            f"{name} keys must be exactly {sorted(expected)}, got {sorted(actual)}"
        )


def _validate_identifier(value: str, *, field_name: str) -> str | None:
    """Validate a portable single-path-component identifier."""
    if not value:
        return f"{field_name} must be non-empty"
    if IDENTIFIER_PATTERN.fullmatch(value) is None:
        return (
            f"{field_name} must start with an ASCII letter or digit and contain "
            f"only letters, digits, '.', '_' or '-', got {value!r}"
        )
    return None


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

    def to_dict(self, resolver: PathResolver) -> dict[str, Any]:
        resolved = resolver.validate(PathRole.DATA, self.path)
        return {
            "path": resolved.relative_to(resolver.roots.data_root).as_posix(),
            "camera_id": self.camera_id,
            "offset_sec": float(self.offset_sec),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        resolver: PathResolver,
    ) -> ClipSource:
        _require_exact_keys(
            data, {"path", "camera_id", "offset_sec"}, name="clip source"
        )
        raw_path = data["path"]
        if type(raw_path) is not str or not raw_path:
            raise ValueError("clip source path must be a non-empty role-relative string")
        return cls(
            path=resolver.resolve(PathRole.DATA, raw_path),
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
        _require_exact_keys(data, {"name", "start_sec", "end_sec"}, name="clip")
        return cls(
            name=str(data["name"]),
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
        )


@dataclass
class ClipStudioProject:
    """Sources and clips of one editing session."""

    recording_id: str = ""
    sources: list[ClipSource] = field(default_factory=list)
    clips: list[Clip] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Return human-readable consistency errors (empty when valid)."""
        errors: list[str] = []
        recording_error = _validate_identifier(
            self.recording_id, field_name="recording_id"
        )
        if recording_error is not None:
            errors.append(recording_error)
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
            clip_name_error = _validate_identifier(clip.name, field_name="clip name")
            if clip_name_error is not None:
                errors.append(clip_name_error)
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

    def to_dict(self, resolver: PathResolver) -> dict[str, Any]:
        return {
            "version": PROJECT_SCHEMA_VERSION,
            "recording_id": self.recording_id,
            "sources": [source.to_dict(resolver) for source in self.sources],
            "clips": [clip.to_dict() for clip in self.clips],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        resolver: PathResolver,
    ) -> ClipStudioProject:
        version = data["version"]
        if version != PROJECT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported project version {version!r}; "
                f"expected {PROJECT_SCHEMA_VERSION}"
            )
        _require_exact_keys(
            data, {"version", "recording_id", "sources", "clips"}, name="project"
        )
        return cls(
            recording_id=str(data["recording_id"]),
            sources=[ClipSource.from_dict(item, resolver) for item in data["sources"]],
            clips=[Clip.from_dict(item) for item in data["clips"]],
        )

    def save(self, path: Path, resolver: PathResolver) -> Path:
        """Validate and atomically write the project JSON."""
        project_path = resolver.validate(PathRole.ARTIFACT, path)
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid project: {errors}")
        saved_path: Path = save_json_atomic(self.to_dict(resolver), project_path)
        return saved_path

    @classmethod
    def load(cls, path: Path, resolver: PathResolver) -> ClipStudioProject:
        """Load one ARTIFACT project with DATA-role source paths."""
        project_path = resolver.validate(PathRole.ARTIFACT, path)
        project = cls.from_dict(load_json(project_path), resolver)
        errors = project.validate()
        if errors:
            raise ValueError(f"Invalid project at {project_path}: {errors}")
        return project


__all__ = ["Clip", "ClipSource", "ClipStudioProject", "PROJECT_SCHEMA_VERSION"]
