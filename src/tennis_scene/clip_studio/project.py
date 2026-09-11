"""Aggregate project model for dataset/video/clip editing.

One ``projects.json`` owns all independently synchronized videos in a dataset.
Each video project has its own camera sources, offsets, and temporal clips.

Sync convention: ``local_time = global_time + offset_sec``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.configuration import PathResolver, PathRole
from src.utils.io import load_json, save_json_atomic

PROJECTS_SCHEMA_VERSION = 1
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _require_exact_keys(data: dict[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(data)
    if actual != expected:
        raise ValueError(
            f"{name} keys must be exactly {sorted(expected)}, got {sorted(actual)}"
        )


def _validate_identifier(value: str, *, field_name: str) -> str | None:
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
    """One camera source and its offset on the video's shared timeline."""

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
    def from_dict(cls, data: dict[str, Any], resolver: PathResolver) -> ClipSource:
        _require_exact_keys(
            data, {"path", "camera_id", "offset_sec"}, name="clip source"
        )
        raw_path = data["path"]
        if type(raw_path) is not str or not raw_path:
            raise ValueError(
                "clip source path must be a non-empty role-relative string"
            )
        return cls(
            path=resolver.resolve(PathRole.DATA, raw_path),
            camera_id=str(data["camera_id"]),
            offset_sec=float(data["offset_sec"]),
        )


@dataclass
class Clip:
    """A half-open segment ``[start_sec, end_sec)`` in one video."""

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
    """Editable state for one raw multi-camera video."""

    dataset_id: str = ""
    video_id: str = ""
    sources: list[ClipSource] = field(default_factory=list)
    clips: list[Clip] = field(default_factory=list)

    def validate(self) -> list[str]:
        errors: list[str] = []
        for value, name in (
            (self.dataset_id, "dataset_id"),
            (self.video_id, "video_id"),
        ):
            error = _validate_identifier(value, field_name=name)
            if error is not None:
                errors.append(error)
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
                    f"offset_sec must be finite for {source.camera_id}, got {source.offset_sec}"
                )
        clip_names = [clip.name for clip in self.clips]
        if len(set(clip_names)) != len(clip_names):
            errors.append(f"clip names must be unique, got {clip_names}")
        for clip in self.clips:
            clip_error = _validate_identifier(clip.name, field_name="clip name")
            if clip_error is not None:
                errors.append(clip_error)
            if not (math.isfinite(clip.start_sec) and math.isfinite(clip.end_sec)):
                errors.append(f"clip '{clip.name}' has non-finite bounds")
            elif clip.end_sec <= clip.start_sec:
                errors.append(
                    f"clip '{clip.name}' must have end_sec > start_sec, "
                    f"got [{clip.start_sec}, {clip.end_sec})"
                )
        return errors

    def next_clip_name(self) -> str:
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
            "sources": [source.to_dict(resolver) for source in self.sources],
            "clips": [clip.to_dict() for clip in self.clips],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        resolver: PathResolver,
        *,
        dataset_id: str,
        video_id: str,
    ) -> ClipStudioProject:
        _require_exact_keys(data, {"sources", "clips"}, name=f"project {video_id}")
        project = cls(
            dataset_id=dataset_id,
            video_id=video_id,
            sources=[ClipSource.from_dict(item, resolver) for item in data["sources"]],
            clips=[Clip.from_dict(item) for item in data["clips"]],
        )
        errors = project.validate()
        if errors:
            raise ValueError(f"Invalid project {video_id}: {errors}")
        return project

    def save(self, path: Path, resolver: PathResolver) -> Path:
        return ClipStudioProjects.save_project(self, path, resolver)

    @classmethod
    def load(
        cls,
        path: Path,
        resolver: PathResolver,
        *,
        dataset_id: str,
        video_id: str,
    ) -> ClipStudioProject:
        projects = ClipStudioProjects.load(path, resolver)
        if projects.dataset_id != dataset_id:
            raise ValueError(
                f"projects dataset_id {projects.dataset_id!r} != {dataset_id!r}"
            )
        try:
            return projects.projects[video_id]
        except KeyError:
            raise KeyError(f"video project {video_id!r} not found in {path}") from None


@dataclass
class ClipStudioProjects:
    """All independently editable video projects in one dataset."""

    dataset_id: str
    projects: dict[str, ClipStudioProject] = field(default_factory=dict)

    def to_dict(self, resolver: PathResolver) -> dict[str, Any]:
        return {
            "version": PROJECTS_SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "projects": {
                video_id: self.projects[video_id].to_dict(resolver)
                for video_id in sorted(self.projects)
            },
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], resolver: PathResolver
    ) -> ClipStudioProjects:
        _require_exact_keys(
            data, {"version", "dataset_id", "projects"}, name="projects"
        )
        if data["version"] != PROJECTS_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported projects version {data['version']!r}; "
                f"expected {PROJECTS_SCHEMA_VERSION}"
            )
        dataset_id = str(data["dataset_id"])
        dataset_error = _validate_identifier(dataset_id, field_name="dataset_id")
        if dataset_error is not None:
            raise ValueError(dataset_error)
        raw_projects = data["projects"]
        if not isinstance(raw_projects, dict):
            raise ValueError("projects must be an object keyed by video_id")
        projects = {
            str(video_id): ClipStudioProject.from_dict(
                value,
                resolver,
                dataset_id=dataset_id,
                video_id=str(video_id),
            )
            for video_id, value in raw_projects.items()
        }
        return cls(dataset_id=dataset_id, projects=projects)

    @classmethod
    def load(cls, path: Path, resolver: PathResolver) -> ClipStudioProjects:
        projects_path = resolver.validate(PathRole.DATA, path)
        payload = load_json(projects_path)
        if not isinstance(payload, dict):
            raise ValueError(f"{projects_path} must contain a JSON object")
        return cls.from_dict(payload, resolver)

    @classmethod
    def save_project(
        cls,
        project: ClipStudioProject,
        path: Path,
        resolver: PathResolver,
    ) -> Path:
        projects_path = resolver.validate(PathRole.DATA, path)
        errors = project.validate()
        if errors:
            raise ValueError(f"Invalid project: {errors}")
        if projects_path.exists():
            collection = cls.load(projects_path, resolver)
            if collection.dataset_id != project.dataset_id:
                raise ValueError(
                    f"projects dataset_id {collection.dataset_id!r} != "
                    f"project dataset_id {project.dataset_id!r}"
                )
        else:
            collection = cls(dataset_id=project.dataset_id)
        collection.projects[project.video_id] = project
        saved_path: Path = save_json_atomic(collection.to_dict(resolver), projects_path)
        return saved_path


__all__ = [
    "Clip",
    "ClipSource",
    "ClipStudioProject",
    "ClipStudioProjects",
    "PROJECTS_SCHEMA_VERSION",
]
