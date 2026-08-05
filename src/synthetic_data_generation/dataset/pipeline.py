"""Role-preserving path manifest for the generic synthetic-data pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from src.utils.configuration import (
    PathContractError,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.io import save_json_atomic

PATH_PIPELINE_SCHEMA = "synthetic_data_path_pipeline_v2"

PATH_PIPELINE_FIELDS = (
    "source_root",
    "artifact_root",
    "execution_root",
    "dataset_root",
    "alignment_observations",
    "render_jobs",
    "pipeline_manifest",
    "alignment_metrics",
    "dataset_plan",
    "render_manifest",
    "quality_metrics",
    "visualization",
)

PATH_PIPELINE_FIELD_ROLES = {
    "source_root": PathRole.EXTERNAL_ASSET,
    "artifact_root": PathRole.ARTIFACT,
    "execution_root": PathRole.OUTPUT,
    "dataset_root": PathRole.DATA,
    "alignment_observations": PathRole.EXTERNAL_ASSET,
    "render_jobs": PathRole.EXTERNAL_ASSET,
    "pipeline_manifest": PathRole.OUTPUT,
    "alignment_metrics": PathRole.ARTIFACT,
    "dataset_plan": PathRole.ARTIFACT,
    "render_manifest": PathRole.ARTIFACT,
    "quality_metrics": PathRole.ARTIFACT,
    "visualization": PathRole.OUTPUT,
}

PATH_PIPELINE_ROOT_ROLES = {
    name: PATH_PIPELINE_FIELD_ROLES[name]
    for name in ("source_root", "artifact_root", "execution_root", "dataset_root")
}

_PATH_PIPELINE_PARENTS = {
    "alignment_observations": "source_root",
    "render_jobs": "source_root",
    "pipeline_manifest": "execution_root",
    "visualization": "execution_root",
    "alignment_metrics": "artifact_root",
    "dataset_plan": "artifact_root",
    "render_manifest": "artifact_root",
    "quality_metrics": "artifact_root",
}

_RENDER_JOB_PATHS = {
    "input": (PathRole.EXTERNAL_ASSET, "source_root"),
    "reference": (PathRole.EXTERNAL_ASSET, "source_root"),
    "output": (PathRole.DATA, "dataset_root"),
}


def _save_json_path(value: object, path: Path) -> Path:
    """Publish JSON while narrowing an untyped changed-file import to Path."""
    published: object = save_json_atomic(value, path)
    if not isinstance(published, Path):
        raise TypeError("save_json_atomic must return pathlib.Path.")
    return published


def _path_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise TypeError(f"paths.{name} must be a non-empty trimmed path string.")
    return value


def _resolve_path(
    value: object,
    *,
    resolver: PathResolver,
    role: PathRole,
    name: str,
) -> Path:
    resolved: Path = resolver.resolve(role, _path_text(value, name=name))
    return resolved


def _resolve_beneath(
    value: object,
    *,
    resolver: PathResolver,
    role: PathRole,
    root: Path,
    name: str,
) -> Path:
    resolved: Path = resolver.resolve_beneath(
        role,
        root,
        _path_text(value, name=name),
    )
    return resolved


@dataclass(frozen=True, slots=True)
class PathPipelineManifest:
    """All role-tagged paths consumed and produced by one pipeline run."""

    runtime_roots: RuntimePathRoots
    source_root: Path
    artifact_root: Path
    execution_root: Path
    dataset_root: Path
    alignment_observations: Path
    render_jobs: Path
    pipeline_manifest: Path
    alignment_metrics: Path
    dataset_plan: Path
    render_manifest: Path
    quality_metrics: Path
    visualization: Path

    def __post_init__(self) -> None:
        resolver = self.resolver
        for name in PATH_PIPELINE_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, Path):
                raise PathContractError(
                    f"Pipeline path {name!r} must be a pathlib.Path."
                )
            resolved = resolver.validate(PATH_PIPELINE_FIELD_ROLES[name], value)
            if resolved != value:
                raise PathContractError(
                    f"Pipeline path {name!r} must be resolved: {value}."
                )

        for name, role in PATH_PIPELINE_ROOT_ROLES.items():
            value = getattr(self, name)
            if value == self.runtime_roots.root(role):
                raise PathContractError(
                    f"Pipeline path {name!r} must be below its {role.value} root."
                )

        for name, parent_name in _PATH_PIPELINE_PARENTS.items():
            value = getattr(self, name)
            parent = getattr(self, parent_name)
            if value == parent or not value.is_relative_to(parent):
                raise PathContractError(
                    f"Pipeline path {name!r} must be below {parent_name!r}."
                )

        leaf_paths = tuple(
            getattr(self, name) for name in _PATH_PIPELINE_PARENTS
        )
        if len(set(leaf_paths)) != len(leaf_paths):
            raise PathContractError("Pipeline leaf paths must be distinct.")

    @property
    def resolver(self) -> PathResolver:
        """Return the resolver reconstructed from the persisted root contract."""
        return PathResolver(self.runtime_roots)

    @classmethod
    def from_config(
        cls,
        value: Mapping[str, object],
        *,
        resolver: PathResolver,
    ) -> Self:
        """Resolve exactly one configured paths mapping."""
        expected = set(PATH_PIPELINE_FIELDS)
        actual = set(value)
        if actual != expected:
            raise ValueError(
                "Pipeline path fields differ: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}."
            )
        resolved_roots = {
            name: _resolve_path(value[name], resolver=resolver, role=role, name=name)
            for name, role in PATH_PIPELINE_ROOT_ROLES.items()
        }
        resolved = dict(resolved_roots)
        for name, parent_name in _PATH_PIPELINE_PARENTS.items():
            resolved[name] = _resolve_beneath(
                value[name],
                resolver=resolver,
                role=PATH_PIPELINE_FIELD_ROLES[name],
                root=resolved_roots[parent_name],
                name=name,
            )
        return cls(runtime_roots=resolver.roots, **resolved)

    @classmethod
    def from_dict(cls, value: object, *, resolver: PathResolver) -> Self:
        """Parse one role-preserving path manifest and revalidate containment."""
        expected_sections = {"schema", "roots", "roles", "paths"}
        if not isinstance(value, dict) or set(value) != expected_sections:
            raise ValueError(
                "Pipeline manifest must contain only schema, roots, roles, and paths."
            )
        if value["schema"] != PATH_PIPELINE_SCHEMA:
            raise ValueError("Unsupported pipeline manifest schema.")

        expected_roles = {
            name: role.value for name, role in PATH_PIPELINE_FIELD_ROLES.items()
        }
        if value["roles"] != expected_roles:
            raise ValueError("Pipeline manifest path roles differ from the schema.")

        raw_roots = value["roots"]
        if not isinstance(raw_roots, dict):
            raise TypeError("Pipeline manifest roots must be an object.")
        expected_root_fields = {
            f"{role.value}_root" for role in PathRole
        }
        if set(raw_roots) != expected_root_fields:
            raise ValueError("Pipeline manifest root fields differ.")
        if raw_roots != dict(resolver.roots.as_mapping()):
            raise PathContractError(
                "Pipeline manifest roots differ from the active runtime root contract."
            )

        raw_paths = value["paths"]
        expected_paths = set(PATH_PIPELINE_FIELDS)
        if not isinstance(raw_paths, dict) or set(raw_paths) != expected_paths:
            raise ValueError("Pipeline manifest path fields differ.")
        parsed: dict[str, Path] = {}
        for name in PATH_PIPELINE_FIELDS:
            item = raw_paths[name]
            if type(item) is not str or not item or item != item.strip():
                raise TypeError(
                    f"Pipeline path {name!r} must be a non-empty trimmed string."
                )
            path = Path(item)
            if not path.is_absolute():
                raise ValueError(f"Pipeline path {name!r} must be absolute.")
            parsed[name] = path
        return cls(runtime_roots=resolver.roots, **parsed)

    @classmethod
    def read(cls, path: Path, *, resolver: PathResolver) -> Self:
        """Read and validate the manifest at its declared output location."""
        if not path.is_file():
            raise FileNotFoundError(f"Pipeline manifest does not exist: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Pipeline manifest is malformed JSON: {path}") from error
        manifest = cls.from_dict(value, resolver=resolver)
        if path.resolve(strict=False) != manifest.pipeline_manifest:
            raise PathContractError(
                "Pipeline manifest was read from a path other than its declared "
                f"pipeline_manifest location: {path}."
            )
        return manifest

    def resolve_render_job_path(self, name: str, value: object) -> Path:
        """Resolve one role-relative render-job path below its declared subroot."""
        if name not in _RENDER_JOB_PATHS:
            raise KeyError(name)
        role, root_name = _RENDER_JOB_PATHS[name]
        root = getattr(self, root_name)
        if not isinstance(root, Path):
            raise AssertionError(f"Pipeline root {root_name!r} is not a Path.")
        resolved: Path = self.resolver.resolve_beneath(
            role,
            root,
            _path_text(value, name=f"render_job.{name}"),
        )
        return resolved

    def validate_render_job_path(self, name: str, value: object) -> Path:
        """Revalidate one absolute path recovered from a persisted dataset plan."""
        if name not in _RENDER_JOB_PATHS:
            raise KeyError(name)
        if type(value) is not str or not value or value != value.strip():
            raise TypeError(
                f"Persisted render-job {name} must be a non-empty trimmed string."
            )
        path = Path(value)
        if not path.is_absolute():
            raise PathContractError(
                f"Persisted render-job {name} path must be absolute: {path}."
            )
        role, root_name = _RENDER_JOB_PATHS[name]
        resolved: Path = self.resolver.validate(role, path)
        root = getattr(self, root_name)
        if not isinstance(root, Path):
            raise AssertionError(f"Pipeline root {root_name!r} is not a Path.")
        if resolved == root or not resolved.is_relative_to(root):
            raise PathContractError(
                f"Persisted render-job {name} path must be below {root_name}: "
                f"{resolved}."
            )
        return resolved

    def to_dict(self) -> dict[str, object]:
        """Return the role-preserving JSON representation."""
        return {
            "schema": PATH_PIPELINE_SCHEMA,
            "roots": dict(self.runtime_roots.as_mapping()),
            "roles": {
                name: role.value for name, role in PATH_PIPELINE_FIELD_ROLES.items()
            },
            "paths": {
                name: str(getattr(self, name)) for name in PATH_PIPELINE_FIELDS
            },
        }

    def write(self) -> Path:
        """Publish the path manifest atomically."""
        return _save_json_path(self.to_dict(), self.pipeline_manifest)
