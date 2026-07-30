"""Path-only manifest for the generic synthetic-data pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Self, cast

from src.utils.io import save_json_atomic

PATH_PIPELINE_SCHEMA = "synthetic_data_path_pipeline_v1"


def _resolve_path(value: object, *, project_root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"paths.{name} must be a non-empty path string.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


@dataclass(frozen=True)
class PathPipelineManifest:
    """All configured paths consumed and produced by one pipeline run."""

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

    @classmethod
    def from_config(
        cls,
        value: Mapping[str, object],
        *,
        project_root: Path,
    ) -> Self:
        """Resolve exactly one configured paths mapping."""
        expected = {field.name for field in fields(cls)}
        actual = set(value)
        if actual != expected:
            raise ValueError(
                "Pipeline path fields differ: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}."
            )
        root = project_root.resolve()
        resolved = {
            name: _resolve_path(value[name], project_root=root, name=name)
            for name in sorted(expected)
        }
        return cls(**resolved)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse one path-only manifest."""
        if not isinstance(value, dict) or set(value) != {"schema", "paths"}:
            raise ValueError("Pipeline manifest must contain only schema and paths.")
        if value["schema"] != PATH_PIPELINE_SCHEMA:
            raise ValueError("Unsupported pipeline manifest schema.")
        raw_paths = value["paths"]
        expected = {field.name for field in fields(cls)}
        if not isinstance(raw_paths, dict) or set(raw_paths) != expected:
            raise ValueError("Pipeline manifest path fields differ.")
        parsed: dict[str, Path] = {}
        for name in sorted(expected):
            item = raw_paths[name]
            if not isinstance(item, str) or not item:
                raise TypeError(f"Pipeline path {name!r} must be a non-empty string.")
            path = Path(item)
            if not path.is_absolute():
                raise ValueError(f"Pipeline path {name!r} must be absolute.")
            parsed[name] = path
        return cls(**parsed)

    @classmethod
    def read(cls, path: Path) -> Self:
        """Read a manifest with explicit missing-file and malformed-JSON errors."""
        if not path.is_file():
            raise FileNotFoundError(f"Pipeline manifest does not exist: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Pipeline manifest is malformed JSON: {path}") from error
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, object]:
        """Return the path-only JSON representation."""
        return {
            "schema": PATH_PIPELINE_SCHEMA,
            "paths": {
                field.name: str(getattr(self, field.name)) for field in fields(self)
            },
        }

    def write(self) -> Path:
        """Publish the path manifest atomically."""
        return cast(
            Path,
            save_json_atomic(self.to_dict(), self.pipeline_manifest),
        )
