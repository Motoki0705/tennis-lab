"""Typed request contract for the external ``nht-reconstruct`` command."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

NHT_RECONSTRUCT_COMMAND = "nht-reconstruct"
NHT_PIPELINE_CONFIG_SCHEMA = "nht_pipeline_config_v1"

_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_NHT_PIPELINE_CONFIG_KEYS = frozenset(
    {
        "schema",
        "seed",
        "frames",
        "preprocess",
        "sfm",
        "nht_training",
        "export",
        "operations",
    }
)


@dataclass(frozen=True, slots=True)
class NHTPipelineConfig:
    """Resolved public ``nht-reconstruct --config`` file authority."""

    path: Path
    schema: str

    @classmethod
    def load(cls, path: Path) -> NHTPipelineConfig:
        """Validate the public config envelope without importing NHT internals."""
        if not isinstance(path, Path):
            raise TypeError("NHT pipeline config path must be a pathlib.Path.")
        if not path.is_absolute() or path.resolve(strict=False) != path:
            raise ValueError(
                "NHT pipeline config path must be a resolved absolute path."
            )
        if path.is_symlink():
            raise ValueError("NHT pipeline config must not be a symbolic link.")
        if not path.exists():
            raise FileNotFoundError(f"NHT pipeline config does not exist: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"NHT pipeline config is not a file: {path}")
        try:
            loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ValueError(f"NHT pipeline config is not valid YAML: {path}") from error
        if not isinstance(loaded, Mapping) or any(
            not isinstance(key, str) for key in loaded
        ):
            raise TypeError(
                "NHT pipeline config must contain a string-keyed mapping."
            )
        unknown = sorted(set(loaded) - _NHT_PIPELINE_CONFIG_KEYS)
        if unknown:
            raise ValueError(
                "Unknown NHT pipeline config key(s): " + ", ".join(unknown) + "."
            )
        if loaded.get("schema") != NHT_PIPELINE_CONFIG_SCHEMA:
            raise ValueError(
                "NHT pipeline config schema must be "
                f"{NHT_PIPELINE_CONFIG_SCHEMA!r}."
            )
        return cls(path=path, schema=NHT_PIPELINE_CONFIG_SCHEMA)

    def validate(self) -> None:
        """Revalidate the mutable external file immediately before use."""
        current = type(self).load(self.path)
        if current.schema != self.schema:  # pragma: no cover - fixed schema guard
            raise ValueError("NHT pipeline config schema changed after resolution.")

    def provenance(self) -> Mapping[str, str]:
        """Return JSON-safe resolved provenance for the canonical run manifest."""
        return {"path": str(self.path), "schema": self.schema}


@dataclass(frozen=True, slots=True)
class ReconstructionCommandRequest:
    """One full reconstruction in the scene's fixed NHT-owned workspace."""

    scene_id: str
    input_video: Path
    workspace: Path
    pipeline_config: NHTPipelineConfig
    executable: str | Path = NHT_RECONSTRUCT_COMMAND

    def __post_init__(self) -> None:
        if _PORTABLE_ID.fullmatch(self.scene_id) is None:
            raise ValueError(
                f"scene_id is not a portable identifier: {self.scene_id!r}."
            )
        for name, path in (
            ("input_video", self.input_video),
            ("workspace", self.workspace),
        ):
            if not isinstance(path, Path):
                raise TypeError(f"{name} must be a pathlib.Path.")
            if not path.is_absolute():
                raise ValueError(f"{name} must be an absolute path: {path}")
        if not self.input_video.is_file():
            raise FileNotFoundError(
                f"NHT input video does not exist: {self.input_video}"
            )
        if (
            self.workspace.name != "reconstruction"
            or self.workspace.parent.name != self.scene_id
        ):
            raise ValueError(
                "NHT workspace must be the fixed <scene_id>/reconstruction directory."
            )
        if self.workspace.is_symlink():
            raise ValueError(
                "NHT reconstruction workspace must not be a symbolic link."
            )
        if self.workspace.exists() and not self.workspace.is_dir():
            raise NotADirectoryError(
                f"NHT workspace is not a directory: {self.workspace}"
            )
        if not isinstance(self.pipeline_config, NHTPipelineConfig):
            raise TypeError("pipeline_config must be an NHTPipelineConfig.")
        self.pipeline_config.validate()
        _validate_executable(self.executable)

    @property
    def scene_path(self) -> Path:
        """Return the sole standard scene entry path produced by NHT."""
        return self.workspace / "export" / "scene.json"

    @property
    def run_manifest_path(self) -> Path:
        """Return NHT's fixed mutable run manifest path."""
        return self.workspace / "run.json"

    def argv(self) -> tuple[str, ...]:
        """Return the exact shell-free public command argument vector."""
        return (
            str(self.executable),
            "--scene-id",
            self.scene_id,
            "--input-video",
            str(self.input_video),
            "--workspace",
            str(self.workspace),
            "--config",
            str(self.pipeline_config.path),
        )


def _validate_executable(executable: str | Path) -> None:
    if isinstance(executable, str):
        if executable != NHT_RECONSTRUCT_COMMAND:
            raise ValueError(
                "String reconstruction executable must be exactly nht-reconstruct."
            )
        return
    if not isinstance(executable, Path) or not executable.is_absolute():
        raise ValueError(
            "Path reconstruction executable must be an absolute pathlib.Path."
        )
    if executable.name != NHT_RECONSTRUCT_COMMAND:
        raise ValueError("Reconstruction executable basename must be nht-reconstruct.")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(
            f"nht-reconstruct executable is unavailable: {executable}"
        )


__all__ = [
    "NHT_PIPELINE_CONFIG_SCHEMA",
    "NHT_RECONSTRUCT_COMMAND",
    "NHTPipelineConfig",
    "ReconstructionCommandRequest",
]
