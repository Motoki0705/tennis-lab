"""Typed request contract for the external ``nht-reconstruct`` command."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

NHT_RECONSTRUCT_COMMAND = "nht-reconstruct"

_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class ReconstructionCommandRequest:
    """One full reconstruction in the scene's fixed NHT-owned workspace."""

    scene_id: str
    input_video: Path
    workspace: Path
    config_path: Path
    executable: str | Path = NHT_RECONSTRUCT_COMMAND

    def __post_init__(self) -> None:
        if _PORTABLE_ID.fullmatch(self.scene_id) is None:
            raise ValueError(
                f"scene_id is not a portable identifier: {self.scene_id!r}."
            )
        for name, path in (
            ("input_video", self.input_video),
            ("workspace", self.workspace),
            ("config_path", self.config_path),
        ):
            if not isinstance(path, Path):
                raise TypeError(f"{name} must be a pathlib.Path.")
            if not path.is_absolute():
                raise ValueError(f"{name} must be an absolute path: {path}")
        if not self.input_video.is_file():
            raise FileNotFoundError(
                f"NHT input video does not exist: {self.input_video}"
            )
        if not self.config_path.is_file():
            raise FileNotFoundError(f"NHT config does not exist: {self.config_path}")
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
            str(self.config_path),
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


__all__ = ["NHT_RECONSTRUCT_COMMAND", "ReconstructionCommandRequest"]
