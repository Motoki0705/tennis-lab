"""Shell-free execution of NHT in its fixed canonical reconstruction workspace."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from src.synthetic_data_generation.reconstruction.contracts import (
    ReconstructionCommandRequest,
)
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
    validate_standard_scene_export,
)

if TYPE_CHECKING:
    from src.synthetic_data_generation.pipeline.contracts import (
        StageExecutionContext,
        StageExecutionSummary,
    )


def run_nht_reconstruction(
    request: ReconstructionCommandRequest,
    *,
    environment: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> StandardSceneExport:
    """Run ``nht-reconstruct`` as argv and validate only its public files."""
    if timeout_seconds is not None and timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive when provided.")
    public_environment = _public_environment(environment)
    child_environment = None
    if public_environment:
        child_environment = dict(os.environ)
        child_environment.update(public_environment)
    subprocess.run(
        list(request.argv()),
        check=True,
        shell=False,
        timeout=timeout_seconds,
        env=child_environment,
    )
    if not request.run_manifest_path.is_file():
        raise FileNotFoundError(
            f"nht-reconstruct completed without its fixed run.json: {request.run_manifest_path}"
        )
    scene = validate_standard_scene_export(request.scene_path)
    if scene.scene_id != request.scene_id:
        raise ValueError(
            f"NHT export scene_id {scene.scene_id!r} disagrees with {request.scene_id!r}."
        )
    return scene


@dataclass(frozen=True, slots=True)
class NHTReconstructionHandler:
    """Canonical scene-pipeline handler backed only by the NHT command boundary."""

    executable: str | Path
    environment: Mapping[str, str]
    timeout_seconds: float

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0.0:
            raise ValueError("NHT reconstruction timeout_seconds must be positive.")
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(_public_environment(self.environment)),
        )

    def _command_request(
        self,
        context: StageExecutionContext,
    ) -> ReconstructionCommandRequest:
        source_video = context.owner_path.parent / "source" / "video.mp4"
        return ReconstructionCommandRequest(
            scene_id=context.request.scene_id,
            input_video=source_video,
            workspace=context.owner_path,
            executable=self.executable,
        )

    def preflight(self, context: StageExecutionContext) -> None:
        """Validate fixed paths and command inputs before execution."""
        self._command_request(context)

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        """Execute NHT and summarize its semantically validated public export."""
        from src.synthetic_data_generation.pipeline.contracts import (
            StageExecutionSummary,
        )

        scene = run_nht_reconstruction(
            self._command_request(context),
            environment=self.environment,
            timeout_seconds=self.timeout_seconds,
        )
        return StageExecutionSummary(
            {
                "scene_id": scene.scene_id,
                "camera_count": len(scene.cameras),
                "point_count": scene.point_count,
                "scene_path": "export/scene.json",
            }
        )

    def validate(self, context: StageExecutionContext) -> None:
        """Require the fixed standard export after NHT's own atomic publication."""
        scene = validate_standard_scene_export(
            context.owner_path / "export" / "scene.json"
        )
        if scene.scene_id != context.request.scene_id:
            raise ValueError(
                "Validated reconstruction export belongs to another scene."
            )


def _public_environment(
    environment: Mapping[str, str] | None,
) -> dict[str, str]:
    if environment is None:
        return {}
    unknown = sorted(set(environment) - {"CUDA_VISIBLE_DEVICES"})
    if unknown:
        raise ValueError(
            "nht-reconstruct environment contains unsupported private key(s): "
            + ", ".join(unknown)
            + "."
        )
    result: dict[str, str] = {}
    for key, value in environment.items():
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(
                f"nht-reconstruct environment {key} must be a trimmed non-empty string."
            )
        result[key] = value
    return result


__all__ = ["NHTReconstructionHandler", "run_nht_reconstruction"]
