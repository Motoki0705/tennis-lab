"""Shell-free execution of NHT in its fixed canonical reconstruction workspace."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.synthetic_data_generation.reconstruction.contracts import (
    ReconstructionCommandRequest,
)
from src.synthetic_data_generation.reconstruction.runtime_config import (
    NHTTrainingRuntime,
    resolved_nht_runtime_config,
    write_nht_runtime_config,
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
    child_environment = None
    if environment is not None:
        child_environment = dict(os.environ)
        child_environment.update(environment)
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

    config_path: Path
    executable: str | Path
    training_runtime: NHTTrainingRuntime
    environment: Mapping[str, str]
    timeout_seconds: float

    def _command_request(
        self,
        context: StageExecutionContext,
        *,
        config_path: Path,
    ) -> ReconstructionCommandRequest:
        source_video = context.owner_path.parent / "source" / "video.mp4"
        return ReconstructionCommandRequest(
            scene_id=context.request.scene_id,
            input_video=source_video,
            workspace=context.owner_path,
            config_path=config_path,
            executable=self.executable,
        )

    def preflight(self, context: StageExecutionContext) -> None:
        """Validate fixed paths and command inputs before execution."""
        self._command_request(context, config_path=self.config_path)
        resolved_nht_runtime_config(
            self.config_path,
            runtime=self.training_runtime,
        )

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        """Execute NHT and summarize its semantically validated public export."""
        from src.synthetic_data_generation.pipeline.contracts import (
            StageExecutionSummary,
        )

        input_config = write_nht_runtime_config(
            self.config_path,
            context.owner_path / "input-config.yaml",
            runtime=self.training_runtime,
        )
        scene = run_nht_reconstruction(
            self._command_request(context, config_path=input_config),
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
        resolved_nht_runtime_config(
            context.owner_path / "input-config.yaml",
            runtime=self.training_runtime,
        )
        if scene.scene_id != context.request.scene_id:
            raise ValueError(
                "Validated reconstruction export belongs to another scene."
            )


__all__ = ["NHTReconstructionHandler", "run_nht_reconstruction"]
