"""Shell-free execution of NHT in its fixed canonical reconstruction workspace."""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import yaml

from src.synthetic_data_generation.reconstruction.contracts import (
    NHTPipelineConfig,
    NHTTrainingRuntime,
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
    training_runtime: NHTTrainingRuntime | None = None,
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
    with _runtime_bound_request(
        request, training_runtime, environment=public_environment
    ) as effective_request:
        subprocess.run(
            list(effective_request.argv()),
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
    pipeline_config: NHTPipelineConfig
    training_runtime: NHTTrainingRuntime
    environment: Mapping[str, str]
    timeout_seconds: float

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0.0:
            raise ValueError("NHT reconstruction timeout_seconds must be positive.")
        if not isinstance(self.pipeline_config, NHTPipelineConfig):
            raise TypeError("pipeline_config must be an NHTPipelineConfig.")
        if not isinstance(self.training_runtime, NHTTrainingRuntime):
            raise TypeError("training_runtime must be an NHTTrainingRuntime.")
        self.pipeline_config.validate()
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
            pipeline_config=self.pipeline_config,
            executable=self.executable,
        )

    def preflight(self, context: StageExecutionContext) -> None:
        """Validate fixed paths and command inputs before execution."""
        self._command_request(context)
        self.training_runtime.validate()

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        """Execute NHT and summarize its semantically validated public export."""
        from src.synthetic_data_generation.pipeline.contracts import (
            StageExecutionSummary,
        )

        scene = run_nht_reconstruction(
            self._command_request(context),
            training_runtime=self.training_runtime,
            environment=self.environment,
            timeout_seconds=self.timeout_seconds,
        )
        return StageExecutionSummary(
            {
                "scene_id": scene.scene_id,
                "camera_count": len(scene.cameras),
                "point_count": scene.point_count,
                "scene_path": "export/scene.json",
                "pipeline_config": dict(self.pipeline_config.provenance()),
                "training_runtime": dict(self.training_runtime.provenance()),
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
    unknown = sorted(
        set(environment)
        - {
            "CUDA_VISIBLE_DEVICES",
            "TENNIS_LAB_NHT_MINIMUM_MEDIAN_TRACK_LENGTH",
            "TENNIS_LAB_NHT_MINIMUM_SPARSE_POINTS",
        }
    )
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


@contextmanager
def _runtime_bound_request(
    request: ReconstructionCommandRequest,
    runtime: NHTTrainingRuntime | None,
    *,
    environment: Mapping[str, str],
) -> Iterator[ReconstructionCommandRequest]:
    """Bind the portable NHT config to one machine-local trainer environment."""
    track_override = environment.get(
        "TENNIS_LAB_NHT_MINIMUM_MEDIAN_TRACK_LENGTH"
    )
    sparse_points_override = environment.get(
        "TENNIS_LAB_NHT_MINIMUM_SPARSE_POINTS"
    )
    if runtime is None and track_override is None and sparse_points_override is None:
        yield request
        return
    if runtime is not None:
        runtime.validate()
    loaded: object = yaml.safe_load(
        request.pipeline_config.path.read_text(encoding="utf-8")
    )
    if not isinstance(loaded, Mapping):  # pragma: no cover - validated by contract
        raise TypeError("NHT pipeline config must contain a mapping.")
    effective = dict(loaded)
    if runtime is not None:
        training_value = effective.get("nht_training", {})
        if not isinstance(training_value, Mapping):
            raise TypeError("NHT pipeline config nht_training must be a mapping.")
        training = dict(training_value)
        training.update(
            {
                "python": str(runtime.python),
                "trainer": str(runtime.trainer),
            }
        )
        effective["nht_training"] = training
    if track_override is not None:
        minimum_track_length = float(track_override)
        if not math.isfinite(minimum_track_length) or not (
            2.0 <= minimum_track_length <= 3.0
        ):
            raise ValueError(
                "TENNIS_LAB_NHT_MINIMUM_MEDIAN_TRACK_LENGTH must be between "
                "2.0 and 3.0."
            )
        sfm_value = effective.get("sfm")
        if not isinstance(sfm_value, Mapping):
            raise TypeError("NHT pipeline config sfm must be a mapping.")
        sfm = dict(sfm_value)
        gates_value = sfm.get("quality_gates")
        if not isinstance(gates_value, Mapping):
            raise TypeError("NHT pipeline config sfm.quality_gates must be a mapping.")
        gates = dict(gates_value)
        gates["minimum_median_track_length"] = minimum_track_length
        sfm["quality_gates"] = gates
        effective["sfm"] = sfm
    if sparse_points_override is not None:
        try:
            minimum_sparse_points = int(sparse_points_override)
        except ValueError as error:
            raise ValueError(
                "TENNIS_LAB_NHT_MINIMUM_SPARSE_POINTS must be an integer."
            ) from error
        if not 40_000 <= minimum_sparse_points <= 50_000:
            raise ValueError(
                "TENNIS_LAB_NHT_MINIMUM_SPARSE_POINTS must be between "
                "40000 and 50000."
            )
        sfm_value = effective.get("sfm")
        if not isinstance(sfm_value, Mapping):
            raise TypeError("NHT pipeline config sfm must be a mapping.")
        sfm = dict(sfm_value)
        gates_value = sfm.get("quality_gates")
        if not isinstance(gates_value, Mapping):
            raise TypeError("NHT pipeline config sfm.quality_gates must be a mapping.")
        gates = dict(gates_value)
        gates["minimum_sparse_points"] = minimum_sparse_points
        sfm["quality_gates"] = gates
        effective["sfm"] = sfm
    with tempfile.TemporaryDirectory(prefix="tennis-lab-nht-") as directory:
        path = Path(directory).joinpath("pipeline.yaml")
        path.write_text(
            yaml.safe_dump(effective, sort_keys=False),
            encoding="utf-8",
        )
        yield replace(
            request,
            pipeline_config=NHTPipelineConfig.load(path.resolve()),
        )


__all__ = ["NHTReconstructionHandler", "run_nht_reconstruction"]
