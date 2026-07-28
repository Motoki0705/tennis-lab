"""Court dataset pipeline planning with explicit algorithm selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.synthetic_data_generation.dataset.court.algorithms import (
    CAMERA_SAMPLING_ALGORITHMS,
    LABEL_ALGORITHMS,
)
from src.synthetic_data_generation.dataset.pipeline import (
    DatasetPipelinePlan,
    PipelineCommand,
    configured_command,
    require_mapping,
    require_string,
)


@dataclass(frozen=True)
class CourtDatasetPipeline:
    """Build reproducible commands for camera sampling, rendering, and labels."""

    @property
    def dataset_name(self) -> str:
        """Return the central dataset registry key."""
        return "court"

    def build_plan(self, config: Mapping[str, object]) -> DatasetPipelinePlan:
        """Validate court configuration and build its exact command plan."""
        algorithms = require_mapping(config.get("algorithms"), name="algorithms")
        camera_name = require_string(
            algorithms.get("camera_sampling"),
            name="algorithms.camera_sampling",
        )
        label_name = require_string(
            algorithms.get("labels"),
            name="algorithms.labels",
        )
        camera_algorithm = CAMERA_SAMPLING_ALGORITHMS.resolve(camera_name)
        LABEL_ALGORITHMS.resolve(label_name)
        stages = require_mapping(config.get("stages"), name="stages")
        sample_module = (
            "src.synthetic_data_generation.dataset.court.components."
            "camera_sampling.support_probe"
            if camera_algorithm == "sfm_neighborhood"
            else (
                "src.synthetic_data_generation.dataset.court.components."
                "camera_sampling.orbit_plan"
            )
        )
        definitions = (
            (
                "runtime_probe",
                "nht",
                "src.synthetic_data_generation.rendering.nht.runtime_probe",
            ),
            ("sample_cameras", "project", sample_module),
            (
                "render",
                "nht",
                "src.synthetic_data_generation.dataset.court.rendering.nht",
            ),
            (
                "validate",
                "project",
                "src.synthetic_data_generation.dataset.court.validation.dataset",
            ),
            (
                "report",
                "project",
                "src.synthetic_data_generation.dataset.court.reporting.dataset_preview",
            ),
        )
        commands: list[PipelineCommand] = []
        for stage, runtime, module in definitions:
            command = configured_command(
                stages,
                stage=stage,
                runtime=runtime,  # type: ignore[arg-type]
                module=module,
            )
            if command is not None:
                commands.append(command)
        return DatasetPipelinePlan(
            dataset=self.dataset_name,
            selected_algorithms={
                "camera_sampling": camera_name,
                "labels": label_name,
            },
            commands=tuple(commands),
        )


def create_pipeline() -> CourtDatasetPipeline:
    """Construct the built-in court pipeline."""
    return CourtDatasetPipeline()
