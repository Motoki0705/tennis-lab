"""PLCS dataset pipeline planning with explicit algorithm selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.synthetic_data_generation.dataset.pipeline import (
    DatasetPipelinePlan,
    PipelineCommand,
    configured_command,
    require_mapping,
    require_string,
)
from src.synthetic_data_generation.dataset.plcs.algorithms import (
    AVATAR_CONTROL_ALGORITHMS,
    MOTION_ALGORITHMS,
)


@dataclass(frozen=True)
class PLCSDatasetPipeline:
    """Build reproducible commands for PLCS assets, scenes, and validation."""

    @property
    def dataset_name(self) -> str:
        """Return the central dataset registry key."""
        return "plcs"

    def build_plan(self, config: Mapping[str, object]) -> DatasetPipelinePlan:
        """Validate PLCS configuration and build its exact command plan."""
        algorithms = require_mapping(config.get("algorithms"), name="algorithms")
        control_name = require_string(
            algorithms.get("avatar_control"),
            name="algorithms.avatar_control",
        )
        motion_name = require_string(
            algorithms.get("motion"),
            name="algorithms.motion",
        )
        AVATAR_CONTROL_ALGORITHMS.resolve(control_name)
        MOTION_ALGORITHMS.resolve(motion_name)
        stages = require_mapping(config.get("stages"), name="stages")
        definitions = (
            (
                "runtime_probe",
                "nht",
                "src.synthetic_data_generation.rendering.nht.runtime_probe",
            ),
            (
                "prepare_avatar",
                "project",
                "src.synthetic_data_generation.dataset.plcs.components."
                "avatar_asset_builder",
            ),
            (
                "fit_appearance",
                "nht",
                "src.synthetic_data_generation.dataset.plcs.rendering.avatar_fit",
            ),
            (
                "plan_sequences",
                "project",
                "src.synthetic_data_generation.dataset.plcs.components."
                "scene_plan_builder",
            ),
            (
                "render",
                "nht",
                "src.synthetic_data_generation.dataset.plcs.rendering.nht",
            ),
            (
                "validate",
                "project",
                "src.synthetic_data_generation.dataset.plcs.validation.dataset",
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
                "avatar_control": control_name,
                "motion": motion_name,
            },
            commands=tuple(commands),
        )


def create_pipeline() -> PLCSDatasetPipeline:
    """Construct the built-in PLCS pipeline."""
    return PLCSDatasetPipeline()
