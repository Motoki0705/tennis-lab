"""BLCS dataset pipeline planning with explicit algorithm selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.synthetic_data_generation.dataset.blcs.algorithms import (
    BALL_ASSET_ALGORITHMS,
    TRAJECTORY_ALGORITHMS,
)
from src.synthetic_data_generation.dataset.pipeline import (
    DatasetPipelinePlan,
    PipelineCommand,
    configured_command,
    require_mapping,
    require_string,
)


@dataclass(frozen=True)
class BLCSDatasetPipeline:
    """Build reproducible commands for BLCS preparation, render, and validation."""

    @property
    def dataset_name(self) -> str:
        """Return the central dataset registry key."""
        return "blcs"

    def build_plan(self, config: Mapping[str, object]) -> DatasetPipelinePlan:
        """Validate BLCS configuration and build its exact command plan."""
        algorithms = require_mapping(config.get("algorithms"), name="algorithms")
        asset_name = require_string(
            algorithms.get("ball_asset"),
            name="algorithms.ball_asset",
        )
        trajectory_name = require_string(
            algorithms.get("trajectory"),
            name="algorithms.trajectory",
        )
        asset_algorithm = BALL_ASSET_ALGORITHMS.resolve(asset_name)
        TRAJECTORY_ALGORITHMS.resolve(trajectory_name)
        stages = require_mapping(config.get("stages"), name="stages")
        prepare_module = (
            "src.synthetic_data_generation.dataset.blcs.components."
            "procedural_ball_asset_builder"
            if asset_algorithm == "procedural_fibonacci"
            else (
                "src.synthetic_data_generation.dataset.blcs.components."
                "asset_preparation"
            )
        )
        definitions = (
            (
                "runtime_probe",
                "nht",
                "src.synthetic_data_generation.rendering.nht.runtime_probe",
            ),
            ("prepare_assets", "nht", prepare_module),
            (
                "plan_sequences",
                "project",
                "src.synthetic_data_generation.dataset.blcs.components."
                "scene_plan_builder",
            ),
            (
                "render",
                "nht",
                "src.synthetic_data_generation.dataset.blcs.rendering.nht",
            ),
            (
                "validate",
                "project",
                "src.synthetic_data_generation.dataset.blcs.validation.dataset",
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
                "ball_asset": asset_name,
                "trajectory": trajectory_name,
            },
            commands=tuple(commands),
        )


def create_pipeline() -> BLCSDatasetPipeline:
    """Construct the built-in BLCS pipeline."""
    return BLCSDatasetPipeline()
