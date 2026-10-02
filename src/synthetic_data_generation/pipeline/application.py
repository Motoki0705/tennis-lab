"""Composition root for the sole canonical scene-pipeline application."""

from __future__ import annotations

from typing import cast

from src.synthetic_data_generation.alignment import (
    create_production_alignment_handler,
)
from src.synthetic_data_generation.alignment.line_inputs import (
    NHTRenderedAlignmentLineInputSource,
)
from src.synthetic_data_generation.configuration import ScenePipelineConfiguration
from src.synthetic_data_generation.pipeline.contracts import (
    DatasetTarget,
    StageExecutionSummary,
    StageHandler,
)
from src.synthetic_data_generation.pipeline.handlers import (
    DeferredStageHandler,
    IngestStageHandler,
    ReportStageHandler,
)
from src.synthetic_data_generation.pipeline.registry import (
    CanonicalStageHandlers,
    StageRegistry,
    canonical_registry,
)
from src.synthetic_data_generation.pipeline.runner import ScenePipelineRunner
from src.synthetic_data_generation.reconstruction import NHTReconstructionHandler
from src.synthetic_data_generation.rendering.nht import NHTRenderClient


def build_stage_registry(
    runtime: ScenePipelineConfiguration,
) -> StageRegistry:
    """Bind every modular handler into the exhaustive typed definitions."""
    nht = runtime.nht
    alignment = create_production_alignment_handler(
        settings=runtime.alignment.evidence,
        policy=runtime.alignment.acceptance,
        resolver=runtime.resolver,
        input_source=NHTRenderedAlignmentLineInputSource(
            client=NHTRenderClient(),
            executable=nht.render_executable,
            environment=nht.environment,
            timeout_seconds=nht.render_timeout_seconds,
        ),
    )
    handlers = CanonicalStageHandlers(
        ingest=IngestStageHandler(),
        reconstruction=NHTReconstructionHandler(
            executable=nht.reconstruct_executable,
            pipeline_config=nht.pipeline_config,
            training_runtime=nht.training_runtime,
            environment=dict(nht.environment),
            timeout_seconds=nht.reconstruction_timeout_seconds,
        ),
        alignment=alignment,
        court_dataset=DeferredStageHandler(lambda: _build_court_handler(runtime)),
        report=DeferredStageHandler(lambda: _build_report_handler(runtime)),
    )
    return canonical_registry(handlers)


def _build_court_handler(
    runtime: ScenePipelineConfiguration,
) -> StageHandler[StageExecutionSummary]:
    from src.synthetic_data_generation.dataset.court.handler import (
        CourtDatasetStageHandler,
    )
    from src.synthetic_data_generation.dataset.court.rendering import CourtNHTRenderer
    from src.synthetic_data_generation.rendering.nht import NHTRenderClient

    nht = runtime.nht
    return cast(
        StageHandler[StageExecutionSummary],
        CourtDatasetStageHandler(
            configuration=runtime.court,
            profile=runtime.profile,
            renderer=CourtNHTRenderer(
                executable=nht.render_executable,
                client=NHTRenderClient(),
                environment=dict(nht.environment),
                timeout_seconds=nht.render_timeout_seconds,
            ),
        ),
    )


def _build_report_handler(
    runtime: ScenePipelineConfiguration,
) -> StageHandler[StageExecutionSummary]:
    dataset_manifests = {
        target: runtime.workspace.root / "datasets" / target.value / "dataset.json"
        for target in DatasetTarget
    }
    return ReportStageHandler(
        alignment_directory=runtime.workspace.root / "alignment",
        dataset_manifests=dataset_manifests,
    )


def build_scene_pipeline_runner(
    runtime: ScenePipelineConfiguration,
    *,
    resolved_config_yaml: str,
) -> ScenePipelineRunner:
    """Construct the runner after the Hydra boundary has resolved all values."""
    return ScenePipelineRunner(
        workspace=runtime.workspace,
        registry=build_stage_registry(runtime),
        resolved_config_yaml=resolved_config_yaml,
    )


__all__ = ["build_scene_pipeline_runner", "build_stage_registry"]
