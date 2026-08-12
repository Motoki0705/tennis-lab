"""Composition root for the sole canonical scene-pipeline application."""

from __future__ import annotations

from src.synthetic_data_generation.alignment import (
    create_production_alignment_handler,
)
from src.synthetic_data_generation.configuration import ScenePipelineConfiguration
from src.synthetic_data_generation.dataset.blcs.handler import BLCSDatasetStageHandler
from src.synthetic_data_generation.dataset.blcs.rendering import BLCSNHTRenderer
from src.synthetic_data_generation.dataset.blcs.source import (
    PhysicsBLCSTrajectoryProvider,
)
from src.synthetic_data_generation.dataset.court.handler import (
    CourtDatasetStageHandler,
)
from src.synthetic_data_generation.dataset.court.rendering import CourtNHTRenderer
from src.synthetic_data_generation.dataset.plcs.handler import PLCSStageHandler
from src.synthetic_data_generation.dataset.plcs.rendering import NHTPLCSRenderer
from src.synthetic_data_generation.pipeline.contracts import DatasetTarget
from src.synthetic_data_generation.pipeline.handlers import (
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
from src.tasks.plcs.generate_dataset.sampling.motion_source import ACCADMotionLibrary


def build_stage_registry(
    runtime: ScenePipelineConfiguration,
) -> StageRegistry:
    """Bind every modular handler into the exhaustive typed definitions."""
    nht = runtime.nht
    render_environment = dict(nht.environment)
    alignment = create_production_alignment_handler(
        settings=runtime.alignment.evidence,
        policy=runtime.alignment.acceptance,
        resolver=runtime.resolver,
    )
    court = CourtDatasetStageHandler(
        configuration=runtime.court,
        profile=runtime.profile,
        renderer=CourtNHTRenderer(
            executable=nht.render_executable,
            client=NHTRenderClient(),
            environment=render_environment,
            timeout_seconds=nht.render_timeout_seconds,
        ),
    )
    blcs = BLCSDatasetStageHandler(
        workspace=runtime.workspace,
        configuration=runtime.blcs,
        camera_configuration=runtime.camera,
        seed=runtime.stages.seed,
        assets=runtime.blcs.assets,
        trajectory_provider=PhysicsBLCSTrajectoryProvider(
            generator_config=runtime.blcs.generator,
            settings=runtime.blcs.trajectory_source,
        ),
        renderer=BLCSNHTRenderer(
            assets=runtime.blcs.assets,
            client=NHTRenderClient(),
            executable=nht.render_executable,
            environment=render_environment,
            timeout_seconds=runtime.blcs.render_timeout_seconds,
            execution_device=runtime.blcs.performance.execution_device,
            maximum_batch_frames=runtime.blcs.performance.maximum_batch_frames,
        ),
    )
    plcs = PLCSStageHandler(
        configuration=runtime.plcs,
        camera_configuration=runtime.camera,
        motion_library=ACCADMotionLibrary.from_root(runtime.plcs.accad_root),
        avatar_appearance_source=runtime.plcs.appearance,
        renderer=NHTPLCSRenderer(
            client=NHTRenderClient(),
            compositor=runtime.plcs.foreground_compositor,
            executable=nht.render_executable,
            environment=render_environment,
            timeout_seconds=runtime.plcs.render_timeout_seconds,
        ),
        parameters=runtime.plcs.build_stage_parameters(seed=runtime.stages.seed),
    )
    dataset_manifests = {
        target: runtime.workspace.root / "datasets" / target.value / "dataset.json"
        for target in DatasetTarget
    }
    handlers = CanonicalStageHandlers(
        ingest=IngestStageHandler(),
        reconstruction=NHTReconstructionHandler(
            executable=nht.reconstruct_executable,
            pipeline_config=nht.pipeline_config,
            environment=dict(nht.environment),
            timeout_seconds=nht.reconstruction_timeout_seconds,
        ),
        alignment=alignment,
        court_dataset=court,
        blcs_dataset=blcs,
        plcs_dataset=plcs,
        report=ReportStageHandler(
            alignment_directory=runtime.workspace.root / "alignment",
            dataset_manifests=dataset_manifests,
        ),
    )
    return canonical_registry(handlers)


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
