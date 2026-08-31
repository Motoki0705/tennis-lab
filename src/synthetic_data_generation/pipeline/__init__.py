"""Canonical mutable scene pipeline public API."""

from src.synthetic_data_generation.pipeline.contracts import (
    DatasetTarget,
    ScenePipelineRequest,
    StageDefinition,
    StageExecutionSummary,
    StageInput,
    StageName,
    StageStatus,
)
from src.synthetic_data_generation.pipeline.handlers import (
    DeferredStageHandler,
    IngestStageHandler,
    ReportStageHandler,
    VideoProperties,
)
from src.synthetic_data_generation.pipeline.registry import (
    CanonicalStageHandlers,
    StageRegistry,
    canonical_registry,
)
from src.synthetic_data_generation.pipeline.runner import ScenePipelineRunner
from src.synthetic_data_generation.pipeline.workspace import SceneWorkspace

__all__ = [
    "CanonicalStageHandlers",
    "DatasetTarget",
    "DeferredStageHandler",
    "IngestStageHandler",
    "ReportStageHandler",
    "ScenePipelineRequest",
    "ScenePipelineRunner",
    "SceneWorkspace",
    "StageDefinition",
    "StageExecutionSummary",
    "StageInput",
    "StageName",
    "StageRegistry",
    "StageStatus",
    "VideoProperties",
    "canonical_registry",
]
