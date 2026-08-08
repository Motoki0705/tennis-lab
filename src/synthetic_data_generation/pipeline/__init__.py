"""Canonical mutable scene pipeline public API."""

from src.synthetic_data_generation.pipeline.contracts import (
    DatasetTarget,
    ScenePipelineRequest,
    StageExecutionSummary,
    StageName,
    StageSpec,
    StageStatus,
)
from src.synthetic_data_generation.pipeline.handlers import (
    IngestStageHandler,
    ReportStageHandler,
    VideoProperties,
)
from src.synthetic_data_generation.pipeline.registry import (
    StageRegistry,
    canonical_registry,
)
from src.synthetic_data_generation.pipeline.runner import ScenePipelineRunner
from src.synthetic_data_generation.pipeline.workspace import SceneWorkspace

__all__ = [
    "DatasetTarget",
    "IngestStageHandler",
    "ReportStageHandler",
    "ScenePipelineRequest",
    "ScenePipelineRunner",
    "SceneWorkspace",
    "StageExecutionSummary",
    "StageName",
    "StageRegistry",
    "StageSpec",
    "StageStatus",
    "VideoProperties",
    "canonical_registry",
]
