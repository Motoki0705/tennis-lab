"""Domain datasets published by the canonical mutable scene pipeline."""

from src.synthetic_data_generation.dataset.contracts import (
    DatasetDomain,
    DatasetManifest,
    FrameInventory,
    TargetCourtBinding,
)
from src.synthetic_data_generation.dataset.runtime import (
    BackgroundArrays,
    ChunkReader,
    ChunkWriter,
    DatasetPerformanceBudget,
    DatasetPerformanceMetrics,
    FinalDatasetAssembler,
    ForegroundDelta,
    ForegroundDeltaBatch,
    LogicalRenderSample,
    PerformanceTimer,
    RenderSampleKey,
    RenderSession,
    SharedBackgroundStore,
    ValidatedChunk,
    load_performance_metrics,
    write_performance_metrics,
)

__all__ = [
    "DatasetDomain",
    "DatasetManifest",
    "DatasetPerformanceBudget",
    "DatasetPerformanceMetrics",
    "BackgroundArrays",
    "ChunkReader",
    "ChunkWriter",
    "FinalDatasetAssembler",
    "FrameInventory",
    "ForegroundDelta",
    "ForegroundDeltaBatch",
    "LogicalRenderSample",
    "load_performance_metrics",
    "PerformanceTimer",
    "RenderSampleKey",
    "RenderSession",
    "SharedBackgroundStore",
    "TargetCourtBinding",
    "ValidatedChunk",
    "write_performance_metrics",
]
