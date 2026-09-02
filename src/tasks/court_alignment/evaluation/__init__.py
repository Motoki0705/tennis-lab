"""Evaluation adapters for measured court-alignment evidence."""

from src.tasks.court_alignment.evaluation.real_heatmap import (
    AlignmentGroundPlane,
    DecoderOptions,
    MetricOptions,
    PixelUVTransform,
    PreprocessOptions,
    RealHeatmapArchive,
    RealHeatmapEvaluationRequest,
    evaluate_real_heatmap,
    write_evaluation_artifacts,
)

__all__ = [
    "AlignmentGroundPlane",
    "DecoderOptions",
    "MetricOptions",
    "PixelUVTransform",
    "PreprocessOptions",
    "RealHeatmapArchive",
    "RealHeatmapEvaluationRequest",
    "evaluate_real_heatmap",
    "write_evaluation_artifacts",
]
