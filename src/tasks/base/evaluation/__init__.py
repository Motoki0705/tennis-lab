"""Shared task evaluation contracts."""

from src.tasks.base.evaluation.track_query_reference import (
    AxisWisePositionError,
    PairedReferenceEvaluationError,
    PairedReferencePositionMetrics,
    compute_axis_wise_position_error,
    compute_heading_error_radians,
    compute_paired_reference_position_metrics,
    compute_y_sign_accuracy,
    stratify_metric_by_reference_view_index,
)

__all__ = [
    "AxisWisePositionError",
    "PairedReferenceEvaluationError",
    "PairedReferencePositionMetrics",
    "compute_axis_wise_position_error",
    "compute_heading_error_radians",
    "compute_paired_reference_position_metrics",
    "compute_y_sign_accuracy",
    "stratify_metric_by_reference_view_index",
]
