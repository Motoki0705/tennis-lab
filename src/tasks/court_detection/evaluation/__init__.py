"""Homography-based court annotation quality evaluation."""

from src.tasks.court_detection.evaluation.contracts import (
    CourtAnnotationDatasetSpec,
    HomographyEvaluationCriteria,
)
from src.tasks.court_detection.evaluation.pipeline import (
    DatasetEvaluationResult,
    evaluate_annotation_dataset,
    evaluate_annotation_datasets,
    write_evaluation_results,
)

__all__ = [
    "CourtAnnotationDatasetSpec",
    "DatasetEvaluationResult",
    "HomographyEvaluationCriteria",
    "evaluate_annotation_dataset",
    "evaluate_annotation_datasets",
    "write_evaluation_results",
]
