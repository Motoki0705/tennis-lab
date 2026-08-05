"""Configuration contracts for court annotation evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CourtAnnotationDatasetSpec:
    """One ``data_train.json``-compatible dataset to evaluate."""

    name: str
    annotation_json: Path
    image_dir: Path
    image_extensions: tuple[str, ...]

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name) is None:
            raise ValueError(
                f"Dataset name must be a non-empty path-safe name, got {self.name!r}."
            )
        if not self.image_extensions:
            raise ValueError("image_extensions must contain at least one suffix.")
        normalized = tuple(
            _normalize_extension(value) for value in self.image_extensions
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"image_extensions contains duplicates: {normalized}.")
        object.__setattr__(self, "image_extensions", normalized)


@dataclass(frozen=True)
class HomographyEvaluationCriteria:
    """Thresholds used to accept a 14-keypoint court annotation."""

    ransac_reproj_threshold_normalized: float
    min_inliers: int
    min_template_x_span_ratio: float
    min_template_y_span_ratio: float
    max_inlier_rms_normalized: float
    min_visible_fraction: float
    min_court_area_ratio: float
    max_court_area_ratio: float
    min_line_edge_support: float
    line_distance_tolerance_px: float
    line_evidence_max_side: int
    require_ground_view: bool
    max_opposite_edge_ratio: float

    def __post_init__(self) -> None:
        _require_positive(
            "ransac_reproj_threshold_normalized",
            self.ransac_reproj_threshold_normalized,
        )
        if not 4 <= self.min_inliers <= 14:
            raise ValueError(f"min_inliers must be in [4, 14], got {self.min_inliers}.")
        for name, value in (
            ("min_template_x_span_ratio", self.min_template_x_span_ratio),
            ("min_template_y_span_ratio", self.min_template_y_span_ratio),
            ("min_visible_fraction", self.min_visible_fraction),
            ("min_court_area_ratio", self.min_court_area_ratio),
            ("max_court_area_ratio", self.max_court_area_ratio),
            ("min_line_edge_support", self.min_line_edge_support),
            ("max_opposite_edge_ratio", self.max_opposite_edge_ratio),
        ):
            _require_unit_interval(name, value)
        _require_positive("max_inlier_rms_normalized", self.max_inlier_rms_normalized)
        _require_positive("line_distance_tolerance_px", self.line_distance_tolerance_px)
        if self.line_evidence_max_side <= 0:
            raise ValueError(
                "line_evidence_max_side must be positive, "
                f"got {self.line_evidence_max_side}."
            )
        if self.min_court_area_ratio >= self.max_court_area_ratio:
            raise ValueError(
                "min_court_area_ratio must be smaller than max_court_area_ratio, "
                f"got {self.min_court_area_ratio} and {self.max_court_area_ratio}."
            )


def _normalize_extension(value: str) -> str:
    suffix = value.lower()
    if not suffix.startswith(".") or len(suffix) <= 1:
        raise ValueError(f"Image extension must start with '.', got {value!r}.")
    return suffix


def _require_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _require_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")
