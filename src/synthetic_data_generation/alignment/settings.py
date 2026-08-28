"""Explicit production settings for measured NHT court-alignment evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from src.utils.schema.court import (
    DOUBLES_WIDTH,
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    SERVICE_LINE_DISTANCE,
    SINGLES_WIDTH,
)


def _positive_float(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    if not math.isfinite(float(value)) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")


def _fraction(value: float, *, name: str, inclusive_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    lower_valid = value >= 0.0 if inclusive_zero else value > 0.0
    if not math.isfinite(float(value)) or not lower_valid or value > 1.0:
        lower = "[0" if inclusive_zero else "(0"
        raise ValueError(f"{name} must lie in {lower}, 1].")


def _positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TypeError(f"{name} must be a positive integer.")


@dataclass(frozen=True, slots=True)
class CourtLineArchitectureSettings:
    """Exact trained court-line network architecture required for strict loading."""

    backbone_name: str
    backbone_strict: bool
    backbone_train_mode: str
    backbone_last_n_blocks: int
    backbone_out_indices: tuple[int, ...]
    backbone_layer_mode: str
    lora_enabled: bool
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    decoder_channels: int
    decoder_reassemble_factors: tuple[float, ...]
    line_bce_weight: float
    line_dice_weight: float
    line_positive_weight: float

    def __post_init__(self) -> None:
        for name in ("backbone_name", "backbone_train_mode", "backbone_layer_mode"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"{name} must be a non-empty string.")
        if (
            type(self.backbone_strict) is not bool
            or type(self.lora_enabled) is not bool
        ):
            raise TypeError("Backbone strictness and LoRA enablement must be booleans.")
        if self.backbone_train_mode not in {"frozen", "last_n", "full"}:
            raise ValueError("backbone_train_mode is invalid.")
        if self.backbone_layer_mode not in {"uniform", "last"}:
            raise ValueError("backbone_layer_mode is invalid.")
        if (
            isinstance(self.backbone_last_n_blocks, bool)
            or not isinstance(self.backbone_last_n_blocks, int)
            or self.backbone_last_n_blocks < 0
        ):
            raise TypeError("backbone_last_n_blocks must be a non-negative integer.")
        if len(self.backbone_out_indices) != 4 or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in self.backbone_out_indices
        ):
            raise ValueError(
                "backbone_out_indices must contain four non-negative integers."
            )
        _positive_int(self.lora_rank, name="lora_rank")
        _positive_float(self.lora_alpha, name="lora_alpha")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must lie in [0, 1).")
        if not self.lora_target_modules or any(
            not isinstance(item, str) or not item for item in self.lora_target_modules
        ):
            raise ValueError("lora_target_modules must contain non-empty strings.")
        _positive_int(self.decoder_channels, name="decoder_channels")
        if len(self.decoder_reassemble_factors) != 4:
            raise ValueError("decoder_reassemble_factors must contain four values.")
        for index, value in enumerate(self.decoder_reassemble_factors):
            _positive_float(value, name=f"decoder_reassemble_factors[{index}]")
        for name in ("line_bce_weight", "line_dice_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite.")
        if self.line_bce_weight == 0.0 and self.line_dice_weight == 0.0:
            raise ValueError("At least one court-line loss weight must be positive.")
        _positive_float(self.line_positive_weight, name="line_positive_weight")


@dataclass(frozen=True, slots=True)
class CourtLineModelSettings:
    """Required trained line-model and deterministic image extraction settings."""

    checkpoint_path: Path
    backbone_repository_path: Path
    backbone_checkpoint_path: Path
    device: str
    expected_short_side: int
    probability_threshold: float
    maximum_selected_pixels_per_camera: int
    architecture: CourtLineArchitectureSettings

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_path",
            "backbone_repository_path",
            "backbone_checkpoint_path",
        ):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{name} must be an explicit absolute Path.")
        if not isinstance(self.device, str) or not self.device.strip():
            raise TypeError("device must be a non-empty string.")
        _positive_int(self.expected_short_side, name="expected_short_side")
        _fraction(
            self.probability_threshold,
            name="probability_threshold",
            inclusive_zero=True,
        )
        _positive_int(
            self.maximum_selected_pixels_per_camera,
            name="maximum_selected_pixels_per_camera",
        )


@dataclass(frozen=True, slots=True)
class GroundPlaneSettings:
    """Required robust ground-plane settings in normalized NHT scene units."""

    footprint_quantile: float
    footprint_margin: float
    minimum_camera_height: float
    maximum_camera_height: float
    histogram_bin_width: float
    candidate_half_width: float
    ransac_threshold: float
    refine_threshold: float
    ransac_iterations: int
    ransac_sample_limit: int
    refine_iterations: int
    minimum_candidate_points: int
    minimum_support_points: int
    minimum_normal_up_cosine: float
    minimum_positive_camera_fraction: float
    support_bounds_quantile: float

    def __post_init__(self) -> None:
        for name in ("footprint_quantile", "support_bounds_quantile"):
            value = getattr(self, name)
            if not 0.0 <= value < 0.5:
                raise ValueError(f"{name} must lie in [0, 0.5).")
        _fraction(self.minimum_normal_up_cosine, name="minimum_normal_up_cosine")
        _fraction(
            self.minimum_positive_camera_fraction,
            name="minimum_positive_camera_fraction",
        )
        for name in (
            "footprint_margin",
            "minimum_camera_height",
            "maximum_camera_height",
            "histogram_bin_width",
            "candidate_half_width",
            "ransac_threshold",
            "refine_threshold",
        ):
            _positive_float(getattr(self, name), name=name)
        if self.minimum_camera_height >= self.maximum_camera_height:
            raise ValueError(
                "minimum_camera_height must be below maximum_camera_height."
            )
        for name in (
            "ransac_iterations",
            "ransac_sample_limit",
            "refine_iterations",
            "minimum_candidate_points",
            "minimum_support_points",
        ):
            _positive_int(getattr(self, name), name=name)


@dataclass(frozen=True, slots=True)
class LineProjectionSettings:
    """Required ray/plane projection, weighting, raster, and evidence gates."""

    minimum_ray_plane_cosine: float
    maximum_ray_distance: float
    bounds_margin: float
    proximity_scale: float
    proximity_power: float
    grid_spacing: float
    minimum_projected_points_per_camera: int

    def __post_init__(self) -> None:
        for name in (
            "minimum_ray_plane_cosine",
            "maximum_ray_distance",
            "bounds_margin",
            "proximity_scale",
            "proximity_power",
            "grid_spacing",
        ):
            _positive_float(getattr(self, name), name=name)
        if self.minimum_ray_plane_cosine >= 1.0:
            raise ValueError("minimum_ray_plane_cosine must be below one.")
        _positive_int(
            self.minimum_projected_points_per_camera,
            name="minimum_projected_points_per_camera",
        )


@dataclass(frozen=True, slots=True)
class CourtCandidateFitSettings:
    """Required deterministic regulation-court Sim(3) search settings."""

    maximum_candidate_count: int
    maximum_retained_state_count: int
    minimum_explained_evidence_fraction: float
    samples_per_metre: float
    minimum_nht_scene_units_per_metre: float
    maximum_nht_scene_units_per_metre: float
    orientation_minimum_radians: float
    orientation_maximum_radians: float
    score_distance_metres: float
    minimum_template_score: float
    family_orientation_tolerance_radians: float
    family_scale_relative_tolerance: float
    minimum_center_separation_metres: float
    optimizer_maximum_iterations: int
    optimizer_population_size: int
    optimizer_tolerance: float
    maximum_fit_points: int
    common_scale_relative_tolerance: float
    scale_bound_margin_relative: float
    evidence_assignment_distance_metres: float
    whole_template_inlier_distance_metres: float
    minimum_whole_template_inlier_fraction: float
    maximum_whole_template_q95_error_metres: float
    minimum_semantic_segment_inlier_fraction: float
    maximum_court_footprint_overlap_fraction: float

    def __post_init__(self) -> None:
        _positive_int(
            self.maximum_candidate_count,
            name="maximum_candidate_count",
        )
        _positive_int(
            self.maximum_retained_state_count,
            name="maximum_retained_state_count",
        )
        _fraction(
            self.minimum_explained_evidence_fraction,
            name="minimum_explained_evidence_fraction",
        )
        if self.minimum_explained_evidence_fraction >= 1.0:
            raise ValueError("minimum_explained_evidence_fraction must be below one.")
        for name in (
            "samples_per_metre",
            "minimum_nht_scene_units_per_metre",
            "maximum_nht_scene_units_per_metre",
            "score_distance_metres",
            "minimum_template_score",
            "family_orientation_tolerance_radians",
            "minimum_center_separation_metres",
            "optimizer_tolerance",
            "evidence_assignment_distance_metres",
            "whole_template_inlier_distance_metres",
            "maximum_whole_template_q95_error_metres",
        ):
            _positive_float(getattr(self, name), name=name)
        if (
            self.minimum_nht_scene_units_per_metre
            >= self.maximum_nht_scene_units_per_metre
        ):
            raise ValueError("Court candidate scale bounds are invalid.")
        if self.orientation_minimum_radians >= self.orientation_maximum_radians:
            raise ValueError("Court candidate orientation bounds are invalid.")
        for name in (
            "family_scale_relative_tolerance",
            "common_scale_relative_tolerance",
            "scale_bound_margin_relative",
        ):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1).")
        if self.scale_bound_margin_relative >= 0.5:
            raise ValueError("scale_bound_margin_relative must be below 0.5.")
        _fraction(
            self.minimum_whole_template_inlier_fraction,
            name="minimum_whole_template_inlier_fraction",
        )
        _fraction(
            self.minimum_semantic_segment_inlier_fraction,
            name="minimum_semantic_segment_inlier_fraction",
        )
        _fraction(
            self.maximum_court_footprint_overlap_fraction,
            name="maximum_court_footprint_overlap_fraction",
            inclusive_zero=True,
        )
        _positive_int(
            self.optimizer_maximum_iterations,
            name="optimizer_maximum_iterations",
        )
        _positive_int(self.optimizer_population_size, name="optimizer_population_size")
        _positive_int(self.maximum_fit_points, name="maximum_fit_points")

    def whole_court_evidence(
        self,
        *,
        required_court_count: int,
        minimum_matches_per_offset_level: int,
    ) -> WholeCourtEvidenceSettings:
        """Freeze geometry-derived identifiability and diagnostic policy."""
        _positive_int(required_court_count, name="required_court_count")
        return WholeCourtEvidenceSettings(
            required_court_count=required_court_count,
            maximum_common_scale_relative_deviation=(
                self.common_scale_relative_tolerance
            ),
            maximum_center_refit_displacement_metres=(
                self.maximum_center_refit_displacement_metres()
            ),
            minimum_distinct_offset_levels=2,
            minimum_matches_per_offset_level=minimum_matches_per_offset_level,
            minimum_level_camera_count=2,
            minimum_secondary_tangential_span_metres=(
                2.0 * self.whole_template_inlier_distance_metres
            ),
            minimum_longitudinal_offset_span_metres=SINGLES_WIDTH,
            minimum_longitudinal_tangential_span_metres=(2.0 * SERVICE_LINE_DISTANCE),
            minimum_transverse_offset_span_metres=(2.0 * SERVICE_LINE_DISTANCE),
            minimum_transverse_tangential_span_metres=SINGLES_WIDTH,
            samples_per_metre=self.samples_per_metre,
            inlier_distance_metres=self.whole_template_inlier_distance_metres,
            minimum_inlier_fraction=self.minimum_whole_template_inlier_fraction,
            maximum_q95_error_metres=self.maximum_whole_template_q95_error_metres,
            minimum_semantic_segment_inlier_fraction=(
                self.minimum_semantic_segment_inlier_fraction
            ),
            minimum_center_separation_metres=self.minimum_center_separation_metres,
            maximum_footprint_overlap_fraction=(
                self.maximum_court_footprint_overlap_fraction
            ),
        )

    def orientation_family_count(self) -> int:
        """Return the number of deterministic <=pi/2 orientation basins."""
        span = self.orientation_maximum_radians - self.orientation_minimum_radians
        return max(1, math.ceil((span - 1.0e-12) / (math.pi / 2.0)))

    def maximum_center_refit_displacement_metres(self) -> float:
        """Return the scale/localization-derived native-center search radius."""
        court_half_diagonal = math.hypot(HALF_DOUBLES_WIDTH, HALF_LENGTH)
        return (
            self.common_scale_relative_tolerance * court_half_diagonal
            + self.whole_template_inlier_distance_metres
        )


@dataclass(frozen=True, slots=True)
class WholeCourtEvidenceSettings:
    """Persistable identifiability policy and whole-template diagnostic policy."""

    required_court_count: int
    maximum_common_scale_relative_deviation: float
    maximum_center_refit_displacement_metres: float
    minimum_distinct_offset_levels: int
    minimum_matches_per_offset_level: int
    minimum_level_camera_count: int
    minimum_secondary_tangential_span_metres: float
    minimum_longitudinal_offset_span_metres: float
    minimum_longitudinal_tangential_span_metres: float
    minimum_transverse_offset_span_metres: float
    minimum_transverse_tangential_span_metres: float
    samples_per_metre: float
    inlier_distance_metres: float
    minimum_inlier_fraction: float
    maximum_q95_error_metres: float
    minimum_semantic_segment_inlier_fraction: float
    minimum_center_separation_metres: float
    maximum_footprint_overlap_fraction: float

    def __post_init__(self) -> None:
        _positive_int(self.required_court_count, name="required_court_count")
        _positive_int(
            self.minimum_distinct_offset_levels,
            name="minimum_distinct_offset_levels",
        )
        if self.minimum_distinct_offset_levels != 2:
            raise ValueError("minimum_distinct_offset_levels must equal two.")
        _positive_int(
            self.minimum_matches_per_offset_level,
            name="minimum_matches_per_offset_level",
        )
        _positive_int(
            self.minimum_level_camera_count,
            name="minimum_level_camera_count",
        )
        if self.minimum_level_camera_count != 2:
            raise ValueError(
                "minimum_level_camera_count must equal two for independent "
                "multiview support."
            )
        for name in (
            "maximum_common_scale_relative_deviation",
            "maximum_center_refit_displacement_metres",
            "minimum_longitudinal_offset_span_metres",
            "minimum_longitudinal_tangential_span_metres",
            "minimum_transverse_offset_span_metres",
            "minimum_transverse_tangential_span_metres",
            "minimum_secondary_tangential_span_metres",
            "samples_per_metre",
            "inlier_distance_metres",
            "maximum_q95_error_metres",
            "minimum_center_separation_metres",
        ):
            _positive_float(getattr(self, name), name=name)
        if self.maximum_common_scale_relative_deviation >= 1.0:
            raise ValueError(
                "maximum_common_scale_relative_deviation must be below one."
            )
        maximum_scale_deviation = (
            2.0
            * self.inlier_distance_metres
            / self.minimum_longitudinal_offset_span_metres
        )
        if (
            self.maximum_common_scale_relative_deviation
            > maximum_scale_deviation + 1.0e-12
        ):
            raise ValueError(
                "maximum_common_scale_relative_deviation exceeds the "
                "localization/semantic-separation bound."
            )
        expected_displacement = (
            self.maximum_common_scale_relative_deviation
            * math.hypot(HALF_DOUBLES_WIDTH, HALF_LENGTH)
            + self.inlier_distance_metres
        )
        if not math.isclose(
            self.maximum_center_refit_displacement_metres,
            expected_displacement,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        ):
            raise ValueError(
                "maximum_center_refit_displacement_metres must equal the "
                "scale/localization-derived court-center bound."
            )
        expected_spans = {
            "minimum_longitudinal_offset_span_metres": SINGLES_WIDTH,
            "minimum_longitudinal_tangential_span_metres": (
                2.0 * SERVICE_LINE_DISTANCE
            ),
            "minimum_transverse_offset_span_metres": (2.0 * SERVICE_LINE_DISTANCE),
            "minimum_transverse_tangential_span_metres": SINGLES_WIDTH,
        }
        for name, expected in expected_spans.items():
            if not math.isclose(
                getattr(self, name), expected, abs_tol=1.0e-12, rel_tol=0.0
            ):
                raise ValueError(
                    f"{name} must equal the regulation-derived value {expected}."
                )
        expected_secondary_span = 2.0 * self.inlier_distance_metres
        if not math.isclose(
            self.minimum_secondary_tangential_span_metres,
            expected_secondary_span,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        ):
            raise ValueError(
                "minimum_secondary_tangential_span_metres must equal twice the "
                "maximum localization error."
            )
        if not math.isclose(
            self.minimum_center_separation_metres,
            DOUBLES_WIDTH,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        ):
            raise ValueError(
                "minimum_center_separation_metres is diagnostic and must equal "
                f"the regulation doubles width {DOUBLES_WIDTH}."
            )
        _fraction(self.minimum_inlier_fraction, name="minimum_inlier_fraction")
        _fraction(
            self.minimum_semantic_segment_inlier_fraction,
            name="minimum_semantic_segment_inlier_fraction",
        )
        _fraction(
            self.maximum_footprint_overlap_fraction,
            name="maximum_footprint_overlap_fraction",
            inclusive_zero=True,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the exact policy persisted with measured ground-line evidence."""
        return {
            "required_court_count": self.required_court_count,
            "maximum_common_scale_relative_deviation": (
                self.maximum_common_scale_relative_deviation
            ),
            "maximum_center_refit_displacement_metres": (
                self.maximum_center_refit_displacement_metres
            ),
            "minimum_distinct_offset_levels": self.minimum_distinct_offset_levels,
            "minimum_matches_per_offset_level": (self.minimum_matches_per_offset_level),
            "minimum_level_camera_count": self.minimum_level_camera_count,
            "minimum_secondary_tangential_span_metres": (
                self.minimum_secondary_tangential_span_metres
            ),
            "minimum_longitudinal_offset_span_metres": (
                self.minimum_longitudinal_offset_span_metres
            ),
            "minimum_longitudinal_tangential_span_metres": (
                self.minimum_longitudinal_tangential_span_metres
            ),
            "minimum_transverse_offset_span_metres": (
                self.minimum_transverse_offset_span_metres
            ),
            "minimum_transverse_tangential_span_metres": (
                self.minimum_transverse_tangential_span_metres
            ),
            "samples_per_metre": self.samples_per_metre,
            "inlier_distance_metres": self.inlier_distance_metres,
            "minimum_inlier_fraction": self.minimum_inlier_fraction,
            "maximum_q95_error_metres": self.maximum_q95_error_metres,
            "minimum_semantic_segment_inlier_fraction": (
                self.minimum_semantic_segment_inlier_fraction
            ),
            "minimum_center_separation_metres": (self.minimum_center_separation_metres),
            "maximum_footprint_overlap_fraction": (
                self.maximum_footprint_overlap_fraction
            ),
        }


@dataclass(frozen=True, slots=True)
class CorrespondenceSettings:
    """Required measured nearest-line correspondence settings."""

    maximum_match_distance_metres: float
    maximum_correspondences_per_camera: int
    minimum_correspondences_per_camera: int

    def __post_init__(self) -> None:
        _positive_float(
            self.maximum_match_distance_metres,
            name="maximum_match_distance_metres",
        )
        _positive_int(
            self.maximum_correspondences_per_camera,
            name="maximum_correspondences_per_camera",
        )
        _positive_int(
            self.minimum_correspondences_per_camera,
            name="minimum_correspondences_per_camera",
        )
        if self.minimum_correspondences_per_camera < 3:
            raise ValueError(
                "minimum_correspondences_per_camera must be at least three."
            )
        if (
            self.minimum_correspondences_per_camera
            > self.maximum_correspondences_per_camera
        ):
            raise ValueError("Minimum correspondences cannot exceed the maximum.")


@dataclass(frozen=True, slots=True)
class AlignmentEvidenceSettings:
    """No-default production authority for deterministic measured evidence."""

    seed: int
    fit_fraction: float
    holdout_fraction: float
    minimum_fit_cameras: int
    minimum_holdout_cameras: int
    camera_prefix_count: int
    line_model: CourtLineModelSettings
    ground_plane: GroundPlaneSettings
    projection: LineProjectionSettings
    candidate_fit: CourtCandidateFitSettings
    correspondences: CorrespondenceSettings

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise TypeError("seed must be a non-negative integer.")
        _fraction(self.fit_fraction, name="fit_fraction")
        _fraction(self.holdout_fraction, name="holdout_fraction")
        if not math.isclose(
            self.fit_fraction + self.holdout_fraction,
            1.0,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        ):
            raise ValueError("fit_fraction and holdout_fraction must sum to one.")
        for name in (
            "minimum_fit_cameras",
            "minimum_holdout_cameras",
            "camera_prefix_count",
        ):
            _positive_int(getattr(self, name), name=name)
        minimum_unit = self.minimum_fit_cameras + self.minimum_holdout_cameras
        expected_fit = minimum_unit * self.fit_fraction
        expected_holdout = minimum_unit * self.holdout_fraction
        if not math.isclose(
            expected_fit,
            self.minimum_fit_cameras,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        ) or not math.isclose(
            expected_holdout,
            self.minimum_holdout_cameras,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        ):
            raise ValueError(
                "Camera-unit partition counts disagree with the "
                "configured fit/holdout fractions."
            )
        camera_unit = self.minimum_fit_cameras + self.minimum_holdout_cameras
        if self.camera_prefix_count % camera_unit != 0:
            raise ValueError(
                "camera_prefix_count must be an integer number of immutable "
                "fit/holdout partition units."
            )
        if (
            self.camera_partition_unit_count()
            < self.candidate_fit.orientation_family_count()
        ):
            raise ValueError(
                "camera_prefix_count must provide at least one immutable camera "
                "unit per orientation search family."
            )
        assignment_distance = self.candidate_fit.evidence_assignment_distance_metres
        minimum_assignment_distance = max(
            self.correspondences.maximum_match_distance_metres,
            self.candidate_fit.whole_template_inlier_distance_metres,
        )
        half_minimum_line_separation = (HALF_DOUBLES_WIDTH - HALF_SINGLES_WIDTH) / 2.0
        if assignment_distance < minimum_assignment_distance:
            raise ValueError(
                "evidence_assignment_distance_metres must cover both the "
                "correspondence and localization gates."
            )
        if assignment_distance >= half_minimum_line_separation:
            raise ValueError(
                "evidence_assignment_distance_metres must be below half the "
                "minimum distinct regulation-line separation."
            )

    def camera_partition_unit_count(self) -> int:
        """Return the candidate-count-independent camera partition capacity."""
        return self.camera_prefix_count // (
            self.minimum_fit_cameras + self.minimum_holdout_cameras
        )

    def require_available_cameras(self, *, available_camera_count: int) -> None:
        """Reject scenes that cannot supply the fixed selection prefix."""
        _positive_int(available_camera_count, name="available_camera_count")
        if available_camera_count < self.camera_prefix_count:
            raise ValueError(
                "Available cameras cannot satisfy the fixed alignment selection: "
                f"{available_camera_count} < {self.camera_prefix_count}."
            )


__all__ = [
    "AlignmentEvidenceSettings",
    "CorrespondenceSettings",
    "CourtCandidateFitSettings",
    "CourtLineArchitectureSettings",
    "CourtLineModelSettings",
    "GroundPlaneSettings",
    "LineProjectionSettings",
    "WholeCourtEvidenceSettings",
]
