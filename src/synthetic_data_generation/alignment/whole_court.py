"""Whole-template evidence and physical topology checks for court alignment."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from src.synthetic_data_generation.alignment.contracts import CorrespondenceSet
from src.synthetic_data_generation.alignment.settings import (
    WholeCourtEvidenceSettings,
)
from src.synthetic_data_generation.scene_contract import RigidTransform
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    SERVICE_LINE_DISTANCE,
)

_SEMANTIC_ATOL_METRES = 1.0e-8
_LONGITUDINAL_OFFSETS = (
    -HALF_DOUBLES_WIDTH,
    -HALF_SINGLES_WIDTH,
    0.0,
    HALF_SINGLES_WIDTH,
    HALF_DOUBLES_WIDTH,
)
_TRANSVERSE_OFFSETS = (
    -HALF_LENGTH,
    -SERVICE_LINE_DISTANCE,
    SERVICE_LINE_DISTANCE,
    HALF_LENGTH,
)


@dataclass(frozen=True, slots=True)
class CourtLineSegment:
    """One named regulation-court line segment and its semantic family."""

    name: str
    family: str
    start: tuple[float, float]
    end: tuple[float, float]


COURT_LINE_SEGMENTS = (
    CourtLineSegment(
        "doubles_sideline_left",
        "sidelines",
        (-HALF_DOUBLES_WIDTH, -HALF_LENGTH),
        (-HALF_DOUBLES_WIDTH, HALF_LENGTH),
    ),
    CourtLineSegment(
        "doubles_sideline_right",
        "sidelines",
        (HALF_DOUBLES_WIDTH, -HALF_LENGTH),
        (HALF_DOUBLES_WIDTH, HALF_LENGTH),
    ),
    CourtLineSegment(
        "singles_sideline_left",
        "sidelines",
        (-HALF_SINGLES_WIDTH, -HALF_LENGTH),
        (-HALF_SINGLES_WIDTH, HALF_LENGTH),
    ),
    CourtLineSegment(
        "singles_sideline_right",
        "sidelines",
        (HALF_SINGLES_WIDTH, -HALF_LENGTH),
        (HALF_SINGLES_WIDTH, HALF_LENGTH),
    ),
    CourtLineSegment(
        "baseline_near",
        "baselines",
        (-HALF_DOUBLES_WIDTH, -HALF_LENGTH),
        (HALF_DOUBLES_WIDTH, -HALF_LENGTH),
    ),
    CourtLineSegment(
        "baseline_far",
        "baselines",
        (-HALF_DOUBLES_WIDTH, HALF_LENGTH),
        (HALF_DOUBLES_WIDTH, HALF_LENGTH),
    ),
    CourtLineSegment(
        "service_line_near",
        "service_lines",
        (-HALF_SINGLES_WIDTH, -SERVICE_LINE_DISTANCE),
        (HALF_SINGLES_WIDTH, -SERVICE_LINE_DISTANCE),
    ),
    CourtLineSegment(
        "service_line_far",
        "service_lines",
        (-HALF_SINGLES_WIDTH, SERVICE_LINE_DISTANCE),
        (HALF_SINGLES_WIDTH, SERVICE_LINE_DISTANCE),
    ),
    CourtLineSegment(
        "center_service_t",
        "center_service_t",
        (0.0, -SERVICE_LINE_DISTANCE),
        (0.0, SERVICE_LINE_DISTANCE),
    ),
)


@dataclass(frozen=True, slots=True)
class WholeTemplateMetrics:
    """Distances for every template sample, including unmatched line regions."""

    template_sample_count: int
    supported_sample_count: int
    inlier_fraction: float
    rms_error_m: float
    q95_error_m: float
    maximum_error_m: float
    segment_inlier_fractions: Mapping[str, float]
    family_inlier_fractions: Mapping[str, float]

    def threshold_checks(
        self,
        settings: WholeCourtEvidenceSettings,
    ) -> dict[str, bool]:
        """Evaluate all whole-template and semantic gates without fallback."""
        return {
            "minimum_whole_template_inlier_fraction": (
                bool(self.inlier_fraction >= settings.minimum_inlier_fraction)
            ),
            "maximum_whole_template_q95_error_metres": (
                bool(self.q95_error_m <= settings.maximum_q95_error_metres)
            ),
            "minimum_semantic_segment_inlier_fraction": all(
                fraction >= settings.minimum_semantic_segment_inlier_fraction
                for fraction in self.segment_inlier_fractions.values()
            ),
        }

    def to_dict(
        self,
        *,
        settings: WholeCourtEvidenceSettings,
    ) -> dict[str, object]:
        """Return deterministic quantitative whole-court diagnostics."""
        return {
            "template_sample_count": self.template_sample_count,
            "supported_sample_count": self.supported_sample_count,
            "inlier_fraction": self.inlier_fraction,
            "rms_error_m": self.rms_error_m,
            "q95_error_m": self.q95_error_m,
            "maximum_error_m": self.maximum_error_m,
            "segment_inlier_fractions": dict(self.segment_inlier_fractions),
            "family_inlier_fractions": dict(self.family_inlier_fractions),
            "diagnostic_threshold_checks": self.threshold_checks(settings),
        }


@dataclass(frozen=True, slots=True)
class SemanticOffsetLevelMetrics:
    """Exclusive matches assigned to one regulation-line offset level."""

    offset_metres: float
    match_count: int
    unique_template_sample_count: int
    tangential_span_metres: float
    tangential_bin_size_metres: float
    unique_tangential_bin_count: int
    required_unique_tangential_bin_count: int
    camera_ids: tuple[str, ...]
    threshold_checks: Mapping[str, bool]
    anchor_eligible: bool
    secondary_eligible: bool
    supported: bool

    def to_dict(self) -> dict[str, object]:
        """Return stable per-level evidence diagnostics."""
        return {
            "offset_metres": self.offset_metres,
            "match_count": self.match_count,
            "unique_template_sample_count": self.unique_template_sample_count,
            "tangential_span_metres": self.tangential_span_metres,
            "tangential_bin_size_metres": self.tangential_bin_size_metres,
            "unique_tangential_bin_count": self.unique_tangential_bin_count,
            "required_unique_tangential_bin_count": (
                self.required_unique_tangential_bin_count
            ),
            "camera_ids": list(self.camera_ids),
            "threshold_checks": dict(self.threshold_checks),
            "anchor_eligible": self.anchor_eligible,
            "secondary_eligible": self.secondary_eligible,
            "supported": self.supported,
        }


@dataclass(frozen=True, slots=True)
class QualifyingOffsetPairMetrics:
    """Deterministically selected anchor/secondary semantic offset pair."""

    anchor_offset_metres: float
    secondary_offset_metres: float
    offset_separation_metres: float
    camera_ids: tuple[str, ...]
    threshold_checks: Mapping[str, bool]
    rejection_reasons: tuple[str, ...]
    accepted: bool

    def to_dict(self) -> dict[str, object]:
        """Return the selected levels and every positive-evidence reason."""
        return {
            "anchor_offset_metres": self.anchor_offset_metres,
            "secondary_offset_metres": self.secondary_offset_metres,
            "offset_separation_metres": self.offset_separation_metres,
            "camera_ids": list(self.camera_ids),
            "threshold_checks": dict(self.threshold_checks),
            "rejection_reasons": list(self.rejection_reasons),
            "accepted": self.accepted,
        }


@dataclass(frozen=True, slots=True)
class LineFamilyIdentifiabilityMetrics:
    """Positive geometric evidence for one parallel regulation-line family."""

    family: str
    match_count: int
    camera_ids: tuple[str, ...]
    offset_levels: tuple[SemanticOffsetLevelMetrics, ...]
    supported_offset_level_count: int
    offset_span_metres: float
    qualifying_pair: QualifyingOffsetPairMetrics

    def threshold_checks(
        self,
        *,
        minimum_offset_span_metres: float,
        settings: WholeCourtEvidenceSettings,
    ) -> dict[str, bool]:
        """Evaluate the persisted anchor/secondary evidence requirements."""
        del minimum_offset_span_metres, settings
        return dict(self.qualifying_pair.threshold_checks)

    def to_dict(
        self,
        *,
        minimum_camera_count: int,
        minimum_offset_span_metres: float,
        minimum_tangential_span_metres: float,
        settings: WholeCourtEvidenceSettings,
    ) -> dict[str, object]:
        """Return quantitative family evidence and its acceptance checks."""
        checks = self.threshold_checks(
            minimum_offset_span_metres=minimum_offset_span_metres,
            settings=settings,
        )
        return {
            "family": self.family,
            "match_count": self.match_count,
            "camera_ids": list(self.camera_ids),
            "offset_levels": [item.to_dict() for item in self.offset_levels],
            "supported_offset_level_count": self.supported_offset_level_count,
            "offset_span_metres": self.offset_span_metres,
            "minimum_family_camera_count": minimum_camera_count,
            "minimum_anchor_tangential_span_metres": (minimum_tangential_span_metres),
            "minimum_multiview_camera_count_per_level": (
                settings.minimum_level_camera_count
            ),
            "minimum_secondary_unique_template_samples": (
                settings.minimum_matches_per_offset_level
            ),
            "minimum_secondary_tangential_span_metres": (
                settings.minimum_secondary_tangential_span_metres
            ),
            "qualifying_anchor_secondary_pair": self.qualifying_pair.to_dict(),
            "qualifying_camera_ids": list(self.qualifying_pair.camera_ids),
            "threshold_checks": checks,
            "accepted": all(checks.values()),
        }


@dataclass(frozen=True, slots=True)
class CourtIdentifiabilityMetrics:
    """Exclusive two-family geometric identifiability for one court partition."""

    longitudinal: LineFamilyIdentifiabilityMetrics
    transverse: LineFamilyIdentifiabilityMetrics

    def to_dict(
        self,
        *,
        minimum_camera_count: int,
        settings: WholeCourtEvidenceSettings,
    ) -> dict[str, object]:
        """Return persisted positive-evidence diagnostics and acceptance."""
        longitudinal = self.longitudinal.to_dict(
            minimum_camera_count=minimum_camera_count,
            minimum_offset_span_metres=(
                settings.minimum_longitudinal_offset_span_metres
            ),
            minimum_tangential_span_metres=(
                settings.minimum_longitudinal_tangential_span_metres
            ),
            settings=settings,
        )
        transverse = self.transverse.to_dict(
            minimum_camera_count=minimum_camera_count,
            minimum_offset_span_metres=(settings.minimum_transverse_offset_span_metres),
            minimum_tangential_span_metres=(
                settings.minimum_transverse_tangential_span_metres
            ),
            settings=settings,
        )
        return {
            "minimum_camera_count": minimum_camera_count,
            "longitudinal": longitudinal,
            "transverse": transverse,
            "accepted": (
                longitudinal["accepted"] is True and transverse["accepted"] is True
            ),
        }


@dataclass(frozen=True, slots=True)
class CourtPairTopologyMetrics:
    """Physical separation and footprint intersection for one court pair."""

    first_candidate_id: str
    second_candidate_id: str
    center_separation_metres: float
    footprint_overlap_fraction: float

    def threshold_checks(
        self,
        settings: WholeCourtEvidenceSettings,
    ) -> dict[str, bool]:
        """Evaluate configured multi-court topology gates."""
        return {
            "diagnostic_minimum_center_separation_metres": (
                bool(
                    self.center_separation_metres
                    >= settings.minimum_center_separation_metres
                )
            ),
            "maximum_footprint_overlap_fraction": (
                bool(
                    self.footprint_overlap_fraction
                    <= settings.maximum_footprint_overlap_fraction
                )
            ),
        }

    def to_dict(
        self,
        *,
        settings: WholeCourtEvidenceSettings,
    ) -> dict[str, object]:
        """Return deterministic quantitative pair-topology diagnostics."""
        checks = self.threshold_checks(settings)
        return {
            "candidate_ids": [self.first_candidate_id, self.second_candidate_id],
            "center_separation_metres": self.center_separation_metres,
            "footprint_overlap_fraction": self.footprint_overlap_fraction,
            "threshold_checks": checks,
            "accepted": all(checks.values()),
        }


def evaluate_court_identifiability(
    correspondences: CorrespondenceSet,
    *,
    minimum_camera_count: int,
    settings: WholeCourtEvidenceSettings,
) -> CourtIdentifiabilityMetrics:
    """Evaluate exclusive regulation geometry without treating absence as evidence."""
    if (
        isinstance(minimum_camera_count, bool)
        or not isinstance(minimum_camera_count, int)
        or minimum_camera_count < 1
    ):
        raise TypeError("minimum_camera_count must be a positive integer.")
    points = correspondences.points_court
    longitudinal_membership = _longitudinal_membership(points)
    transverse_membership = _transverse_membership(points)
    ambiguous = longitudinal_membership & transverse_membership
    longitudinal = _line_family_metrics(
        family="longitudinal",
        points=points,
        camera_ids=correspondences.camera_ids,
        family_mask=longitudinal_membership & ~ambiguous,
        offset_axis=0,
        tangential_axis=1,
        semantic_offsets=_LONGITUDINAL_OFFSETS,
        minimum_camera_count=minimum_camera_count,
        minimum_tangential_span_metres=(
            settings.minimum_longitudinal_tangential_span_metres
        ),
        settings=settings,
    )
    transverse = _line_family_metrics(
        family="transverse",
        points=points,
        camera_ids=correspondences.camera_ids,
        family_mask=transverse_membership & ~ambiguous,
        offset_axis=1,
        tangential_axis=0,
        semantic_offsets=_TRANSVERSE_OFFSETS,
        minimum_camera_count=minimum_camera_count,
        minimum_tangential_span_metres=(
            settings.minimum_transverse_tangential_span_metres
        ),
        settings=settings,
    )
    return CourtIdentifiabilityMetrics(
        longitudinal=longitudinal,
        transverse=transverse,
    )


def _longitudinal_membership(
    points: NDArray[np.float64],
) -> NDArray[np.bool_]:
    x = points[:, 0]
    y = points[:, 1]
    sideline = _matches_any(
        x, _LONGITUDINAL_OFFSETS[:2] + _LONGITUDINAL_OFFSETS[3:]
    ) & (np.abs(y) <= HALF_LENGTH + _SEMANTIC_ATOL_METRES)
    center_service = np.isclose(x, 0.0, atol=_SEMANTIC_ATOL_METRES, rtol=0.0) & (
        np.abs(y) <= SERVICE_LINE_DISTANCE + _SEMANTIC_ATOL_METRES
    )
    return np.asarray(sideline | center_service, dtype=np.bool_)


def _transverse_membership(
    points: NDArray[np.float64],
) -> NDArray[np.bool_]:
    x = points[:, 0]
    y = points[:, 1]
    baseline = _matches_any(y, (-HALF_LENGTH, HALF_LENGTH)) & (
        np.abs(x) <= HALF_DOUBLES_WIDTH + _SEMANTIC_ATOL_METRES
    )
    service_line = _matches_any(y, (-SERVICE_LINE_DISTANCE, SERVICE_LINE_DISTANCE)) & (
        np.abs(x) <= HALF_SINGLES_WIDTH + _SEMANTIC_ATOL_METRES
    )
    return np.asarray(baseline | service_line, dtype=np.bool_)


def _matches_any(
    values: NDArray[np.float64],
    targets: Sequence[float],
) -> NDArray[np.bool_]:
    return np.asarray(
        np.any(
            np.isclose(
                values[:, None],
                np.asarray(targets, dtype=np.float64)[None, :],
                atol=_SEMANTIC_ATOL_METRES,
                rtol=0.0,
            ),
            axis=1,
        ),
        dtype=np.bool_,
    )


def _line_family_metrics(
    *,
    family: str,
    points: NDArray[np.float64],
    camera_ids: tuple[str, ...],
    family_mask: NDArray[np.bool_],
    offset_axis: int,
    tangential_axis: int,
    semantic_offsets: tuple[float, ...],
    minimum_camera_count: int,
    minimum_tangential_span_metres: float,
    settings: WholeCourtEvidenceSettings,
) -> LineFamilyIdentifiabilityMetrics:
    family_indices = np.flatnonzero(family_mask)
    family_points = points[family_indices]
    family_cameras = np.asarray(camera_ids, dtype=np.str_)[family_indices]
    levels: list[SemanticOffsetLevelMetrics] = []
    bin_size = 2.0 * settings.inlier_distance_metres
    required_anchor_bin_count = (
        math.floor(minimum_tangential_span_metres / bin_size) + 1
    )
    for offset in semantic_offsets:
        mask = np.isclose(
            family_points[:, offset_axis],
            offset,
            atol=_SEMANTIC_ATOL_METRES,
            rtol=0.0,
        )
        level_points = family_points[mask]
        level_cameras = tuple(dict.fromkeys(family_cameras[mask].tolist()))
        unique_tangential = np.unique(level_points[:, tangential_axis])
        tangential_span = (
            float(np.ptp(unique_tangential)) if len(unique_tangential) else 0.0
        )
        tangential_minimum = _tangential_minimum(family=family, offset=offset)
        occupied_bins = np.unique(
            np.floor(
                (unique_tangential - tangential_minimum) / bin_size + 1.0e-10
            ).astype(np.int64)
        )
        checks: dict[str, bool] = {
            "minimum_multiview_camera_count": bool(
                len(level_cameras) >= settings.minimum_level_camera_count
            ),
            "minimum_anchor_tangential_span_metres": (
                bool(tangential_span >= minimum_tangential_span_metres)
            ),
            "minimum_anchor_unique_tangential_bins": (
                bool(len(occupied_bins) >= required_anchor_bin_count)
            ),
            "minimum_secondary_unique_template_samples": (
                bool(
                    len(unique_tangential) >= settings.minimum_matches_per_offset_level
                )
            ),
            "minimum_secondary_unique_tangential_bins": (
                bool(len(occupied_bins) >= settings.minimum_matches_per_offset_level)
            ),
            "minimum_secondary_tangential_span_metres": (
                bool(
                    tangential_span >= settings.minimum_secondary_tangential_span_metres
                )
            ),
        }
        anchor_eligible = all(
            checks[name]
            for name in (
                "minimum_multiview_camera_count",
                "minimum_anchor_tangential_span_metres",
                "minimum_anchor_unique_tangential_bins",
            )
        )
        secondary_eligible = all(
            checks[name]
            for name in (
                "minimum_multiview_camera_count",
                "minimum_secondary_unique_template_samples",
                "minimum_secondary_unique_tangential_bins",
                "minimum_secondary_tangential_span_metres",
            )
        )
        levels.append(
            SemanticOffsetLevelMetrics(
                offset_metres=offset,
                match_count=len(level_points),
                unique_template_sample_count=len(unique_tangential),
                tangential_span_metres=tangential_span,
                tangential_bin_size_metres=bin_size,
                unique_tangential_bin_count=len(occupied_bins),
                required_unique_tangential_bin_count=required_anchor_bin_count,
                camera_ids=level_cameras,
                threshold_checks=checks,
                anchor_eligible=anchor_eligible,
                secondary_eligible=secondary_eligible,
                supported=anchor_eligible or secondary_eligible,
            )
        )
    family_camera_ids = tuple(dict.fromkeys(family_cameras.tolist()))
    qualifying_pair = _select_anchor_secondary_pair(
        levels,
        minimum_family_camera_count=minimum_camera_count,
        minimum_offset_separation_metres=(
            settings.minimum_longitudinal_offset_span_metres
            if family == "longitudinal"
            else settings.minimum_transverse_offset_span_metres
        ),
    )
    return LineFamilyIdentifiabilityMetrics(
        family=family,
        match_count=len(family_points),
        camera_ids=family_camera_ids,
        offset_levels=tuple(levels),
        supported_offset_level_count=sum(item.supported for item in levels),
        offset_span_metres=qualifying_pair.offset_separation_metres,
        qualifying_pair=qualifying_pair,
    )


def _select_anchor_secondary_pair(
    levels: Sequence[SemanticOffsetLevelMetrics],
    *,
    minimum_family_camera_count: int,
    minimum_offset_separation_metres: float,
) -> QualifyingOffsetPairMetrics:
    """Choose one stable pair, retaining exact reasons when no pair qualifies."""
    candidates: list[QualifyingOffsetPairMetrics] = []
    for anchor in levels:
        for secondary in levels:
            if anchor.offset_metres == secondary.offset_metres:
                continue
            separation = abs(anchor.offset_metres - secondary.offset_metres)
            qualifying_camera_ids = tuple(
                dict.fromkeys((*anchor.camera_ids, *secondary.camera_ids))
            )
            checks = {
                "minimum_family_camera_count": bool(
                    len(qualifying_camera_ids) >= minimum_family_camera_count
                ),
                "anchor_level_eligible": anchor.anchor_eligible,
                "secondary_level_eligible": secondary.secondary_eligible,
                "minimum_offset_separation_metres": bool(
                    separation >= minimum_offset_separation_metres
                ),
            }
            reasons = tuple(name for name, accepted in checks.items() if not accepted)
            candidates.append(
                QualifyingOffsetPairMetrics(
                    anchor_offset_metres=anchor.offset_metres,
                    secondary_offset_metres=secondary.offset_metres,
                    offset_separation_metres=separation,
                    camera_ids=qualifying_camera_ids,
                    threshold_checks=checks,
                    rejection_reasons=reasons,
                    accepted=all(checks.values()),
                )
            )
    if not candidates:
        raise RuntimeError("A regulation line family must expose two semantic levels.")
    candidates.sort(
        key=lambda item: (
            not item.accepted,
            len(item.rejection_reasons),
            item.anchor_offset_metres,
            item.secondary_offset_metres,
        )
    )
    return candidates[0]


def _tangential_minimum(*, family: str, offset: float) -> float:
    if family == "longitudinal":
        return float(
            -SERVICE_LINE_DISTANCE
            if math.isclose(offset, 0.0, abs_tol=_SEMANTIC_ATOL_METRES)
            else -HALF_LENGTH
        )
    if family == "transverse":
        return float(
            -HALF_DOUBLES_WIDTH
            if math.isclose(abs(offset), HALF_LENGTH, abs_tol=_SEMANTIC_ATOL_METRES)
            else -HALF_SINGLES_WIDTH
        )
    raise ValueError(f"Unknown line family {family!r}.")


def sample_court_line_segments(
    samples_per_metre: float,
) -> tuple[tuple[CourtLineSegment, NDArray[np.float64]], ...]:
    """Sample every semantic regulation-court segment independently."""
    if not math.isfinite(samples_per_metre) or samples_per_metre <= 0.0:
        raise ValueError("samples_per_metre must be positive and finite.")
    sampled: list[tuple[CourtLineSegment, NDArray[np.float64]]] = []
    for segment in COURT_LINE_SEGMENTS:
        start = np.asarray(segment.start, dtype=np.float64)
        end = np.asarray(segment.end, dtype=np.float64)
        count = max(16, int(np.linalg.norm(end - start) * samples_per_metre))
        fraction = np.linspace(0.0, 1.0, count, dtype=np.float64)[:, None]
        points = start * (1.0 - fraction) + end * fraction
        sampled.append((segment, np.asarray(points, dtype=np.float64)))
    return tuple(sampled)


def sample_court_line_template(samples_per_metre: float) -> NDArray[np.float64]:
    """Return the complete sampled regulation-court line template."""
    return np.concatenate(
        [points for _segment, points in sample_court_line_segments(samples_per_metre)]
    )


def transform_template_2d(
    template: NDArray[np.float64],
    parameters: Sequence[float] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply center, orientation, and physical scale to a 2-D court template."""
    points = np.asarray(template, dtype=np.float64)
    values = np.asarray(parameters, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("template must be a finite (N, 2) array.")
    if values.shape != (4,) or not np.isfinite(values).all() or values[3] <= 0.0:
        raise ValueError(
            "parameters must contain finite center/orientation/positive scale."
        )
    center_u, center_v, orientation, scale = values
    cosine = math.cos(float(orientation))
    sine = math.sin(float(orientation))
    rotation_transpose = np.asarray(
        ((cosine, sine), (-sine, cosine)),
        dtype=np.float64,
    )
    return np.asarray(
        points @ rotation_transpose * float(scale) + np.asarray((center_u, center_v)),
        dtype=np.float64,
    )


def evaluate_whole_template(
    *,
    scene_from_court: RigidTransform,
    measured_points_scene: NDArray[np.float64],
    settings: WholeCourtEvidenceSettings,
) -> WholeTemplateMetrics:
    """Measure all template samples against one independent evidence partition."""
    measured = np.asarray(measured_points_scene, dtype=np.float64)
    if (
        measured.ndim != 2
        or measured.shape[1] != 3
        or len(measured) == 0
        or not np.isfinite(measured).all()
    ):
        raise ValueError(
            "measured_points_scene must be a non-empty finite (N, 3) array."
        )
    tree = cKDTree(measured)
    segment_distances: dict[str, NDArray[np.float64]] = {}
    family_parts: dict[str, list[NDArray[np.float64]]] = {}
    for segment, points_2d in sample_court_line_segments(settings.samples_per_metre):
        points_court = np.column_stack((points_2d, np.zeros(len(points_2d))))
        predicted = scene_from_court.apply(points_court)
        distances, _indices = tree.query(predicted, k=1, workers=1)
        finite = np.asarray(distances, dtype=np.float64)
        segment_distances[segment.name] = finite
        family_parts.setdefault(segment.family, []).append(finite)
    distances = np.concatenate(list(segment_distances.values()))
    threshold = settings.inlier_distance_metres
    supported = int(np.count_nonzero(distances <= threshold))
    segment_fractions = {
        name: float(np.mean(values <= threshold))
        for name, values in segment_distances.items()
    }
    family_fractions = {
        name: float(np.mean(np.concatenate(parts) <= threshold))
        for name, parts in family_parts.items()
    }
    return WholeTemplateMetrics(
        template_sample_count=len(distances),
        supported_sample_count=supported,
        inlier_fraction=supported / len(distances),
        rms_error_m=float(np.sqrt(np.mean(np.square(distances)))),
        q95_error_m=float(np.quantile(distances, 0.95)),
        maximum_error_m=float(np.max(distances)),
        segment_inlier_fractions=segment_fractions,
        family_inlier_fractions=family_fractions,
    )


def evaluate_court_topology(
    courts: Sequence[tuple[str, RigidTransform]],
) -> tuple[CourtPairTopologyMetrics, ...]:
    """Measure center separation and exact oriented-footprint overlap per pair."""
    court_tuple = tuple(courts)
    pairs: list[CourtPairTopologyMetrics] = []
    for first_index, (first_id, first_transform) in enumerate(court_tuple):
        first_inverse = first_transform.inverse()
        first_center = first_transform.apply(np.zeros((1, 3), dtype=np.float64))[0]
        for second_id, second_transform in court_tuple[first_index + 1 :]:
            second_center = second_transform.apply(np.zeros((1, 3), dtype=np.float64))[
                0
            ]
            second_corners_scene = second_transform.apply(_footprint_corners_3d())
            second_in_first = first_inverse.apply(second_corners_scene)[:, :2]
            intersection = _convex_intersection(
                _footprint_corners_3d()[:, :2],
                second_in_first,
            )
            footprint_area = 4.0 * HALF_DOUBLES_WIDTH * HALF_LENGTH
            overlap = _polygon_area(intersection) / footprint_area
            pairs.append(
                CourtPairTopologyMetrics(
                    first_candidate_id=first_id,
                    second_candidate_id=second_id,
                    center_separation_metres=float(
                        np.linalg.norm(first_center - second_center)
                    ),
                    footprint_overlap_fraction=float(overlap),
                )
            )
    return tuple(pairs)


def _footprint_corners_3d() -> NDArray[np.float64]:
    return np.asarray(
        [
            (-HALF_DOUBLES_WIDTH, -HALF_LENGTH, 0.0),
            (HALF_DOUBLES_WIDTH, -HALF_LENGTH, 0.0),
            (HALF_DOUBLES_WIDTH, HALF_LENGTH, 0.0),
            (-HALF_DOUBLES_WIDTH, HALF_LENGTH, 0.0),
        ],
        dtype=np.float64,
    )


def _convex_intersection(
    subject: NDArray[np.float64],
    clip: NDArray[np.float64],
) -> NDArray[np.float64]:
    subject_ccw = _counterclockwise(subject)
    clip_ccw = _counterclockwise(clip)
    output = [np.asarray(point, dtype=np.float64) for point in subject_ccw]
    for edge_index, edge_start in enumerate(clip_ccw):
        edge_end = clip_ccw[(edge_index + 1) % len(clip_ccw)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            current_inside = _inside_clip_edge(current, edge_start, edge_end)
            previous_inside = _inside_clip_edge(previous, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(previous, current, edge_start, edge_end)
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    _line_intersection(previous, current, edge_start, edge_end)
                )
            previous = current
    if not output:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(output, dtype=np.float64)


def _counterclockwise(points: NDArray[np.float64]) -> NDArray[np.float64]:
    polygon = np.asarray(points, dtype=np.float64)
    if _signed_polygon_area(polygon) < 0.0:
        return np.asarray(polygon[::-1], dtype=np.float64)
    return polygon


def _inside_clip_edge(
    point: NDArray[np.float64],
    edge_start: NDArray[np.float64],
    edge_end: NDArray[np.float64],
) -> bool:
    edge = edge_end - edge_start
    relative = point - edge_start
    return float(edge[0] * relative[1] - edge[1] * relative[0]) >= -1.0e-10


def _line_intersection(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    edge_start: NDArray[np.float64],
    edge_end: NDArray[np.float64],
) -> NDArray[np.float64]:
    direction = second - first
    edge = edge_end - edge_start
    denominator = direction[0] * edge[1] - direction[1] * edge[0]
    if abs(float(denominator)) <= 1.0e-14:
        return np.asarray(second, dtype=np.float64)
    relative = edge_start - first
    fraction = (relative[0] * edge[1] - relative[1] * edge[0]) / denominator
    return np.asarray(first + fraction * direction, dtype=np.float64)


def _polygon_area(points: NDArray[np.float64]) -> float:
    return abs(_signed_polygon_area(points))


def _signed_polygon_area(points: NDArray[np.float64]) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float((np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


__all__ = [
    "COURT_LINE_SEGMENTS",
    "CourtIdentifiabilityMetrics",
    "CourtLineSegment",
    "CourtPairTopologyMetrics",
    "LineFamilyIdentifiabilityMetrics",
    "QualifyingOffsetPairMetrics",
    "SemanticOffsetLevelMetrics",
    "WholeTemplateMetrics",
    "evaluate_court_identifiability",
    "evaluate_court_topology",
    "evaluate_whole_template",
    "sample_court_line_segments",
    "sample_court_line_template",
    "transform_template_2d",
]
