"""Strict semantic contracts for fit/holdout court alignment.

The contracts describe geometry and observable quality only.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.scene_contract import (
    COURT_AXES_METRES,
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
)

ALIGNMENT_SCHEMA = "semantic_multi_court_alignment_v2"
ALIGNMENT_COORDINATE_CONVENTION = (
    f"metric_scene_from_court_column_vectors;{COURT_AXES_METRES}"
)

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_INVERSE_ATOL = 1.0e-6


class AlignmentStatus(StrEnum):
    """Independent fit and holdout acceptance states."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class MetricSceneAdapter:
    """Explicit uniform similarity between NHT-normalized and metric scene frames.

    ``nht_scene_from_metric_scene`` is intentionally kept separate from every
    court ``RigidTransform``.  NHT exports use normalized scene units, whereas
    the alignment and dataset contracts use metres and proper SE(3).
    """

    nht_scene_from_metric_scene: tuple[float, ...]
    metric_scene_from_nht_scene: tuple[float, ...]
    nht_scene_units_per_metre: float

    def __post_init__(self) -> None:
        forward = _similarity_matrix(
            self.nht_scene_from_metric_scene,
            name="nht_scene_from_metric_scene",
        )
        inverse = _similarity_matrix(
            self.metric_scene_from_nht_scene,
            name="metric_scene_from_nht_scene",
        )
        scale = _finite_float(
            self.nht_scene_units_per_metre,
            name="nht_scene_units_per_metre",
        )
        if scale <= 0.0:
            raise ValueError("nht_scene_units_per_metre must be positive.")
        measured_scale = _similarity_scale(forward)
        if not math.isclose(scale, measured_scale, abs_tol=1.0e-10, rel_tol=1.0e-8):
            raise ValueError(
                "Declared NHT scene scale disagrees with the similarity matrix."
            )
        if not np.allclose(forward @ inverse, np.eye(4), atol=_INVERSE_ATOL, rtol=0.0):
            raise ValueError("Metric/NHT scene similarities must be reciprocal.")
        if not np.allclose(inverse @ forward, np.eye(4), atol=_INVERSE_ATOL, rtol=0.0):
            raise ValueError("Metric/NHT scene similarities must be reciprocal.")
        object.__setattr__(
            self,
            "nht_scene_from_metric_scene",
            tuple(float(value) for value in forward.ravel()),
        )
        object.__setattr__(
            self,
            "metric_scene_from_nht_scene",
            tuple(float(value) for value in inverse.ravel()),
        )
        object.__setattr__(self, "nht_scene_units_per_metre", scale)

    @classmethod
    def from_nht_scene_from_metric_scene(
        cls,
        matrix: NDArray[np.floating[Any]],
    ) -> Self:
        """Construct from one validated metric-to-NHT uniform similarity."""
        forward = _similarity_matrix(matrix, name="nht_scene_from_metric_scene")
        inverse = np.linalg.inv(forward)
        return cls(
            nht_scene_from_metric_scene=tuple(
                float(value) for value in forward.ravel()
            ),
            metric_scene_from_nht_scene=tuple(
                float(value) for value in inverse.ravel()
            ),
            nht_scene_units_per_metre=_similarity_scale(forward),
        )

    def nht_from_metric_points(
        self,
        points_metric_scene: NDArray[np.floating[Any]],
    ) -> NDArray[np.float64]:
        """Map metric-scene points to the public NHT normalized scene."""
        return _apply_matrix(self.nht_matrix(), points_metric_scene)

    def metric_from_nht_points(
        self,
        points_nht_scene: NDArray[np.floating[Any]],
    ) -> NDArray[np.float64]:
        """Map public NHT normalized-scene points into metres."""
        return _apply_matrix(self.metric_matrix(), points_nht_scene)

    def metric_from_nht_camera(
        self, camera_to_nht_scene: RigidTransform
    ) -> RigidTransform:
        """Convert a public NHT camera pose to the rigid metric scene frame."""
        return _camera_pose_through_similarity(
            camera_to_nht_scene,
            target_from_source=self.metric_matrix(),
        )

    def nht_from_metric_camera(
        self,
        camera_to_metric_scene: RigidTransform,
    ) -> RigidTransform:
        """Convert a metric camera pose for the independent NHT renderer boundary."""
        return _camera_pose_through_similarity(
            camera_to_metric_scene,
            target_from_source=self.nht_matrix(),
        )

    def nht_matrix(self) -> NDArray[np.float64]:
        """Return the metric-to-NHT similarity matrix."""
        return np.asarray(self.nht_scene_from_metric_scene, dtype=np.float64).reshape(
            4, 4
        )

    def metric_matrix(self) -> NDArray[np.float64]:
        """Return the NHT-to-metric similarity matrix."""
        return np.asarray(self.metric_scene_from_nht_scene, dtype=np.float64).reshape(
            4, 4
        )

    def to_dict(self) -> dict[str, object]:
        """Return the strict persisted frame-adapter representation."""
        return {
            "nht_scene_from_metric_scene": list(self.nht_scene_from_metric_scene),
            "metric_scene_from_nht_scene": list(self.metric_scene_from_nht_scene),
            "nht_scene_units_per_metre": self.nht_scene_units_per_metre,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse and cross-check both persisted similarities and their scale."""
        raw = _strict_mapping(
            value,
            keys={
                "nht_scene_from_metric_scene",
                "metric_scene_from_nht_scene",
                "nht_scene_units_per_metre",
            },
            name="metric scene adapter",
        )
        return cls(
            nht_scene_from_metric_scene=_finite_tuple(
                raw["nht_scene_from_metric_scene"],
                size=16,
                name="nht_scene_from_metric_scene",
            ),
            metric_scene_from_nht_scene=_finite_tuple(
                raw["metric_scene_from_nht_scene"],
                size=16,
                name="metric_scene_from_nht_scene",
            ),
            nht_scene_units_per_metre=_finite_float(
                raw["nht_scene_units_per_metre"],
                name="nht_scene_units_per_metre",
            ),
        )


@dataclass(frozen=True, slots=True)
class CameraLineDiagnostics:
    """Measured line-evidence inventory for one real exported camera."""

    camera_id: str
    selected_line_pixel_count: int
    projected_line_point_count: int

    def __post_init__(self) -> None:
        _identifier(self.camera_id, name="camera_id")
        _integer(
            self.selected_line_pixel_count,
            name="selected_line_pixel_count",
            minimum=1,
        )
        _integer(
            self.projected_line_point_count,
            name="projected_line_point_count",
            minimum=1,
        )

    def to_dict(self) -> dict[str, object]:
        """Return strict persisted diagnostics."""
        return {
            "camera_id": self.camera_id,
            "selected_line_pixel_count": self.selected_line_pixel_count,
            "projected_line_point_count": self.projected_line_point_count,
        }


@dataclass(frozen=True, slots=True)
class CandidateScaleDiagnostics:
    """Measured Sim(3) scale and image-line score for one fitted court."""

    candidate_id: str
    nht_scene_units_per_metre: float
    template_score: float

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, name="candidate_id")
        scale = _finite_float(
            self.nht_scene_units_per_metre,
            name="nht_scene_units_per_metre",
        )
        score = _finite_float(self.template_score, name="template_score")
        if scale <= 0.0 or score <= 0.0:
            raise ValueError("Candidate scale and template score must be positive.")
        object.__setattr__(self, "nht_scene_units_per_metre", scale)
        object.__setattr__(self, "template_score", score)

    def to_dict(self) -> dict[str, object]:
        """Return strict persisted diagnostics."""
        return {
            "candidate_id": self.candidate_id,
            "nht_scene_units_per_metre": self.nht_scene_units_per_metre,
            "template_score": self.template_score,
        }


@dataclass(frozen=True, slots=True)
class AlignmentEvidenceDiagnostics:
    """Measured image-line and common-scale evidence retained for audit."""

    cameras: tuple[CameraLineDiagnostics, ...]
    candidate_scales: tuple[CandidateScaleDiagnostics, ...]
    common_nht_scene_units_per_metre: float
    maximum_relative_scale_deviation: float

    def __post_init__(self) -> None:
        cameras = tuple(self.cameras)
        candidate_scales = tuple(self.candidate_scales)
        if not cameras or not candidate_scales:
            raise ValueError(
                "Evidence diagnostics must include cameras and candidates."
            )
        camera_ids = [item.camera_id for item in cameras]
        candidate_ids = [item.candidate_id for item in candidate_scales]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("Diagnostic camera IDs must be unique.")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Diagnostic candidate IDs must be unique.")
        common_scale = _finite_float(
            self.common_nht_scene_units_per_metre,
            name="common_nht_scene_units_per_metre",
        )
        maximum_deviation = _finite_float(
            self.maximum_relative_scale_deviation,
            name="maximum_relative_scale_deviation",
        )
        if common_scale <= 0.0 or maximum_deviation < 0.0:
            raise ValueError(
                "Common scale must be positive and deviation non-negative."
            )
        measured = max(
            abs(item.nht_scene_units_per_metre / common_scale - 1.0)
            for item in candidate_scales
        )
        if not math.isclose(
            measured, maximum_deviation, abs_tol=1.0e-10, rel_tol=1.0e-8
        ):
            raise ValueError(
                "Maximum relative scale deviation disagrees with candidates."
            )
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(self, "candidate_scales", candidate_scales)
        object.__setattr__(self, "common_nht_scene_units_per_metre", common_scale)
        object.__setattr__(self, "maximum_relative_scale_deviation", maximum_deviation)

    def to_dict(self) -> dict[str, object]:
        """Return machine-readable measured evidence diagnostics."""
        return {
            "schema": "alignment_measured_evidence_v1",
            "cameras": [item.to_dict() for item in self.cameras],
            "candidate_scales": [item.to_dict() for item in self.candidate_scales],
            "common_nht_scene_units_per_metre": self.common_nht_scene_units_per_metre,
            "maximum_relative_scale_deviation": self.maximum_relative_scale_deviation,
        }


@dataclass(frozen=True, slots=True)
class AlignmentPartitions:
    """One explicit, non-overlapping camera split used by every candidate."""

    fit_camera_ids: tuple[str, ...]
    holdout_camera_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        fit = _camera_ids(self.fit_camera_ids, name="fit_camera_ids")
        holdout = _camera_ids(self.holdout_camera_ids, name="holdout_camera_ids")
        overlap = set(fit).intersection(holdout)
        if overlap:
            raise ValueError(f"Fit and holdout camera IDs overlap: {sorted(overlap)}.")
        object.__setattr__(self, "fit_camera_ids", fit)
        object.__setattr__(self, "holdout_camera_ids", holdout)

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""
        return {
            "fit_camera_ids": list(self.fit_camera_ids),
            "holdout_camera_ids": list(self.holdout_camera_ids),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse a split and reject missing or unknown fields."""
        raw = _strict_mapping(
            value,
            keys={"fit_camera_ids", "holdout_camera_ids"},
            name="partitions",
        )
        return cls(
            fit_camera_ids=_string_tuple(raw["fit_camera_ids"], name="fit_camera_ids"),
            holdout_camera_ids=_string_tuple(
                raw["holdout_camera_ids"], name="holdout_camera_ids"
            ),
        )


@dataclass(frozen=True, slots=True)
class PartitionThresholds:
    """Quantitative acceptance policy for one evidence partition."""

    minimum_camera_count: int
    minimum_correspondence_count: int
    inlier_distance_m: float
    minimum_inlier_fraction: float
    maximum_rms_error_m: float
    maximum_q95_error_m: float

    def __post_init__(self) -> None:
        minimum_cameras = _integer(
            self.minimum_camera_count, name="minimum_camera_count", minimum=1
        )
        minimum_correspondences = _integer(
            self.minimum_correspondence_count,
            name="minimum_correspondence_count",
            minimum=3,
        )
        inlier_distance = _finite_float(
            self.inlier_distance_m, name="inlier_distance_m"
        )
        inlier_fraction = _finite_float(
            self.minimum_inlier_fraction, name="minimum_inlier_fraction"
        )
        rms = _finite_float(self.maximum_rms_error_m, name="maximum_rms_error_m")
        q95 = _finite_float(self.maximum_q95_error_m, name="maximum_q95_error_m")
        if inlier_distance <= 0.0 or rms <= 0.0 or q95 <= 0.0:
            raise ValueError("Distance and error thresholds must be positive.")
        if not 0.0 <= inlier_fraction <= 1.0:
            raise ValueError("minimum_inlier_fraction must lie in [0, 1].")
        object.__setattr__(self, "minimum_camera_count", minimum_cameras)
        object.__setattr__(
            self, "minimum_correspondence_count", minimum_correspondences
        )
        object.__setattr__(self, "inlier_distance_m", inlier_distance)
        object.__setattr__(self, "minimum_inlier_fraction", inlier_fraction)
        object.__setattr__(self, "maximum_rms_error_m", rms)
        object.__setattr__(self, "maximum_q95_error_m", q95)

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""
        return {
            "minimum_camera_count": self.minimum_camera_count,
            "minimum_correspondence_count": self.minimum_correspondence_count,
            "inlier_distance_m": self.inlier_distance_m,
            "minimum_inlier_fraction": self.minimum_inlier_fraction,
            "maximum_rms_error_m": self.maximum_rms_error_m,
            "maximum_q95_error_m": self.maximum_q95_error_m,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse a strict partition policy."""
        keys = {
            "minimum_camera_count",
            "minimum_correspondence_count",
            "inlier_distance_m",
            "minimum_inlier_fraction",
            "maximum_rms_error_m",
            "maximum_q95_error_m",
        }
        raw = _strict_mapping(value, keys=keys, name="partition thresholds")
        return cls(
            minimum_camera_count=_integer(
                raw["minimum_camera_count"], name="minimum_camera_count", minimum=1
            ),
            minimum_correspondence_count=_integer(
                raw["minimum_correspondence_count"],
                name="minimum_correspondence_count",
                minimum=3,
            ),
            inlier_distance_m=_finite_float(
                raw["inlier_distance_m"], name="inlier_distance_m"
            ),
            minimum_inlier_fraction=_finite_float(
                raw["minimum_inlier_fraction"], name="minimum_inlier_fraction"
            ),
            maximum_rms_error_m=_finite_float(
                raw["maximum_rms_error_m"], name="maximum_rms_error_m"
            ),
            maximum_q95_error_m=_finite_float(
                raw["maximum_q95_error_m"], name="maximum_q95_error_m"
            ),
        )


@dataclass(frozen=True, slots=True)
class AlignmentAcceptancePolicy:
    """Separate fit and holdout policies; neither partition substitutes for the other."""

    fit: PartitionThresholds
    holdout: PartitionThresholds

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""
        return {"fit": self.fit.to_dict(), "holdout": self.holdout.to_dict()}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse a strict two-part policy."""
        raw = _strict_mapping(value, keys={"fit", "holdout"}, name="policy")
        return cls(
            fit=PartitionThresholds.from_dict(raw["fit"]),
            holdout=PartitionThresholds.from_dict(raw["holdout"]),
        )


@dataclass(frozen=True, slots=True)
class PartitionMetrics:
    """Measured residual quality for exactly one declared camera partition."""

    camera_ids: tuple[str, ...]
    correspondence_count: int
    inlier_count: int
    inlier_fraction: float
    rms_error_m: float
    q95_error_m: float
    maximum_error_m: float

    def __post_init__(self) -> None:
        camera_ids = _camera_ids(self.camera_ids, name="metrics.camera_ids")
        count = _integer(
            self.correspondence_count, name="correspondence_count", minimum=1
        )
        inliers = _integer(self.inlier_count, name="inlier_count", minimum=0)
        if inliers > count:
            raise ValueError("inlier_count cannot exceed correspondence_count.")
        fraction = _finite_float(self.inlier_fraction, name="inlier_fraction")
        expected_fraction = inliers / count
        if not math.isclose(fraction, expected_fraction, abs_tol=1.0e-12, rel_tol=0.0):
            raise ValueError(
                "inlier_fraction is inconsistent with the declared counts."
            )
        errors = (
            _finite_float(self.rms_error_m, name="rms_error_m"),
            _finite_float(self.q95_error_m, name="q95_error_m"),
            _finite_float(self.maximum_error_m, name="maximum_error_m"),
        )
        if any(value < 0.0 for value in errors):
            raise ValueError("Alignment residual metrics must be non-negative.")
        if self.q95_error_m > self.maximum_error_m + 1.0e-12:
            raise ValueError("q95_error_m cannot exceed maximum_error_m.")
        object.__setattr__(self, "camera_ids", camera_ids)
        object.__setattr__(self, "correspondence_count", count)
        object.__setattr__(self, "inlier_count", inliers)
        object.__setattr__(self, "inlier_fraction", fraction)
        object.__setattr__(self, "rms_error_m", errors[0])
        object.__setattr__(self, "q95_error_m", errors[1])
        object.__setattr__(self, "maximum_error_m", errors[2])

    def threshold_checks(self, thresholds: PartitionThresholds) -> dict[str, bool]:
        """Evaluate all gates from measured fields without descriptive fallback."""
        return {
            "minimum_camera_count": len(self.camera_ids)
            >= thresholds.minimum_camera_count,
            "minimum_correspondence_count": (
                self.correspondence_count >= thresholds.minimum_correspondence_count
            ),
            "minimum_inlier_fraction": (
                self.inlier_fraction >= thresholds.minimum_inlier_fraction
            ),
            "maximum_rms_error_m": self.rms_error_m <= thresholds.maximum_rms_error_m,
            "maximum_q95_error_m": self.q95_error_m <= thresholds.maximum_q95_error_m,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""
        return {
            "camera_ids": list(self.camera_ids),
            "correspondence_count": self.correspondence_count,
            "inlier_count": self.inlier_count,
            "inlier_fraction": self.inlier_fraction,
            "rms_error_m": self.rms_error_m,
            "q95_error_m": self.q95_error_m,
            "maximum_error_m": self.maximum_error_m,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse measured metrics and reject unknown or non-finite values."""
        keys = {
            "camera_ids",
            "correspondence_count",
            "inlier_count",
            "inlier_fraction",
            "rms_error_m",
            "q95_error_m",
            "maximum_error_m",
        }
        raw = _strict_mapping(value, keys=keys, name="partition metrics")
        return cls(
            camera_ids=_string_tuple(raw["camera_ids"], name="metrics.camera_ids"),
            correspondence_count=_integer(
                raw["correspondence_count"], name="correspondence_count", minimum=1
            ),
            inlier_count=_integer(raw["inlier_count"], name="inlier_count", minimum=0),
            inlier_fraction=_finite_float(
                raw["inlier_fraction"], name="inlier_fraction"
            ),
            rms_error_m=_finite_float(raw["rms_error_m"], name="rms_error_m"),
            q95_error_m=_finite_float(raw["q95_error_m"], name="q95_error_m"),
            maximum_error_m=_finite_float(
                raw["maximum_error_m"], name="maximum_error_m"
            ),
        )


@dataclass(frozen=True, slots=True)
class PartitionAssessment:
    """Metrics and status kept independently for fit or holdout."""

    status: AlignmentStatus
    metrics: PartitionMetrics
    threshold_checks: Mapping[str, bool]

    def __post_init__(self) -> None:
        if not isinstance(self.status, AlignmentStatus):
            raise TypeError("status must be an AlignmentStatus.")
        checks = dict(self.threshold_checks)
        expected_names = {
            "minimum_camera_count",
            "minimum_correspondence_count",
            "minimum_inlier_fraction",
            "maximum_rms_error_m",
            "maximum_q95_error_m",
        }
        if set(checks) != expected_names or any(
            type(value) is not bool for value in checks.values()
        ):
            raise ValueError(
                "threshold_checks must contain exactly the five boolean gates."
            )
        accepted = all(checks.values())
        if accepted != (self.status is AlignmentStatus.ACCEPTED):
            raise ValueError("Partition status disagrees with its threshold checks.")
        object.__setattr__(self, "threshold_checks", checks)

    @classmethod
    def evaluate(
        cls,
        metrics: PartitionMetrics,
        thresholds: PartitionThresholds,
    ) -> Self:
        """Create an assessment by applying every threshold."""
        checks = metrics.threshold_checks(thresholds)
        status = (
            AlignmentStatus.ACCEPTED
            if all(checks.values())
            else AlignmentStatus.REJECTED
        )
        return cls(status=status, metrics=metrics, threshold_checks=checks)

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""
        return {
            "status": self.status.value,
            "metrics": self.metrics.to_dict(),
            "threshold_checks": dict(self.threshold_checks),
        }

    @classmethod
    def from_dict(cls, value: object, *, thresholds: PartitionThresholds) -> Self:
        """Parse and recompute all status gates from persisted metrics."""
        raw = _strict_mapping(
            value,
            keys={"status", "metrics", "threshold_checks"},
            name="partition assessment",
        )
        status = AlignmentStatus(_string(raw["status"], name="status"))
        metrics = PartitionMetrics.from_dict(raw["metrics"])
        checks_raw = _strict_mapping(
            raw["threshold_checks"],
            keys={
                "minimum_camera_count",
                "minimum_correspondence_count",
                "minimum_inlier_fraction",
                "maximum_rms_error_m",
                "maximum_q95_error_m",
            },
            name="threshold_checks",
        )
        checks = {
            key: _boolean(item, name=f"threshold_checks.{key}")
            for key, item in checks_raw.items()
        }
        expected = metrics.threshold_checks(thresholds)
        if checks != expected:
            raise ValueError(
                "Persisted threshold checks disagree with measured metrics."
            )
        return cls(status=status, metrics=metrics, threshold_checks=checks)


@dataclass(frozen=True, slots=True)
class CorrespondenceSet:
    """Court/scene point pairs belonging to exactly one camera partition."""

    points_court: NDArray[np.float64]
    points_scene: NDArray[np.float64]
    camera_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        court = _point_array(self.points_court, name="points_court")
        scene = _point_array(self.points_scene, name="points_scene")
        if court.shape != scene.shape:
            raise ValueError(
                "Court and scene correspondence arrays must have the same shape."
            )
        if len(court) < 3:
            raise ValueError("At least three correspondences are required.")
        camera_ids = _string_tuple(self.camera_ids, name="correspondence camera_ids")
        if len(camera_ids) != len(court):
            raise ValueError("There must be one camera ID per correspondence.")
        for camera_id in camera_ids:
            _identifier(camera_id, name="camera_id")
        court.setflags(write=False)
        scene.setflags(write=False)
        object.__setattr__(self, "points_court", court)
        object.__setattr__(self, "points_scene", scene)
        object.__setattr__(self, "camera_ids", camera_ids)

    @property
    def observed_camera_ids(self) -> tuple[str, ...]:
        """Return stable first-seen camera IDs."""
        return tuple(dict.fromkeys(self.camera_ids))


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """Disjoint fit and holdout correspondences for one physical court candidate."""

    court_instance_id: str
    candidate_id: str
    fit: CorrespondenceSet
    holdout: CorrespondenceSet

    def __post_init__(self) -> None:
        _identifier(self.court_instance_id, name="court_instance_id")
        _identifier(self.candidate_id, name="candidate_id")


@dataclass(frozen=True, slots=True)
class MeasuredCameraLines:
    """All measured line/ground intersections retained for one public camera."""

    camera_id: str
    points_nht_scene: NDArray[np.float64]

    def __post_init__(self) -> None:
        _identifier(self.camera_id, name="camera_id")
        points = _point_array(self.points_nht_scene, name="points_nht_scene")
        points.setflags(write=False)
        object.__setattr__(self, "points_nht_scene", points)


@dataclass(frozen=True, slots=True)
class AlignmentEvidence:
    """All semantic observations returned by a standard-export evidence source."""

    partitions: AlignmentPartitions
    candidates: tuple[CandidateEvidence, ...]
    measured_camera_lines: tuple[MeasuredCameraLines, ...]
    complex_points_scene: NDArray[np.float64]
    primary_candidate_id: str | None
    metric_adapter: MetricSceneAdapter
    diagnostics: AlignmentEvidenceDiagnostics

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError(
                "Alignment evidence must contain at least one court candidate."
            )
        court_ids = [candidate.court_instance_id for candidate in candidates]
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(court_ids) != len(set(court_ids)):
            raise ValueError("Evidence court_instance_id values must be unique.")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Evidence candidate_id values must be unique.")
        for candidate in candidates:
            _require_ordered_camera_subset(
                candidate.fit.observed_camera_ids,
                declared=self.partitions.fit_camera_ids,
                name="Candidate fit evidence",
            )
            _require_ordered_camera_subset(
                candidate.holdout.observed_camera_ids,
                declared=self.partitions.holdout_camera_ids,
                name="Candidate holdout evidence",
            )
        if self.primary_candidate_id is not None:
            _identifier(self.primary_candidate_id, name="primary_candidate_id")
            if self.primary_candidate_id not in candidate_ids:
                raise ValueError(
                    "primary_candidate_id does not reference an evidence candidate."
                )
        expected_camera_ids = (
            self.partitions.fit_camera_ids + self.partitions.holdout_camera_ids
        )
        measured_camera_lines = tuple(self.measured_camera_lines)
        measured_camera_ids = tuple(item.camera_id for item in measured_camera_lines)
        if measured_camera_ids != expected_camera_ids:
            raise ValueError(
                "Measured camera lines do not match the declared camera partitions."
            )
        diagnostic_camera_ids = tuple(
            item.camera_id for item in self.diagnostics.cameras
        )
        if diagnostic_camera_ids != expected_camera_ids:
            raise ValueError(
                "Evidence diagnostics do not match the declared camera partitions."
            )
        for measured, diagnostic in zip(
            measured_camera_lines,
            self.diagnostics.cameras,
            strict=True,
        ):
            if len(measured.points_nht_scene) != diagnostic.projected_line_point_count:
                raise ValueError(
                    "Projected line diagnostics disagree with retained measured points."
                )
        diagnostic_candidate_ids = tuple(
            item.candidate_id for item in self.diagnostics.candidate_scales
        )
        if diagnostic_candidate_ids != tuple(candidate_ids):
            raise ValueError("Evidence diagnostics do not match the candidate order.")
        if not math.isclose(
            self.metric_adapter.nht_scene_units_per_metre,
            self.diagnostics.common_nht_scene_units_per_metre,
            abs_tol=1.0e-10,
            rel_tol=1.0e-8,
        ):
            raise ValueError(
                "Metric adapter scale disagrees with measured scale diagnostics."
            )
        complex_points = _point_array(
            self.complex_points_scene,
            name="complex_points_scene",
            minimum_count=2,
        )
        if np.any(np.ptp(complex_points, axis=0) <= 0.0):
            raise ValueError("Complex support points must have positive X/Y/Z extent.")
        complex_points.setflags(write=False)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "measured_camera_lines", measured_camera_lines)
        object.__setattr__(self, "complex_points_scene", complex_points)


@dataclass(frozen=True, slots=True)
class CandidateAlignment:
    """One fitted transform with independently evaluated fit and holdout evidence."""

    court_instance_id: str
    candidate_id: str
    scene_from_court: RigidTransform
    court_from_scene: RigidTransform
    fit: PartitionAssessment
    holdout: PartitionAssessment

    def __post_init__(self) -> None:
        _identifier(self.court_instance_id, name="court_instance_id")
        _identifier(self.candidate_id, name="candidate_id")
        forward = self.court_from_scene.matrix() @ self.scene_from_court.matrix()
        reverse = self.scene_from_court.matrix() @ self.court_from_scene.matrix()
        if not np.allclose(
            forward, np.eye(4), atol=_INVERSE_ATOL, rtol=0.0
        ) or not np.allclose(reverse, np.eye(4), atol=_INVERSE_ATOL, rtol=0.0):
            raise ValueError("Candidate court transforms must be reciprocal.")
        if set(self.fit.metrics.camera_ids).intersection(
            self.holdout.metrics.camera_ids
        ):
            raise ValueError("Candidate fit and holdout metrics must remain disjoint.")

    @property
    def accepted(self) -> bool:
        """Return true only when both independent partitions pass."""
        return (
            self.fit.status is AlignmentStatus.ACCEPTED
            and self.holdout.status is AlignmentStatus.ACCEPTED
        )

    def to_court_instance(self) -> CourtInstance:
        """Create the shared dataset-facing court contract, failing closed."""
        if not self.accepted:
            raise ValueError("A rejected candidate cannot enter MultiCourtLayout.")
        return CourtInstance(
            court_instance_id=self.court_instance_id,
            candidate_id=self.candidate_id,
            scene_from_court=self.scene_from_court,
            court_from_scene=self.court_from_scene,
            fit_status=self.fit.status.value,
            fit_metrics=_assessment_metrics(self.fit),
            holdout_status=self.holdout.status.value,
            holdout_metrics=_assessment_metrics(self.holdout),
        )

    def to_dict(self) -> dict[str, object]:
        """Return all candidate evidence, including rejected candidates."""
        return {
            "court_instance_id": self.court_instance_id,
            "candidate_id": self.candidate_id,
            "scene_from_court": self.scene_from_court.to_list(),
            "court_from_scene": self.court_from_scene.to_list(),
            "fit": self.fit.to_dict(),
            "holdout": self.holdout.to_dict(),
            "accepted": self.accepted,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        policy: AlignmentAcceptancePolicy,
        partitions: AlignmentPartitions,
    ) -> Self:
        """Parse one candidate and recompute transform/status invariants."""
        raw = _strict_mapping(
            value,
            keys={
                "court_instance_id",
                "candidate_id",
                "scene_from_court",
                "court_from_scene",
                "fit",
                "holdout",
                "accepted",
            },
            name="candidate alignment",
        )
        fit = PartitionAssessment.from_dict(raw["fit"], thresholds=policy.fit)
        holdout = PartitionAssessment.from_dict(
            raw["holdout"], thresholds=policy.holdout
        )
        _require_ordered_camera_subset(
            fit.metrics.camera_ids,
            declared=partitions.fit_camera_ids,
            name="Candidate fit metrics",
        )
        _require_ordered_camera_subset(
            holdout.metrics.camera_ids,
            declared=partitions.holdout_camera_ids,
            name="Candidate holdout metrics",
        )
        result = cls(
            court_instance_id=_string(
                raw["court_instance_id"], name="court_instance_id"
            ),
            candidate_id=_string(raw["candidate_id"], name="candidate_id"),
            scene_from_court=_transform(
                raw["scene_from_court"], name="scene_from_court"
            ),
            court_from_scene=_transform(
                raw["court_from_scene"], name="court_from_scene"
            ),
            fit=fit,
            holdout=holdout,
        )
        if _boolean(raw["accepted"], name="accepted") != result.accepted:
            raise ValueError(
                "Candidate accepted flag disagrees with fit/holdout status."
            )
        return result


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Complete final alignment and the accepted multi-court authority."""

    partitions: AlignmentPartitions
    policy: AlignmentAcceptancePolicy
    candidates: tuple[CandidateAlignment, ...]
    layout: MultiCourtLayout
    metric_adapter: MetricSceneAdapter

    def __post_init__(self) -> None:
        if not isinstance(self.metric_adapter, MetricSceneAdapter):
            raise TypeError("metric_adapter must be a MetricSceneAdapter.")
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("Alignment result must retain every evaluated candidate.")
        court_ids = [candidate.court_instance_id for candidate in candidates]
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(court_ids) != len(set(court_ids)) or len(candidate_ids) != len(
            set(candidate_ids)
        ):
            raise ValueError("Alignment candidate and court IDs must be unique.")
        for candidate in candidates:
            _require_ordered_camera_subset(
                candidate.fit.metrics.camera_ids,
                declared=self.partitions.fit_camera_ids,
                name="Candidate fit metrics",
            )
            _require_ordered_camera_subset(
                candidate.holdout.metrics.camera_ids,
                declared=self.partitions.holdout_camera_ids,
                name="Candidate holdout metrics",
            )
        accepted = tuple(candidate for candidate in candidates if candidate.accepted)
        if not accepted:
            raise ValueError("Holdout acceptance failed for every court candidate.")
        expected_courts = tuple(candidate.to_court_instance() for candidate in accepted)
        if [court.to_dict() for court in self.layout.courts] != [
            court.to_dict() for court in expected_courts
        ]:
            raise ValueError(
                "MultiCourtLayout must contain exactly all accepted candidates."
            )
        object.__setattr__(self, "candidates", candidates)

    def to_dict(self) -> dict[str, object]:
        """Return the canonical fixed-path alignment document."""
        return {
            "schema": ALIGNMENT_SCHEMA,
            "coordinate_convention": ALIGNMENT_COORDINATE_CONVENTION,
            "metric_scene_adapter": self.metric_adapter.to_dict(),
            "partitions": self.partitions.to_dict(),
            "policy": self.policy.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "layout": self.layout.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Strictly parse the document and cross-check its duplicated layout view."""
        raw = _strict_mapping(
            value,
            keys={
                "schema",
                "coordinate_convention",
                "metric_scene_adapter",
                "partitions",
                "policy",
                "candidates",
                "layout",
            },
            name="alignment result",
        )
        if raw["schema"] != ALIGNMENT_SCHEMA:
            raise ValueError(f"Unsupported alignment schema: {raw['schema']!r}.")
        if raw["coordinate_convention"] != ALIGNMENT_COORDINATE_CONVENTION:
            raise ValueError("Unsupported alignment coordinate convention.")
        partitions = AlignmentPartitions.from_dict(raw["partitions"])
        policy = AlignmentAcceptancePolicy.from_dict(raw["policy"])
        candidates_raw = _sequence(raw["candidates"], name="candidates")
        candidates = tuple(
            CandidateAlignment.from_dict(
                candidate,
                policy=policy,
                partitions=partitions,
            )
            for candidate in candidates_raw
        )
        layout_raw = _strict_mapping(
            raw["layout"],
            keys={
                "schema",
                "courts",
                "complex_bounds_scene",
                "primary_court_instance_id",
            },
            name="layout",
        )
        if layout_raw["schema"] != "multi_court_layout_v1":
            raise ValueError("Unsupported multi-court layout schema.")
        bounds = _finite_tuple(
            layout_raw["complex_bounds_scene"],
            size=6,
            name="complex_bounds_scene",
        )
        primary_raw = layout_raw["primary_court_instance_id"]
        if primary_raw is not None and not isinstance(primary_raw, str):
            raise TypeError("primary_court_instance_id must be a string or null.")
        layout = MultiCourtLayout(
            courts=tuple(
                candidate.to_court_instance()
                for candidate in candidates
                if candidate.accepted
            ),
            complex_bounds_scene=bounds,
            primary_court_instance_id=primary_raw,
        )
        result = cls(
            partitions=partitions,
            policy=policy,
            candidates=candidates,
            layout=layout,
            metric_adapter=MetricSceneAdapter.from_dict(raw["metric_scene_adapter"]),
        )
        if layout_raw != result.layout.to_dict():
            raise ValueError(
                "Serialized layout disagrees with accepted candidate evidence."
            )
        return result


def build_layout(
    candidates: Sequence[CandidateAlignment],
    *,
    complex_points_scene: NDArray[np.floating[Any]],
    primary_candidate_id: str | None,
) -> MultiCourtLayout:
    """Build the complete accepted layout without selecting an implicit fallback."""
    candidate_tuple = tuple(candidates)
    accepted = tuple(candidate for candidate in candidate_tuple if candidate.accepted)
    if not accepted:
        raise ValueError("Holdout acceptance failed for every court candidate.")
    primary_court_id: str | None = None
    if primary_candidate_id is not None:
        matching = [
            candidate
            for candidate in candidate_tuple
            if candidate.candidate_id == primary_candidate_id
        ]
        if len(matching) != 1:
            raise ValueError(
                "primary_candidate_id does not identify exactly one candidate."
            )
        if not matching[0].accepted:
            raise ValueError(
                "The explicitly selected primary candidate failed acceptance."
            )
        primary_court_id = matching[0].court_instance_id
    points = _point_array(
        complex_points_scene, name="complex_points_scene", minimum_count=2
    )
    minimum, maximum = _robust_complex_bounds(points)
    if np.any(minimum >= maximum):
        raise ValueError("Complex bounds must have positive extent on every axis.")
    bounds = tuple(float(value) for value in np.concatenate((minimum, maximum)))
    return MultiCourtLayout(
        courts=tuple(candidate.to_court_instance() for candidate in accepted),
        complex_bounds_scene=bounds,
        primary_court_instance_id=primary_court_id,
    )


def _robust_complex_bounds(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bound dense SfM support without letting isolated points move the complex."""
    if len(points) < 100:
        return np.min(points, axis=0), np.max(points, axis=0)
    quantiles = np.quantile(points, (0.01, 0.99), axis=0)
    minimum = np.asarray(quantiles[0], dtype=np.float64)
    maximum = np.asarray(quantiles[1], dtype=np.float64)
    raw_minimum = np.min(points, axis=0)
    raw_maximum = np.max(points, axis=0)
    collapsed = minimum >= maximum
    minimum[collapsed] = raw_minimum[collapsed]
    maximum[collapsed] = raw_maximum[collapsed]
    return minimum, maximum


def _assessment_metrics(assessment: PartitionAssessment) -> dict[str, object]:
    return {
        **assessment.metrics.to_dict(),
        "threshold_checks": dict(assessment.threshold_checks),
    }


def _point_array(
    value: NDArray[np.floating[Any]],
    *,
    name: str,
    minimum_count: int = 1,
) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.dtype.kind not in {"f", "i", "u"}:
        raise TypeError(f"{name} must have a real numeric dtype.")
    result = np.asarray(array, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 3 or len(result) < minimum_count:
        raise ValueError(f"{name} must have shape (N, 3) with N >= {minimum_count}.")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _similarity_matrix(
    value: Sequence[object] | NDArray[np.floating[Any]],
    *,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.dtype.kind not in {"f", "i", "u"}:
        raise TypeError(f"{name} must have a real numeric dtype.")
    matrix = np.asarray(array, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f"{name} must contain exactly 16 values.")
    matrix = matrix.reshape(4, 4).copy()
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9, rtol=0.0):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1].")
    _similarity_scale(matrix)
    return matrix


def _similarity_scale(matrix: NDArray[np.float64]) -> float:
    linear = matrix[:3, :3]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    scale = float(np.mean(singular_values))
    if scale <= 0.0 or not np.allclose(
        singular_values,
        scale,
        atol=1.0e-9,
        rtol=1.0e-7,
    ):
        raise ValueError("Scene-frame adapter must have one positive uniform scale.")
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-7, rtol=0.0):
        raise ValueError("Scene-frame adapter rotation must be orthonormal.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-7):
        raise ValueError("Scene-frame adapter rotation must be proper-handed.")
    return scale


def _apply_matrix(
    matrix: NDArray[np.float64],
    points: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 3 or not np.isfinite(array).all():
        raise ValueError("Scene-frame points must be a finite (..., 3) array.")
    return array @ matrix[:3, :3].T + matrix[:3, 3]


def _camera_pose_through_similarity(
    camera_to_source: RigidTransform,
    *,
    target_from_source: NDArray[np.float64],
) -> RigidTransform:
    scale = _similarity_scale(target_from_source)
    frame_rotation = target_from_source[:3, :3] / scale
    source_pose = camera_to_source.matrix()
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = frame_rotation @ source_pose[:3, :3]
    result[:3, 3] = (
        target_from_source[:3, :3] @ source_pose[:3, 3] + target_from_source[:3, 3]
    )
    return RigidTransform.from_matrix(result)


def _camera_ids(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(value)
    if not result:
        raise ValueError(f"{name} must not be empty.")
    for camera_id in result:
        if not isinstance(camera_id, str):
            raise TypeError(f"{name} must contain only strings.")
        _identifier(camera_id, name="camera_id")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique values.")
    return result


def _require_ordered_camera_subset(
    observed: tuple[str, ...],
    *,
    declared: tuple[str, ...],
    name: str,
) -> None:
    observed_set = set(observed)
    if not observed_set.issubset(declared):
        raise ValueError(f"{name} contains cameras outside its declared partition.")
    expected_order = tuple(
        camera_id for camera_id in declared if camera_id in observed_set
    )
    if observed != expected_order:
        raise ValueError(f"{name} must preserve declared camera order.")


def _identifier(value: str, *, name: str) -> None:
    if not isinstance(value, str) or _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a portable non-empty identifier: {value!r}.")


def _strict_mapping(value: object, *, keys: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings.")
    result = dict(value)
    if set(result) != keys:
        raise ValueError(
            f"{name} keys do not match the schema; "
            f"missing={sorted(keys - set(result))}, unknown={sorted(set(result) - keys)}."
        )
    return result


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    sequence = _sequence(value, name=name)
    if any(not isinstance(item, str) for item in sequence):
        raise TypeError(f"{name} must contain only strings.")
    return tuple(item for item in sequence if isinstance(item, str))


def _finite_tuple(value: object, *, size: int, name: str) -> tuple[float, ...]:
    sequence = _sequence(value, name=name)
    if len(sequence) != size:
        raise ValueError(f"{name} must contain exactly {size} values.")
    return tuple(_finite_float(item, name=name) for item in sequence)


def _transform(value: object, *, name: str) -> RigidTransform:
    return RigidTransform(_finite_tuple(value, size=16, name=name))


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{name} must be an integer >= {minimum}.")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean.")
    return bool(value)


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


__all__ = [
    "ALIGNMENT_COORDINATE_CONVENTION",
    "ALIGNMENT_SCHEMA",
    "AlignmentAcceptancePolicy",
    "AlignmentEvidence",
    "AlignmentEvidenceDiagnostics",
    "AlignmentPartitions",
    "AlignmentResult",
    "AlignmentStatus",
    "CandidateAlignment",
    "CandidateEvidence",
    "CandidateScaleDiagnostics",
    "CameraLineDiagnostics",
    "CorrespondenceSet",
    "MetricSceneAdapter",
    "MeasuredCameraLines",
    "PartitionAssessment",
    "PartitionMetrics",
    "PartitionThresholds",
    "build_layout",
]
