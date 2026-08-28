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

from src.synthetic_data_generation.alignment.heatmaps import AlignmentLineHeatmaps
from src.synthetic_data_generation.alignment.settings import WholeCourtEvidenceSettings
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


class CameraEvidencePartition(StrEnum):
    """Frozen camera partition assigned before line-evidence measurement."""

    FIT = "fit"
    HOLDOUT = "holdout"


class CameraExclusionReason(StrEnum):
    """Explicit fail-closed reasons for excluding selected cameras."""

    NO_DETECTED_LINE_PIXELS = "no_detected_line_pixels"
    INSUFFICIENT_PROJECTED_POINTS = "insufficient_projected_points"


class CameraSelectionPolicy(StrEnum):
    """Evidence-independent policy used to select the one camera prefix."""

    NESTED_UNIFORM_PREFIX_V1 = "nested_uniform_prefix_v1"


class CameraOwnershipRule(StrEnum):
    """Versioned slot rule assigning the fixed prefix before measurement."""

    FIXED_UNIT_EVEN_HOLDOUT_SLOTS_V1 = "fixed_unit_even_holdout_slots_v1"


class AlignmentEvaluationPolicy(StrEnum):
    """Strict fit/holdout evaluation policy."""

    FIT_SELECT_ONCE_HOLDOUT_EVALUATE_ONCE_V1 = (
        "fit_select_once_holdout_evaluate_once_v1"
    )


class AlignmentEvaluationOutcome(StrEnum):
    """Persistable terminal outcome for a successfully returned evidence object."""

    FULL_VALIDATION_PASS = "full_validation_pass"


class ProposalSearchStopReason(StrEnum):
    """Auditable reason that fit-only court-count inference terminated."""

    RESIDUAL_EVIDENCE_BELOW_MINIMUM = "residual_evidence_below_minimum"
    NO_RELIABLE_PROPOSAL = "no_reliable_proposal"


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
class ExcludedCameraDiagnostics:
    """Selected camera excluded after measurement without repartition or backfill."""

    camera_id: str
    original_partition: CameraEvidencePartition
    selected_line_pixel_count: int
    projected_line_point_count: int
    reason: CameraExclusionReason

    def __post_init__(self) -> None:
        _identifier(self.camera_id, name="camera_id")
        if not isinstance(self.original_partition, CameraEvidencePartition):
            raise TypeError("original_partition must be a CameraEvidencePartition.")
        _integer(
            self.selected_line_pixel_count,
            name="selected_line_pixel_count",
            minimum=0,
        )
        _integer(
            self.projected_line_point_count,
            name="projected_line_point_count",
            minimum=0,
        )
        if self.projected_line_point_count > self.selected_line_pixel_count:
            raise ValueError(
                "Excluded projected point count cannot exceed selected pixels."
            )
        if not isinstance(self.reason, CameraExclusionReason):
            raise TypeError("reason must be a CameraExclusionReason.")
        if self.reason is CameraExclusionReason.NO_DETECTED_LINE_PIXELS:
            if (
                self.selected_line_pixel_count != 0
                or self.projected_line_point_count != 0
            ):
                raise ValueError(
                    "A no-line exclusion must have zero selected and projected points."
                )
        elif self.selected_line_pixel_count == 0:
            raise ValueError(
                "An insufficient-projection exclusion requires detected line pixels."
            )

    def to_dict(self) -> dict[str, object]:
        """Return strict persisted exclusion evidence."""
        return {
            "camera_id": self.camera_id,
            "original_partition": self.original_partition.value,
            "selected_line_pixel_count": self.selected_line_pixel_count,
            "projected_line_point_count": self.projected_line_point_count,
            "reason": self.reason.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse one strict persisted exclusion diagnostic."""
        raw = _strict_mapping(
            value,
            keys={
                "camera_id",
                "original_partition",
                "selected_line_pixel_count",
                "projected_line_point_count",
                "reason",
            },
            name="excluded camera diagnostics",
        )
        return cls(
            camera_id=_string(raw["camera_id"], name="camera_id"),
            original_partition=CameraEvidencePartition(
                _string(raw["original_partition"], name="original_partition")
            ),
            selected_line_pixel_count=_integer(
                raw["selected_line_pixel_count"],
                name="selected_line_pixel_count",
                minimum=0,
            ),
            projected_line_point_count=_integer(
                raw["projected_line_point_count"],
                name="projected_line_point_count",
                minimum=0,
            ),
            reason=CameraExclusionReason(
                _string(raw["reason"], name="camera exclusion reason")
            ),
        )


@dataclass(frozen=True, slots=True)
class FixedCameraSelectionDiagnostics:
    """Exact one-shot camera selection, ownership, observation, and exclusions."""

    policy: CameraSelectionPolicy
    ownership_rule: CameraOwnershipRule
    requested_camera_count: int
    available_camera_count: int
    partition_unit_count: int
    fit_cameras_per_unit: int
    holdout_cameras_per_unit: int
    camera_prefix_ids: tuple[str, ...]
    fit_camera_ids: tuple[str, ...]
    holdout_camera_ids: tuple[str, ...]
    observed_camera_ids: tuple[str, ...]
    excluded_cameras: tuple[ExcludedCameraDiagnostics, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.policy, CameraSelectionPolicy):
            raise TypeError("policy must be a CameraSelectionPolicy.")
        if (
            self.ownership_rule
            is not CameraOwnershipRule.FIXED_UNIT_EVEN_HOLDOUT_SLOTS_V1
        ):
            raise ValueError("Unsupported fixed camera ownership rule.")
        requested = _integer(
            self.requested_camera_count,
            name="requested_camera_count",
            minimum=1,
        )
        available = _integer(
            self.available_camera_count,
            name="available_camera_count",
            minimum=requested,
        )
        unit_count = _integer(
            self.partition_unit_count,
            name="partition_unit_count",
            minimum=1,
        )
        fit_per_unit = _integer(
            self.fit_cameras_per_unit,
            name="fit_cameras_per_unit",
            minimum=1,
        )
        holdout_per_unit = _integer(
            self.holdout_cameras_per_unit,
            name="holdout_cameras_per_unit",
            minimum=1,
        )
        if requested != unit_count * (fit_per_unit + holdout_per_unit):
            raise ValueError(
                "Requested camera count disagrees with immutable partition units."
            )
        selected = _camera_ids(
            self.camera_prefix_ids, name="selection camera_prefix_ids"
        )
        fit = _camera_ids(self.fit_camera_ids, name="selection fit_camera_ids")
        holdout = _camera_ids(
            self.holdout_camera_ids, name="selection holdout_camera_ids"
        )
        observed = _camera_ids(
            self.observed_camera_ids, name="selection observed_camera_ids"
        )
        if len(selected) != requested:
            raise ValueError("Selected camera IDs do not match the requested count.")
        if len(fit) != unit_count * fit_per_unit or len(holdout) != (
            unit_count * holdout_per_unit
        ):
            raise ValueError("Fit/holdout ownership counts disagree with fixed units.")
        expected_fit, expected_holdout = _fixed_unit_camera_ownership(
            selected,
            fit_cameras_per_unit=fit_per_unit,
            holdout_cameras_per_unit=holdout_per_unit,
        )
        if fit != expected_fit or holdout != expected_holdout:
            raise ValueError(
                "Fixed fit/holdout ownership violates the persisted unit slot rule."
            )
        exclusions = tuple(self.excluded_cameras)
        excluded_ids = tuple(item.camera_id for item in exclusions)
        if len(excluded_ids) != len(set(excluded_ids)):
            raise ValueError("Fixed selection exclusion IDs must be unique.")
        if set(observed).intersection(excluded_ids) or set(observed).union(
            excluded_ids
        ) != set(selected):
            raise ValueError(
                "Observed and excluded IDs must partition the fixed selection."
            )
        if observed != tuple(
            item for item in selected if item not in set(excluded_ids)
        ):
            raise ValueError("Observed camera IDs must preserve fixed selection order.")
        ownership = {item: CameraEvidencePartition.FIT for item in fit} | {
            item: CameraEvidencePartition.HOLDOUT for item in holdout
        }
        if any(
            ownership[item.camera_id] is not item.original_partition
            for item in exclusions
        ):
            raise ValueError("Exclusion ownership disagrees with fixed partitioning.")
        object.__setattr__(self, "requested_camera_count", requested)
        object.__setattr__(self, "available_camera_count", available)
        object.__setattr__(self, "partition_unit_count", unit_count)
        object.__setattr__(self, "fit_cameras_per_unit", fit_per_unit)
        object.__setattr__(self, "holdout_cameras_per_unit", holdout_per_unit)
        object.__setattr__(self, "camera_prefix_ids", selected)
        object.__setattr__(self, "fit_camera_ids", fit)
        object.__setattr__(self, "holdout_camera_ids", holdout)
        object.__setattr__(self, "observed_camera_ids", observed)
        object.__setattr__(self, "excluded_cameras", exclusions)

    def to_dict(self) -> dict[str, object]:
        """Return the strict fixed selection representation."""
        return {
            "policy": self.policy.value,
            "ownership_rule": self.ownership_rule.value,
            "requested_camera_count": self.requested_camera_count,
            "available_camera_count": self.available_camera_count,
            "partition_unit_count": self.partition_unit_count,
            "fit_cameras_per_unit": self.fit_cameras_per_unit,
            "holdout_cameras_per_unit": self.holdout_cameras_per_unit,
            "camera_prefix_ids": list(self.camera_prefix_ids),
            "fit_camera_ids": list(self.fit_camera_ids),
            "holdout_camera_ids": list(self.holdout_camera_ids),
            "observed_camera_ids": list(self.observed_camera_ids),
            "excluded_cameras": [item.to_dict() for item in self.excluded_cameras],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse strict one-shot selection diagnostics."""
        keys = {
            "policy",
            "ownership_rule",
            "requested_camera_count",
            "available_camera_count",
            "partition_unit_count",
            "fit_cameras_per_unit",
            "holdout_cameras_per_unit",
            "camera_prefix_ids",
            "fit_camera_ids",
            "holdout_camera_ids",
            "observed_camera_ids",
            "excluded_cameras",
        }
        raw = _strict_mapping(value, keys=keys, name="fixed camera selection")
        exclusions = _sequence(raw["excluded_cameras"], name="excluded_cameras")
        return cls(
            policy=CameraSelectionPolicy(_string(raw["policy"], name="policy")),
            ownership_rule=CameraOwnershipRule(
                _string(raw["ownership_rule"], name="ownership_rule")
            ),
            requested_camera_count=_integer(
                raw["requested_camera_count"], name="requested_camera_count", minimum=1
            ),
            available_camera_count=_integer(
                raw["available_camera_count"], name="available_camera_count", minimum=1
            ),
            partition_unit_count=_integer(
                raw["partition_unit_count"],
                name="partition_unit_count",
                minimum=1,
            ),
            fit_cameras_per_unit=_integer(
                raw["fit_cameras_per_unit"], name="fit_cameras_per_unit", minimum=1
            ),
            holdout_cameras_per_unit=_integer(
                raw["holdout_cameras_per_unit"],
                name="holdout_cameras_per_unit",
                minimum=1,
            ),
            camera_prefix_ids=_string_tuple(
                raw["camera_prefix_ids"], name="camera_prefix_ids"
            ),
            fit_camera_ids=_string_tuple(raw["fit_camera_ids"], name="fit_camera_ids"),
            holdout_camera_ids=_string_tuple(
                raw["holdout_camera_ids"], name="holdout_camera_ids"
            ),
            observed_camera_ids=_string_tuple(
                raw["observed_camera_ids"], name="observed_camera_ids"
            ),
            excluded_cameras=tuple(
                ExcludedCameraDiagnostics.from_dict(item) for item in exclusions
            ),
        )


@dataclass(frozen=True, slots=True)
class AlignmentEvaluationDiagnostics:
    """The single fit selection and single post-selection holdout evaluation."""

    policy: AlignmentEvaluationPolicy
    evaluation_index: int
    fit_camera_ids: tuple[str, ...]
    holdout_camera_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    fit_evaluation_count: int
    holdout_evaluation_count: int
    outcome: AlignmentEvaluationOutcome

    def __post_init__(self) -> None:
        if (
            self.policy
            is not AlignmentEvaluationPolicy.FIT_SELECT_ONCE_HOLDOUT_EVALUATE_ONCE_V1
        ):
            raise ValueError("Unsupported alignment evaluation policy.")
        if _integer(self.evaluation_index, name="evaluation_index", minimum=0) != 0:
            raise ValueError("The one-shot alignment evaluation index must be zero.")
        fit = _camera_ids(self.fit_camera_ids, name="evaluation fit_camera_ids")
        holdout = _camera_ids(
            self.holdout_camera_ids, name="evaluation holdout_camera_ids"
        )
        candidates = _camera_ids(self.candidate_ids, name="evaluation candidate_ids")
        if set(fit).intersection(holdout):
            raise ValueError("Evaluation fit and holdout camera IDs overlap.")
        if self.fit_evaluation_count != 1 or self.holdout_evaluation_count != 1:
            raise ValueError("Fit and holdout must each be evaluated exactly once.")
        if self.outcome is not AlignmentEvaluationOutcome.FULL_VALIDATION_PASS:
            raise ValueError("Returned evidence must record one full validation PASS.")
        object.__setattr__(self, "fit_camera_ids", fit)
        object.__setattr__(self, "holdout_camera_ids", holdout)
        object.__setattr__(self, "candidate_ids", candidates)

    def to_dict(self) -> dict[str, object]:
        """Return strict one-evaluation diagnostics."""
        return {
            "policy": self.policy.value,
            "evaluation_index": self.evaluation_index,
            "fit_camera_ids": list(self.fit_camera_ids),
            "holdout_camera_ids": list(self.holdout_camera_ids),
            "candidate_ids": list(self.candidate_ids),
            "fit_evaluation_count": self.fit_evaluation_count,
            "holdout_evaluation_count": self.holdout_evaluation_count,
            "outcome": self.outcome.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse strict one-evaluation diagnostics."""
        keys = {
            "policy",
            "evaluation_index",
            "fit_camera_ids",
            "holdout_camera_ids",
            "candidate_ids",
            "fit_evaluation_count",
            "holdout_evaluation_count",
            "outcome",
        }
        raw = _strict_mapping(value, keys=keys, name="alignment evaluation")
        return cls(
            policy=AlignmentEvaluationPolicy(
                _string(raw["policy"], name="evaluation policy")
            ),
            evaluation_index=_integer(
                raw["evaluation_index"], name="evaluation_index", minimum=0
            ),
            fit_camera_ids=_string_tuple(raw["fit_camera_ids"], name="fit_camera_ids"),
            holdout_camera_ids=_string_tuple(
                raw["holdout_camera_ids"], name="holdout_camera_ids"
            ),
            candidate_ids=_string_tuple(raw["candidate_ids"], name="candidate_ids"),
            fit_evaluation_count=_integer(
                raw["fit_evaluation_count"], name="fit_evaluation_count", minimum=1
            ),
            holdout_evaluation_count=_integer(
                raw["holdout_evaluation_count"],
                name="holdout_evaluation_count",
                minimum=1,
            ),
            outcome=AlignmentEvaluationOutcome(
                _string(raw["outcome"], name="evaluation outcome")
            ),
        )


@dataclass(frozen=True, slots=True)
class LineInferenceDeterminismDiagnostics:
    """Runtime determinism policy and environment, without hardware-portable claims."""

    seed: int
    device: str
    model_eval: bool
    inference_mode: bool
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    cublas_workspace_config: str | None
    torch_version: str
    cuda_version: str | None
    device_name: str
    cross_hardware_bit_identity_claimed: bool

    def __post_init__(self) -> None:
        _integer(self.seed, name="determinism seed", minimum=0)
        for name in ("device", "torch_version", "device_name"):
            _string(getattr(self, name), name=name)
        for name in (
            "model_eval",
            "inference_mode",
            "deterministic_algorithms",
            "deterministic_warn_only",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "cuda_matmul_allow_tf32",
            "cudnn_allow_tf32",
            "cross_hardware_bit_identity_claimed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean.")
        if (
            not self.model_eval
            or not self.inference_mode
            or not self.deterministic_algorithms
        ):
            raise ValueError(
                "Line inference must use eval, inference, and deterministic modes."
            )
        if self.deterministic_warn_only:
            raise ValueError("Nondeterministic operations must fail instead of warn.")
        if self.cudnn_benchmark or not self.cudnn_deterministic:
            raise ValueError("cuDNN determinism policy is not strict.")
        if self.cuda_matmul_allow_tf32 or self.cudnn_allow_tf32:
            raise ValueError("TF32 must be disabled for deterministic inference.")
        if self.device.startswith("cuda"):
            if self.cublas_workspace_config not in {":4096:8", ":16:8"}:
                raise ValueError(
                    "CUDA determinism requires a supported CUBLAS workspace."
                )
            if self.cuda_version is None:
                raise ValueError("CUDA inference must record its CUDA version.")
        elif self.cublas_workspace_config is not None or self.cuda_version is not None:
            raise ValueError("CPU diagnostics must not claim CUDA environment values.")
        if self.cross_hardware_bit_identity_claimed:
            raise ValueError("Cross-hardware bit identity must not be claimed.")

    def to_dict(self) -> dict[str, object]:
        """Return strict determinism diagnostics."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse strict determinism diagnostics."""
        keys = set(cls.__dataclass_fields__)
        raw = _strict_mapping(value, keys=keys, name="line inference determinism")
        cublas = raw["cublas_workspace_config"]
        cuda_version = raw["cuda_version"]
        if cublas is not None and not isinstance(cublas, str):
            raise TypeError("cublas_workspace_config must be a string or null.")
        if cuda_version is not None and not isinstance(cuda_version, str):
            raise TypeError("cuda_version must be a string or null.")
        return cls(
            seed=_integer(raw["seed"], name="seed", minimum=0),
            device=_string(raw["device"], name="device"),
            model_eval=_boolean(raw["model_eval"], name="model_eval"),
            inference_mode=_boolean(raw["inference_mode"], name="inference_mode"),
            deterministic_algorithms=_boolean(
                raw["deterministic_algorithms"], name="deterministic_algorithms"
            ),
            deterministic_warn_only=_boolean(
                raw["deterministic_warn_only"], name="deterministic_warn_only"
            ),
            cudnn_benchmark=_boolean(raw["cudnn_benchmark"], name="cudnn_benchmark"),
            cudnn_deterministic=_boolean(
                raw["cudnn_deterministic"], name="cudnn_deterministic"
            ),
            cuda_matmul_allow_tf32=_boolean(
                raw["cuda_matmul_allow_tf32"], name="cuda_matmul_allow_tf32"
            ),
            cudnn_allow_tf32=_boolean(raw["cudnn_allow_tf32"], name="cudnn_allow_tf32"),
            cublas_workspace_config=cublas,
            torch_version=_string(raw["torch_version"], name="torch_version"),
            cuda_version=cuda_version,
            device_name=_string(raw["device_name"], name="device_name"),
            cross_hardware_bit_identity_claimed=_boolean(
                raw["cross_hardware_bit_identity_claimed"],
                name="cross_hardware_bit_identity_claimed",
            ),
        )


@dataclass(frozen=True, slots=True)
class ProposalSearchDiagnostics:
    """Bounded fit-only court-count inference and selected joint objective."""

    orientation_band_count: int
    center_tile_count: int
    maximum_center_tile_width_scene_units: float
    maximum_candidate_count: int
    maximum_retained_state_count: int
    maximum_tile_state_count: int
    maximum_residual_state_count: int
    residual_state_count: int
    residual_tree_build_count: int
    explored_tile_state_count: int
    geometrically_impossible_tile_state_count: int
    feasible_proposal_count_before_deduplication: int
    duplicate_proposal_count: int
    retained_proposal_count: int
    expanded_state_count: int
    pruned_state_count: int
    feasible_complete_state_count: int
    refinement_attempt_count: int
    refinement_rejected_state_count: int
    selected_complete_state_rank: int
    inferred_candidate_count: int
    stopping_reason: ProposalSearchStopReason
    minimum_explained_evidence_fraction: float
    selected_orientation_band_indices: tuple[int, ...]
    selected_center_tile_indices: tuple[int, ...]
    selected_candidate_explained_evidence_fractions: tuple[float, ...]
    original_point_count: int
    selected_residual_point_count: int
    selected_explained_point_count: int
    original_evidence_sum: float
    selected_residual_evidence_sum: float
    selected_explained_evidence_sum: float
    selected_explained_evidence_fraction: float
    selected_native_score_sum: float

    def __post_init__(self) -> None:
        bands = _integer(
            self.orientation_band_count,
            name="orientation_band_count",
            minimum=1,
        )
        tiles = _integer(self.center_tile_count, name="center_tile_count", minimum=1)
        tile_width = _finite_float(
            self.maximum_center_tile_width_scene_units,
            name="maximum_center_tile_width_scene_units",
        )
        if tile_width <= 0.0:
            raise ValueError("Maximum center tile width must be positive.")
        maximum_candidates = _integer(
            self.maximum_candidate_count,
            name="maximum_candidate_count",
            minimum=1,
        )
        maximum_retained = _integer(
            self.maximum_retained_state_count,
            name="maximum_retained_state_count",
            minimum=1,
        )
        maximum_tile_states = _integer(
            self.maximum_tile_state_count,
            name="maximum_tile_state_count",
            minimum=1,
        )
        maximum_residual_states = _integer(
            self.maximum_residual_state_count,
            name="maximum_residual_state_count",
            minimum=1,
        )
        residual_states = _integer(
            self.residual_state_count,
            name="residual_state_count",
            minimum=1,
        )
        residual_tree_builds = _integer(
            self.residual_tree_build_count,
            name="residual_tree_build_count",
            minimum=1,
        )
        explored = _integer(
            self.explored_tile_state_count,
            name="explored_tile_state_count",
            minimum=1,
        )
        impossible = _integer(
            self.geometrically_impossible_tile_state_count,
            name="geometrically_impossible_tile_state_count",
            minimum=0,
        )
        feasible_proposals = _integer(
            self.feasible_proposal_count_before_deduplication,
            name="feasible_proposal_count_before_deduplication",
            minimum=1,
        )
        duplicates = _integer(
            self.duplicate_proposal_count,
            name="duplicate_proposal_count",
            minimum=0,
        )
        retained = _integer(
            self.retained_proposal_count,
            name="retained_proposal_count",
            minimum=1,
        )
        expanded = _integer(
            self.expanded_state_count,
            name="expanded_state_count",
            minimum=1,
        )
        pruned = _integer(
            self.pruned_state_count,
            name="pruned_state_count",
            minimum=0,
        )
        feasible = _integer(
            self.feasible_complete_state_count,
            name="feasible_complete_state_count",
            minimum=1,
        )
        refinement_attempts = _integer(
            self.refinement_attempt_count,
            name="refinement_attempt_count",
            minimum=1,
        )
        refinement_rejections = _integer(
            self.refinement_rejected_state_count,
            name="refinement_rejected_state_count",
            minimum=0,
        )
        selected_rank = _integer(
            self.selected_complete_state_rank,
            name="selected_complete_state_rank",
            minimum=0,
        )
        if refinement_attempts != refinement_rejections + 1:
            raise ValueError(
                "Refinement attempts must equal rejected states plus the selection."
            )
        if selected_rank != refinement_rejections:
            raise ValueError(
                "Selected complete-state rank must equal prior refinement rejections."
            )
        if selected_rank >= feasible:
            raise ValueError(
                "Selected complete-state rank exceeds feasible complete states."
            )
        inferred = _integer(
            self.inferred_candidate_count,
            name="inferred_candidate_count",
            minimum=1,
        )
        if inferred > maximum_candidates:
            raise ValueError("Inferred candidate count exceeds the configured maximum.")
        if not isinstance(self.stopping_reason, ProposalSearchStopReason):
            raise TypeError("stopping_reason must be a ProposalSearchStopReason.")
        minimum_fraction = _finite_float(
            self.minimum_explained_evidence_fraction,
            name="minimum_explained_evidence_fraction",
        )
        if not 0.0 < minimum_fraction < 1.0:
            raise ValueError("minimum_explained_evidence_fraction must lie in (0, 1).")
        branch = tuple(
            _integer(item, name="selected orientation band", minimum=0)
            for item in self.selected_orientation_band_indices
        )
        if len(branch) != inferred or any(item >= bands for item in branch):
            raise ValueError("Selected proposal branch contains an invalid band index.")
        selected_tiles = tuple(
            _integer(item, name="selected center tile", minimum=0)
            for item in self.selected_center_tile_indices
        )
        if len(selected_tiles) != len(branch) or any(
            item >= tiles for item in selected_tiles
        ):
            raise ValueError("Selected proposal branch contains an invalid tile index.")
        branch_factor = bands * tiles
        expected_maximum_residual_states = 1 + (
            (maximum_candidates - 1) * maximum_retained
        )
        if maximum_residual_states != expected_maximum_residual_states:
            raise ValueError(
                "Maximum residual-state count disagrees with the bounded beam."
            )
        if maximum_tile_states != maximum_residual_states * branch_factor:
            raise ValueError(
                "Maximum tile/state count disagrees with the bounded beam."
            )
        if residual_states > maximum_residual_states:
            raise ValueError("Residual states exceed their configured resource bound.")
        if residual_tree_builds != residual_states:
            raise ValueError("Residual proposal search must build one tree per state.")
        if explored + impossible != residual_states * branch_factor:
            raise ValueError(
                "Every residual state must classify each orientation/tile branch."
            )
        if feasible_proposals != retained + duplicates:
            raise ValueError("Proposal deduplication counts do not balance.")
        if expanded > retained:
            raise ValueError("Expanded states exceed retained feasible proposals.")
        if pruned > expanded:
            raise ValueError("Pruned states exceed expanded search states.")
        if feasible > maximum_retained:
            raise ValueError("Complete states exceed the configured beam width.")
        candidate_fractions = tuple(
            _finite_float(item, name="candidate explained evidence fraction")
            for item in self.selected_candidate_explained_evidence_fractions
        )
        if len(candidate_fractions) != inferred or any(
            item + 1.0e-12 < minimum_fraction or item > 1.0
            for item in candidate_fractions
        ):
            raise ValueError(
                "Selected candidates must each satisfy the explained-evidence gate."
            )
        original = _integer(
            self.original_point_count,
            name="original_point_count",
            minimum=3,
        )
        residual = _integer(
            self.selected_residual_point_count,
            name="selected_residual_point_count",
            minimum=0,
        )
        explained = _integer(
            self.selected_explained_point_count,
            name="selected_explained_point_count",
            minimum=1,
        )
        if residual + explained != original:
            raise ValueError(
                "Explained and residual points must partition original evidence."
            )
        original_evidence = _finite_float(
            self.original_evidence_sum,
            name="original_evidence_sum",
        )
        residual_evidence = _finite_float(
            self.selected_residual_evidence_sum,
            name="selected_residual_evidence_sum",
        )
        explained_evidence = _finite_float(
            self.selected_explained_evidence_sum,
            name="selected_explained_evidence_sum",
        )
        explained_fraction = _finite_float(
            self.selected_explained_evidence_fraction,
            name="selected_explained_evidence_fraction",
        )
        if (
            original_evidence <= 0.0
            or residual_evidence < 0.0
            or explained_evidence <= 0.0
            or not math.isclose(
                residual_evidence + explained_evidence,
                original_evidence,
                abs_tol=1.0e-8,
                rel_tol=1.0e-8,
            )
        ):
            raise ValueError(
                "Explained and residual weighted evidence must partition the original."
            )
        measured_fraction = explained_evidence / original_evidence
        if not math.isclose(
            explained_fraction,
            measured_fraction,
            abs_tol=1.0e-10,
            rel_tol=1.0e-8,
        ) or not math.isclose(
            sum(candidate_fractions),
            measured_fraction,
            abs_tol=1.0e-8,
            rel_tol=1.0e-8,
        ):
            raise ValueError("Explained-evidence fractions disagree with the sums.")
        if (
            self.stopping_reason
            is ProposalSearchStopReason.RESIDUAL_EVIDENCE_BELOW_MINIMUM
            and residual_evidence / original_evidence >= minimum_fraction
        ):
            raise ValueError(
                "Residual-evidence stop requires residual evidence below the gate."
            )
        score = _finite_float(
            self.selected_native_score_sum, name="selected_native_score_sum"
        )
        if score <= 0.0:
            raise ValueError("Selected native score sum must be positive.")
        object.__setattr__(self, "selected_orientation_band_indices", branch)
        object.__setattr__(self, "selected_center_tile_indices", selected_tiles)
        object.__setattr__(
            self,
            "selected_candidate_explained_evidence_fractions",
            candidate_fractions,
        )

    def to_dict(self) -> dict[str, object]:
        """Return strict proposal search diagnostics."""
        return {
            "orientation_band_count": self.orientation_band_count,
            "center_tile_count": self.center_tile_count,
            "maximum_center_tile_width_scene_units": (
                self.maximum_center_tile_width_scene_units
            ),
            "maximum_candidate_count": self.maximum_candidate_count,
            "maximum_retained_state_count": self.maximum_retained_state_count,
            "maximum_tile_state_count": self.maximum_tile_state_count,
            "maximum_residual_state_count": self.maximum_residual_state_count,
            "residual_state_count": self.residual_state_count,
            "residual_tree_build_count": self.residual_tree_build_count,
            "explored_tile_state_count": self.explored_tile_state_count,
            "geometrically_impossible_tile_state_count": (
                self.geometrically_impossible_tile_state_count
            ),
            "feasible_proposal_count_before_deduplication": (
                self.feasible_proposal_count_before_deduplication
            ),
            "duplicate_proposal_count": self.duplicate_proposal_count,
            "retained_proposal_count": self.retained_proposal_count,
            "expanded_state_count": self.expanded_state_count,
            "pruned_state_count": self.pruned_state_count,
            "feasible_complete_state_count": self.feasible_complete_state_count,
            "refinement_attempt_count": self.refinement_attempt_count,
            "refinement_rejected_state_count": (self.refinement_rejected_state_count),
            "selected_complete_state_rank": self.selected_complete_state_rank,
            "inferred_candidate_count": self.inferred_candidate_count,
            "stopping_reason": self.stopping_reason.value,
            "minimum_explained_evidence_fraction": (
                self.minimum_explained_evidence_fraction
            ),
            "selected_orientation_band_indices": list(
                self.selected_orientation_band_indices
            ),
            "selected_center_tile_indices": list(self.selected_center_tile_indices),
            "selected_candidate_explained_evidence_fractions": list(
                self.selected_candidate_explained_evidence_fractions
            ),
            "original_point_count": self.original_point_count,
            "selected_residual_point_count": self.selected_residual_point_count,
            "selected_explained_point_count": self.selected_explained_point_count,
            "original_evidence_sum": self.original_evidence_sum,
            "selected_residual_evidence_sum": self.selected_residual_evidence_sum,
            "selected_explained_evidence_sum": self.selected_explained_evidence_sum,
            "selected_explained_evidence_fraction": (
                self.selected_explained_evidence_fraction
            ),
            "selected_native_score_sum": self.selected_native_score_sum,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse strict proposal search diagnostics."""
        keys = set(cls.__dataclass_fields__)
        raw = _strict_mapping(value, keys=keys, name="proposal search diagnostics")
        return cls(
            orientation_band_count=_integer(
                raw["orientation_band_count"], name="orientation_band_count", minimum=1
            ),
            center_tile_count=_integer(
                raw["center_tile_count"], name="center_tile_count", minimum=1
            ),
            maximum_center_tile_width_scene_units=_finite_float(
                raw["maximum_center_tile_width_scene_units"],
                name="maximum_center_tile_width_scene_units",
            ),
            maximum_candidate_count=_integer(
                raw["maximum_candidate_count"],
                name="maximum_candidate_count",
                minimum=1,
            ),
            maximum_retained_state_count=_integer(
                raw["maximum_retained_state_count"],
                name="maximum_retained_state_count",
                minimum=1,
            ),
            maximum_tile_state_count=_integer(
                raw["maximum_tile_state_count"],
                name="maximum_tile_state_count",
                minimum=1,
            ),
            maximum_residual_state_count=_integer(
                raw["maximum_residual_state_count"],
                name="maximum_residual_state_count",
                minimum=1,
            ),
            residual_state_count=_integer(
                raw["residual_state_count"],
                name="residual_state_count",
                minimum=1,
            ),
            residual_tree_build_count=_integer(
                raw["residual_tree_build_count"],
                name="residual_tree_build_count",
                minimum=1,
            ),
            explored_tile_state_count=_integer(
                raw["explored_tile_state_count"],
                name="explored_tile_state_count",
                minimum=1,
            ),
            geometrically_impossible_tile_state_count=_integer(
                raw["geometrically_impossible_tile_state_count"],
                name="geometrically_impossible_tile_state_count",
                minimum=0,
            ),
            feasible_proposal_count_before_deduplication=_integer(
                raw["feasible_proposal_count_before_deduplication"],
                name="feasible_proposal_count_before_deduplication",
                minimum=1,
            ),
            duplicate_proposal_count=_integer(
                raw["duplicate_proposal_count"],
                name="duplicate_proposal_count",
                minimum=0,
            ),
            retained_proposal_count=_integer(
                raw["retained_proposal_count"],
                name="retained_proposal_count",
                minimum=1,
            ),
            expanded_state_count=_integer(
                raw["expanded_state_count"], name="expanded_state_count", minimum=1
            ),
            pruned_state_count=_integer(
                raw["pruned_state_count"], name="pruned_state_count", minimum=0
            ),
            feasible_complete_state_count=_integer(
                raw["feasible_complete_state_count"],
                name="feasible_complete_state_count",
                minimum=1,
            ),
            refinement_attempt_count=_integer(
                raw["refinement_attempt_count"],
                name="refinement_attempt_count",
                minimum=1,
            ),
            refinement_rejected_state_count=_integer(
                raw["refinement_rejected_state_count"],
                name="refinement_rejected_state_count",
                minimum=0,
            ),
            selected_complete_state_rank=_integer(
                raw["selected_complete_state_rank"],
                name="selected_complete_state_rank",
                minimum=0,
            ),
            inferred_candidate_count=_integer(
                raw["inferred_candidate_count"],
                name="inferred_candidate_count",
                minimum=1,
            ),
            stopping_reason=ProposalSearchStopReason(
                _string(raw["stopping_reason"], name="stopping_reason")
            ),
            minimum_explained_evidence_fraction=_finite_float(
                raw["minimum_explained_evidence_fraction"],
                name="minimum_explained_evidence_fraction",
            ),
            selected_orientation_band_indices=tuple(
                _integer(item, name="selected orientation band", minimum=0)
                for item in _sequence(
                    raw["selected_orientation_band_indices"],
                    name="selected_orientation_band_indices",
                )
            ),
            selected_center_tile_indices=tuple(
                _integer(item, name="selected center tile", minimum=0)
                for item in _sequence(
                    raw["selected_center_tile_indices"],
                    name="selected_center_tile_indices",
                )
            ),
            selected_candidate_explained_evidence_fractions=tuple(
                _finite_float(
                    item,
                    name="candidate explained evidence fraction",
                )
                for item in _sequence(
                    raw["selected_candidate_explained_evidence_fractions"],
                    name="selected_candidate_explained_evidence_fractions",
                )
            ),
            original_point_count=_integer(
                raw["original_point_count"], name="original_point_count", minimum=3
            ),
            selected_residual_point_count=_integer(
                raw["selected_residual_point_count"],
                name="selected_residual_point_count",
                minimum=0,
            ),
            selected_explained_point_count=_integer(
                raw["selected_explained_point_count"],
                name="selected_explained_point_count",
                minimum=1,
            ),
            original_evidence_sum=_finite_float(
                raw["original_evidence_sum"], name="original_evidence_sum"
            ),
            selected_residual_evidence_sum=_finite_float(
                raw["selected_residual_evidence_sum"],
                name="selected_residual_evidence_sum",
            ),
            selected_explained_evidence_sum=_finite_float(
                raw["selected_explained_evidence_sum"],
                name="selected_explained_evidence_sum",
            ),
            selected_explained_evidence_fraction=_finite_float(
                raw["selected_explained_evidence_fraction"],
                name="selected_explained_evidence_fraction",
            ),
            selected_native_score_sum=_finite_float(
                raw["selected_native_score_sum"], name="selected_native_score_sum"
            ),
        )


@dataclass(frozen=True, slots=True)
class CandidateScaleDiagnostics:
    """Measured Sim(3) scale and image-line score for one fitted court."""

    candidate_id: str
    nht_scene_units_per_metre: float
    template_score: float
    common_scale_refit_center_displacement_metres: float
    maximum_common_scale_refit_center_displacement_metres: float
    proposal_orientation_band_minimum_radians: float
    proposal_orientation_band_maximum_radians: float
    proposal_residual_point_count_before_suppression: int
    proposal_residual_point_count_after_suppression: int
    native_center_uv: tuple[float, float]
    native_orientation_radians: float

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, name="candidate_id")
        scale = _finite_float(
            self.nht_scene_units_per_metre,
            name="nht_scene_units_per_metre",
        )
        score = _finite_float(self.template_score, name="template_score")
        if scale <= 0.0 or score <= 0.0:
            raise ValueError("Candidate scale and template score must be positive.")
        displacement = _finite_float(
            self.common_scale_refit_center_displacement_metres,
            name="common_scale_refit_center_displacement_metres",
        )
        maximum_displacement = _finite_float(
            self.maximum_common_scale_refit_center_displacement_metres,
            name="maximum_common_scale_refit_center_displacement_metres",
        )
        if displacement < 0.0 or maximum_displacement <= 0.0:
            raise ValueError(
                "Common-scale refit displacement must be non-negative and its "
                "derived maximum must be positive."
            )
        if displacement > maximum_displacement + 1.0e-10:
            raise ValueError(
                "Common-scale refit displacement exceeds its derived maximum."
            )
        band_minimum = _finite_float(
            self.proposal_orientation_band_minimum_radians,
            name="proposal_orientation_band_minimum_radians",
        )
        band_maximum = _finite_float(
            self.proposal_orientation_band_maximum_radians,
            name="proposal_orientation_band_maximum_radians",
        )
        if (
            band_minimum >= band_maximum
            or band_maximum - band_minimum > math.pi / 2.0 + 1.0e-12
        ):
            raise ValueError(
                "Proposal orientation band must be ordered and no wider than pi/2."
            )
        before = _integer(
            self.proposal_residual_point_count_before_suppression,
            name="proposal_residual_point_count_before_suppression",
            minimum=3,
        )
        after = _integer(
            self.proposal_residual_point_count_after_suppression,
            name="proposal_residual_point_count_after_suppression",
            minimum=0,
        )
        if after >= before:
            raise ValueError(
                "Selected proposal must suppress at least one residual point."
            )
        native_center = tuple(
            _finite_float(item, name="native_center_uv")
            for item in self.native_center_uv
        )
        if len(native_center) != 2:
            raise ValueError("native_center_uv must contain exactly two values.")
        native_orientation = _finite_float(
            self.native_orientation_radians,
            name="native_orientation_radians",
        )
        if not band_minimum - 1.0e-12 <= native_orientation <= band_maximum + 1.0e-12:
            raise ValueError(
                "Native proposal orientation lies outside its search band."
            )
        object.__setattr__(
            self,
            "proposal_orientation_band_minimum_radians",
            band_minimum,
        )
        object.__setattr__(
            self,
            "proposal_orientation_band_maximum_radians",
            band_maximum,
        )
        object.__setattr__(self, "native_center_uv", native_center)
        object.__setattr__(
            self,
            "native_orientation_radians",
            native_orientation,
        )
        object.__setattr__(self, "nht_scene_units_per_metre", scale)
        object.__setattr__(self, "template_score", score)
        object.__setattr__(
            self,
            "common_scale_refit_center_displacement_metres",
            displacement,
        )
        object.__setattr__(
            self,
            "maximum_common_scale_refit_center_displacement_metres",
            maximum_displacement,
        )

    def to_dict(self) -> dict[str, object]:
        """Return strict persisted diagnostics."""
        return {
            "candidate_id": self.candidate_id,
            "nht_scene_units_per_metre": self.nht_scene_units_per_metre,
            "template_score": self.template_score,
            "common_scale_refit_center_displacement_metres": (
                self.common_scale_refit_center_displacement_metres
            ),
            "maximum_common_scale_refit_center_displacement_metres": (
                self.maximum_common_scale_refit_center_displacement_metres
            ),
            "proposal_orientation_band_radians": [
                self.proposal_orientation_band_minimum_radians,
                self.proposal_orientation_band_maximum_radians,
            ],
            "proposal_residual_point_count_before_suppression": (
                self.proposal_residual_point_count_before_suppression
            ),
            "proposal_residual_point_count_after_suppression": (
                self.proposal_residual_point_count_after_suppression
            ),
            "native_center_uv": list(self.native_center_uv),
            "native_orientation_radians": self.native_orientation_radians,
        }


@dataclass(frozen=True, slots=True)
class AlignmentEvidenceDiagnostics:
    """Measured image-line and common-scale evidence retained for audit."""

    cameras: tuple[CameraLineDiagnostics, ...]
    candidate_scales: tuple[CandidateScaleDiagnostics, ...]
    common_nht_scene_units_per_metre: float
    maximum_relative_scale_deviation: float
    selection: FixedCameraSelectionDiagnostics
    evaluation: AlignmentEvaluationDiagnostics
    determinism: LineInferenceDeterminismDiagnostics
    proposal_search: ProposalSearchDiagnostics
    excluded_cameras: tuple[ExcludedCameraDiagnostics, ...]

    def __post_init__(self) -> None:
        cameras = tuple(self.cameras)
        candidate_scales = tuple(self.candidate_scales)
        excluded_cameras = tuple(self.excluded_cameras)
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
        excluded_camera_ids = [item.camera_id for item in excluded_cameras]
        if len(excluded_camera_ids) != len(set(excluded_camera_ids)):
            raise ValueError("Excluded diagnostic camera IDs must be unique.")
        overlap = set(camera_ids).intersection(excluded_camera_ids)
        if overlap:
            raise ValueError(
                "Retained and excluded diagnostic camera IDs overlap: "
                f"{sorted(overlap)}."
            )
        if excluded_cameras != self.selection.excluded_cameras:
            raise ValueError("Top-level exclusions disagree with fixed selection.")
        if set(camera_ids) != set(self.selection.observed_camera_ids):
            raise ValueError("Camera diagnostics disagree with observed selection IDs.")
        excluded_set = set(excluded_camera_ids)
        retained_fit = tuple(
            item for item in self.selection.fit_camera_ids if item not in excluded_set
        )
        retained_holdout = tuple(
            item
            for item in self.selection.holdout_camera_ids
            if item not in excluded_set
        )
        if self.evaluation.fit_camera_ids != retained_fit or (
            self.evaluation.holdout_camera_ids != retained_holdout
        ):
            raise ValueError(
                "One-shot evaluation partitions disagree with selection/exclusions."
            )
        if tuple(candidate_ids) != self.evaluation.candidate_ids:
            raise ValueError(
                "Evaluation candidate IDs disagree with scale diagnostics."
            )
        if len(candidate_ids) != self.proposal_search.inferred_candidate_count:
            raise ValueError(
                "Candidate diagnostics disagree with the inferred proposal count."
            )
        if len(self.proposal_search.selected_orientation_band_indices) != len(
            candidate_ids
        ):
            raise ValueError(
                "Selected proposal branch depth disagrees with candidates."
            )
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
        object.__setattr__(self, "excluded_cameras", excluded_cameras)
        object.__setattr__(self, "common_nht_scene_units_per_metre", common_scale)
        object.__setattr__(self, "maximum_relative_scale_deviation", maximum_deviation)

    def to_dict(self) -> dict[str, object]:
        """Return machine-readable measured evidence diagnostics."""
        return {
            "schema": "alignment_measured_evidence_v10",
            "cameras": [item.to_dict() for item in self.cameras],
            "excluded_cameras": [item.to_dict() for item in self.excluded_cameras],
            "selection": self.selection.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "determinism": self.determinism.to_dict(),
            "proposal_search": self.proposal_search.to_dict(),
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
    whole_court_settings: WholeCourtEvidenceSettings

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
        if not isinstance(
            self.whole_court_settings,
            WholeCourtEvidenceSettings,
        ):
            raise TypeError("whole_court_settings must be WholeCourtEvidenceSettings.")
        if self.whole_court_settings.required_court_count != len(candidates):
            raise ValueError(
                "Whole-court policy required count disagrees with evidence candidates."
            )
        if (
            self.diagnostics.maximum_relative_scale_deviation
            > self.whole_court_settings.maximum_common_scale_relative_deviation
        ):
            raise ValueError(
                "Native candidate scale deviation exceeds the whole-court policy."
            )
        for scale_diagnostic in self.diagnostics.candidate_scales:
            if not math.isclose(
                scale_diagnostic.maximum_common_scale_refit_center_displacement_metres,
                self.whole_court_settings.maximum_center_refit_displacement_metres,
                abs_tol=1.0e-10,
                rel_tol=1.0e-8,
            ):
                raise ValueError(
                    "Candidate refit displacement bound disagrees with policy."
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


@dataclass(frozen=True, slots=True)
class EvaluatedAlignment:
    """One immutable evidence/result/heatmap bundle produced by the one-shot gate."""

    evidence: AlignmentEvidence
    result: AlignmentResult
    heatmaps: AlignmentLineHeatmaps

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, AlignmentEvidence):
            raise TypeError("evidence must be AlignmentEvidence.")
        if not isinstance(self.result, AlignmentResult):
            raise TypeError("result must be AlignmentResult.")
        if not isinstance(self.heatmaps, AlignmentLineHeatmaps):
            raise TypeError("heatmaps must be AlignmentLineHeatmaps.")
        if self.result.partitions != self.evidence.partitions:
            raise ValueError("Evaluated result partitions disagree with evidence.")
        if self.result.metric_adapter != self.evidence.metric_adapter:
            raise ValueError("Evaluated result metric adapter disagrees with evidence.")
        evidence_ids = tuple(
            (candidate.court_instance_id, candidate.candidate_id)
            for candidate in self.evidence.candidates
        )
        result_ids = tuple(
            (candidate.court_instance_id, candidate.candidate_id)
            for candidate in self.result.candidates
        )
        if result_ids != evidence_ids:
            raise ValueError("Evaluated result candidates disagree with evidence.")
        selection = self.evidence.diagnostics.selection
        if self.heatmaps.camera_ids != selection.camera_prefix_ids:
            raise ValueError(
                "Evaluated heatmaps disagree with the fixed camera prefix."
            )
        if (
            self.heatmaps.aggregate_camera_ids
            != self.evidence.diagnostics.evaluation.fit_camera_ids
        ):
            raise ValueError(
                "Evaluated aggregate heatmap disagrees with fit-only count evidence."
            )
        projected_counts = {
            item.camera_id: item.projected_line_point_count
            for item in self.evidence.diagnostics.cameras
        }
        projected_counts.update(
            {
                item.camera_id: item.projected_line_point_count
                for item in selection.excluded_cameras
            }
        )
        if set(projected_counts) != set(selection.camera_prefix_ids):
            raise ValueError(
                "Evidence diagnostics do not cover the heatmap camera prefix."
            )
        if any(
            len(view.points_uv) != projected_counts[view.camera_id]
            for view in self.heatmaps.views
        ):
            raise ValueError(
                "Evaluated heatmap point counts disagree with diagnostics."
            )


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


def _fixed_unit_camera_ownership(
    camera_prefix_ids: tuple[str, ...],
    *,
    fit_cameras_per_unit: int,
    holdout_cameras_per_unit: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Recompute the versioned evenly spaced slot ownership from prefix order."""
    unit = fit_cameras_per_unit + holdout_cameras_per_unit
    if len(camera_prefix_ids) % unit != 0:
        raise ValueError(
            "Fixed camera prefix does not contain complete ownership units."
        )
    holdout_slots = {
        (2 * index + 1) * unit // (2 * holdout_cameras_per_unit)
        for index in range(holdout_cameras_per_unit)
    }
    holdout_indices = {
        index
        for index in range(len(camera_prefix_ids))
        if index % unit in holdout_slots
    }
    return (
        tuple(
            camera_id
            for index, camera_id in enumerate(camera_prefix_ids)
            if index not in holdout_indices
        ),
        tuple(
            camera_id
            for index, camera_id in enumerate(camera_prefix_ids)
            if index in holdout_indices
        ),
    )


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
    "AlignmentEvaluationDiagnostics",
    "AlignmentEvaluationOutcome",
    "AlignmentEvaluationPolicy",
    "AlignmentEvidence",
    "AlignmentEvidenceDiagnostics",
    "AlignmentPartitions",
    "AlignmentResult",
    "AlignmentStatus",
    "CandidateAlignment",
    "CandidateEvidence",
    "CandidateScaleDiagnostics",
    "CameraEvidencePartition",
    "CameraExclusionReason",
    "CameraSelectionPolicy",
    "CameraLineDiagnostics",
    "CameraOwnershipRule",
    "CorrespondenceSet",
    "EvaluatedAlignment",
    "MetricSceneAdapter",
    "MeasuredCameraLines",
    "ExcludedCameraDiagnostics",
    "FixedCameraSelectionDiagnostics",
    "LineInferenceDeterminismDiagnostics",
    "PartitionAssessment",
    "PartitionMetrics",
    "PartitionThresholds",
    "ProposalSearchDiagnostics",
    "ProposalSearchStopReason",
    "build_layout",
]
