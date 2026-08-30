"""Deterministic public RGB/alpha/depth/support benchmark logic."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

QUALITY_FEATURE_DEFINITION_ID = "court_public_quality_features_v1"
QUALITY_DECISION_REPORT_SCHEMA = "court_trajectory_quality_decision_v2"
BLIND_ANNOTATION_SCHEMA = "court_trajectory_blind_annotation_v1"
BLIND_ADJUDICATION_SCHEMA = "court_trajectory_blind_adjudication_v1"
CONSENSUS_SCHEMA = "court_trajectory_annotation_consensus_v1"
MINIMUM_HELD_OUT_RECALL = 0.90
MINIMUM_HELD_OUT_PRECISION = 0.80
MAXIMUM_VALID_CONTROL_FALSE_POSITIVE_RATE = 0.10
MINIMUM_HELD_OUT_POSITIVE_LABELS = 12
MINIMUM_HELD_OUT_NEGATIVE_LABELS = 12


class BenchmarkSplit(StrEnum):
    CALIBRATION = "calibration"
    HELD_OUT = "held_out"


class QualityDecision(StrEnum):
    QUALITY_ONLY_PROMOTED = "quality_only_promoted"
    QUALITY_ONLY_REJECTED = "quality_only_rejected"


class RuleOperator(StrEnum):
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"


@dataclass(frozen=True, slots=True)
class PublicQualityFeatures:
    """Features derived only from validated public renderer/support outputs."""

    rgb_gradient_p95: float
    alpha_mean: float
    alpha_coverage_fraction: float
    alpha_transition_fraction: float
    depth_valid_fraction: float
    depth_gradient_p95: float
    support_margin_m: float
    obstacle_clearance_m: float
    captured_camera_distance_m: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        for name in (
            "alpha_mean",
            "alpha_coverage_fraction",
            "alpha_transition_fraction",
            "depth_valid_fraction",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if (
            min(
                self.rgb_gradient_p95,
                self.depth_gradient_p95,
                self.captured_camera_distance_m,
            )
            < 0.0
        ):
            raise ValueError("Gradient and distance features must be non-negative.")

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}

    @classmethod
    def from_mapping(cls, value: object) -> PublicQualityFeatures:
        """Parse the exact public feature inventory without inferring omissions."""
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("Public quality feature keys are invalid.")
        return cls(
            **{
                name: _finite_number(value[name], name=name)
                for name in cls.__dataclass_fields__
            }
        )


@dataclass(frozen=True, slots=True)
class QualityDecisionRule:
    """One calibration-frozen scalar rule; no threshold is inferred here."""

    rule_id: str
    feature_name: str
    operator: RuleOperator
    threshold: float

    def __post_init__(self) -> None:
        if not self.rule_id or self.rule_id != self.rule_id.strip():
            raise ValueError("rule_id must be a non-empty trimmed string.")
        if self.feature_name not in PublicQualityFeatures.__dataclass_fields__:
            raise ValueError("feature_name is not a public quality feature.")
        if not isinstance(self.operator, RuleOperator):
            raise TypeError("operator must be a RuleOperator.")
        threshold = float(self.threshold)
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite.")
        object.__setattr__(self, "threshold", threshold)

    def predicts_artifact(self, features: PublicQualityFeatures) -> bool:
        value = float(getattr(features, self.feature_name))
        if self.operator is RuleOperator.GREATER_THAN_OR_EQUAL:
            return value >= self.threshold
        return value <= self.threshold

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "feature_name": self.feature_name,
            "operator": self.operator.value,
            "threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class QualityObservation:
    """One blind consensus label joined to features only after annotation."""

    opaque_id: str
    trajectory_group_id: str
    stratum: str
    split: BenchmarkSplit
    artifact_heavy: bool
    valid_control: bool
    features: PublicQualityFeatures

    def __post_init__(self) -> None:
        for name in ("opaque_id", "trajectory_group_id", "stratum"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be a non-empty trimmed string.")
        if not isinstance(self.split, BenchmarkSplit):
            raise TypeError("split must be a BenchmarkSplit.")
        if not isinstance(self.artifact_heavy, bool) or not isinstance(
            self.valid_control, bool
        ):
            raise TypeError("artifact_heavy and valid_control must be booleans.")
        if self.artifact_heavy and self.valid_control:
            raise ValueError("An artifact-heavy observation cannot be a valid control.")
        if not isinstance(self.features, PublicQualityFeatures):
            raise TypeError("features must be PublicQualityFeatures.")


@dataclass(frozen=True, slots=True)
class QualityDecisionMetrics:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    held_out_positive_count: int
    held_out_negative_count: int
    valid_control_count: int
    valid_control_false_positive_count: int
    recall: float
    precision: float
    valid_control_false_positive_rate: float

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class BlindAnnotationRecord:
    """One explicit blind-review decision keyed only by opaque pilot ID."""

    opaque_id: str
    artifact_heavy: bool
    note: str

    def __post_init__(self) -> None:
        if not isinstance(self.opaque_id, str) or not self.opaque_id:
            raise ValueError("Blind annotation opaque_id must be non-empty.")
        if not isinstance(self.artifact_heavy, bool):
            raise TypeError("Blind annotation artifact_heavy must be boolean.")
        if not isinstance(self.note, str):
            raise TypeError("Blind annotation note must be a string.")

    @classmethod
    def from_mapping(cls, value: object) -> BlindAnnotationRecord:
        if not isinstance(value, Mapping) or set(value) != {
            "opaque_id",
            "artifact_heavy",
            "note",
        }:
            raise ValueError("Blind annotation record schema is invalid.")
        return cls(
            opaque_id=value["opaque_id"],
            artifact_heavy=value["artifact_heavy"],
            note=value["note"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "opaque_id": self.opaque_id,
            "artifact_heavy": self.artifact_heavy,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class BlindAnnotation:
    """One complete reviewer file bound to the frozen pilot manifest hash."""

    schema: str
    pilot_manifest_sha256: str
    reviewer_id: str
    records: tuple[BlindAnnotationRecord, ...]

    def __post_init__(self) -> None:
        if self.schema not in {BLIND_ANNOTATION_SCHEMA, BLIND_ADJUDICATION_SCHEMA}:
            raise ValueError("Blind annotation schema identity is invalid.")
        if not _is_sha256(self.pilot_manifest_sha256):
            raise ValueError("Blind annotation pilot-manifest hash is invalid.")
        if (
            not isinstance(self.reviewer_id, str)
            or not self.reviewer_id
            or self.reviewer_id != self.reviewer_id.strip()
        ):
            raise ValueError(
                "Blind annotation reviewer_id must be non-empty and trimmed."
            )
        if not self.records:
            raise ValueError("Blind annotation records must be non-empty.")
        opaque_ids = tuple(record.opaque_id for record in self.records)
        if len(opaque_ids) != len(set(opaque_ids)):
            raise ValueError("Blind annotation contains duplicate opaque IDs.")
        if opaque_ids != tuple(sorted(opaque_ids)):
            raise ValueError("Blind annotation records must use opaque-ID order.")

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        expected_schema: str,
        expected_manifest_sha256: str,
        expected_opaque_ids: Sequence[str],
    ) -> BlindAnnotation:
        """Parse a reviewer file and require exact identity, order, and coverage."""
        if not isinstance(value, Mapping) or set(value) != {
            "schema",
            "pilot_manifest_sha256",
            "reviewer_id",
            "records",
        }:
            raise ValueError("Blind annotation file schema is invalid.")
        records_raw = value["records"]
        if not isinstance(records_raw, Sequence) or isinstance(
            records_raw, (str, bytes)
        ):
            raise TypeError("Blind annotation records must be a sequence.")
        annotation = cls(
            schema=value["schema"],
            pilot_manifest_sha256=value["pilot_manifest_sha256"],
            reviewer_id=value["reviewer_id"],
            records=tuple(
                BlindAnnotationRecord.from_mapping(record) for record in records_raw
            ),
        )
        if annotation.schema != expected_schema:
            raise ValueError("Blind annotation has the wrong schema identity.")
        if annotation.pilot_manifest_sha256 != expected_manifest_sha256:
            raise ValueError("Blind annotation pilot-manifest hash changed.")
        expected = tuple(expected_opaque_ids)
        observed = tuple(record.opaque_id for record in annotation.records)
        if observed != expected:
            raise ValueError("Blind annotation coverage or opaque-ID order changed.")
        return annotation


@dataclass(frozen=True, slots=True)
class AnnotationConsensus:
    """Deterministic direct-agreement plus exact-disagreement adjudication."""

    reviewer_ids: tuple[str, str]
    adjudicator_id: str
    disagreement_ids: tuple[str, ...]
    records: tuple[BlindAnnotationRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": CONSENSUS_SCHEMA,
            "reviewer_ids": list(self.reviewer_ids),
            "adjudicator_id": self.adjudicator_id,
            "disagreement_ids": list(self.disagreement_ids),
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True, slots=True)
class QualityRuleCalibration:
    """Calibration-only midpoint selection evidence for one scalar rule."""

    rule: QualityDecisionRule | None
    metrics: QualityDecisionMetrics | None
    threshold_lower_bound: float | None
    threshold_upper_bound: float | None
    eligible_feature_names: tuple[str, ...]
    evaluated_candidate_count: int

    def __post_init__(self) -> None:
        selected = self.rule is not None
        if selected != (self.metrics is not None) or selected != (
            self.threshold_lower_bound is not None
            and self.threshold_upper_bound is not None
        ):
            raise ValueError("Calibration rule, metrics, and bounds must be all-or-none.")
        if self.evaluated_candidate_count < 0:
            raise ValueError("evaluated_candidate_count must be non-negative.")

    def to_dict(self) -> dict[str, object]:
        return {
            "selection": "best_frozen_gate_passing_adjacent_midpoint_v1",
            "status": (
                "eligible_rule_selected"
                if self.rule is not None
                else "no_calibration_threshold_family_passes_frozen_gates"
            ),
            "evaluated_candidate_count": self.evaluated_candidate_count,
            "eligible_feature_names": list(self.eligible_feature_names),
            "rule": self.rule.to_dict() if self.rule is not None else None,
            "predictive_metrics": (
                self.metrics.to_dict() if self.metrics is not None else None
            ),
            "threshold_bounds": (
                {
                    "lower": self.threshold_lower_bound,
                    "upper": self.threshold_upper_bound,
                }
                if self.threshold_lower_bound is not None
                and self.threshold_upper_bound is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class QualityDecisionReport:
    """Frozen report result; rejection keeps geometry as production authority."""

    rule: QualityDecisionRule | None
    metrics: QualityDecisionMetrics | None
    decision: QualityDecision
    failure_reasons: tuple[str, ...]
    calibration_group_ids: tuple[str, ...]
    held_out_group_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": QUALITY_DECISION_REPORT_SCHEMA,
            "feature_definition_id": QUALITY_FEATURE_DEFINITION_ID,
            "thresholds": {
                "minimum_recall": MINIMUM_HELD_OUT_RECALL,
                "minimum_precision": MINIMUM_HELD_OUT_PRECISION,
                "maximum_valid_control_false_positive_rate": (
                    MAXIMUM_VALID_CONTROL_FALSE_POSITIVE_RATE
                ),
                "minimum_positive_labels": MINIMUM_HELD_OUT_POSITIVE_LABELS,
                "minimum_negative_labels": MINIMUM_HELD_OUT_NEGATIVE_LABELS,
            },
            "rule": self.rule.to_dict() if self.rule is not None else None,
            "predictive_metrics": (
                self.metrics.to_dict() if self.metrics is not None else None
            ),
            "decision": self.decision.value,
            "failure_reasons": list(self.failure_reasons),
            "calibration_group_ids": list(self.calibration_group_ids),
            "held_out_group_ids": list(self.held_out_group_ids),
            "production_authority": (
                "quality_plus_geometry"
                if self.decision is QualityDecision.QUALITY_ONLY_PROMOTED
                else "geometry_only"
            ),
        }


def extract_public_quality_features(
    *,
    rgb: NDArray[np.floating],
    alpha: NDArray[np.floating],
    depth: NDArray[np.floating],
    support_margin_m: float,
    obstacle_clearance_m: float,
    captured_camera_distance_m: float,
) -> PublicQualityFeatures:
    """Extract deterministic features from public arrays without assigning labels."""
    rgb_array = np.asarray(rgb, dtype=np.float64)
    alpha_array = np.asarray(alpha, dtype=np.float64)
    depth_array = np.asarray(depth, dtype=np.float64)
    if (
        rgb_array.ndim != 3
        or rgb_array.shape[2] != 3
        or alpha_array.shape != (*rgb_array.shape[:2], 1)
        or depth_array.shape != alpha_array.shape
    ):
        raise ValueError("Public quality arrays have incompatible HWC shapes.")
    if not all(
        np.isfinite(value).all() for value in (rgb_array, alpha_array, depth_array)
    ):
        raise ValueError("Public quality arrays must be finite.")
    if (
        np.any(rgb_array < 0.0)
        or np.any(rgb_array > 1.0)
        or np.any(alpha_array < 0.0)
        or np.any(alpha_array > 1.0)
        or np.any(depth_array < 0.0)
    ):
        raise ValueError("Public RGB/alpha/depth arrays are outside their contracts.")
    luminance = (
        0.2126 * rgb_array[..., 0]
        + 0.7152 * rgb_array[..., 1]
        + 0.0722 * rgb_array[..., 2]
    )
    rgb_gradient = _gradient_magnitudes(luminance)
    alpha_plane = alpha_array[..., 0]
    depth_plane = depth_array[..., 0]
    valid_depth = (alpha_plane >= 0.01) & (depth_plane > 0.0)
    depth_gradient = _valid_depth_gradients(depth_plane, valid_depth)
    return PublicQualityFeatures(
        rgb_gradient_p95=_p95(rgb_gradient),
        alpha_mean=float(np.mean(alpha_plane)),
        alpha_coverage_fraction=float(np.mean(alpha_plane >= 0.01)),
        alpha_transition_fraction=float(
            np.mean((alpha_plane > 0.01) & (alpha_plane < 0.99))
        ),
        depth_valid_fraction=float(np.mean(valid_depth)),
        depth_gradient_p95=_p95(depth_gradient),
        support_margin_m=float(support_margin_m),
        obstacle_clearance_m=float(obstacle_clearance_m),
        captured_camera_distance_m=float(captured_camera_distance_m),
    )


def assign_group_held_out_splits(
    group_ids: Sequence[str],
    *,
    seed: int,
    calibration_fraction: float,
) -> Mapping[str, BenchmarkSplit]:
    """Assign complete groups deterministically before feature outcomes are visible."""
    identifiers = tuple(group_ids)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("group_ids must be non-empty and unique.")
    if len(identifiers) < 2 or not 0.0 < calibration_fraction < 1.0:
        raise ValueError(
            "Group-held-out assignment requires >=2 groups and a proper fraction."
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    ordered = sorted(
        identifiers,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    calibration_count = min(
        len(ordered) - 1,
        max(1, round(len(ordered) * calibration_fraction)),
    )
    calibration = set(ordered[:calibration_count])
    return {
        group_id: (
            BenchmarkSplit.CALIBRATION
            if group_id in calibration
            else BenchmarkSplit.HELD_OUT
        )
        for group_id in identifiers
    }


def opaque_review_ids(
    source_ids_by_stratum: Mapping[str, Sequence[str]],
    *,
    seed: int,
) -> tuple[tuple[str, str, str], ...]:
    """Create stable opaque IDs while preserving exact stratified membership."""
    if not source_ids_by_stratum:
        raise ValueError("At least one pilot stratum is required.")
    result: list[tuple[str, str, str]] = []
    observed_sources: set[str] = set()
    observed_opaque: set[str] = set()
    for stratum in sorted(source_ids_by_stratum):
        source_ids = tuple(source_ids_by_stratum[stratum])
        if not source_ids:
            raise ValueError(f"Pilot stratum {stratum!r} is empty.")
        for source_id in sorted(source_ids):
            if source_id in observed_sources:
                raise ValueError("A pilot source ID occurs in multiple strata.")
            observed_sources.add(source_id)
            opaque_id = (
                "review-"
                + hashlib.sha256(f"{seed}:{source_id}".encode()).hexdigest()[:16]
            )
            if opaque_id in observed_opaque:
                raise ValueError("Opaque review ID collision.")
            observed_opaque.add(opaque_id)
            result.append((opaque_id, stratum, source_id))
    return tuple(sorted(result))


def derive_annotation_consensus(
    *,
    reviewer_a: BlindAnnotation,
    reviewer_b: BlindAnnotation,
    adjudication: BlindAnnotation,
) -> AnnotationConsensus:
    """Join full reviewers directly and use adjudication for disagreements only."""
    if reviewer_a.schema != BLIND_ANNOTATION_SCHEMA or reviewer_b.schema != (
        BLIND_ANNOTATION_SCHEMA
    ):
        raise ValueError("Consensus requires two complete blind annotations.")
    if adjudication.schema != BLIND_ADJUDICATION_SCHEMA:
        raise ValueError("Consensus requires the exact adjudication schema.")
    reviewer_ids = (reviewer_a.reviewer_id, reviewer_b.reviewer_id)
    if len(set((*reviewer_ids, adjudication.reviewer_id))) != 3:
        raise ValueError("Reviewers and adjudicator must have distinct identities.")
    if reviewer_a.pilot_manifest_sha256 != reviewer_b.pilot_manifest_sha256 or (
        reviewer_a.pilot_manifest_sha256 != adjudication.pilot_manifest_sha256
    ):
        raise ValueError("Annotation files disagree on the frozen pilot-manifest hash.")
    opaque_ids = tuple(record.opaque_id for record in reviewer_a.records)
    if tuple(record.opaque_id for record in reviewer_b.records) != opaque_ids:
        raise ValueError("Blind reviewer coverage or order disagrees.")
    decisions_a = {
        record.opaque_id: record.artifact_heavy for record in reviewer_a.records
    }
    decisions_b = {
        record.opaque_id: record.artifact_heavy for record in reviewer_b.records
    }
    disagreement_ids = tuple(
        opaque_id
        for opaque_id in opaque_ids
        if decisions_a[opaque_id] != decisions_b[opaque_id]
    )
    if tuple(record.opaque_id for record in adjudication.records) != disagreement_ids:
        raise ValueError(
            "Adjudication must cover exactly the ordered blind-review disagreement set."
        )
    adjudicated = {
        record.opaque_id: record.artifact_heavy for record in adjudication.records
    }
    consensus = tuple(
        BlindAnnotationRecord(
            opaque_id=opaque_id,
            artifact_heavy=(
                decisions_a[opaque_id]
                if decisions_a[opaque_id] == decisions_b[opaque_id]
                else adjudicated[opaque_id]
            ),
            note="",
        )
        for opaque_id in opaque_ids
    )
    return AnnotationConsensus(
        reviewer_ids=reviewer_ids,
        adjudicator_id=adjudication.reviewer_id,
        disagreement_ids=disagreement_ids,
        records=consensus,
    )


def calibrate_quality_only_rule(
    observations: Sequence[QualityObservation],
) -> QualityRuleCalibration:
    """Select the strongest frozen-gate-passing midpoint on calibration groups."""
    values = tuple(observations)
    _validate_observation_inventory(values)
    calibration = tuple(
        item for item in values if item.split is BenchmarkSplit.CALIBRATION
    )
    if not calibration:
        raise ValueError("Quality calibration requires calibration observations.")
    candidates: list[
        tuple[
            QualityDecisionMetrics,
            QualityDecisionRule,
            float,
            float,
        ]
    ] = []
    evaluated_candidate_count = 0
    for feature_name in sorted(PublicQualityFeatures.__dataclass_fields__):
        feature_values = sorted(
            {float(getattr(item.features, feature_name)) for item in calibration}
        )
        for lower, upper in zip(feature_values, feature_values[1:], strict=False):
            threshold = lower + (upper - lower) / 2.0
            for operator in RuleOperator:
                evaluated_candidate_count += 1
                rule = QualityDecisionRule(
                    rule_id=(
                        f"calibration-midpoint-v1:{feature_name}:{operator.value}"
                    ),
                    feature_name=feature_name,
                    operator=operator,
                    threshold=threshold,
                )
                metrics = _decision_metrics(calibration, rule=rule)
                if (
                    metrics.recall >= MINIMUM_HELD_OUT_RECALL
                    and metrics.precision >= MINIMUM_HELD_OUT_PRECISION
                    and metrics.valid_control_false_positive_rate
                    <= MAXIMUM_VALID_CONTROL_FALSE_POSITIVE_RATE
                ):
                    candidates.append((metrics, rule, lower, upper))
    if not candidates:
        return QualityRuleCalibration(
            rule=None,
            metrics=None,
            threshold_lower_bound=None,
            threshold_upper_bound=None,
            eligible_feature_names=(),
            evaluated_candidate_count=evaluated_candidate_count,
        )
    eligible_feature_names = tuple(
        sorted({rule.feature_name for _metrics, rule, _lower, _upper in candidates})
    )
    metrics, rule, lower, upper = min(
        candidates,
        key=lambda value: (
            -value[0].precision,
            -value[0].recall,
            value[0].valid_control_false_positive_rate,
            value[1].feature_name,
            value[1].operator.value,
            value[1].threshold,
        ),
    )
    return QualityRuleCalibration(
        rule=rule,
        metrics=metrics,
        threshold_lower_bound=lower,
        threshold_upper_bound=upper,
        eligible_feature_names=eligible_feature_names,
        evaluated_candidate_count=evaluated_candidate_count,
    )


def evaluate_quality_only_rule(
    observations: Sequence[QualityObservation],
    *,
    rule: QualityDecisionRule | None,
) -> QualityDecisionReport:
    """Apply fixed Issue thresholds only to held-out trajectory groups."""
    values = tuple(observations)
    split_by_group = _validate_observation_inventory(values)
    held_out = tuple(item for item in values if item.split is BenchmarkSplit.HELD_OUT)
    if not held_out or not any(
        item.split is BenchmarkSplit.CALIBRATION for item in values
    ):
        raise ValueError(
            "Quality evaluation requires calibration and held-out observations."
        )
    failures: list[str] = []
    held_out_positive_count = sum(item.artifact_heavy for item in held_out)
    held_out_negative_count = len(held_out) - held_out_positive_count
    if rule is None:
        failures.append("no_calibration_threshold_family_passes_frozen_gates")
    if held_out_positive_count < MINIMUM_HELD_OUT_POSITIVE_LABELS:
        failures.append("insufficient_held_out_positive_labels")
    if held_out_negative_count < MINIMUM_HELD_OUT_NEGATIVE_LABELS:
        failures.append("insufficient_held_out_negative_labels")
    metrics = _decision_metrics(held_out, rule=rule) if rule is not None else None
    if metrics is not None:
        if metrics.recall < MINIMUM_HELD_OUT_RECALL:
            failures.append("held_out_recall_below_threshold")
        if metrics.precision < MINIMUM_HELD_OUT_PRECISION:
            failures.append("held_out_precision_below_threshold")
        if (
            metrics.valid_control_false_positive_rate
            > MAXIMUM_VALID_CONTROL_FALSE_POSITIVE_RATE
        ):
            failures.append("valid_control_false_positive_rate_above_threshold")
    calibration_groups = tuple(
        sorted(
            group_id
            for group_id, split in split_by_group.items()
            if split is BenchmarkSplit.CALIBRATION
        )
    )
    held_out_groups = tuple(
        sorted(
            group_id
            for group_id, split in split_by_group.items()
            if split is BenchmarkSplit.HELD_OUT
        )
    )
    return QualityDecisionReport(
        rule=rule,
        metrics=metrics,
        decision=(
            QualityDecision.QUALITY_ONLY_REJECTED
            if failures
            else QualityDecision.QUALITY_ONLY_PROMOTED
        ),
        failure_reasons=tuple(failures),
        calibration_group_ids=calibration_groups,
        held_out_group_ids=held_out_groups,
    )


def _validate_observation_inventory(
    values: Sequence[QualityObservation],
) -> dict[str, BenchmarkSplit]:
    if not values or len({item.opaque_id for item in values}) != len(values):
        raise ValueError(
            "Quality observations must be non-empty with unique opaque IDs."
        )
    split_by_group: dict[str, BenchmarkSplit] = {}
    for item in values:
        previous = split_by_group.setdefault(item.trajectory_group_id, item.split)
        if previous is not item.split:
            raise ValueError(
                "Calibration/held-out split leaks within a trajectory group."
            )
    return split_by_group


def _decision_metrics(
    observations: Sequence[QualityObservation],
    *,
    rule: QualityDecisionRule,
) -> QualityDecisionMetrics:
    counts: Counter[str] = Counter()
    valid_controls = 0
    valid_control_false_positives = 0
    for item in observations:
        prediction = rule.predicts_artifact(item.features)
        counts[
            (
                "tp"
                if prediction and item.artifact_heavy
                else "fp"
                if prediction
                else "fn"
                if item.artifact_heavy
                else "tn"
            )
        ] += 1
        if item.valid_control:
            valid_controls += 1
            valid_control_false_positives += int(prediction)
    positive_count = counts["tp"] + counts["fn"]
    negative_count = counts["tn"] + counts["fp"]
    recall = counts["tp"] / positive_count if positive_count else 0.0
    precision_denominator = counts["tp"] + counts["fp"]
    precision = counts["tp"] / precision_denominator if precision_denominator else 0.0
    control_fpr = (
        valid_control_false_positives / valid_controls if valid_controls else 1.0
    )
    return QualityDecisionMetrics(
        true_positive=counts["tp"],
        false_positive=counts["fp"],
        true_negative=counts["tn"],
        false_negative=counts["fn"],
        held_out_positive_count=positive_count,
        held_out_negative_count=negative_count,
        valid_control_count=valid_controls,
        valid_control_false_positive_count=valid_control_false_positives,
        recall=recall,
        precision=precision,
        valid_control_false_positive_rate=control_fpr,
    )


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _gradient_magnitudes(values: NDArray[np.float64]) -> NDArray[np.float64]:
    horizontal = np.abs(np.diff(values, axis=1)).ravel()
    vertical = np.abs(np.diff(values, axis=0)).ravel()
    return np.concatenate((horizontal, vertical))


def _valid_depth_gradients(
    depth: NDArray[np.float64], valid: NDArray[np.bool_]
) -> NDArray[np.float64]:
    horizontal_valid = valid[:, 1:] & valid[:, :-1]
    vertical_valid = valid[1:, :] & valid[:-1, :]
    horizontal = np.abs(depth[:, 1:] - depth[:, :-1])[horizontal_valid]
    vertical = np.abs(depth[1:, :] - depth[:-1, :])[vertical_valid]
    return np.concatenate((horizontal, vertical))


def _p95(values: NDArray[np.float64]) -> float:
    return float(np.quantile(values, 0.95)) if values.size else 0.0


__all__ = [
    "AnnotationConsensus",
    "BenchmarkSplit",
    "BLIND_ADJUDICATION_SCHEMA",
    "BLIND_ANNOTATION_SCHEMA",
    "BlindAnnotation",
    "BlindAnnotationRecord",
    "CONSENSUS_SCHEMA",
    "MAXIMUM_VALID_CONTROL_FALSE_POSITIVE_RATE",
    "MINIMUM_HELD_OUT_NEGATIVE_LABELS",
    "MINIMUM_HELD_OUT_POSITIVE_LABELS",
    "MINIMUM_HELD_OUT_PRECISION",
    "MINIMUM_HELD_OUT_RECALL",
    "PublicQualityFeatures",
    "QUALITY_DECISION_REPORT_SCHEMA",
    "QUALITY_FEATURE_DEFINITION_ID",
    "QualityDecision",
    "QualityDecisionMetrics",
    "QualityDecisionReport",
    "QualityDecisionRule",
    "QualityObservation",
    "QualityRuleCalibration",
    "RuleOperator",
    "assign_group_held_out_splits",
    "calibrate_quality_only_rule",
    "derive_annotation_consensus",
    "evaluate_quality_only_rule",
    "extract_public_quality_features",
    "opaque_review_ids",
]
