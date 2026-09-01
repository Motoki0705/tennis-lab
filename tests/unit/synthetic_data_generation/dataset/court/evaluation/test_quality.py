from __future__ import annotations

import numpy as np
import pytest

from src.synthetic_data_generation.dataset.court.evaluation.quality import (
    BLIND_ADJUDICATION_SCHEMA,
    BLIND_ANNOTATION_SCHEMA,
    BenchmarkSplit,
    BlindAnnotation,
    QualityDecision,
    QualityDecisionRule,
    QualityObservation,
    RuleOperator,
    assign_group_held_out_splits,
    calibrate_quality_only_rule,
    derive_annotation_consensus,
    evaluate_quality_only_rule,
    extract_public_quality_features,
    opaque_review_ids,
)


def _features(alpha_mean: float, *, captured_camera_distance_m: float = 0.25):
    return extract_public_quality_features(
        rgb=np.full((4, 5, 3), 0.25, dtype=np.float32),
        alpha=np.full((4, 5, 1), alpha_mean, dtype=np.float32),
        depth=np.full((4, 5, 1), 3.0, dtype=np.float32),
        support_margin_m=0.5,
        obstacle_clearance_m=1.0,
        captured_camera_distance_m=captured_camera_distance_m,
    )


def _observation(
    index: int,
    *,
    split: BenchmarkSplit,
    artifact_heavy: bool,
    valid_control: bool,
    alpha_mean: float,
) -> QualityObservation:
    return QualityObservation(
        opaque_id=f"review-{index:016x}",
        trajectory_group_id=(
            "group-calibration"
            if split is BenchmarkSplit.CALIBRATION
            else "group-held-out"
        ),
        stratum="captured_control" if valid_control else "safe_v4_candidate",
        split=split,
        artifact_heavy=artifact_heavy,
        valid_control=valid_control,
        features=_features(alpha_mean),
    )


def test_public_feature_extraction_and_group_assignment_are_deterministic() -> None:
    features = _features(0.5)
    assert features.rgb_gradient_p95 == 0.0
    assert features.alpha_mean == pytest.approx(0.5)
    assert features.alpha_coverage_fraction == 1.0
    assert features.depth_valid_fraction == 1.0

    first = assign_group_held_out_splits(
        ("group-a", "group-b", "group-c", "group-d"),
        seed=823,
        calibration_fraction=0.5,
    )
    second = assign_group_held_out_splits(
        ("group-a", "group-b", "group-c", "group-d"),
        seed=823,
        calibration_fraction=0.5,
    )
    assert first == second
    assert set(first.values()) == set(BenchmarkSplit)

    opaque = opaque_review_ids(
        {"captured_control": ("source-b",), "safe_v4_candidate": ("source-a",)},
        seed=823,
    )
    assert opaque == opaque_review_ids(
        {"safe_v4_candidate": ("source-a",), "captured_control": ("source-b",)},
        seed=823,
    )
    assert all(item[0].startswith("review-") and len(item[0]) == 23 for item in opaque)

    serialized = features.to_dict()
    assert type(features).from_mapping(serialized) == features
    serialized.pop("captured_camera_distance_m")
    with pytest.raises(ValueError, match="feature keys"):
        type(features).from_mapping(serialized)


def test_quality_only_promotion_requires_every_frozen_held_out_gate() -> None:
    observations = [
        _observation(
            0,
            split=BenchmarkSplit.CALIBRATION,
            artifact_heavy=False,
            valid_control=True,
            alpha_mean=0.1,
        )
    ]
    observations.extend(
        _observation(
            index,
            split=BenchmarkSplit.HELD_OUT,
            artifact_heavy=True,
            valid_control=False,
            alpha_mean=0.9,
        )
        for index in range(1, 13)
    )
    observations.extend(
        _observation(
            index,
            split=BenchmarkSplit.HELD_OUT,
            artifact_heavy=False,
            valid_control=True,
            alpha_mean=0.1,
        )
        for index in range(13, 25)
    )
    rule = QualityDecisionRule(
        rule_id="alpha-rule-v1",
        feature_name="alpha_mean",
        operator=RuleOperator.GREATER_THAN_OR_EQUAL,
        threshold=0.5,
    )

    report = evaluate_quality_only_rule(observations, rule=rule)

    assert report.decision is QualityDecision.QUALITY_ONLY_PROMOTED
    assert report.failure_reasons == ()
    assert report.metrics is not None
    assert report.metrics.recall == 1.0
    assert report.metrics.precision == 1.0
    assert report.metrics.valid_control_false_positive_rate == 0.0
    assert report.to_dict()["production_authority"] == "quality_plus_geometry"

    insufficient = evaluate_quality_only_rule(
        (observations[0], *observations[1:7], *observations[13:19]),
        rule=rule,
    )
    assert insufficient.decision is QualityDecision.QUALITY_ONLY_REJECTED
    assert set(insufficient.failure_reasons) >= {
        "insufficient_held_out_positive_labels",
        "insufficient_held_out_negative_labels",
    }


def test_quality_observations_reject_split_leakage_and_control_label_conflicts() -> (
    None
):
    with pytest.raises(ValueError, match="valid control"):
        _observation(
            0,
            split=BenchmarkSplit.CALIBRATION,
            artifact_heavy=True,
            valid_control=True,
            alpha_mean=0.9,
        )

    calibration = _observation(
        1,
        split=BenchmarkSplit.CALIBRATION,
        artifact_heavy=False,
        valid_control=True,
        alpha_mean=0.1,
    )
    leaked = QualityObservation(
        opaque_id="review-0000000000000002",
        trajectory_group_id=calibration.trajectory_group_id,
        stratum="safe_v4_candidate",
        split=BenchmarkSplit.HELD_OUT,
        artifact_heavy=True,
        valid_control=False,
        features=_features(0.9),
    )
    with pytest.raises(ValueError, match="leaks"):
        evaluate_quality_only_rule(
            (calibration, leaked),
            rule=QualityDecisionRule(
                rule_id="alpha-rule-v1",
                feature_name="alpha_mean",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.5,
            ),
        )


def _annotation(
    *,
    schema: str,
    reviewer_id: str,
    labels: tuple[bool, ...],
) -> BlindAnnotation:
    opaque_ids = tuple(f"review-{index:016x}" for index in range(len(labels)))
    return BlindAnnotation.from_mapping(
        {
            "schema": schema,
            "pilot_manifest_sha256": "a" * 64,
            "reviewer_id": reviewer_id,
            "records": [
                {
                    "opaque_id": opaque_id,
                    "artifact_heavy": label,
                    "note": "",
                }
                for opaque_id, label in zip(opaque_ids, labels, strict=True)
            ],
        },
        expected_schema=schema,
        expected_manifest_sha256="a" * 64,
        expected_opaque_ids=opaque_ids,
    )


def test_annotations_require_exact_hash_schema_coverage_order_and_unique_ids() -> None:
    opaque_ids = ("review-0000000000000000", "review-0000000000000001")
    value = {
        "schema": BLIND_ANNOTATION_SCHEMA,
        "pilot_manifest_sha256": "a" * 64,
        "reviewer_id": "reviewer-a",
        "records": [
            {"opaque_id": opaque_id, "artifact_heavy": False, "note": ""}
            for opaque_id in opaque_ids
        ],
    }
    assert (
        len(
            BlindAnnotation.from_mapping(
                value,
                expected_schema=BLIND_ANNOTATION_SCHEMA,
                expected_manifest_sha256="a" * 64,
                expected_opaque_ids=opaque_ids,
            ).records
        )
        == 2
    )

    wrong_hash = dict(value, pilot_manifest_sha256="b" * 64)
    with pytest.raises(ValueError, match="hash changed"):
        BlindAnnotation.from_mapping(
            wrong_hash,
            expected_schema=BLIND_ANNOTATION_SCHEMA,
            expected_manifest_sha256="a" * 64,
            expected_opaque_ids=opaque_ids,
        )
    wrong_schema = dict(value, schema=BLIND_ADJUDICATION_SCHEMA)
    with pytest.raises(ValueError, match="wrong schema"):
        BlindAnnotation.from_mapping(
            wrong_schema,
            expected_schema=BLIND_ANNOTATION_SCHEMA,
            expected_manifest_sha256="a" * 64,
            expected_opaque_ids=opaque_ids,
        )
    for records, match in (
        (list(reversed(value["records"])), "opaque-ID order"),
        ((value["records"][0], value["records"][0]), "duplicate"),
        ((value["records"][0],), "coverage"),
    ):
        with pytest.raises(ValueError, match=match):
            BlindAnnotation.from_mapping(
                dict(value, records=records),
                expected_schema=BLIND_ANNOTATION_SCHEMA,
                expected_manifest_sha256="a" * 64,
                expected_opaque_ids=opaque_ids,
            )


def test_consensus_uses_adjudication_for_exact_disagreement_set_only() -> None:
    reviewer_a = _annotation(
        schema=BLIND_ANNOTATION_SCHEMA,
        reviewer_id="reviewer-a",
        labels=(False, True, False),
    )
    reviewer_b = _annotation(
        schema=BLIND_ANNOTATION_SCHEMA,
        reviewer_id="reviewer-b",
        labels=(False, False, False),
    )
    adjudication = BlindAnnotation.from_mapping(
        {
            "schema": BLIND_ADJUDICATION_SCHEMA,
            "pilot_manifest_sha256": "a" * 64,
            "reviewer_id": "reviewer-c",
            "records": [
                {
                    "opaque_id": "review-0000000000000001",
                    "artifact_heavy": True,
                    "note": "material tearing",
                }
            ],
        },
        expected_schema=BLIND_ADJUDICATION_SCHEMA,
        expected_manifest_sha256="a" * 64,
        expected_opaque_ids=("review-0000000000000001",),
    )
    consensus = derive_annotation_consensus(
        reviewer_a=reviewer_a,
        reviewer_b=reviewer_b,
        adjudication=adjudication,
    )
    assert consensus.disagreement_ids == ("review-0000000000000001",)
    assert [record.artifact_heavy for record in consensus.records] == [
        False,
        True,
        False,
    ]

    extra = _annotation(
        schema=BLIND_ADJUDICATION_SCHEMA,
        reviewer_id="reviewer-c",
        labels=(False, True, False),
    )
    with pytest.raises(ValueError, match="exactly"):
        derive_annotation_consensus(
            reviewer_a=reviewer_a,
            reviewer_b=reviewer_b,
            adjudication=extra,
        )


def test_calibration_selects_an_exact_adjacent_midpoint_without_held_out_tuning() -> (
    None
):
    observations: list[QualityObservation] = []
    for index, distance in enumerate((*range(11), 21.0)):
        observations.append(
            QualityObservation(
                opaque_id=f"review-{index:016x}",
                trajectory_group_id=f"negative-{index:02d}",
                stratum="captured_control",
                split=BenchmarkSplit.CALIBRATION,
                artifact_heavy=False,
                valid_control=True,
                features=_features(
                    0.5,
                    captured_camera_distance_m=float(distance),
                ),
            )
        )
    for offset, distance in enumerate((20.0, *range(30, 40)), start=len(observations)):
        observations.append(
            QualityObservation(
                opaque_id=f"review-{offset:016x}",
                trajectory_group_id=f"positive-{offset:02d}",
                stratum="legacy_orbit",
                split=BenchmarkSplit.CALIBRATION,
                artifact_heavy=True,
                valid_control=False,
                features=_features(
                    0.5,
                    captured_camera_distance_m=float(distance),
                ),
            )
        )

    calibration = calibrate_quality_only_rule(observations)

    assert calibration.eligible_feature_names == ("captured_camera_distance_m",)
    assert calibration.threshold_lower_bound == 21.0
    assert calibration.threshold_upper_bound == 30.0
    assert calibration.rule is not None
    assert calibration.metrics is not None
    assert calibration.rule.threshold == 25.5
    assert calibration.metrics.true_positive == 10
    assert calibration.metrics.false_positive == 0

    held_out_decoys = tuple(
        QualityObservation(
            opaque_id=f"review-{100 + index:016x}",
            trajectory_group_id=f"held-out-decoy-{index}",
            stratum="legacy_orbit",
            split=BenchmarkSplit.HELD_OUT,
            artifact_heavy=bool(index % 2),
            valid_control=False,
            features=_features(
                0.99 if index == 0 else 0.01,
                captured_camera_distance_m=1_000.0 - index,
            ),
        )
        for index in range(2)
    )
    assert calibrate_quality_only_rule(
        (*observations, *held_out_decoys)
    ) == calibration


def test_zero_calibration_passing_candidates_is_an_explicit_no_rule_rejection() -> (
    None
):
    observations = [
        _observation(
            0,
            split=BenchmarkSplit.CALIBRATION,
            artifact_heavy=False,
            valid_control=True,
            alpha_mean=0.5,
        ),
        _observation(
            1,
            split=BenchmarkSplit.CALIBRATION,
            artifact_heavy=True,
            valid_control=False,
            alpha_mean=0.5,
        ),
    ]
    observations.extend(
        _observation(
            index,
            split=BenchmarkSplit.HELD_OUT,
            artifact_heavy=index < 14,
            valid_control=index >= 14,
            alpha_mean=0.5,
        )
        for index in range(2, 26)
    )

    calibration = calibrate_quality_only_rule(observations)
    report = evaluate_quality_only_rule(observations, rule=calibration.rule)

    assert calibration.rule is None
    assert calibration.metrics is None
    assert calibration.evaluated_candidate_count == 0
    assert report.decision is QualityDecision.QUALITY_ONLY_REJECTED
    assert report.rule is None
    assert report.metrics is None
    assert report.failure_reasons == (
        "no_calibration_threshold_family_passes_frozen_gates",
    )
    assert report.to_dict()["rule"] is None
    assert report.to_dict()["predictive_metrics"] is None
