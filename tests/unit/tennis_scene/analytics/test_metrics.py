from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from src.tennis_scene.analytics.metrics import (
    METRIC_DEFINITIONS,
    MatchPointId,
    MetricRecord,
    SourceId,
    TerminalCause,
    counted_error_assessment,
    counted_terminal_cause,
    unique_point_bins,
)

EXPECTED_IDS = {
    "point.played_count",
    "point.win_rate",
    "serve.point_win_rate",
    "serve.first_in_rate",
    "serve.first_point_win_rate",
    "serve.second_point_win_rate",
    "serve.double_fault_count",
    "serve.double_fault_rate",
    "serve.ace_count",
    "serve.ace_rate",
    "return.point_win_rate",
    "return.in_play_rate",
    "shot.count",
    "shot.type_share",
    "shot.terminal_error_rate",
    "shot.unreturned_point_count",
    "rally.contact_count",
    "rally.completed_shot_count",
    "conditional.point_win_rate",
    "game.service_win_rate",
    "game.return_win_rate",
    "score.break_point_conversion_rate",
    "score.break_point_saved_rate",
    "score.game_point_conversion_rate",
    "score.match_point_conversion_rate",
    "point.terminal_cause_count",
    "point.forced_error_count",
    "point.unforced_error_count",
    "conditional.shot_error_rate",
    "conditional.landing_out_rate",
    "conditional.landing_point_win_rate",
}


def test_metric_dictionary_covers_the_required_ids_with_versioned_definitions() -> None:
    assert set(METRIC_DEFINITIONS) == EXPECTED_IDS
    definitions = json.loads(
        json.dumps(
            [definition.to_record() for definition in METRIC_DEFINITIONS.values()]
        )
    )
    for definition in definitions:
        assert definition["numerator"] and definition["denominator"]
        assert definition["definition_version"] == "1"
        assert definition["unit"] in {"point", "game", "serve_attempt", "shot_attempt"}
        assert definition["subject_types"]
        assert definition["count_relation"] in {
            "exact_count",
            "subset_count",
            "ratio",
            "multi_per_unit",
        }
        assert set(definition["evidence_keys"]) == {"numerator", "denominator"}
        assert definition["rules"]["availability"]["empty_denominator"] == {
            "value": None,
            "status": "unavailable",
        }


def test_serves_and_the_two_rally_lengths_preserve_the_fault_and_let_rules() -> None:
    serve = METRIC_DEFINITIONS["serve.first_in_rate"].to_record()["rules"][
        "serve_attempts"
    ]
    assert serve["let_advances_serve_number"] is False
    assert serve["let_in_first_in_denominator"] is False
    assert serve["first_fault_requires_second_serve"] is True
    assert serve["second_fault_terminal_cause"] == "double_fault"
    assert serve["double_fault_in_second_serve_win_denominator"] is True
    assert serve["double_fault_in_return_win_denominator"] is True
    assert serve["double_fault_in_return_in_play_denominator"] is False
    assert METRIC_DEFINITIONS["return.in_play_rate"].unit == "serve_attempt"

    length = METRIC_DEFINITIONS["rally.contact_count"].to_record()["rules"][
        "rally_contact"
    ]
    completed = METRIC_DEFINITIONS["rally.completed_shot_count"].to_record()["rules"][
        "rally_completed"
    ]
    assert length["includes_terminal_error_contact"] is True
    assert completed["includes_terminal_error_contact"] is False
    assert (
        length["confirmed_double_fault_count"]
        == completed["confirmed_double_fault_count"]
        == 0
    )
    assert length["starts_at"] == completed["starts_at"] == "last_legal_serve"


def test_conditional_point_keys_deduplicate_only_within_subject_and_bin() -> None:
    point = MatchPointId(10, "point-5")
    same_point_from_other_clip = MatchPointId(10, "point-5")
    next_point = MatchPointId(10, "point-6")
    same_key_in_other_match = MatchPointId(20, "point-5")
    assert unique_point_bins(
        [
            (point, "team-A", "low"),
            (same_point_from_other_clip, "team-A", "low"),
            (point, "team-A", "high"),
            (point, "player-2", "low"),
            (next_point, "team-A", "low"),
            (same_key_in_other_match, "team-A", "low"),
        ]
    ) == {
        (point, "team-A", "low"),
        (point, "team-A", "high"),
        (point, "player-2", "low"),
        (next_point, "team-A", "low"),
        (same_key_in_other_match, "team-A", "low"),
    }
    clip_point_row_id: Any = 15
    with pytest.raises(ValueError, match="Pass MatchPointId"):
        unique_point_bins([(clip_point_row_id, "team-A", "low")])
    with pytest.raises(ValueError, match="canonical point requires"):
        MatchPointId(10, " ")
    unresolved_key: Any = None
    with pytest.raises(ValueError, match="canonical point requires"):
        MatchPointId(10, unresolved_key)
    points = METRIC_DEFINITIONS["conditional.point_win_rate"].to_record()
    assert points["unit"] == "point"
    assert points["rules"]["conditional_points"]["additive_across_bins"] is False
    assert points["rules"]["conditional_points"]["idempotency_key"] == [
        "match_context_id",
        "match_point_key",
        "subject_id",
        "bin_id",
    ]
    assert METRIC_DEFINITIONS["conditional.shot_error_rate"].unit == "shot_attempt"


def test_landing_bins_are_used_only_for_observed_landing_metrics() -> None:
    filters = {"bins": {"landing_depth": "out-long", "landing_course": "crosscourt"}}
    record = _partial_win_rate()
    for metric_id in (
        "conditional.landing_out_rate",
        "conditional.landing_point_win_rate",
    ):
        rule = METRIC_DEFINITIONS[metric_id].to_record()["rules"]["observed_landing"]
        assert rule["population"] == "shots_with_observed_landing"
        assert rule["landing_source_kind"] == "observed"
        source_ids: dict[str, list[SourceId]] = (
            {
                "numerator_ball_event_ids": [11],
                "denominator_ball_event_ids": [11, 12, 13, 14],
            }
            if metric_id == "conditional.landing_out_rate"
            else record.source_ids
        )
        assert (
            replace(
                record,
                metric_id=metric_id,
                filters=filters,
                bin_version="bins-v1",
                source_ids=source_ids,
            ).value
            == 0.25
        )
    with pytest.raises(ValueError, match="Unsupported metric bin"):
        replace(
            record,
            metric_id="conditional.shot_error_rate",
            filters={"shot_role": "rally", **filters},
            bin_version="bins-v1",
        )
    with pytest.raises(ValueError, match="require condition bins and bin_version"):
        replace(record, metric_id="conditional.landing_out_rate", filters=filters)


def test_terminal_cause_and_manual_error_assessment_keep_their_provenance() -> None:
    point_cause = counted_terminal_cause("double_fault", "derived_v4_score")
    counts = {
        cause: int(point_cause == cause) for cause in ("double_fault", "net_error")
    }
    assert counts == {"double_fault": 1, "net_error": 0}
    assert counted_terminal_cause("net_error", "manual_confirmed") == "net_error"
    assert counted_terminal_cause("net_error", "derived_v4_score") == "net_error"
    assert counted_terminal_cause("net_error", "unknown") is None
    assert counted_terminal_cause("unknown", "manual_confirmed") is None
    assert counted_error_assessment("undetermined", "manual_confirmed") is None
    assert counted_error_assessment("unforced", "derived_v4_score") is None
    assert counted_error_assessment("forced", "manual_confirmed") == "forced"


@pytest.mark.parametrize(
    "cause", ["out_long", "out_wide", "out_both", "unreturned", "other"]
)
def test_unconfirmed_terminal_causes_cannot_be_counted_as_derived(
    cause: TerminalCause,
) -> None:
    assert counted_terminal_cause(cause, "derived_v4_score") is None
    assert counted_terminal_cause(cause, "manual_confirmed") == cause


def _partial_win_rate() -> MetricRecord:
    return MetricRecord(
        metric_id="point.win_rate",
        definition_version="1",
        subject_type="team",
        subject_id="A",
        filters={},
        bin_version=None,
        numerator=1,
        denominator=4,
        value=0.25,
        candidate_count=7,
        resolved_count=4,
        unknown_count=2,
        excluded_by_reason={"not_point": 1},
        status="partial",
        source_ids={
            "numerator_point_ids": ["p-1"],
            "denominator_point_ids": ["p-1", "p-2", "p-3", "p-4"],
        },
    )


def test_metric_record_preserves_raw_counts_and_source_ids() -> None:
    assert json.loads(json.dumps(_partial_win_rate().to_record())) == {
        "metric_id": "point.win_rate",
        "definition_version": "1",
        "subject_type": "team",
        "subject_id": "A",
        "filters": {},
        "bin_version": None,
        "numerator": 1,
        "denominator": 4,
        "value": 0.25,
        "candidate_count": 7,
        "resolved_count": 4,
        "unknown_count": 2,
        "excluded_by_reason": {"not_point": 1},
        "status": "partial",
        "source_ids": {
            "numerator_point_ids": ["p-1"],
            "denominator_point_ids": ["p-1", "p-2", "p-3", "p-4"],
        },
    }


def test_count_metrics_distinguish_unassessed_from_confirmed_zero() -> None:
    unavailable = replace(
        _partial_win_rate(),
        metric_id="point.unforced_error_count",
        numerator=0,
        denominator=0,
        resolved_count=0,
        candidate_count=3,
        unknown_count=3,
        excluded_by_reason={},
        value=None,
        status="unavailable",
        source_ids={},
    )
    assert unavailable.value is None
    with pytest.raises(ValueError, match="null value"):
        replace(unavailable, value=0)
    confirmed = replace(
        unavailable,
        denominator=3,
        resolved_count=3,
        unknown_count=0,
        value=0,
        status="complete",
        source_ids={
            "numerator_point_ids": [],
            "denominator_point_ids": ["p-1", "p-2", "p-3"],
        },
    )
    assert confirmed.value == 0
    assert (
        replace(
            confirmed,
            numerator=2,
            value=2,
            denominator=1,
            resolved_count=1,
            candidate_count=1,
            metric_id="rally.contact_count",
            subject_type="point",
            source_ids={
                "numerator_ball_event_ids": [31, 33],
                "denominator_point_ids": ["p-1"],
            },
        ).value
        == 2
    )


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"status": "complete"}, "status disagrees"),
        ({"status": "unavailable"}, "status disagrees"),
        ({"candidate_count": 6}, "account for"),
        ({"excluded_by_reason": {"not_point": -1}}, "nonnegative integers"),
        ({"denominator": 3}, "equal resolved_count"),
        ({"numerator": 5}, "exceeds denominator"),
        ({"value": 0.5}, "value disagrees"),
        ({"value": float("nan")}, "value disagrees"),
        ({"subject_type": "player"}, "Invalid metric subject"),
        (
            {"definition_version": "unreleased"},
            "Unknown metric ID or definition version",
        ),
        ({"metric_id": "point.terminal_cause_count"}, "Missing required metric filter"),
        (
            {
                "metric_id": "point.terminal_cause_count",
                "filters": {"terminal_cause": "unknown"},
            },
            "requires a known terminal cause",
        ),
        ({"source_ids": {}}, "Missing source_ids"),
        (
            {
                "source_ids": {
                    "numerator_point_ids": ["p-1"],
                    "denominator_point_ids": [],
                }
            },
            "count must match denominator",
        ),
        (
            {
                "source_ids": {
                    "numerator_point_ids": ["p-1"],
                    "denominator_point_ids": ["p-1", "p-2", "p-1", "p-3"],
                }
            },
            "must contain unique IDs",
        ),
        (
            {
                "source_ids": {
                    "numerator_point_ids": ["p-9"],
                    "denominator_point_ids": ["p-1", "p-2", "p-3", "p-4"],
                }
            },
            "must be a subset",
        ),
        ({"source_ids": {"numerator_point_ids": [True]}}, "must contain valid IDs"),
        ({"metric_id": "point.played_count"}, "Exact count"),
        (
            {"metric_id": "point.unforced_error_count", "numerator": 5},
            "exceeds denominator",
        ),
    ],
)
def test_metric_record_rejects_inconsistent_inputs(changes: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        replace(_partial_win_rate(), **changes)


def test_exact_count_uses_each_resolved_point_once() -> None:
    record = _partial_win_rate()
    evidence = record.source_ids["denominator_point_ids"]
    counted = replace(
        record,
        metric_id="point.played_count",
        numerator=4,
        value=4,
        source_ids={"numerator_point_ids": evidence, "denominator_point_ids": evidence},
    )
    assert counted.numerator == counted.denominator == counted.value == 4
    with pytest.raises(ValueError, match="Exact count"):
        replace(
            counted,
            numerator=3,
            denominator=1,
            value=3,
            candidate_count=1,
            resolved_count=1,
            status="complete",
            unknown_count=0,
            excluded_by_reason={},
            source_ids={},
        )


def test_conditional_shot_error_rate_requires_a_single_role_population() -> None:
    metric = METRIC_DEFINITIONS["conditional.shot_error_rate"]
    bins = {"contact_height": "low"}
    with pytest.raises(ValueError, match="Missing required metric filter shot_role"):
        replace(
            _partial_win_rate(),
            metric_id=metric.metric_id,
            filters={"bins": bins},
            bin_version="b1",
        )
    with pytest.raises(ValueError, match="shot_role must be"):
        replace(
            _partial_win_rate(),
            metric_id=metric.metric_id,
            filters={"shot_role": "all", "bins": bins},
            bin_version="b1",
        )

    # Each attempt below is from the same subject and height bin.
    attempts = [
        (11, "serve", "fault"),
        (12, "rally", "legal"),
        (13, "serve", "let"),
        (14, "return", "net_error"),
    ]
    rules = metric.to_record()["rules"]["shot_error_role"]
    expected = {
        "serve": (1, 1, 1.0, [11], [11]),
        "rally": (0, 1, 0.0, [], [12]),
        "return": (1, 1, 1.0, [14], [14]),
        "non_serve": (1, 2, 0.5, [14], [12, 14]),
    }
    for role, hand_count in expected.items():
        scoped = [
            (idx, outcome)
            for idx, kind, outcome in attempts
            if kind in rules["roles_by_filter"][role]
        ]
        outcomes = rules["serve" if role == "serve" else "non_serve"]
        resolved: list[SourceId] = [
            idx for idx, outcome in scoped if outcome in outcomes["resolved_outcomes"]
        ]
        errors: list[SourceId] = [
            idx for idx, outcome in scoped if outcome in outcomes["error_outcomes"]
        ]
        record = MetricRecord(
            metric_id=metric.metric_id,
            definition_version="1",
            subject_type="team",
            subject_id="A",
            filters={"shot_role": role, "bins": bins},
            bin_version="b1",
            numerator=len(errors),
            denominator=len(resolved),
            value=len(errors) / len(resolved),
            candidate_count=len(scoped),
            resolved_count=len(resolved),
            unknown_count=0,
            status="complete",
            excluded_by_reason={"let": 1} if role == "serve" else {},
            source_ids={
                "numerator_ball_event_ids": errors,
                "denominator_ball_event_ids": resolved,
            },
        )
        assert (
            record.numerator,
            record.denominator,
            record.value,
            errors,
            resolved,
        ) == hand_count
