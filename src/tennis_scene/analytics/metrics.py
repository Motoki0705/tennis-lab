"""Metric definitions and output validation for the Phase 2 aggregators."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from math import isclose
from types import MappingProxyType
from typing import Any, Literal, Self, TypeAlias, get_args

DEFINITION_VERSION = "1"

MetricStatus = Literal["complete", "partial", "unavailable"]
SubjectType = Literal["team", "player", "point"]
MetricUnit = Literal["point", "game", "serve_attempt", "shot_attempt"]
CountRelation = Literal["exact_count", "subset_count", "ratio", "multi_per_unit"]
TerminalCause = Literal[
    "net_error",
    "out_long",
    "out_wide",
    "out_both",
    "double_fault",
    "unreturned",
    "other",
    "unknown",
]
TerminalCauseSource = Literal["manual_confirmed", "derived_v4_score", "unknown"]
ErrorAssessment = Literal["forced", "unforced", "undetermined"]
SourceId: TypeAlias = int | str

TERMINAL_CAUSES_BY_SOURCE: Mapping[TerminalCauseSource, tuple[TerminalCause, ...]] = (
    MappingProxyType(
        {
            "manual_confirmed": tuple(
                cause for cause in get_args(TerminalCause) if cause != "unknown"
            ),
            "derived_v4_score": ("net_error", "double_fault"),
            "unknown": ("unknown",),
        }
    )
)


@dataclass(frozen=True)
class MatchPointId:
    """Identity shared by clip-level records of the same point in one match."""

    match_context_id: int
    match_point_key: str

    def __post_init__(self) -> None:
        if (
            type(self.match_context_id) is not int
            or self.match_context_id <= 0
            or not isinstance(self.match_point_key, str)
            or not self.match_point_key.strip()
        ):
            raise ValueError(
                "A canonical point requires a match ID and match point key"
            )


PointBinKey: TypeAlias = tuple[MatchPointId, SourceId, str]

_TEAM: tuple[SubjectType, ...] = ("team",)
_PLAYER_TEAM: tuple[SubjectType, ...] = ("player", "team")
_POINT: tuple[SubjectType, ...] = ("point",)
_SHOT_AXES = (
    "contact_height",
    "shot_subtype",
    "shot_role",
    "incoming_speed",
    "incoming_landing_depth",
    "contact_position",
    "hitter_movement",
)
_LANDING_AXES = ("landing_depth", "landing_lane", "landing_course")
_SHOT_ROLE_FILTERS = {
    "serve": ("serve",),
    "return": ("return",),
    "rally": ("rally",),
    "non_serve": ("return", "rally"),
}
_NON_SERVE_ERRORS = ("net_error", "out_long", "out_wide", "out_both")
_UNIT_ID_GROUPS: dict[MetricUnit, str] = {
    "point": "point_ids",
    "game": "game_ids",
    "serve_attempt": "serve_attempt_ids",
    "shot_attempt": "ball_event_ids",
}
_MATCH_POINT_FIELDS = ["match_context_id", "match_point_key"]

_RULES: dict[str, dict[str, Any]] = {
    "availability": {
        "empty_denominator": {"value": None, "status": "unavailable"},
        "complete": "Evaluable candidates exist and none are unknown for this metric.",
        "partial": "Evaluable candidates and unknown candidates both exist.",
        "counts": "candidate_count = resolved_count + unknown_count + sum(excluded_by_reason.values())",
        "denominator": "resolved_count",
        "missing": "Determine eligibility and missing inputs per metric. A known point result can be used when shot features are missing.",
    },
    "point": {
        "idempotency_key": _MATCH_POINT_FIELDS,
        "winner_score_status": "confirmed",
        "eligibility": "Use valid points with confirmed boundaries where the metric requires them. Keep truncated or multiple points unresolved when a required boundary is missing. Deduplicate overlapping clips by point identity.",
        "team_credit": "Credit the team once. Credit a player only in the role specified by the metric.",
    },
    "serve_attempts": {
        "valid_numbers": [1, 2],
        "unknown_number": None,
        "let_advances_serve_number": False,
        "let_in_first_in_denominator": False,
        "first_fault_requires_second_serve": True,
        "second_fault_terminal_cause": "double_fault",
        "double_fault_in_second_serve_win_denominator": True,
        "double_fault_in_return_win_denominator": True,
        "double_fault_in_return_in_play_denominator": False,
        "eligibility": "Identify attempts from confirmed context. A let replays the same serve number and its first bounce is outside the rally. A first observed shot can have an unknown serve number.",
    },
    "ace": {
        "requires": [
            "legal_serve",
            "confirmed_no_receiver_contact",
            "confirmed_point_win",
        ],
        "missing": "An absent next shot with truncation or tracking gaps leaves ace status unknown.",
    },
    "shot_attempts": {
        "excludes": ["dead_ball", "let_replay"],
        "serve_faults_in_shot_count": True,
        "eligibility": "Use in-point racket contacts with established serve, return or rally role. A known contact with unknown subtype can contribute to shot count and remains unknown for subtype share.",
    },
    "shot_error": {
        "denominator_unit": "shot_attempt",
        "serve_faults_in_nonserve_error_rate": False,
        "serve_faults_in_serve_bin": True,
        "eligibility": "Use a known immediate legal or failed result. Count a net or out loss only for the attributed terminal non-serve shot. For serve bins use known fault/in on actual numbered attempts, excluding lets.",
    },
    "shot_error_role": {
        "role_filter": "shot_role",
        "roles_by_filter": _SHOT_ROLE_FILTERS,
        "serve": {
            "resolved_outcomes": ("in", "fault"),
            "error_outcomes": ("fault",),
            "excluded_outcomes": ("let",),
        },
        "non_serve": {
            "resolved_outcomes": ("legal", *_NON_SERVE_ERRORS),
            "error_outcomes": _NON_SERVE_ERRORS,
        },
    },
    "rally_contact": {
        "starts_at": "last_legal_serve",
        "includes_terminal_error_contact": True,
        "confirmed_double_fault_count": 0,
        "excludes": ["earlier_serve_fault", "let_replay", "dead_ball"],
    },
    "rally_completed": {
        "starts_at": "last_legal_serve",
        "includes_terminal_error_contact": False,
        "confirmed_double_fault_count": 0,
        "excludes": ["earlier_serve_fault", "let_replay", "dead_ball"],
    },
    "score": {
        "requires": [
            "confirmed_game_boundaries",
            "confirmed_score_before",
            "scoring_rules",
        ],
        "eligibility": "Evaluate actual played opportunities under the recorded rules. Separate regular games from tiebreaks or final games. Repeated break points after deuce count as separate played points.",
    },
    "conditional_points": {
        "idempotency_key": [*_MATCH_POINT_FIELDS, "subject_id", "bin_id"],
        "denominator_unit": "point",
        "additive_across_bins": False,
        "player_value": "The player's team's result in points containing the player's qualifying shot.",
    },
    "observed_landing": {
        "population": "shots_with_observed_landing",
        "landing_source_kind": "observed",
        "axes": list(_LANDING_AXES),
        "assume_landing_when_missing": False,
        "eligibility": "Keep missing landing reasons, such as net stop, volley interception, tracking gap and truncation. Preserve signed out distances and report coverage alongside observed-landing metrics.",
    },
    "terminal_cause": {
        "idempotency_key": _MATCH_POINT_FIELDS,
        "causes_by_source": dict(TERMINAL_CAUSES_BY_SOURCE),
        "max_causes_per_point": 1,
        "double_fault_can_also_count_as_net_error": False,
        "eligibility": "Use confirmed or deterministic V4-plus-score provenance. The physical cause of a second serve fault belongs to the serve attempt.",
    },
    "error_assessment": {
        "source": "manual_confirmed",
        "unassessed": "undetermined",
        "zero_when_entirely_unassessed": False,
        "eligibility": "Require manual forced/unforced assessment. Keep terminal cause independent of assessment.",
    },
}


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    numerator: str
    denominator: str
    unit: MetricUnit
    subject_types: tuple[SubjectType, ...]
    count_relation: CountRelation = "ratio"
    rules: tuple[str, ...] = ()
    bin_axes: tuple[str, ...] = ()
    required_filters: tuple[str, ...] = ()
    definition_version: str = DEFINITION_VERSION

    @property
    def value_kind(self) -> Literal["count", "rate"]:
        return "rate" if self.count_relation == "ratio" else "count"

    @property
    def evidence_keys(self) -> dict[str, str]:
        denominator_group = _UNIT_ID_GROUPS[self.unit]
        numerator_group = (
            "ball_event_ids"
            if self.count_relation == "multi_per_unit"
            else denominator_group
        )
        return {
            "numerator": f"numerator_{numerator_group}",
            "denominator": f"denominator_{denominator_group}",
        }

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["value_kind"] = self.value_kind
        record["evidence_keys"] = self.evidence_keys
        record["rules"] = deepcopy(
            {rule_id: _RULES[rule_id] for rule_id in ("availability", *self.rules)}
        )
        return record


_DEFINITIONS = (
    MetricDefinition(
        "point.played_count",
        "Valid points played by the team.",
        "Points with confirmed valid point eligibility.",
        "point",
        _TEAM,
        count_relation="exact_count",
        rules=("point",),
    ),
    MetricDefinition(
        "point.win_rate",
        "Valid points won by the team.",
        "Valid points with a known winner.",
        "point",
        _TEAM,
        rules=("point",),
    ),
    MetricDefinition(
        "serve.point_win_rate",
        "Service points won.",
        "Service points with a known winner.",
        "point",
        _PLAYER_TEAM,
        rules=("point", "serve_attempts"),
    ),
    MetricDefinition(
        "serve.first_in_rate",
        "Points started with a legal first serve.",
        "Points with a known in/fault result on the first serve after any replays.",
        "point",
        _PLAYER_TEAM,
        rules=("point", "serve_attempts"),
    ),
    MetricDefinition(
        "serve.first_point_win_rate",
        "Points won after a legal first serve.",
        "Points with a legal first serve and a known winner.",
        "point",
        _PLAYER_TEAM,
        rules=("point", "serve_attempts"),
    ),
    MetricDefinition(
        "serve.second_point_win_rate",
        "Points won when a second serve was required.",
        "Points requiring a second serve with a known winner, including double faults.",
        "point",
        _PLAYER_TEAM,
        rules=("point", "serve_attempts"),
    ),
    MetricDefinition(
        "serve.double_fault_count",
        "Confirmed double faults.",
        "Service points with known double fault status.",
        "point",
        _PLAYER_TEAM,
        count_relation="subset_count",
        rules=("point", "serve_attempts"),
    ),
    MetricDefinition(
        "serve.double_fault_rate",
        "Confirmed double faults.",
        "Service points with known double fault status.",
        "point",
        _PLAYER_TEAM,
        rules=("point", "serve_attempts"),
    ),
    MetricDefinition(
        "serve.ace_count",
        "Confirmed aces.",
        "Service points with known ace status.",
        "point",
        _PLAYER_TEAM,
        count_relation="subset_count",
        rules=("point", "ace"),
    ),
    MetricDefinition(
        "serve.ace_rate",
        "Confirmed aces.",
        "Service points with known ace status.",
        "point",
        _PLAYER_TEAM,
        rules=("point", "ace"),
    ),
    MetricDefinition(
        "return.point_win_rate",
        "Points won by the return side, including opponent double faults.",
        "Return points with a known winner. Player attribution requires a known designated receiver.",
        "point",
        _PLAYER_TEAM,
        rules=("point", "serve_attempts"),
    ),
    MetricDefinition(
        "return.in_play_rate",
        "Legal returns put in play.",
        "Legal serve opportunities with a known return result, including aces as failed returns.",
        "serve_attempt",
        _PLAYER_TEAM,
        rules=("serve_attempts", "ace"),
    ),
    MetricDefinition(
        "shot.count",
        "In-point serve, return and rally contacts.",
        "Eligible contacts with established shot attribution.",
        "shot_attempt",
        _PLAYER_TEAM,
        count_relation="exact_count",
        rules=("shot_attempts",),
    ),
    MetricDefinition(
        "shot.type_share",
        "Contacts of the selected detailed shot subtype.",
        "Eligible contacts with a known subtype.",
        "shot_attempt",
        _PLAYER_TEAM,
        rules=("shot_attempts",),
        required_filters=("shot_subtype",),
    ),
    MetricDefinition(
        "shot.terminal_error_rate",
        "Non-serve shots that immediately lost the point by net or out.",
        "Non-serve shots with known immediate legal or failed outcome.",
        "shot_attempt",
        _PLAYER_TEAM,
        rules=("shot_attempts", "shot_error"),
    ),
    MetricDefinition(
        "shot.unreturned_point_count",
        "Points won by a legal shot with no legal return. Winner attribution additionally requires confirmed no contact.",
        "Points with a known unreturned outcome and attributed shot.",
        "point",
        _PLAYER_TEAM,
        count_relation="subset_count",
        rules=("point", "shot_attempts"),
    ),
    MetricDefinition(
        "rally.contact_count",
        "Racket contacts from the last legal serve through point end, including the terminal error contact.",
        "The point when its rally length can be resolved, including a confirmed double fault as zero contacts.",
        "point",
        _POINT,
        count_relation="multi_per_unit",
        rules=("rally_contact",),
    ),
    MetricDefinition(
        "rally.completed_shot_count",
        "Racket contacts from the last legal serve through point end, excluding the terminal error contact.",
        "The point when its rally length can be resolved, including a confirmed double fault as zero completed shots.",
        "point",
        _POINT,
        count_relation="multi_per_unit",
        rules=("rally_completed",),
    ),
    MetricDefinition(
        "conditional.point_win_rate",
        "Unique points won that contain a shot in the specified condition bins.",
        "Unique points containing a qualifying shot, with known winners.",
        "point",
        _PLAYER_TEAM,
        rules=("point", "conditional_points"),
        bin_axes=_SHOT_AXES,
    ),
    MetricDefinition(
        "game.service_win_rate",
        "Regular service games won.",
        "Completed regular service games with a known server and winner.",
        "game",
        _TEAM,
        rules=("score",),
    ),
    MetricDefinition(
        "game.return_win_rate",
        "Regular return games won.",
        "Completed regular return games with a known server and winner.",
        "game",
        _TEAM,
        rules=("score",),
    ),
    MetricDefinition(
        "score.break_point_conversion_rate",
        "Points won by the return side when it could win the regular game with the next point.",
        "Played break-point opportunities with a known winner.",
        "point",
        _TEAM,
        rules=("point", "score"),
    ),
    MetricDefinition(
        "score.break_point_saved_rate",
        "Points won by the service side when the return side could win the regular game with the next point.",
        "Played break-point opportunities with a known winner.",
        "point",
        _TEAM,
        rules=("point", "score"),
    ),
    MetricDefinition(
        "score.game_point_conversion_rate",
        "Points won when the team could win the game with the next point.",
        "Played game-point opportunities for the team with a known winner.",
        "point",
        _TEAM,
        rules=("point", "score"),
    ),
    MetricDefinition(
        "score.match_point_conversion_rate",
        "Points won when the team could win the match with the next point.",
        "Played match-point opportunities for the team with a known winner.",
        "point",
        _TEAM,
        rules=("point", "score"),
    ),
    MetricDefinition(
        "point.terminal_cause_count",
        "Points that ended with the selected confirmed or derived terminal cause.",
        "Points with a known terminal cause.",
        "point",
        _TEAM,
        count_relation="subset_count",
        rules=("point", "terminal_cause"),
        required_filters=("terminal_cause",),
    ),
    MetricDefinition(
        "point.forced_error_count",
        "Points lost by the team with a manually confirmed forced error.",
        "Points lost by the team with a manually confirmed error assessment.",
        "point",
        _TEAM,
        count_relation="subset_count",
        rules=("point", "error_assessment"),
    ),
    MetricDefinition(
        "point.unforced_error_count",
        "Points lost by the team with a manually confirmed unforced error.",
        "Points lost by the team with a manually confirmed error assessment.",
        "point",
        _TEAM,
        count_relation="subset_count",
        rules=("point", "error_assessment"),
    ),
    MetricDefinition(
        "conditional.shot_error_rate",
        "Immediate terminal net/out errors in non-serve bins, or faults in serve-attempt bins.",
        "Actual shot attempts in the selected bins with a known immediate outcome.",
        "shot_attempt",
        _PLAYER_TEAM,
        rules=("shot_attempts", "shot_error", "shot_error_role"),
        bin_axes=tuple(axis for axis in _SHOT_AXES if axis != "shot_role"),
        required_filters=("shot_role",),
    ),
    MetricDefinition(
        "conditional.landing_out_rate",
        "Observed landings judged out, optionally for a selected out subtype.",
        "Shots with an observed landing in the selected zone and a known line decision.",
        "shot_attempt",
        _PLAYER_TEAM,
        rules=("observed_landing",),
        bin_axes=_LANDING_AXES,
    ),
    MetricDefinition(
        "conditional.landing_point_win_rate",
        "Unique points won that contain a qualifying shot with an observed landing in the selected zone.",
        "Unique points containing a qualifying observed landing in the selected zone, with known winners.",
        "point",
        _PLAYER_TEAM,
        rules=("point", "conditional_points", "observed_landing"),
        bin_axes=_LANDING_AXES,
    ),
)

METRIC_DEFINITIONS: Mapping[str, MetricDefinition] = MappingProxyType(
    {definition.metric_id: definition for definition in _DEFINITIONS}
)
METRIC_DEFINITION_FIELDS = (
    *MetricDefinition.__dataclass_fields__,
    "value_kind",
    "evidence_keys",
)


def unique_point_bins(keys: Iterable[PointBinKey]) -> frozenset[PointBinKey]:
    """Deduplicate points independently for each subject and condition bin."""
    unique: set[PointBinKey] = set()
    for key in keys:
        if not isinstance(key[0], MatchPointId):
            raise ValueError("Pass MatchPointId with the match ID and match point key")
        unique.add(key)
    return frozenset(unique)


def counted_terminal_cause(
    cause: TerminalCause, source: TerminalCauseSource
) -> TerminalCause | None:
    """Return the one eligible point cause, independent of serve fault causes."""
    if cause != "unknown" and cause in TERMINAL_CAUSES_BY_SOURCE.get(source, ()):
        return cause
    return None


def counted_error_assessment(
    assessment: ErrorAssessment, source: str | None
) -> ErrorAssessment | None:
    return (
        assessment
        if assessment != "undetermined" and source == "manual_confirmed"
        else None
    )


def metric_status(denominator: int, unknown_count: int) -> MetricStatus:
    if denominator == 0:
        return "unavailable"
    return "partial" if unknown_count else "complete"


@dataclass(frozen=True)
class MetricRecord:
    """An aggregate with the evidence required to interpret its coverage."""

    metric_id: str
    definition_version: str
    subject_type: SubjectType
    subject_id: str
    filters: dict[str, Any]
    bin_version: str | None
    numerator: int
    denominator: int
    value: float | None
    candidate_count: int
    resolved_count: int
    unknown_count: int
    excluded_by_reason: dict[str, int]
    status: MetricStatus
    source_ids: dict[str, list[SourceId]]

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_record(cls, row: Mapping[str, Any]) -> Self:
        return cls(**{field.name: row[field.name] for field in fields(cls)})

    def validate(self) -> None:
        definition = METRIC_DEFINITIONS.get(self.metric_id)
        if (
            definition is None
            or self.definition_version != definition.definition_version
        ):
            raise ValueError("Unknown metric ID or definition version")
        if self.subject_type not in definition.subject_types or not self.subject_id:
            raise ValueError("Invalid metric subject")

        if not isinstance(self.filters, dict):
            raise ValueError("Metric filters must be an object")
        for key in definition.required_filters:
            if not self.filters.get(key):
                raise ValueError(f"Missing required metric filter {key}")
        if self.metric_id == "point.terminal_cause_count":
            cause = self.filters["terminal_cause"]
            if cause == "unknown" or cause not in get_args(TerminalCause):
                raise ValueError(
                    "A terminal cause count requires a known terminal cause"
                )
        if self.metric_id == "conditional.shot_error_rate":
            role = self.filters["shot_role"]
            if not isinstance(role, str) or role not in _SHOT_ROLE_FILTERS:
                raise ValueError("shot_role must be serve, return, rally or non_serve")
        bins = self.filters.get("bins", {})
        if not isinstance(bins, dict) or any(
            axis not in definition.bin_axes or not isinstance(key, str) or not key
            for axis, key in bins.items()
        ):
            raise ValueError("Unsupported metric bin")
        if definition.bin_axes and (not bins or not self.bin_version):
            raise ValueError("Binned metrics require condition bins and bin_version")

        if not isinstance(self.excluded_by_reason, dict):
            raise ValueError("excluded_by_reason must be an object")
        counts = (
            self.numerator,
            self.denominator,
            self.candidate_count,
            self.resolved_count,
            self.unknown_count,
            *self.excluded_by_reason.values(),
        )
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("Metric counts must be nonnegative integers")
        excluded_count = sum(self.excluded_by_reason.values())
        if (
            self.candidate_count
            != self.resolved_count + self.unknown_count + excluded_count
        ):
            raise ValueError(
                "Candidate counts must account for resolved, unknown and excluded cases"
            )
        if self.denominator != self.resolved_count:
            raise ValueError("denominator must equal resolved_count")
        if self.status != metric_status(self.denominator, self.unknown_count):
            raise ValueError("Metric status disagrees with resolved and unknown counts")
        if self.denominator == 0:
            if self.value is not None or self.numerator:
                raise ValueError(
                    "An empty denominator requires a null value and zero numerator"
                )
        elif (
            definition.count_relation == "exact_count"
            and self.numerator != self.denominator
        ):
            raise ValueError("Exact count requires numerator equal to denominator")
        elif (
            definition.count_relation != "multi_per_unit"
            and self.numerator > self.denominator
        ):
            raise ValueError("Metric numerator exceeds denominator")
        else:
            self._validate_value(definition)
        self._validate_evidence(definition)

    def _validate_value(self, definition: MetricDefinition) -> None:
        expected = (
            self.numerator / self.denominator
            if definition.count_relation == "ratio"
            else self.numerator
        )
        if (
            self.value is None
            or isinstance(self.value, bool)
            or not isclose(self.value, expected, rel_tol=1e-12)
        ):
            raise ValueError(
                "Metric value disagrees with its numerator and denominator"
            )

    def _validate_evidence(self, definition: MetricDefinition) -> None:
        if not isinstance(self.source_ids, dict):
            raise ValueError("source_ids must be an object")
        for group, ids in self.source_ids.items():
            if not isinstance(ids, list) or any(
                not (type(item) is int and item > 0)
                and not (isinstance(item, str) and item.strip())
                for item in ids
            ):
                raise ValueError(f"source_ids[{group}] must contain valid IDs")
            if len(set(ids)) != len(ids):
                raise ValueError(f"source_ids[{group}] must contain unique IDs")
        keys = definition.evidence_keys
        evidence: dict[str, set[SourceId]] = {}
        for role, count in (
            ("numerator", self.numerator),
            ("denominator", self.denominator),
        ):
            key = keys[role]
            group_ids = self.source_ids.get(key)
            if group_ids is None and self.denominator:
                raise ValueError(f"Missing source_ids evidence group {key}")
            evidence[role] = set(group_ids or ())
            if len(evidence[role]) != count:
                raise ValueError(f"source_ids[{key}] count must match {role}")
        if (
            definition.count_relation != "multi_per_unit"
            and not evidence["numerator"] <= evidence["denominator"]
        ):
            raise ValueError(
                "Numerator evidence must be a subset of denominator evidence"
            )

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
