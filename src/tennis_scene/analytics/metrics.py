"""Metric definitions and output validation for the Phase 2 aggregators."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isclose
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, get_args

DEFINITION_VERSION = "1"

MetricStatus = Literal["complete", "partial", "unavailable"]
SubjectType = Literal["team", "player", "point"]
MetricUnit = Literal["point", "game", "serve_attempt", "shot_attempt"]
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
PointBinKey: TypeAlias = tuple[SourceId, SourceId, str]

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
        "idempotency_key": ["point_id"],
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
        "idempotency_key": ["point_id", "subject_id", "bin_id"],
        "denominator_unit": "point",
        "additive_across_bins": False,
        "player_value": "The player's team's result in points containing the player's qualifying shot.",
    },
    "observed_landing": {
        "population": "shots_with_observed_landing",
        "axes": list(_LANDING_AXES),
        "assume_landing_when_missing": False,
        "eligibility": "Keep missing landing reasons, such as net stop, volley interception, tracking gap and truncation. Preserve signed out distances and report coverage alongside observed-landing metrics.",
    },
    "terminal_cause": {
        "idempotency_key": ["point_id"],
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
    value_kind: Literal["count", "rate"] = "rate"
    rules: tuple[str, ...] = ()
    bin_axes: tuple[str, ...] = ()
    required_filters: tuple[str, ...] = ()
    definition_version: str = DEFINITION_VERSION

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
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
        value_kind="count",
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
        value_kind="count",
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
        value_kind="count",
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
        value_kind="count",
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
        value_kind="count",
        rules=("point", "shot_attempts"),
    ),
    MetricDefinition(
        "rally.contact_count",
        "Racket contacts from the last legal serve through point end, including the terminal error contact.",
        "The point when its rally length can be resolved, including a confirmed double fault as zero contacts.",
        "point",
        _POINT,
        value_kind="count",
        rules=("rally_contact",),
    ),
    MetricDefinition(
        "rally.completed_shot_count",
        "Racket contacts from the last legal serve through point end, excluding the terminal error contact.",
        "The point when its rally length can be resolved, including a confirmed double fault as zero completed shots.",
        "point",
        _POINT,
        value_kind="count",
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
        value_kind="count",
        rules=("point", "terminal_cause"),
        required_filters=("terminal_cause",),
    ),
    MetricDefinition(
        "point.forced_error_count",
        "Points lost by the team with a manually confirmed forced error.",
        "Points lost by the team with a manually confirmed error assessment.",
        "point",
        _TEAM,
        value_kind="count",
        rules=("point", "error_assessment"),
    ),
    MetricDefinition(
        "point.unforced_error_count",
        "Points lost by the team with a manually confirmed unforced error.",
        "Points lost by the team with a manually confirmed error assessment.",
        "point",
        _TEAM,
        value_kind="count",
        rules=("point", "error_assessment"),
    ),
    MetricDefinition(
        "conditional.shot_error_rate",
        "Immediate terminal net/out errors in non-serve bins, or faults in serve-attempt bins.",
        "Actual shot attempts in the selected bins with a known immediate outcome.",
        "shot_attempt",
        _PLAYER_TEAM,
        rules=("shot_attempts", "shot_error"),
        bin_axes=_SHOT_AXES,
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
METRIC_DEFINITION_FIELDS = tuple(MetricDefinition.__dataclass_fields__)


def unique_point_bins(keys: Iterable[PointBinKey]) -> frozenset[PointBinKey]:
    """Deduplicate points independently for each subject and condition bin."""
    return frozenset(keys)


def counted_terminal_cause(
    cause: TerminalCause, source: TerminalCauseSource
) -> TerminalCause | None:
    """Return the one eligible point cause, independent of serve fault causes."""
    return cause if cause != "unknown" and source != "unknown" else None


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
        definition = METRIC_DEFINITIONS.get(self.metric_id)
        if (
            definition is None
            or self.definition_version != definition.definition_version
        ):
            raise ValueError("Unknown metric ID or definition version")
        if self.subject_type not in definition.subject_types or not self.subject_id:
            raise ValueError("Invalid metric subject")

        if any(not self.filters.get(key) for key in definition.required_filters):
            raise ValueError("Missing required metric filter")
        if self.metric_id == "point.terminal_cause_count":
            cause = self.filters["terminal_cause"]
            if cause == "unknown" or cause not in get_args(TerminalCause):
                raise ValueError(
                    "A terminal cause count requires a known terminal cause"
                )
        bins = self.filters.get("bins", {})
        if not isinstance(bins, dict) or any(
            axis not in definition.bin_axes or not isinstance(key, str) or not key
            for axis, key in bins.items()
        ):
            raise ValueError("Unsupported metric bin")
        if definition.bin_axes and (not bins or not self.bin_version):
            raise ValueError("Binned metrics require condition bins and bin_version")

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
            return

        expected_value: float = self.numerator
        if definition.value_kind == "rate":
            if self.numerator > self.denominator:
                raise ValueError("Rate numerator exceeds denominator")
            expected_value = self.numerator / self.denominator
        if (
            self.value is None
            or isinstance(self.value, bool)
            or not isclose(self.value, expected_value, rel_tol=1e-12)
        ):
            raise ValueError(
                "Metric value disagrees with its numerator and denominator"
            )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
