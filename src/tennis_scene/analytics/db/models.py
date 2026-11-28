"""SQLAlchemy models for the versioned analytics store."""

from __future__ import annotations

from datetime import datetime
from typing import Any, get_args

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Connection,
    DateTime,
    Double,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    MetaData,
    Text,
    UniqueConstraint,
    event,
    false,
    func,
    text,
    true,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Mapper, mapped_column

from src.tennis_scene.analytics.metrics import (
    DEFINITION_VERSION,
    METRIC_DEFINITIONS,
    TERMINAL_CAUSES_BY_SOURCE,
    ErrorAssessment,
    MetricRecord,
    MetricStatus,
    TerminalCause,
    TerminalCauseSource,
)
from src.tennis_scene.ball_event_labels import (
    EVENT_CLASSES_V4,
    EVENT_SOURCES,
    EVENT_STATUS,
    EVENT_SUBTYPES_BY_KIND_V4,
    EVENT_SUBTYPES_V4,
)

SCHEMA_VERSION = 1
VISIBILITIES = ("private", "team", "org", "public")
SPORT_TYPES = ("soft_tennis", "tennis")
PROFILE_SOURCES = ("rule_default", "measured", "custom")
GEOMETRY_SOURCES = ("observed", "estimated")
RESOLUTION_STATUSES = ("resolved", "ambiguous", "unresolved")

_JSON = JSONB(none_as_null=True)
_EMPTY_OBJECT = text("'{}'::jsonb")
_EMPTY_ARRAY = text("'[]'::jsonb")


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "pk": "pk_%(table_name)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "ix": "ix_%(table_name)s_%(column_0_N_name)s",
            "fk": "fk_%(table_name)s_%(column_0_N_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
        }
    )


class _Record:
    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class _OwnedRecord(_Record):
    owner_team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("teams.id"))
    visibility: Mapped[str] = mapped_column(Text, server_default="team")


def _choices(column: str, choices: tuple[str | int, ...]) -> CheckConstraint:
    values = ", ".join(repr(value) for value in choices)
    return CheckConstraint(f"{column} IN ({values})", name=column)


def _owned_index(*columns: str) -> Index:
    return Index(None, "owner_team_id", *columns)


def _ref(column: str, target: str, *scope: str) -> ForeignKeyConstraint:
    return ForeignKeyConstraint(
        ["owner_team_id", *scope, column],
        [f"{target}.{key}" for key in ("owner_team_id", *scope, "id")],
    )


def _sha256(column: str) -> CheckConstraint:
    return CheckConstraint(f"{column} ~ '^[a-f0-9]{{64}}$'", name=column)


def _frame_bounds() -> CheckConstraint:
    return CheckConstraint(
        "start_frame >= 0 AND end_frame >= 0 AND start_frame <= end_frame",
        name="frame_bounds",
    )


class Organization(_Record, Base):
    __tablename__ = "organizations"

    slug: Mapped[str] = mapped_column(Text, unique=True)
    name: Mapped[str] = mapped_column(Text)


class Team(_Record, Base):
    __tablename__ = "teams"
    __table_args__ = (UniqueConstraint("organization_id", "slug"),)

    organization_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("organizations.id")
    )
    slug: Mapped[str] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text)


class User(_Record, Base):
    __tablename__ = "users"

    external_subject: Mapped[str] = mapped_column(Text, unique=True)
    email: Mapped[str | None] = mapped_column(Text)
    display_name: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(server_default=true())


class TeamMembership(_Record, Base):
    __tablename__ = "team_memberships"
    __table_args__ = (UniqueConstraint("team_id", "user_id"), Index(None, "user_id"))

    team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("teams.id"))
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"))
    role: Mapped[str] = mapped_column(Text)


class CourtNetProfile(_OwnedRecord, Base):
    __tablename__ = "court_net_profiles"
    __table_args__ = (
        _choices("visibility", VISIBILITIES),
        UniqueConstraint("owner_team_id", "id"),
        UniqueConstraint("owner_team_id", "source_kind", "id"),
        UniqueConstraint("owner_team_id", "sport_type", "id"),
        UniqueConstraint("owner_team_id", "profile_key"),
        _owned_index("sport_type"),
        _choices("sport_type", SPORT_TYPES),
        _choices("net_profile_kind", ("flat", "itf_parabolic", "measured_curve")),
        _choices("source_kind", PROFILE_SOURCES),
        CheckConstraint(
            "net_height_center_m > 0 AND net_height_post_m > 0 "
            "AND net_post_offset_x_m > 0 AND ball_radius_m > 0",
            name="positive_dimensions",
        ),
    )

    profile_key: Mapped[str] = mapped_column(Text)
    sport_type: Mapped[str] = mapped_column(Text)
    net_height_center_m: Mapped[float] = mapped_column(Double)
    net_height_post_m: Mapped[float] = mapped_column(Double)
    net_profile_kind: Mapped[str] = mapped_column(Text)
    net_post_offset_x_m: Mapped[float] = mapped_column(Double)
    ball_radius_m: Mapped[float] = mapped_column(Double)
    source_kind: Mapped[str] = mapped_column(Text)
    parameters_json: Mapped[dict[str, Any]] = mapped_column(
        _JSON, server_default=_EMPTY_OBJECT
    )


class IngestRun(_OwnedRecord, Base):
    __tablename__ = "ingest_runs"
    __table_args__ = (
        _choices("visibility", VISIBILITIES),
        UniqueConstraint("owner_team_id", "dataset_id", "video_id", "id"),
        UniqueConstraint(
            "owner_team_id",
            "dataset_id",
            "video_id",
            "v4_manifest_sha256",
            "projects_json_sha256",
            "manual_supplement_sha256",
            "schema_version",
            "definition_version",
            name="uq_ingest_runs_inputs",
            postgresql_nulls_not_distinct=True,
        ),
        _owned_index("dataset_id", "video_id"),
        _sha256("v4_manifest_sha256"),
        _sha256("projects_json_sha256"),
        _sha256("manual_supplement_sha256"),
    )

    dataset_id: Mapped[str] = mapped_column(Text)
    video_id: Mapped[str] = mapped_column(Text)
    v4_manifest_sha256: Mapped[str] = mapped_column(Text)
    projects_json_sha256: Mapped[str] = mapped_column(Text)
    manual_supplement_sha256: Mapped[str | None] = mapped_column(Text)
    schema_version: Mapped[int]
    definition_version: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    summary_json: Mapped[dict[str, Any]] = mapped_column(
        _JSON, server_default=_EMPTY_OBJECT
    )


class MatchContext(_OwnedRecord, Base):
    __tablename__ = "match_contexts"
    __table_args__ = (
        _choices("visibility", VISIBILITIES),
        _ref("ingest_run_id", "ingest_runs", "dataset_id", "video_id"),
        _ref("court_net_profile_id", "court_net_profiles", "sport_type"),
        UniqueConstraint("owner_team_id", "id"),
        UniqueConstraint("owner_team_id", "ingest_run_id", "match_key"),
        _owned_index("dataset_id", "video_id"),
        _choices("sport_type", SPORT_TYPES),
        _choices("first_server", ("A", "B")),
    )

    ingest_run_id: Mapped[int] = mapped_column(BigInteger)
    dataset_id: Mapped[str] = mapped_column(Text)
    video_id: Mapped[str] = mapped_column(Text)
    match_key: Mapped[str] = mapped_column(Text)
    sport_type: Mapped[str] = mapped_column(Text)
    rules_json: Mapped[dict[str, Any]] = mapped_column(_JSON)
    side_a_name: Mapped[str | None] = mapped_column(Text)
    side_b_name: Mapped[str | None] = mapped_column(Text)
    first_server: Mapped[str | None] = mapped_column(Text)
    court_net_profile_id: Mapped[int | None] = mapped_column(BigInteger)
    time_conversion_json: Mapped[dict[str, Any]] = mapped_column(_JSON)


class Clip(_OwnedRecord, Base):
    __tablename__ = "clips"
    __table_args__ = (
        _choices("visibility", VISIBILITIES),
        _ref("match_context_id", "match_contexts"),
        UniqueConstraint("owner_team_id", "id"),
        UniqueConstraint("owner_team_id", "match_context_id", "id"),
        UniqueConstraint("owner_team_id", "match_context_id", "clip_id"),
        _owned_index("match_context_id", "global_start_sec"),
        _choices("split", ("train", "val", "test", "unassigned")),
        CheckConstraint(
            "global_start_sec >= 0 AND global_start_sec < global_end_sec "
            "AND fps > 0 AND num_frames > 0",
            name="timebase",
        ),
        _sha256("scene_npz_sha256"),
        _sha256("annotation_sha256"),
    )

    match_context_id: Mapped[int] = mapped_column(BigInteger)
    clip_id: Mapped[str] = mapped_column(Text)
    clip_name: Mapped[str] = mapped_column(Text)
    split: Mapped[str] = mapped_column(Text, server_default="unassigned")
    global_start_sec: Mapped[float] = mapped_column(Double)
    global_end_sec: Mapped[float] = mapped_column(Double)
    fps: Mapped[float] = mapped_column(Double)
    num_frames: Mapped[int]
    scene_schema_version: Mapped[int]
    scene_npz_sha256: Mapped[str] = mapped_column(Text)
    annotation_sha256: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        _JSON, server_default=_EMPTY_OBJECT
    )


_TERMINAL_CAUSE_SOURCE_PAIRS = ", ".join(
    f"({cause!r}, {source!r})"
    for source, causes in TERMINAL_CAUSES_BY_SOURCE.items()
    for cause in causes
)


class Point(_OwnedRecord, Base):
    __tablename__ = "points"
    __table_args__ = (
        _choices("visibility", VISIBILITIES),
        _ref("clip_id", "clips", "match_context_id"),
        UniqueConstraint("owner_team_id", "clip_id", "id"),
        UniqueConstraint("owner_team_id", "clip_id", "point_key"),
        _owned_index("clip_id", "start_frame"),
        _owned_index("match_context_id", "match_point_key"),
        _frame_bounds(),
        CheckConstraint(
            "match_point_key IS NULL OR btrim(match_point_key) <> ''",
            name="match_point_key",
        ),
        _choices(
            "completeness",
            (
                "complete",
                "truncated_start",
                "truncated_end",
                "truncated_both",
                "unknown",
            ),
        ),
        _choices(
            "score_status",
            ("confirmed", "auto", "unknown", "not_point", "multiple_points"),
        ),
        _choices("server_team", ("A", "B")),
        _choices("winner_team", ("A", "B")),
        _choices("error_assessment", get_args(ErrorAssessment)),
        CheckConstraint(
            f"(terminal_cause, terminal_cause_source) IN ({_TERMINAL_CAUSE_SOURCE_PAIRS})",
            name="terminal_cause_provenance",
        ),
    )

    clip_id: Mapped[int] = mapped_column(BigInteger)
    match_context_id: Mapped[int] = mapped_column(BigInteger)
    point_key: Mapped[str] = mapped_column(Text)
    match_point_key: Mapped[str | None] = mapped_column(Text)
    start_frame: Mapped[int | None]
    end_frame: Mapped[int | None]
    completeness: Mapped[str] = mapped_column(Text, server_default="unknown")
    score_status: Mapped[str] = mapped_column(Text, server_default="unknown")
    score_before_json: Mapped[dict[str, Any] | None] = mapped_column(_JSON)
    score_after_json: Mapped[dict[str, Any] | None] = mapped_column(_JSON)
    server_team: Mapped[str | None] = mapped_column(Text)
    winner_team: Mapped[str | None] = mapped_column(Text)
    score_source: Mapped[str | None] = mapped_column(Text)
    terminal_cause: Mapped[TerminalCause] = mapped_column(
        Text, server_default="unknown"
    )
    terminal_cause_source: Mapped[TerminalCauseSource] = mapped_column(
        Text, server_default="unknown"
    )
    error_assessment: Mapped[ErrorAssessment] = mapped_column(
        Text, server_default="undetermined"
    )
    error_assessment_source: Mapped[str | None] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text)


_V4_KIND_SUBTYPE_PAIRS = ", ".join(
    f"({kind!r}, {EVENT_SUBTYPES_V4[subtype]!r})"
    for kind, subtypes in zip(EVENT_CLASSES_V4, EVENT_SUBTYPES_BY_KIND_V4, strict=True)
    for subtype in subtypes
)


class BallEvent(_OwnedRecord, Base):
    __tablename__ = "ball_events"
    __table_args__ = (
        _choices("visibility", VISIBILITIES),
        _ref("clip_id", "clips"),
        _ref("point_id", "points", "clip_id"),
        UniqueConstraint("owner_team_id", "id"),
        UniqueConstraint("owner_team_id", "clip_id", "id"),
        UniqueConstraint("owner_team_id", "clip_id", "point_id", "id"),
        UniqueConstraint("owner_team_id", "clip_id", "source_event_id"),
        _owned_index("clip_id", "frame"),
        _choices("kind", EVENT_CLASSES_V4),
        _choices("status", EVENT_STATUS),
        _choices("court_side", (-1, 0, 1)),
        _choices("event_source", EVENT_SOURCES),
        CheckConstraint(
            f"subtype IS NULL OR (kind, subtype) IN ({_V4_KIND_SUBTYPE_PAIRS})",
            name="kind_subtype",
        ),
        CheckConstraint(
            "(event_source = 'annotation') = (adopted_in_v4 IS NOT NULL)",
            name="v4_adoption_source",
        ),
        CheckConstraint(
            "adopted_in_v4 IS NOT TRUE OR "
            "(subtype IS NOT NULL AND status IS NOT NULL AND court_side IS NOT NULL "
            "AND frame_uncertainty IS NOT NULL)",
            name="annotation_resolution",
        ),
        CheckConstraint(
            "(adopted_in_v4 IS FALSE AND NOT in_point AND nullif(btrim(excluded_reason), '') IS NOT NULL) "
            "OR (adopted_in_v4 IS TRUE AND excluded_reason IS NULL) "
            "OR (adopted_in_v4 IS NULL AND (NOT in_point OR excluded_reason IS NULL))",
            name="exclusion",
        ),
        CheckConstraint(
            "frame >= 0 AND time_s >= 0 AND frame_uncertainty >= 0",
            name="event_time",
        ),
    )

    clip_id: Mapped[int] = mapped_column(BigInteger)
    point_id: Mapped[int | None] = mapped_column(BigInteger)
    source_event_id: Mapped[str] = mapped_column(Text)
    frame: Mapped[int]
    time_s: Mapped[float] = mapped_column(Double)
    frame_uncertainty: Mapped[int | None]
    kind: Mapped[str] = mapped_column(Text)
    subtype: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str | None] = mapped_column(Text)
    court_side: Mapped[int | None]
    event_source: Mapped[str] = mapped_column(Text)
    adopted_in_v4: Mapped[bool | None]
    in_point: Mapped[bool]
    excluded_reason: Mapped[str | None] = mapped_column(Text)
    provenance_json: Mapped[dict[str, Any]] = mapped_column(
        _JSON, server_default=_EMPTY_OBJECT
    )


class ShotHitterEstimate(_OwnedRecord, Base):
    __tablename__ = "shot_hitter_estimates"
    __table_args__ = (
        _choices("visibility", VISIBILITIES),
        _ref("ball_event_id", "ball_events"),
        UniqueConstraint("owner_team_id", "ball_event_id", "rule_version"),
        _owned_index("resolution_status", "rule_version"),
        _choices("resolution_status", RESOLUTION_STATUSES),
        _choices("selected_landmark", ("left_wrist", "right_wrist", "root")),
    )

    ball_event_id: Mapped[int] = mapped_column(BigInteger)
    rule_version: Mapped[str] = mapped_column(Text)
    resolution_status: Mapped[str] = mapped_column(Text)
    local_player_index: Mapped[int | None]
    local_player_id: Mapped[int | None]
    selected_landmark: Mapped[str | None] = mapped_column(Text)
    best_score: Mapped[float | None] = mapped_column(Double)
    margin_to_second: Mapped[float | None] = mapped_column(Double)
    rejection_reason: Mapped[str | None] = mapped_column(Text)
    candidates_json: Mapped[list[dict[str, Any]]] = mapped_column(
        _JSON, server_default=_EMPTY_ARRAY
    )
    evidence_json: Mapped[dict[str, Any]] = mapped_column(
        _JSON, server_default=_EMPTY_OBJECT
    )


class FlightArc(_OwnedRecord, Base):
    __tablename__ = "flight_arcs"
    __table_args__ = (
        _choices("visibility", VISIBILITIES),
        _ref("clip_id", "clips"),
        _ref("point_id", "points", "clip_id"),
        _ref("shot_event_id", "ball_events", "clip_id"),
        _ref("start_event_id", "ball_events", "clip_id"),
        _ref("end_event_id", "ball_events", "clip_id"),
        UniqueConstraint("owner_team_id", "clip_id", "id"),
        UniqueConstraint("owner_team_id", "clip_id", "arc_index"),
        _owned_index("clip_id", "start_frame"),
        _frame_bounds(),
        _choices("source_kind", GEOMETRY_SOURCES),
        CheckConstraint(
            "arc_index >= 0 AND valid_frame_count >= 0 "
            "AND valid_frame_count <= total_frame_count",
            name="frame_counts",
        ),
    )

    clip_id: Mapped[int] = mapped_column(BigInteger)
    point_id: Mapped[int | None] = mapped_column(BigInteger)
    shot_event_id: Mapped[int | None] = mapped_column(BigInteger)
    arc_index: Mapped[int]
    start_event_id: Mapped[int | None] = mapped_column(BigInteger)
    end_event_id: Mapped[int | None] = mapped_column(BigInteger)
    start_frame: Mapped[int]
    end_frame: Mapped[int]
    valid_frame_count: Mapped[int]
    total_frame_count: Mapped[int]
    source_kind: Mapped[str] = mapped_column(Text)
    quality_json: Mapped[dict[str, Any]] = mapped_column(
        _JSON, server_default=_EMPTY_OBJECT
    )


class ShotGeometryFeature(_OwnedRecord, Base):
    __tablename__ = "shot_geometry_features"
    __table_args__ = (
        _choices("visibility", VISIBILITIES),
        _ref("ball_event_id", "ball_events", "clip_id"),
        _ref("flight_arc_id", "flight_arcs", "clip_id"),
        _ref("profile_id", "court_net_profiles"),
        UniqueConstraint(
            "owner_team_id",
            "ball_event_id",
            "profile_id",
            "definition_version",
            "bin_version",
            postgresql_nulls_not_distinct=True,
        ),
        _owned_index("profile_id", "definition_version"),
        _choices("landing_source_kind", GEOMETRY_SOURCES),
        _choices(
            "landing_result",
            ("in", "out_long", "out_wide", "out_both", "uncertain", "not_applicable"),
        ),
        CheckConstraint(
            "(landing_source_kind IS NOT NULL AND landing_missing_reason IS NULL "
            "AND first_bounce_frame IS NOT NULL) "
            "OR (landing_source_kind IS NULL AND nullif(btrim(landing_missing_reason), '') IS NOT NULL "
            "AND first_bounce_frame IS NULL AND first_bounce_time_s IS NULL "
            "AND first_bounce_x_m IS NULL AND first_bounce_y_m IS NULL AND first_bounce_z_m IS NULL "
            "AND landing_depth_m IS NULL AND landing_lane IS NULL AND landing_course IS NULL "
            "AND line_margin_m IS NULL AND (landing_result IS NULL OR landing_result = 'not_applicable'))",
            name="landing_presence",
        ),
    )

    clip_id: Mapped[int] = mapped_column(BigInteger)
    ball_event_id: Mapped[int] = mapped_column(BigInteger)
    flight_arc_id: Mapped[int | None] = mapped_column(BigInteger)
    profile_id: Mapped[int] = mapped_column(BigInteger)
    definition_version: Mapped[str] = mapped_column(Text)
    contact_frame: Mapped[int]
    contact_time_s: Mapped[float] = mapped_column(Double)
    contact_frame_uncertainty: Mapped[float | None] = mapped_column(Double)
    contact_x_m: Mapped[float | None] = mapped_column(Double)
    contact_y_m: Mapped[float | None] = mapped_column(Double)
    contact_z_m: Mapped[float | None] = mapped_column(Double)
    contact_height_m: Mapped[float | None] = mapped_column(Double)
    outgoing_speed_mps: Mapped[float | None] = mapped_column(Double)
    incoming_speed_mps: Mapped[float | None] = mapped_column(Double)
    speed_window_frames: Mapped[dict[str, Any] | None] = mapped_column(_JSON)
    speed_method: Mapped[str | None] = mapped_column(Text)
    outgoing_heading_rad: Mapped[float | None] = mapped_column(Double)
    first_bounce_frame: Mapped[int | None]
    first_bounce_time_s: Mapped[float | None] = mapped_column(Double)
    first_bounce_x_m: Mapped[float | None] = mapped_column(Double)
    first_bounce_y_m: Mapped[float | None] = mapped_column(Double)
    first_bounce_z_m: Mapped[float | None] = mapped_column(Double)
    landing_depth_m: Mapped[float | None] = mapped_column(Double)
    landing_lane: Mapped[str | None] = mapped_column(Text)
    landing_course: Mapped[str | None] = mapped_column(Text)
    line_margin_m: Mapped[float | None] = mapped_column(Double)
    landing_result: Mapped[str | None] = mapped_column(Text)
    landing_source_kind: Mapped[str | None] = mapped_column(Text)
    landing_missing_reason: Mapped[str | None] = mapped_column(Text)
    bin_version: Mapped[str | None] = mapped_column(Text)
    bins_json: Mapped[dict[str, Any]] = mapped_column(
        _JSON, server_default=_EMPTY_OBJECT
    )
    quality_json: Mapped[dict[str, Any]] = mapped_column(
        _JSON, server_default=_EMPTY_OBJECT
    )


class NetPlaneCrossing(_OwnedRecord, Base):
    __tablename__ = "net_plane_crossings"
    __table_args__ = (
        _choices("visibility", VISIBILITIES),
        _ref("flight_arc_id", "flight_arcs", "clip_id"),
        _ref("ball_event_id", "ball_events", "clip_id"),
        ForeignKeyConstraint(
            ["owner_team_id", "profile_source_kind", "court_net_profile_id"],
            [
                "court_net_profiles.owner_team_id",
                "court_net_profiles.source_kind",
                "court_net_profiles.id",
            ],
        ),
        UniqueConstraint("owner_team_id", "flight_arc_id", "crossing_index"),
        _owned_index("ball_event_id", "is_primary"),
        _choices(
            "status",
            ("crossed", "no_crossing_confirmed", "unknown_gap", "not_applicable"),
        ),
        _choices("direction_y", (-1, 1)),
        _choices("source_kind", GEOMETRY_SOURCES),
        _choices("profile_source_kind", PROFILE_SOURCES),
        CheckConstraint(
            "status = 'crossed' OR (NOT is_primary AND crossing_in_net_span IS NULL "
            "AND fractional_frame IS NULL AND time_s IS NULL AND time_to_net_s IS NULL "
            "AND x_m IS NULL AND y_m IS NULL AND z_center_m IS NULL AND direction_y IS NULL "
            "AND net_height_m IS NULL AND net_clearance_center_m IS NULL "
            "AND net_clearance_bottom_m IS NULL)",
            name="non_crossing",
        ),
        CheckConstraint(
            "status <> 'crossed' OR (fractional_frame IS NOT NULL AND time_s IS NOT NULL "
            "AND x_m IS NOT NULL AND y_m IS NOT NULL AND z_center_m IS NOT NULL "
            "AND court_net_profile_id IS NOT NULL)",
            name="crossing_observation",
        ),
        CheckConstraint(
            "(court_net_profile_id IS NULL) = (profile_source_kind IS NULL)",
            name="profile_provenance",
        ),
    )

    clip_id: Mapped[int] = mapped_column(BigInteger)
    flight_arc_id: Mapped[int] = mapped_column(BigInteger)
    ball_event_id: Mapped[int | None] = mapped_column(BigInteger)
    crossing_index: Mapped[int]
    is_primary: Mapped[bool] = mapped_column(server_default=false())
    status: Mapped[str] = mapped_column(Text)
    crossing_in_net_span: Mapped[bool | None]
    fractional_frame: Mapped[float | None] = mapped_column(Double)
    time_s: Mapped[float | None] = mapped_column(Double)
    time_to_net_s: Mapped[float | None] = mapped_column(Double)
    x_m: Mapped[float | None] = mapped_column(Double)
    y_m: Mapped[float | None] = mapped_column(Double)
    z_center_m: Mapped[float | None] = mapped_column(Double)
    direction_y: Mapped[int | None]
    net_height_m: Mapped[float | None] = mapped_column(Double)
    net_clearance_center_m: Mapped[float | None] = mapped_column(Double)
    net_clearance_bottom_m: Mapped[float | None] = mapped_column(Double)
    court_net_profile_id: Mapped[int | None] = mapped_column(BigInteger)
    profile_source_kind: Mapped[str | None] = mapped_column(Text)
    source_kind: Mapped[str] = mapped_column(Text)
    used_frames_json: Mapped[list[int]] = mapped_column(
        _JSON, server_default=_EMPTY_ARRAY
    )
    fit_error_m: Mapped[float | None] = mapped_column(Double)


class PlayerIdentityMap(_OwnedRecord, Base):
    __tablename__ = "player_identity_maps"
    __table_args__ = (
        _choices("visibility", VISIBILITIES),
        _ref("clip_id", "clips"),
        UniqueConstraint("owner_team_id", "clip_id", "id"),
        UniqueConstraint(
            "owner_team_id",
            "clip_id",
            "local_player_id",
            "start_frame",
            "mapping_version",
        ),
        _owned_index("clip_id", "local_player_id"),
        _choices("resolution_status", RESOLUTION_STATUSES),
        _sha256("source_sha256"),
        _frame_bounds(),
    )

    clip_id: Mapped[int] = mapped_column(BigInteger)
    local_player_id: Mapped[int]
    global_player_id: Mapped[str | None] = mapped_column(Text)
    start_frame: Mapped[int]
    end_frame: Mapped[int]
    mapping_version: Mapped[str] = mapped_column(Text)
    resolution_status: Mapped[str] = mapped_column(Text)
    source_sha256: Mapped[str | None] = mapped_column(Text)
    evidence_json: Mapped[dict[str, Any]] = mapped_column(
        _JSON, server_default=_EMPTY_OBJECT
    )


class TeamPairAssignment(_OwnedRecord, Base):
    __tablename__ = "team_pair_assignments"
    __table_args__ = (
        _choices("visibility", VISIBILITIES),
        _ref("clip_id", "clips"),
        _ref("first_identity_map_id", "player_identity_maps", "clip_id"),
        _ref("second_identity_map_id", "player_identity_maps", "clip_id"),
        _owned_index("clip_id", "start_frame"),
        _choices("scoring_team", ("A", "B")),
        _choices("court_side", (-1, 1)),
        _choices("resolution_status", RESOLUTION_STATUSES),
        _frame_bounds(),
        CheckConstraint(
            "second_identity_map_id IS NULL OR "
            "(first_identity_map_id IS NOT NULL AND first_identity_map_id <> second_identity_map_id)",
            name="pair_members",
        ),
    )

    clip_id: Mapped[int] = mapped_column(BigInteger)
    pair_key: Mapped[str | None] = mapped_column(Text)
    scoring_team: Mapped[str | None] = mapped_column(Text)
    court_side: Mapped[int | None]
    start_frame: Mapped[int]
    end_frame: Mapped[int]
    first_identity_map_id: Mapped[int | None] = mapped_column(BigInteger)
    second_identity_map_id: Mapped[int | None] = mapped_column(BigInteger)
    assignment_version: Mapped[str] = mapped_column(Text)
    resolution_status: Mapped[str] = mapped_column(Text)
    evidence_json: Mapped[dict[str, Any]] = mapped_column(
        _JSON, server_default=_EMPTY_OBJECT
    )


class ServeAttempt(_OwnedRecord, Base):
    __tablename__ = "serve_attempts"
    __table_args__ = (
        _choices("visibility", VISIBILITIES),
        _ref("point_id", "points", "clip_id"),
        _ref("ball_event_id", "ball_events", "clip_id", "point_id"),
        UniqueConstraint("owner_team_id", "point_id", "attempt_index"),
        _owned_index("ball_event_id", "serve_number"),
        _choices("serve_number", (1, 2)),
        _choices("result", ("in", "fault", "let", "unknown")),
        CheckConstraint("attempt_index >= 0 AND replay_index >= 0", name="order"),
        CheckConstraint("result = 'fault' OR fault_cause IS NULL", name="fault_cause"),
    )

    clip_id: Mapped[int] = mapped_column(BigInteger)
    point_id: Mapped[int] = mapped_column(BigInteger)
    ball_event_id: Mapped[int | None] = mapped_column(BigInteger)
    attempt_index: Mapped[int]
    serve_number: Mapped[int | None]
    replay_index: Mapped[int | None]
    result: Mapped[str] = mapped_column(Text, server_default="unknown")
    fault_cause: Mapped[str | None] = mapped_column(Text)
    source_kind: Mapped[str] = mapped_column(Text)
    evidence_json: Mapped[dict[str, Any]] = mapped_column(
        _JSON, server_default=_EMPTY_OBJECT
    )


def _metric_count_relation() -> CheckConstraint:
    exact_counts = ", ".join(
        repr(metric.metric_id)
        for metric in METRIC_DEFINITIONS.values()
        if metric.count_relation == "exact_count"
    )
    multi_per_unit = ", ".join(
        repr(metric.metric_id)
        for metric in METRIC_DEFINITIONS.values()
        if metric.count_relation == "multi_per_unit"
    )
    return CheckConstraint(
        f"definition_version <> {DEFINITION_VERSION!r} OR "
        f"((metric_id IN ({multi_per_unit}) OR numerator <= denominator) "
        f"AND (metric_id NOT IN ({exact_counts}) OR numerator = denominator))",
        name="count_relation",
    )


class MetricSnapshot(_OwnedRecord, Base):
    __tablename__ = "metric_snapshots"
    __table_args__ = (
        _choices("visibility", VISIBILITIES),
        _ref("match_context_id", "match_contexts"),
        _owned_index("match_context_id", "metric_id"),
        _choices("status", get_args(MetricStatus)),
        _metric_count_relation(),
        CheckConstraint(
            "jsonb_typeof(source_ids) = 'object' AND "
            "(denominator = 0 OR source_ids <> '{}'::jsonb)",
            name="source_ids",
        ),
        CheckConstraint(
            "numerator >= 0 AND denominator >= 0 AND unknown_count >= 0 "
            "AND resolved_count = denominator "
            "AND candidate_count >= resolved_count + unknown_count",
            name="counts",
        ),
        CheckConstraint(
            "(denominator = 0 AND numerator = 0 AND value IS NULL AND status = 'unavailable') "
            "OR (denominator > 0 AND value IS NOT NULL AND "
            "((unknown_count = 0 AND status = 'complete') "
            "OR (unknown_count > 0 AND status = 'partial')))",
            name="availability",
        ),
    )

    match_context_id: Mapped[int] = mapped_column(BigInteger)
    metric_id: Mapped[str] = mapped_column(Text)
    definition_version: Mapped[str] = mapped_column(Text)
    subject_type: Mapped[str] = mapped_column(Text)
    subject_id: Mapped[str] = mapped_column(Text)
    filters: Mapped[dict[str, Any]] = mapped_column(_JSON)
    bin_version: Mapped[str | None] = mapped_column(Text)
    numerator: Mapped[int]
    denominator: Mapped[int]
    value: Mapped[float | None] = mapped_column(Double)
    candidate_count: Mapped[int]
    resolved_count: Mapped[int]
    unknown_count: Mapped[int]
    excluded_by_reason: Mapped[dict[str, int]] = mapped_column(_JSON)
    status: Mapped[MetricStatus] = mapped_column(Text)
    source_ids: Mapped[dict[str, list[int | str]]] = mapped_column(_JSON)


def _validate_metric_snapshot(
    _mapper: Mapper[MetricSnapshot], _connection: Connection, snapshot: MetricSnapshot
) -> None:
    MetricRecord.from_record(
        {name: getattr(snapshot, name) for name in MetricRecord.__dataclass_fields__}
    )


event.listen(MetricSnapshot, "before_insert", _validate_metric_snapshot)
event.listen(MetricSnapshot, "before_update", _validate_metric_snapshot)
