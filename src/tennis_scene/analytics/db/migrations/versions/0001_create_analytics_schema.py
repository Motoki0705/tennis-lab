"""Create analytics schema."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_organizations")),
        sa.UniqueConstraint("slug", name=op.f("uq_organizations_slug")),
    )
    op.create_table(
        "users",
        sa.Column("external_subject", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=True),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column(
            "is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False
        ),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("external_subject", name=op.f("uq_users_external_subject")),
    )
    op.create_table(
        "teams",
        sa.Column("organization_id", sa.BigInteger(), nullable=False),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["organizations.id"],
            name=op.f("fk_teams_organization_id"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_teams")),
        sa.UniqueConstraint(
            "organization_id", "slug", name=op.f("uq_teams_organization_id_slug")
        ),
    )
    op.create_table(
        "court_net_profiles",
        sa.Column("profile_key", sa.Text(), nullable=False),
        sa.Column("sport_type", sa.Text(), nullable=False),
        sa.Column("net_height_center_m", sa.Double(), nullable=False),
        sa.Column("net_height_post_m", sa.Double(), nullable=False),
        sa.Column("net_profile_kind", sa.Text(), nullable=False),
        sa.Column("net_post_offset_x_m", sa.Double(), nullable=False),
        sa.Column("ball_radius_m", sa.Double(), nullable=False),
        sa.Column("source_kind", sa.Text(), nullable=False),
        sa.Column(
            "parameters_json",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("owner_team_id", sa.BigInteger(), nullable=False),
        sa.Column("visibility", sa.Text(), server_default="team", nullable=False),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "net_profile_kind IN ('flat', 'itf_parabolic', 'measured_curve')",
            name=op.f("ck_court_net_profiles_net_profile_kind"),
        ),
        sa.CheckConstraint(
            "source_kind IN ('rule_default', 'measured', 'custom')",
            name=op.f("ck_court_net_profiles_source_kind"),
        ),
        sa.CheckConstraint(
            "sport_type IN ('soft_tennis', 'tennis')",
            name=op.f("ck_court_net_profiles_sport_type"),
        ),
        sa.CheckConstraint(
            "visibility IN ('private', 'team', 'org', 'public')",
            name=op.f("ck_court_net_profiles_visibility"),
        ),
        sa.CheckConstraint(
            "net_height_center_m > 0 AND net_height_post_m > 0 AND net_post_offset_x_m > 0 AND ball_radius_m > 0",
            name=op.f("ck_court_net_profiles_positive_dimensions"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id"],
            ["teams.id"],
            name=op.f("fk_court_net_profiles_owner_team_id"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_court_net_profiles")),
        sa.UniqueConstraint(
            "owner_team_id", "id", name=op.f("uq_court_net_profiles_owner_team_id_id")
        ),
        sa.UniqueConstraint(
            "owner_team_id",
            "profile_key",
            name=op.f("uq_court_net_profiles_owner_team_id_profile_key"),
        ),
        sa.UniqueConstraint(
            "owner_team_id",
            "source_kind",
            "id",
            name=op.f("uq_court_net_profiles_owner_team_id_source_kind_id"),
        ),
        sa.UniqueConstraint(
            "owner_team_id",
            "sport_type",
            "id",
            name=op.f("uq_court_net_profiles_owner_team_id_sport_type_id"),
        ),
    )
    op.create_index(
        op.f("ix_court_net_profiles_owner_team_id_sport_type"),
        "court_net_profiles",
        ["owner_team_id", "sport_type"],
        unique=False,
    )
    op.create_table(
        "ingest_runs",
        sa.Column("dataset_id", sa.Text(), nullable=False),
        sa.Column("video_id", sa.Text(), nullable=False),
        sa.Column("v4_manifest_sha256", sa.Text(), nullable=False),
        sa.Column("projects_json_sha256", sa.Text(), nullable=False),
        sa.Column("manual_supplement_sha256", sa.Text(), nullable=True),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("definition_version", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "summary_json",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("owner_team_id", sa.BigInteger(), nullable=False),
        sa.Column("visibility", sa.Text(), server_default="team", nullable=False),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "manual_supplement_sha256 ~ '^[a-f0-9]{64}$'",
            name=op.f("ck_ingest_runs_manual_supplement_sha256"),
        ),
        sa.CheckConstraint(
            "projects_json_sha256 ~ '^[a-f0-9]{64}$'",
            name=op.f("ck_ingest_runs_projects_json_sha256"),
        ),
        sa.CheckConstraint(
            "v4_manifest_sha256 ~ '^[a-f0-9]{64}$'",
            name=op.f("ck_ingest_runs_v4_manifest_sha256"),
        ),
        sa.CheckConstraint(
            "visibility IN ('private', 'team', 'org', 'public')",
            name=op.f("ck_ingest_runs_visibility"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id"], ["teams.id"], name=op.f("fk_ingest_runs_owner_team_id")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingest_runs")),
        sa.UniqueConstraint(
            "owner_team_id",
            "dataset_id",
            "video_id",
            "id",
            name=op.f("uq_ingest_runs_owner_team_id_dataset_id_video_id_id"),
        ),
        sa.UniqueConstraint(
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
    )
    op.create_index(
        op.f("ix_ingest_runs_owner_team_id_dataset_id_video_id"),
        "ingest_runs",
        ["owner_team_id", "dataset_id", "video_id"],
        unique=False,
    )
    op.create_table(
        "team_memberships",
        sa.Column("team_id", sa.BigInteger(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"], name=op.f("fk_team_memberships_team_id")
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_team_memberships_user_id")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_team_memberships")),
        sa.UniqueConstraint(
            "team_id", "user_id", name=op.f("uq_team_memberships_team_id_user_id")
        ),
    )
    op.create_index(
        op.f("ix_team_memberships_user_id"),
        "team_memberships",
        ["user_id"],
        unique=False,
    )
    op.create_table(
        "match_contexts",
        sa.Column("ingest_run_id", sa.BigInteger(), nullable=False),
        sa.Column("dataset_id", sa.Text(), nullable=False),
        sa.Column("video_id", sa.Text(), nullable=False),
        sa.Column("match_key", sa.Text(), nullable=False),
        sa.Column("sport_type", sa.Text(), nullable=False),
        sa.Column(
            "rules_json",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("side_a_name", sa.Text(), nullable=True),
        sa.Column("side_b_name", sa.Text(), nullable=True),
        sa.Column("first_server", sa.Text(), nullable=True),
        sa.Column("court_net_profile_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "time_conversion_json",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("owner_team_id", sa.BigInteger(), nullable=False),
        sa.Column("visibility", sa.Text(), server_default="team", nullable=False),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "first_server IN ('A', 'B')", name=op.f("ck_match_contexts_first_server")
        ),
        sa.CheckConstraint(
            "sport_type IN ('soft_tennis', 'tennis')",
            name=op.f("ck_match_contexts_sport_type"),
        ),
        sa.CheckConstraint(
            "visibility IN ('private', 'team', 'org', 'public')",
            name=op.f("ck_match_contexts_visibility"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "dataset_id", "video_id", "ingest_run_id"],
            [
                "ingest_runs.owner_team_id",
                "ingest_runs.dataset_id",
                "ingest_runs.video_id",
                "ingest_runs.id",
            ],
            name=op.f(
                "fk_match_contexts_owner_team_id_dataset_id_video_id_ingest_run_id"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "sport_type", "court_net_profile_id"],
            [
                "court_net_profiles.owner_team_id",
                "court_net_profiles.sport_type",
                "court_net_profiles.id",
            ],
            name=op.f(
                "fk_match_contexts_owner_team_id_sport_type_court_net_profile_id"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id"],
            ["teams.id"],
            name=op.f("fk_match_contexts_owner_team_id"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_match_contexts")),
        sa.UniqueConstraint(
            "owner_team_id", "id", name=op.f("uq_match_contexts_owner_team_id_id")
        ),
        sa.UniqueConstraint(
            "owner_team_id",
            "ingest_run_id",
            "match_key",
            name=op.f("uq_match_contexts_owner_team_id_ingest_run_id_match_key"),
        ),
    )
    op.create_index(
        op.f("ix_match_contexts_owner_team_id_dataset_id_video_id"),
        "match_contexts",
        ["owner_team_id", "dataset_id", "video_id"],
        unique=False,
    )
    op.create_table(
        "clips",
        sa.Column("match_context_id", sa.BigInteger(), nullable=False),
        sa.Column("clip_id", sa.Text(), nullable=False),
        sa.Column("clip_name", sa.Text(), nullable=False),
        sa.Column("split", sa.Text(), server_default="unassigned", nullable=False),
        sa.Column("global_start_sec", sa.Double(), nullable=False),
        sa.Column("global_end_sec", sa.Double(), nullable=False),
        sa.Column("fps", sa.Double(), nullable=False),
        sa.Column("num_frames", sa.Integer(), nullable=False),
        sa.Column("scene_schema_version", sa.Integer(), nullable=False),
        sa.Column("scene_npz_sha256", sa.Text(), nullable=False),
        sa.Column("annotation_sha256", sa.Text(), nullable=False),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("owner_team_id", sa.BigInteger(), nullable=False),
        sa.Column("visibility", sa.Text(), server_default="team", nullable=False),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "annotation_sha256 ~ '^[a-f0-9]{64}$'",
            name=op.f("ck_clips_annotation_sha256"),
        ),
        sa.CheckConstraint(
            "scene_npz_sha256 ~ '^[a-f0-9]{64}$'",
            name=op.f("ck_clips_scene_npz_sha256"),
        ),
        sa.CheckConstraint(
            "split IN ('train', 'val', 'test', 'unassigned')",
            name=op.f("ck_clips_split"),
        ),
        sa.CheckConstraint(
            "visibility IN ('private', 'team', 'org', 'public')",
            name=op.f("ck_clips_visibility"),
        ),
        sa.CheckConstraint(
            "global_start_sec >= 0 AND global_start_sec < global_end_sec AND fps > 0 AND num_frames > 0",
            name=op.f("ck_clips_timebase"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "match_context_id"],
            ["match_contexts.owner_team_id", "match_contexts.id"],
            name=op.f("fk_clips_owner_team_id_match_context_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id"], ["teams.id"], name=op.f("fk_clips_owner_team_id")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_clips")),
        sa.UniqueConstraint(
            "owner_team_id", "id", name=op.f("uq_clips_owner_team_id_id")
        ),
        sa.UniqueConstraint(
            "owner_team_id",
            "match_context_id",
            "clip_id",
            name=op.f("uq_clips_owner_team_id_match_context_id_clip_id"),
        ),
        sa.UniqueConstraint(
            "owner_team_id",
            "match_context_id",
            "id",
            name=op.f("uq_clips_owner_team_id_match_context_id_id"),
        ),
    )
    op.create_index(
        op.f("ix_clips_owner_team_id_match_context_id_global_start_sec"),
        "clips",
        ["owner_team_id", "match_context_id", "global_start_sec"],
        unique=False,
    )
    op.create_table(
        "metric_snapshots",
        sa.Column("match_context_id", sa.BigInteger(), nullable=False),
        sa.Column("metric_id", sa.Text(), nullable=False),
        sa.Column("definition_version", sa.Text(), nullable=False),
        sa.Column("subject_type", sa.Text(), nullable=False),
        sa.Column("subject_id", sa.Text(), nullable=False),
        sa.Column(
            "filters",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("bin_version", sa.Text(), nullable=True),
        sa.Column("numerator", sa.Integer(), nullable=False),
        sa.Column("denominator", sa.Integer(), nullable=False),
        sa.Column("value", sa.Double(), nullable=True),
        sa.Column("candidate_count", sa.Integer(), nullable=False),
        sa.Column("resolved_count", sa.Integer(), nullable=False),
        sa.Column("unknown_count", sa.Integer(), nullable=False),
        sa.Column(
            "excluded_by_reason",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "source_ids",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("owner_team_id", sa.BigInteger(), nullable=False),
        sa.Column("visibility", sa.Text(), server_default="team", nullable=False),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(denominator = 0 AND numerator = 0 AND value IS NULL AND status = 'unavailable') OR (denominator > 0 AND value IS NOT NULL AND ((unknown_count = 0 AND status = 'complete') OR (unknown_count > 0 AND status = 'partial')))",
            name=op.f("ck_metric_snapshots_availability"),
        ),
        sa.CheckConstraint(
            "definition_version <> '1' OR ((metric_id IN ('rally.contact_count', 'rally.completed_shot_count') OR numerator <= denominator) AND (metric_id NOT IN ('point.played_count', 'shot.count') OR numerator = denominator))",
            name=op.f("ck_metric_snapshots_count_relation"),
        ),
        sa.CheckConstraint(
            "jsonb_typeof(source_ids) = 'object' AND (denominator = 0 OR source_ids <> '{}'::jsonb)",
            name=op.f("ck_metric_snapshots_source_ids"),
        ),
        sa.CheckConstraint(
            "status IN ('complete', 'partial', 'unavailable')",
            name=op.f("ck_metric_snapshots_status"),
        ),
        sa.CheckConstraint(
            "visibility IN ('private', 'team', 'org', 'public')",
            name=op.f("ck_metric_snapshots_visibility"),
        ),
        sa.CheckConstraint(
            "numerator >= 0 AND denominator >= 0 AND unknown_count >= 0 AND resolved_count = denominator AND candidate_count >= resolved_count + unknown_count",
            name=op.f("ck_metric_snapshots_counts"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "match_context_id"],
            ["match_contexts.owner_team_id", "match_contexts.id"],
            name=op.f("fk_metric_snapshots_owner_team_id_match_context_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id"],
            ["teams.id"],
            name=op.f("fk_metric_snapshots_owner_team_id"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_metric_snapshots")),
    )
    op.create_index(
        op.f("ix_metric_snapshots_owner_team_id_match_context_id_metric_id"),
        "metric_snapshots",
        ["owner_team_id", "match_context_id", "metric_id"],
        unique=False,
    )
    op.create_table(
        "player_identity_maps",
        sa.Column("clip_id", sa.BigInteger(), nullable=False),
        sa.Column("local_player_id", sa.Integer(), nullable=False),
        sa.Column("global_player_id", sa.Text(), nullable=True),
        sa.Column("start_frame", sa.Integer(), nullable=False),
        sa.Column("end_frame", sa.Integer(), nullable=False),
        sa.Column("mapping_version", sa.Text(), nullable=False),
        sa.Column("resolution_status", sa.Text(), nullable=False),
        sa.Column("source_sha256", sa.Text(), nullable=True),
        sa.Column(
            "evidence_json",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("owner_team_id", sa.BigInteger(), nullable=False),
        sa.Column("visibility", sa.Text(), server_default="team", nullable=False),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "resolution_status IN ('resolved', 'ambiguous', 'unresolved')",
            name=op.f("ck_player_identity_maps_resolution_status"),
        ),
        sa.CheckConstraint(
            "source_sha256 ~ '^[a-f0-9]{64}$'",
            name=op.f("ck_player_identity_maps_source_sha256"),
        ),
        sa.CheckConstraint(
            "visibility IN ('private', 'team', 'org', 'public')",
            name=op.f("ck_player_identity_maps_visibility"),
        ),
        sa.CheckConstraint(
            "start_frame >= 0 AND end_frame >= 0 AND start_frame <= end_frame",
            name=op.f("ck_player_identity_maps_frame_bounds"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "clip_id"],
            ["clips.owner_team_id", "clips.id"],
            name=op.f("fk_player_identity_maps_owner_team_id_clip_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id"],
            ["teams.id"],
            name=op.f("fk_player_identity_maps_owner_team_id"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_player_identity_maps")),
        sa.UniqueConstraint(
            "owner_team_id",
            "clip_id",
            "id",
            name=op.f("uq_player_identity_maps_owner_team_id_clip_id_id"),
        ),
        sa.UniqueConstraint(
            "owner_team_id",
            "clip_id",
            "local_player_id",
            "start_frame",
            "mapping_version",
            name=op.f(
                "uq_player_identity_maps_owner_team_id_clip_id_local_player_id_start_frame_mapping_version"
            ),
        ),
    )
    op.create_index(
        op.f("ix_player_identity_maps_owner_team_id_clip_id_local_player_id"),
        "player_identity_maps",
        ["owner_team_id", "clip_id", "local_player_id"],
        unique=False,
    )
    op.create_table(
        "points",
        sa.Column("clip_id", sa.BigInteger(), nullable=False),
        sa.Column("match_context_id", sa.BigInteger(), nullable=False),
        sa.Column("point_key", sa.Text(), nullable=False),
        sa.Column("match_point_key", sa.Text(), nullable=True),
        sa.Column("start_frame", sa.Integer(), nullable=True),
        sa.Column("end_frame", sa.Integer(), nullable=True),
        sa.Column("completeness", sa.Text(), server_default="unknown", nullable=False),
        sa.Column("score_status", sa.Text(), server_default="unknown", nullable=False),
        sa.Column(
            "score_before_json",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "score_after_json",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("server_team", sa.Text(), nullable=True),
        sa.Column("winner_team", sa.Text(), nullable=True),
        sa.Column("score_source", sa.Text(), nullable=True),
        sa.Column(
            "terminal_cause", sa.Text(), server_default="unknown", nullable=False
        ),
        sa.Column(
            "terminal_cause_source", sa.Text(), server_default="unknown", nullable=False
        ),
        sa.Column(
            "error_assessment", sa.Text(), server_default="undetermined", nullable=False
        ),
        sa.Column("error_assessment_source", sa.Text(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("owner_team_id", sa.BigInteger(), nullable=False),
        sa.Column("visibility", sa.Text(), server_default="team", nullable=False),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(terminal_cause, terminal_cause_source) IN (('net_error', 'manual_confirmed'), ('out_long', 'manual_confirmed'), ('out_wide', 'manual_confirmed'), ('out_both', 'manual_confirmed'), ('double_fault', 'manual_confirmed'), ('unreturned', 'manual_confirmed'), ('other', 'manual_confirmed'), ('net_error', 'derived_v4_score'), ('double_fault', 'derived_v4_score'), ('unknown', 'unknown'))",
            name=op.f("ck_points_terminal_cause_provenance"),
        ),
        sa.CheckConstraint(
            "completeness IN ('complete', 'truncated_start', 'truncated_end', 'truncated_both', 'unknown')",
            name=op.f("ck_points_completeness"),
        ),
        sa.CheckConstraint(
            "error_assessment IN ('forced', 'unforced', 'undetermined')",
            name=op.f("ck_points_error_assessment"),
        ),
        sa.CheckConstraint(
            "match_point_key IS NULL OR btrim(match_point_key) <> ''",
            name=op.f("ck_points_match_point_key"),
        ),
        sa.CheckConstraint(
            "score_status IN ('confirmed', 'auto', 'unknown', 'not_point', 'multiple_points')",
            name=op.f("ck_points_score_status"),
        ),
        sa.CheckConstraint(
            "server_team IN ('A', 'B')", name=op.f("ck_points_server_team")
        ),
        sa.CheckConstraint(
            "visibility IN ('private', 'team', 'org', 'public')",
            name=op.f("ck_points_visibility"),
        ),
        sa.CheckConstraint(
            "winner_team IN ('A', 'B')", name=op.f("ck_points_winner_team")
        ),
        sa.CheckConstraint(
            "start_frame >= 0 AND end_frame >= 0 AND start_frame <= end_frame",
            name=op.f("ck_points_frame_bounds"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "match_context_id", "clip_id"],
            ["clips.owner_team_id", "clips.match_context_id", "clips.id"],
            name=op.f("fk_points_owner_team_id_match_context_id_clip_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id"], ["teams.id"], name=op.f("fk_points_owner_team_id")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_points")),
        sa.UniqueConstraint(
            "owner_team_id",
            "clip_id",
            "id",
            name=op.f("uq_points_owner_team_id_clip_id_id"),
        ),
        sa.UniqueConstraint(
            "owner_team_id",
            "clip_id",
            "point_key",
            name=op.f("uq_points_owner_team_id_clip_id_point_key"),
        ),
    )
    op.create_index(
        op.f("ix_points_owner_team_id_clip_id_start_frame"),
        "points",
        ["owner_team_id", "clip_id", "start_frame"],
        unique=False,
    )
    op.create_index(
        op.f("ix_points_owner_team_id_match_context_id_match_point_key"),
        "points",
        ["owner_team_id", "match_context_id", "match_point_key"],
        unique=False,
    )
    op.create_table(
        "ball_events",
        sa.Column("clip_id", sa.BigInteger(), nullable=False),
        sa.Column("point_id", sa.BigInteger(), nullable=True),
        sa.Column("source_event_id", sa.Text(), nullable=False),
        sa.Column("frame", sa.Integer(), nullable=False),
        sa.Column("time_s", sa.Double(), nullable=False),
        sa.Column("frame_uncertainty", sa.Integer(), nullable=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("subtype", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=True),
        sa.Column("court_side", sa.Integer(), nullable=True),
        sa.Column("event_source", sa.Text(), nullable=False),
        sa.Column("adopted_in_v4", sa.Boolean(), nullable=True),
        sa.Column("in_point", sa.Boolean(), nullable=False),
        sa.Column("excluded_reason", sa.Text(), nullable=True),
        sa.Column(
            "provenance_json",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("owner_team_id", sa.BigInteger(), nullable=False),
        sa.Column("visibility", sa.Text(), server_default="team", nullable=False),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(adopted_in_v4 IS FALSE AND NOT in_point AND nullif(btrim(excluded_reason), '') IS NOT NULL) OR (adopted_in_v4 IS TRUE AND excluded_reason IS NULL) OR (adopted_in_v4 IS NULL AND (NOT in_point OR excluded_reason IS NULL))",
            name=op.f("ck_ball_events_exclusion"),
        ),
        sa.CheckConstraint(
            "(event_source = 'annotation') = (adopted_in_v4 IS NOT NULL)",
            name=op.f("ck_ball_events_v4_adoption_source"),
        ),
        sa.CheckConstraint(
            "event_source IN ('annotation', 'rule', 'refiner')",
            name=op.f("ck_ball_events_event_source"),
        ),
        sa.CheckConstraint(
            "kind IN ('shot', 'bounce', 'net', 'unknown')",
            name=op.f("ck_ball_events_kind"),
        ),
        sa.CheckConstraint(
            "status IN ('certain', 'probable', 'uncertain')",
            name=op.f("ck_ball_events_status"),
        ),
        sa.CheckConstraint(
            "subtype IS NULL OR (kind, subtype) IN (('shot', 'serve_overhead'), ('shot', 'serve_underhand'), ('shot', 'groundstroke_forehand'), ('shot', 'groundstroke_backhand'), ('shot', 'volley_forehand'), ('shot', 'volley_backhand'), ('shot', 'volley_center'), ('shot', 'smash'), ('bounce', 'bounce'), ('net', 'service_let'), ('net', 'net_in'), ('net', 'net_fault'), ('unknown', 'unknown'))",
            name=op.f("ck_ball_events_kind_subtype"),
        ),
        sa.CheckConstraint(
            "visibility IN ('private', 'team', 'org', 'public')",
            name=op.f("ck_ball_events_visibility"),
        ),
        sa.CheckConstraint(
            "adopted_in_v4 IS NOT TRUE OR (subtype IS NOT NULL AND status IS NOT NULL AND court_side IS NOT NULL AND frame_uncertainty IS NOT NULL)",
            name=op.f("ck_ball_events_annotation_resolution"),
        ),
        sa.CheckConstraint(
            "court_side IN (-1, 0, 1)", name=op.f("ck_ball_events_court_side")
        ),
        sa.CheckConstraint(
            "frame >= 0 AND time_s >= 0 AND frame_uncertainty >= 0",
            name=op.f("ck_ball_events_event_time"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "clip_id", "point_id"],
            ["points.owner_team_id", "points.clip_id", "points.id"],
            name=op.f("fk_ball_events_owner_team_id_clip_id_point_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "clip_id"],
            ["clips.owner_team_id", "clips.id"],
            name=op.f("fk_ball_events_owner_team_id_clip_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id"], ["teams.id"], name=op.f("fk_ball_events_owner_team_id")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ball_events")),
        sa.UniqueConstraint(
            "owner_team_id",
            "clip_id",
            "id",
            name=op.f("uq_ball_events_owner_team_id_clip_id_id"),
        ),
        sa.UniqueConstraint(
            "owner_team_id",
            "clip_id",
            "point_id",
            "id",
            name=op.f("uq_ball_events_owner_team_id_clip_id_point_id_id"),
        ),
        sa.UniqueConstraint(
            "owner_team_id",
            "clip_id",
            "source_event_id",
            name=op.f("uq_ball_events_owner_team_id_clip_id_source_event_id"),
        ),
        sa.UniqueConstraint(
            "owner_team_id", "id", name=op.f("uq_ball_events_owner_team_id_id")
        ),
    )
    op.create_index(
        op.f("ix_ball_events_owner_team_id_clip_id_frame"),
        "ball_events",
        ["owner_team_id", "clip_id", "frame"],
        unique=False,
    )
    op.create_table(
        "team_pair_assignments",
        sa.Column("clip_id", sa.BigInteger(), nullable=False),
        sa.Column("pair_key", sa.Text(), nullable=True),
        sa.Column("scoring_team", sa.Text(), nullable=True),
        sa.Column("court_side", sa.Integer(), nullable=True),
        sa.Column("start_frame", sa.Integer(), nullable=False),
        sa.Column("end_frame", sa.Integer(), nullable=False),
        sa.Column("first_identity_map_id", sa.BigInteger(), nullable=True),
        sa.Column("second_identity_map_id", sa.BigInteger(), nullable=True),
        sa.Column("assignment_version", sa.Text(), nullable=False),
        sa.Column("resolution_status", sa.Text(), nullable=False),
        sa.Column(
            "evidence_json",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("owner_team_id", sa.BigInteger(), nullable=False),
        sa.Column("visibility", sa.Text(), server_default="team", nullable=False),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "resolution_status IN ('resolved', 'ambiguous', 'unresolved')",
            name=op.f("ck_team_pair_assignments_resolution_status"),
        ),
        sa.CheckConstraint(
            "scoring_team IN ('A', 'B')",
            name=op.f("ck_team_pair_assignments_scoring_team"),
        ),
        sa.CheckConstraint(
            "visibility IN ('private', 'team', 'org', 'public')",
            name=op.f("ck_team_pair_assignments_visibility"),
        ),
        sa.CheckConstraint(
            "court_side IN (-1, 1)", name=op.f("ck_team_pair_assignments_court_side")
        ),
        sa.CheckConstraint(
            "second_identity_map_id IS NULL OR (first_identity_map_id IS NOT NULL AND first_identity_map_id <> second_identity_map_id)",
            name=op.f("ck_team_pair_assignments_pair_members"),
        ),
        sa.CheckConstraint(
            "start_frame >= 0 AND end_frame >= 0 AND start_frame <= end_frame",
            name=op.f("ck_team_pair_assignments_frame_bounds"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "clip_id", "first_identity_map_id"],
            [
                "player_identity_maps.owner_team_id",
                "player_identity_maps.clip_id",
                "player_identity_maps.id",
            ],
            name=op.f(
                "fk_team_pair_assignments_owner_team_id_clip_id_first_identity_map_id"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "clip_id", "second_identity_map_id"],
            [
                "player_identity_maps.owner_team_id",
                "player_identity_maps.clip_id",
                "player_identity_maps.id",
            ],
            name=op.f(
                "fk_team_pair_assignments_owner_team_id_clip_id_second_identity_map_id"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "clip_id"],
            ["clips.owner_team_id", "clips.id"],
            name=op.f("fk_team_pair_assignments_owner_team_id_clip_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id"],
            ["teams.id"],
            name=op.f("fk_team_pair_assignments_owner_team_id"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_team_pair_assignments")),
    )
    op.create_index(
        op.f("ix_team_pair_assignments_owner_team_id_clip_id_start_frame"),
        "team_pair_assignments",
        ["owner_team_id", "clip_id", "start_frame"],
        unique=False,
    )
    op.create_table(
        "flight_arcs",
        sa.Column("clip_id", sa.BigInteger(), nullable=False),
        sa.Column("point_id", sa.BigInteger(), nullable=True),
        sa.Column("shot_event_id", sa.BigInteger(), nullable=True),
        sa.Column("arc_index", sa.Integer(), nullable=False),
        sa.Column("start_event_id", sa.BigInteger(), nullable=True),
        sa.Column("end_event_id", sa.BigInteger(), nullable=True),
        sa.Column("start_frame", sa.Integer(), nullable=False),
        sa.Column("end_frame", sa.Integer(), nullable=False),
        sa.Column("valid_frame_count", sa.Integer(), nullable=False),
        sa.Column("total_frame_count", sa.Integer(), nullable=False),
        sa.Column("source_kind", sa.Text(), nullable=False),
        sa.Column(
            "quality_json",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("owner_team_id", sa.BigInteger(), nullable=False),
        sa.Column("visibility", sa.Text(), server_default="team", nullable=False),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "source_kind IN ('observed', 'estimated')",
            name=op.f("ck_flight_arcs_source_kind"),
        ),
        sa.CheckConstraint(
            "visibility IN ('private', 'team', 'org', 'public')",
            name=op.f("ck_flight_arcs_visibility"),
        ),
        sa.CheckConstraint(
            "arc_index >= 0 AND valid_frame_count >= 0 AND valid_frame_count <= total_frame_count",
            name=op.f("ck_flight_arcs_frame_counts"),
        ),
        sa.CheckConstraint(
            "start_frame >= 0 AND end_frame >= 0 AND start_frame <= end_frame",
            name=op.f("ck_flight_arcs_frame_bounds"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "clip_id", "end_event_id"],
            ["ball_events.owner_team_id", "ball_events.clip_id", "ball_events.id"],
            name=op.f("fk_flight_arcs_owner_team_id_clip_id_end_event_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "clip_id", "point_id"],
            ["points.owner_team_id", "points.clip_id", "points.id"],
            name=op.f("fk_flight_arcs_owner_team_id_clip_id_point_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "clip_id", "shot_event_id"],
            ["ball_events.owner_team_id", "ball_events.clip_id", "ball_events.id"],
            name=op.f("fk_flight_arcs_owner_team_id_clip_id_shot_event_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "clip_id", "start_event_id"],
            ["ball_events.owner_team_id", "ball_events.clip_id", "ball_events.id"],
            name=op.f("fk_flight_arcs_owner_team_id_clip_id_start_event_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "clip_id"],
            ["clips.owner_team_id", "clips.id"],
            name=op.f("fk_flight_arcs_owner_team_id_clip_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id"], ["teams.id"], name=op.f("fk_flight_arcs_owner_team_id")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_flight_arcs")),
        sa.UniqueConstraint(
            "owner_team_id",
            "clip_id",
            "arc_index",
            name=op.f("uq_flight_arcs_owner_team_id_clip_id_arc_index"),
        ),
        sa.UniqueConstraint(
            "owner_team_id",
            "clip_id",
            "id",
            name=op.f("uq_flight_arcs_owner_team_id_clip_id_id"),
        ),
    )
    op.create_index(
        op.f("ix_flight_arcs_owner_team_id_clip_id_start_frame"),
        "flight_arcs",
        ["owner_team_id", "clip_id", "start_frame"],
        unique=False,
    )
    op.create_table(
        "serve_attempts",
        sa.Column("clip_id", sa.BigInteger(), nullable=False),
        sa.Column("point_id", sa.BigInteger(), nullable=False),
        sa.Column("ball_event_id", sa.BigInteger(), nullable=True),
        sa.Column("attempt_index", sa.Integer(), nullable=False),
        sa.Column("serve_number", sa.Integer(), nullable=True),
        sa.Column("replay_index", sa.Integer(), nullable=True),
        sa.Column("result", sa.Text(), server_default="unknown", nullable=False),
        sa.Column("fault_cause", sa.Text(), nullable=True),
        sa.Column("source_kind", sa.Text(), nullable=False),
        sa.Column(
            "evidence_json",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("owner_team_id", sa.BigInteger(), nullable=False),
        sa.Column("visibility", sa.Text(), server_default="team", nullable=False),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "result = 'fault' OR fault_cause IS NULL",
            name=op.f("ck_serve_attempts_fault_cause"),
        ),
        sa.CheckConstraint(
            "result IN ('in', 'fault', 'let', 'unknown')",
            name=op.f("ck_serve_attempts_result"),
        ),
        sa.CheckConstraint(
            "visibility IN ('private', 'team', 'org', 'public')",
            name=op.f("ck_serve_attempts_visibility"),
        ),
        sa.CheckConstraint(
            "attempt_index >= 0 AND replay_index >= 0",
            name=op.f("ck_serve_attempts_order"),
        ),
        sa.CheckConstraint(
            "serve_number IN (1, 2)", name=op.f("ck_serve_attempts_serve_number")
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "clip_id", "point_id", "ball_event_id"],
            [
                "ball_events.owner_team_id",
                "ball_events.clip_id",
                "ball_events.point_id",
                "ball_events.id",
            ],
            name=op.f("fk_serve_attempts_owner_team_id_clip_id_point_id_ball_event_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "clip_id", "point_id"],
            ["points.owner_team_id", "points.clip_id", "points.id"],
            name=op.f("fk_serve_attempts_owner_team_id_clip_id_point_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id"],
            ["teams.id"],
            name=op.f("fk_serve_attempts_owner_team_id"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_serve_attempts")),
        sa.UniqueConstraint(
            "owner_team_id",
            "point_id",
            "attempt_index",
            name=op.f("uq_serve_attempts_owner_team_id_point_id_attempt_index"),
        ),
    )
    op.create_index(
        op.f("ix_serve_attempts_owner_team_id_ball_event_id_serve_number"),
        "serve_attempts",
        ["owner_team_id", "ball_event_id", "serve_number"],
        unique=False,
    )
    op.create_table(
        "shot_hitter_estimates",
        sa.Column("ball_event_id", sa.BigInteger(), nullable=False),
        sa.Column("rule_version", sa.Text(), nullable=False),
        sa.Column("resolution_status", sa.Text(), nullable=False),
        sa.Column("local_player_index", sa.Integer(), nullable=True),
        sa.Column("local_player_id", sa.Integer(), nullable=True),
        sa.Column("selected_landmark", sa.Text(), nullable=True),
        sa.Column("best_score", sa.Double(), nullable=True),
        sa.Column("margin_to_second", sa.Double(), nullable=True),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        sa.Column(
            "candidates_json",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "evidence_json",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("owner_team_id", sa.BigInteger(), nullable=False),
        sa.Column("visibility", sa.Text(), server_default="team", nullable=False),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "resolution_status IN ('resolved', 'ambiguous', 'unresolved')",
            name=op.f("ck_shot_hitter_estimates_resolution_status"),
        ),
        sa.CheckConstraint(
            "selected_landmark IN ('left_wrist', 'right_wrist', 'root')",
            name=op.f("ck_shot_hitter_estimates_selected_landmark"),
        ),
        sa.CheckConstraint(
            "visibility IN ('private', 'team', 'org', 'public')",
            name=op.f("ck_shot_hitter_estimates_visibility"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "ball_event_id"],
            ["ball_events.owner_team_id", "ball_events.id"],
            name=op.f("fk_shot_hitter_estimates_owner_team_id_ball_event_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id"],
            ["teams.id"],
            name=op.f("fk_shot_hitter_estimates_owner_team_id"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_shot_hitter_estimates")),
        sa.UniqueConstraint(
            "owner_team_id",
            "ball_event_id",
            "rule_version",
            name=op.f(
                "uq_shot_hitter_estimates_owner_team_id_ball_event_id_rule_version"
            ),
        ),
    )
    op.create_index(
        op.f("ix_shot_hitter_estimates_owner_team_id_resolution_status_rule_version"),
        "shot_hitter_estimates",
        ["owner_team_id", "resolution_status", "rule_version"],
        unique=False,
    )
    op.create_table(
        "net_plane_crossings",
        sa.Column("clip_id", sa.BigInteger(), nullable=False),
        sa.Column("flight_arc_id", sa.BigInteger(), nullable=False),
        sa.Column("ball_event_id", sa.BigInteger(), nullable=True),
        sa.Column("crossing_index", sa.Integer(), nullable=False),
        sa.Column(
            "is_primary", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("crossing_in_net_span", sa.Boolean(), nullable=True),
        sa.Column("fractional_frame", sa.Double(), nullable=True),
        sa.Column("time_s", sa.Double(), nullable=True),
        sa.Column("time_to_net_s", sa.Double(), nullable=True),
        sa.Column("x_m", sa.Double(), nullable=True),
        sa.Column("y_m", sa.Double(), nullable=True),
        sa.Column("z_center_m", sa.Double(), nullable=True),
        sa.Column("direction_y", sa.Integer(), nullable=True),
        sa.Column("net_height_m", sa.Double(), nullable=True),
        sa.Column("net_clearance_center_m", sa.Double(), nullable=True),
        sa.Column("net_clearance_bottom_m", sa.Double(), nullable=True),
        sa.Column("court_net_profile_id", sa.BigInteger(), nullable=True),
        sa.Column("profile_source_kind", sa.Text(), nullable=True),
        sa.Column("source_kind", sa.Text(), nullable=False),
        sa.Column(
            "used_frames_json",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("fit_error_m", sa.Double(), nullable=True),
        sa.Column("owner_team_id", sa.BigInteger(), nullable=False),
        sa.Column("visibility", sa.Text(), server_default="team", nullable=False),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "profile_source_kind IN ('rule_default', 'measured', 'custom')",
            name=op.f("ck_net_plane_crossings_profile_source_kind"),
        ),
        sa.CheckConstraint(
            "source_kind IN ('observed', 'estimated')",
            name=op.f("ck_net_plane_crossings_source_kind"),
        ),
        sa.CheckConstraint(
            "status <> 'crossed' OR (fractional_frame IS NOT NULL AND time_s IS NOT NULL AND x_m IS NOT NULL AND y_m IS NOT NULL AND z_center_m IS NOT NULL AND court_net_profile_id IS NOT NULL)",
            name=op.f("ck_net_plane_crossings_crossing_observation"),
        ),
        sa.CheckConstraint(
            "status = 'crossed' OR (NOT is_primary AND crossing_in_net_span IS NULL AND fractional_frame IS NULL AND time_s IS NULL AND time_to_net_s IS NULL AND x_m IS NULL AND y_m IS NULL AND z_center_m IS NULL AND direction_y IS NULL AND net_height_m IS NULL AND net_clearance_center_m IS NULL AND net_clearance_bottom_m IS NULL)",
            name=op.f("ck_net_plane_crossings_non_crossing"),
        ),
        sa.CheckConstraint(
            "status IN ('crossed', 'no_crossing_confirmed', 'unknown_gap', 'not_applicable')",
            name=op.f("ck_net_plane_crossings_status"),
        ),
        sa.CheckConstraint(
            "visibility IN ('private', 'team', 'org', 'public')",
            name=op.f("ck_net_plane_crossings_visibility"),
        ),
        sa.CheckConstraint(
            "(court_net_profile_id IS NULL) = (profile_source_kind IS NULL)",
            name=op.f("ck_net_plane_crossings_profile_provenance"),
        ),
        sa.CheckConstraint(
            "direction_y IN (-1, 1)", name=op.f("ck_net_plane_crossings_direction_y")
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "clip_id", "ball_event_id"],
            ["ball_events.owner_team_id", "ball_events.clip_id", "ball_events.id"],
            name=op.f("fk_net_plane_crossings_owner_team_id_clip_id_ball_event_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "clip_id", "flight_arc_id"],
            ["flight_arcs.owner_team_id", "flight_arcs.clip_id", "flight_arcs.id"],
            name=op.f("fk_net_plane_crossings_owner_team_id_clip_id_flight_arc_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "profile_source_kind", "court_net_profile_id"],
            [
                "court_net_profiles.owner_team_id",
                "court_net_profiles.source_kind",
                "court_net_profiles.id",
            ],
            name=op.f(
                "fk_net_plane_crossings_owner_team_id_profile_source_kind_court_net_profile_id"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id"],
            ["teams.id"],
            name=op.f("fk_net_plane_crossings_owner_team_id"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_net_plane_crossings")),
        sa.UniqueConstraint(
            "owner_team_id",
            "flight_arc_id",
            "crossing_index",
            name=op.f(
                "uq_net_plane_crossings_owner_team_id_flight_arc_id_crossing_index"
            ),
        ),
    )
    op.create_index(
        op.f("ix_net_plane_crossings_owner_team_id_ball_event_id_is_primary"),
        "net_plane_crossings",
        ["owner_team_id", "ball_event_id", "is_primary"],
        unique=False,
    )
    op.create_table(
        "shot_geometry_features",
        sa.Column("clip_id", sa.BigInteger(), nullable=False),
        sa.Column("ball_event_id", sa.BigInteger(), nullable=False),
        sa.Column("flight_arc_id", sa.BigInteger(), nullable=True),
        sa.Column("profile_id", sa.BigInteger(), nullable=False),
        sa.Column("definition_version", sa.Text(), nullable=False),
        sa.Column("contact_frame", sa.Integer(), nullable=False),
        sa.Column("contact_time_s", sa.Double(), nullable=False),
        sa.Column("contact_frame_uncertainty", sa.Double(), nullable=True),
        sa.Column("contact_x_m", sa.Double(), nullable=True),
        sa.Column("contact_y_m", sa.Double(), nullable=True),
        sa.Column("contact_z_m", sa.Double(), nullable=True),
        sa.Column("contact_height_m", sa.Double(), nullable=True),
        sa.Column("outgoing_speed_mps", sa.Double(), nullable=True),
        sa.Column("incoming_speed_mps", sa.Double(), nullable=True),
        sa.Column(
            "speed_window_frames",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("speed_method", sa.Text(), nullable=True),
        sa.Column("outgoing_heading_rad", sa.Double(), nullable=True),
        sa.Column("first_bounce_frame", sa.Integer(), nullable=True),
        sa.Column("first_bounce_time_s", sa.Double(), nullable=True),
        sa.Column("first_bounce_x_m", sa.Double(), nullable=True),
        sa.Column("first_bounce_y_m", sa.Double(), nullable=True),
        sa.Column("first_bounce_z_m", sa.Double(), nullable=True),
        sa.Column("landing_depth_m", sa.Double(), nullable=True),
        sa.Column("landing_lane", sa.Text(), nullable=True),
        sa.Column("landing_course", sa.Text(), nullable=True),
        sa.Column("line_margin_m", sa.Double(), nullable=True),
        sa.Column("landing_result", sa.Text(), nullable=True),
        sa.Column("landing_source_kind", sa.Text(), nullable=True),
        sa.Column("landing_missing_reason", sa.Text(), nullable=True),
        sa.Column("bin_version", sa.Text(), nullable=True),
        sa.Column(
            "bins_json",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "quality_json",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("owner_team_id", sa.BigInteger(), nullable=False),
        sa.Column("visibility", sa.Text(), server_default="team", nullable=False),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(landing_source_kind IS NOT NULL AND landing_missing_reason IS NULL AND first_bounce_frame IS NOT NULL) OR (landing_source_kind IS NULL AND nullif(btrim(landing_missing_reason), '') IS NOT NULL AND first_bounce_frame IS NULL AND first_bounce_time_s IS NULL AND first_bounce_x_m IS NULL AND first_bounce_y_m IS NULL AND first_bounce_z_m IS NULL AND landing_depth_m IS NULL AND landing_lane IS NULL AND landing_course IS NULL AND line_margin_m IS NULL AND (landing_result IS NULL OR landing_result = 'not_applicable'))",
            name=op.f("ck_shot_geometry_features_landing_presence"),
        ),
        sa.CheckConstraint(
            "landing_result IN ('in', 'out_long', 'out_wide', 'out_both', 'uncertain', 'not_applicable')",
            name=op.f("ck_shot_geometry_features_landing_result"),
        ),
        sa.CheckConstraint(
            "landing_source_kind IN ('observed', 'estimated')",
            name=op.f("ck_shot_geometry_features_landing_source_kind"),
        ),
        sa.CheckConstraint(
            "visibility IN ('private', 'team', 'org', 'public')",
            name=op.f("ck_shot_geometry_features_visibility"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "clip_id", "ball_event_id"],
            ["ball_events.owner_team_id", "ball_events.clip_id", "ball_events.id"],
            name=op.f("fk_shot_geometry_features_owner_team_id_clip_id_ball_event_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "clip_id", "flight_arc_id"],
            ["flight_arcs.owner_team_id", "flight_arcs.clip_id", "flight_arcs.id"],
            name=op.f("fk_shot_geometry_features_owner_team_id_clip_id_flight_arc_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id", "profile_id"],
            ["court_net_profiles.owner_team_id", "court_net_profiles.id"],
            name=op.f("fk_shot_geometry_features_owner_team_id_profile_id"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_team_id"],
            ["teams.id"],
            name=op.f("fk_shot_geometry_features_owner_team_id"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_shot_geometry_features")),
        sa.UniqueConstraint(
            "owner_team_id",
            "ball_event_id",
            "profile_id",
            "definition_version",
            "bin_version",
            name=op.f(
                "uq_shot_geometry_features_owner_team_id_ball_event_id_profile_id_definition_version_bin_version"
            ),
            postgresql_nulls_not_distinct=True,
        ),
    )
    op.create_index(
        op.f("ix_shot_geometry_features_owner_team_id_profile_id_definition_version"),
        "shot_geometry_features",
        ["owner_team_id", "profile_id", "definition_version"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_table("shot_geometry_features")
    op.drop_table("net_plane_crossings")
    op.drop_table("shot_hitter_estimates")
    op.drop_table("serve_attempts")
    op.drop_table("flight_arcs")
    op.drop_table("team_pair_assignments")
    op.drop_table("ball_events")
    op.drop_table("points")
    op.drop_table("player_identity_maps")
    op.drop_table("metric_snapshots")
    op.drop_table("clips")
    op.drop_table("match_contexts")
    op.drop_table("team_memberships")
    op.drop_table("ingest_runs")
    op.drop_table("court_net_profiles")
    op.drop_table("teams")
    op.drop_table("users")
    op.drop_table("organizations")
