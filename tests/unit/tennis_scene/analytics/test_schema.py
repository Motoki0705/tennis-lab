from __future__ import annotations

import pytest
from sqlalchemy import CheckConstraint, UniqueConstraint

from src.tennis_scene.analytics.db.models import Base

ACCESS_TABLES = {"organizations", "teams", "users", "team_memberships"}
REFERENCES = {
    "court_net_profiles": set(),
    "ingest_runs": set(),
    "match_contexts": {"ingest_runs", "court_net_profiles"},
    "clips": {"match_contexts"},
    "points": {"clips"},
    "ball_events": {"clips", "points"},
    "shot_hitter_estimates": {"ball_events"},
    "flight_arcs": {"clips", "points", "ball_events"},
    "shot_geometry_features": {"ball_events", "flight_arcs", "court_net_profiles"},
    "net_plane_crossings": {"flight_arcs", "ball_events", "court_net_profiles"},
    "player_identity_maps": {"clips"},
    "team_pair_assignments": {"clips", "player_identity_maps"},
    "serve_attempts": {"points", "ball_events"},
    "metric_snapshots": {"match_contexts"},
}


def test_access_tables_have_unique_external_and_membership_keys() -> None:
    tables = Base.metadata.tables
    assert set(tables) == ACCESS_TABLES | set(REFERENCES)
    keys = {
        "organizations": ("slug",),
        "teams": ("organization_id", "slug"),
        "users": ("external_subject",),
        "team_memberships": ("team_id", "user_id"),
    }
    for table_name, key in keys.items():
        table = tables[table_name]
        assert any(
            isinstance(constraint, UniqueConstraint)
            and tuple(constraint.columns.keys()) == key
            for constraint in table.constraints
        )
    assert set(tables["users"].c.keys()) == {
        "id",
        "external_subject",
        "email",
        "display_name",
        "is_active",
        "created_at",
        "updated_at",
    }
    assert (
        next(iter(tables["teams"].c.organization_id.foreign_keys)).target_fullname
        == "organizations.id"
    )
    assert {fk.target_fullname for fk in tables["team_memberships"].foreign_keys} == {
        "teams.id",
        "users.id",
    }


@pytest.mark.parametrize("table_name", list(REFERENCES))
def test_domain_tables_enforce_ownership_and_index_the_team_prefix(
    table_name: str,
) -> None:
    table = Base.metadata.tables[table_name]
    assert not table.c.owner_team_id.nullable
    assert not table.c.visibility.nullable
    assert table.c.visibility.server_default.arg == "team"
    assert table.c.created_at.type.timezone
    assert table.c.updated_at.type.timezone
    assert any(
        isinstance(constraint, CheckConstraint)
        and str(constraint.sqltext)
        == "visibility IN ('private', 'team', 'org', 'public')"
        for constraint in table.constraints
    )
    assert any(
        len(index.columns) > 1 and list(index.columns)[0] is table.c.owner_team_id
        for index in table.indexes
    )
    direct_fk = [
        constraint
        for constraint in table.foreign_key_constraints
        if constraint.referred_table.name == "teams"
    ]
    assert len(direct_fk) == 1
    assert direct_fk[0].elements[0].target_fullname == "teams.id"
    domain_fks = set(table.foreign_key_constraints) - set(direct_fk)
    assert {fk.referred_table.name for fk in domain_fks} == REFERENCES[table_name]
    for fk in domain_fks:
        assert fk.column_keys[0] == "owner_team_id"
        assert fk.elements[0].column.key == "owner_team_id"
