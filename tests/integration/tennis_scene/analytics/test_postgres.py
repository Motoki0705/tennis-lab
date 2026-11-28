from __future__ import annotations

import csv
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from psycopg import Error
from sqlalchemy import Connection, Engine, Table, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.tennis_scene.analytics.__main__ import main
from src.tennis_scene.analytics.db.models import Base, MetricSnapshot
from src.tennis_scene.analytics.db.snapshots import insert_metric_snapshot
from src.tennis_scene.analytics.export import DOMAIN_TABLES
from src.tennis_scene.analytics.metrics import (
    MatchPointId,
    MetricRecord,
    unique_point_bins,
)
from src.tennis_scene.ball_event_schema import (
    EVENT_CLASSES_V4,
    EVENT_SUBTYPES_BY_KIND_V4,
    EVENT_SUBTYPES_V4,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres]
TABLES = Base.metadata.tables
ALEMBIC_INI = Path(__file__).resolve().parents[4] / "alembic.ini"


@pytest.fixture
def migrated_db(postgres_db: Engine) -> Engine:
    command.upgrade(Config(ALEMBIC_INI), "head")
    return postgres_db


def _insert(connection: Connection, table_name: str, **values: Any) -> int:
    table = TABLES[table_name]
    return int(
        connection.execute(
            table.insert().values(**values).returning(table.c.id)
        ).scalar_one()
    )


def _row(connection: Connection, table: Table, row_id: int) -> dict[str, Any]:
    return dict(
        connection.execute(select(table).where(table.c.id == row_id)).mappings().one()
    )


def _access_rows(connection: Connection) -> dict[str, int]:
    org = _insert(connection, "organizations", slug="club", name="Club")
    team = _insert(
        connection, "teams", organization_id=org, slug="data-team", name="Data team"
    )
    user = _insert(
        connection, "users", external_subject="issuer:user-1", display_name="Reviewer"
    )
    member = _insert(
        connection, "team_memberships", team_id=team, user_id=user, role="member"
    )
    return {
        "organizations": org,
        "teams": team,
        "users": user,
        "team_memberships": member,
    }


def _profile(connection: Connection, team: int, key: str) -> int:
    return _insert(
        connection,
        "court_net_profiles",
        owner_team_id=team,
        profile_key=key,
        sport_type="soft_tennis",
        net_profile_kind="flat",
        net_height_center_m=1.07,
        net_height_post_m=1.07,
        net_post_offset_x_m=5.485,
        ball_radius_m=0.033,
        source_kind="rule_default",
    )


def _match(connection: Connection, team: int, run: int, profile: int, key: str) -> int:
    return _insert(
        connection,
        "match_contexts",
        owner_team_id=team,
        ingest_run_id=run,
        dataset_id="meiji_4cam",
        video_id="video-1",
        match_key=key,
        court_net_profile_id=profile,
        sport_type="soft_tennis",
        rules_json={"format": "doubles"},
        time_conversion_json={"unit": "seconds"},
    )


def _clip(
    connection: Connection,
    team: int,
    match: int,
    clip_key: str = "clip_012",
    *,
    num_frames: int = 90,
) -> int:
    return _insert(
        connection,
        "clips",
        owner_team_id=team,
        match_context_id=match,
        clip_id=clip_key,
        clip_name=clip_key,
        split="val",
        global_start_sec=91.5,
        global_end_sec=91.5 + num_frames / 30,
        fps=30,
        num_frames=num_frames,
        scene_schema_version=4,
        scene_npz_sha256="a" * 64,
        annotation_sha256="b" * 64,
    )


def _point(
    connection: Connection,
    team: int,
    match: int,
    clip: int,
    key: str,
    match_point_key: str,
    **details: Any,
) -> int:
    return _insert(
        connection,
        "points",
        owner_team_id=team,
        match_context_id=match,
        clip_id=clip,
        point_key=key,
        match_point_key=match_point_key,
        **details,
    )


def _graph(connection: Connection, team: int) -> dict[str, int]:
    profile = _profile(connection, team, "soft_tennis_standard")
    run = _insert(
        connection,
        "ingest_runs",
        owner_team_id=team,
        dataset_id="meiji_4cam",
        video_id="video-1",
        v4_manifest_sha256="c" * 64,
        projects_json_sha256="d" * 64,
        schema_version=1,
        definition_version="1",
        status="complete",
    )
    match = _match(connection, team, run, profile, "match-1")
    clip = _clip(connection, team, match)
    point = _point(
        connection,
        team,
        match,
        clip,
        "clip-point-1",
        "match-point-11",
        start_frame=1,
        end_frame=43,
        completeness="complete",
        score_status="confirmed",
        score_before_json={"a": 1, "b": 2},
        server_team="A",
        winner_team="B",
        terminal_cause="double_fault",
        terminal_cause_source="derived_v4_score",
        note="",
    )
    shot = _insert(
        connection,
        "ball_events",
        owner_team_id=team,
        clip_id=clip,
        point_id=point,
        source_event_id="v4,event-027",
        frame=9,
        time_s=0.3,
        frame_uncertainty=1,
        kind="shot",
        subtype="serve_overhead",
        status="certain",
        court_side=-1,
        event_source="annotation",
        adopted_in_v4=True,
        in_point=True,
        provenance_json={"frame": 9, "label": "確認済み"},
    )
    hitter = _insert(
        connection,
        "shot_hitter_estimates",
        owner_team_id=team,
        ball_event_id=shot,
        rule_version="hitter-v1",
        resolution_status="resolved",
        local_player_index=0,
        local_player_id=5,
        selected_landmark="left_wrist",
        best_score=0.2,
        margin_to_second=0.1,
        candidates_json=[{"local_player_id": 5}],
    )
    arc = _insert(
        connection,
        "flight_arcs",
        owner_team_id=team,
        clip_id=clip,
        point_id=point,
        shot_event_id=shot,
        arc_index=0,
        start_event_id=shot,
        start_frame=9,
        end_frame=16,
        valid_frame_count=6,
        total_frame_count=8,
        source_kind="observed",
    )
    geometry = _insert(
        connection,
        "shot_geometry_features",
        owner_team_id=team,
        clip_id=clip,
        ball_event_id=shot,
        flight_arc_id=arc,
        profile_id=profile,
        definition_version="geom-v1",
        contact_frame=9,
        contact_time_s=0.3,
        contact_frame_uncertainty=1,
        contact_height_m=1.4,
        outgoing_speed_mps=14.5,
        landing_missing_reason="tracking_gap",
    )
    crossing = _insert(
        connection,
        "net_plane_crossings",
        owner_team_id=team,
        clip_id=clip,
        flight_arc_id=arc,
        ball_event_id=shot,
        crossing_index=0,
        status="unknown_gap",
        source_kind="observed",
        used_frames_json=[9, 10, 11],
    )
    player = _insert(
        connection,
        "player_identity_maps",
        owner_team_id=team,
        clip_id=clip,
        local_player_id=5,
        global_player_id="p17",
        start_frame=0,
        end_frame=89,
        mapping_version="identity-v1",
        resolution_status="resolved",
        source_sha256="e" * 64,
    )
    pair = _insert(
        connection,
        "team_pair_assignments",
        owner_team_id=team,
        clip_id=clip,
        pair_key="pair-4",
        scoring_team="A",
        court_side=-1,
        first_identity_map_id=player,
        start_frame=0,
        end_frame=43,
        assignment_version="pair-v1",
        resolution_status="ambiguous",
    )
    attempt = _insert(
        connection,
        "serve_attempts",
        owner_team_id=team,
        clip_id=clip,
        point_id=point,
        ball_event_id=shot,
        attempt_index=1,
        serve_number=2,
        replay_index=0,
        result="fault",
        fault_cause="net",
        source_kind="manual_confirmed",
    )
    metric = insert_metric_snapshot(
        connection,
        owner_team_id=team,
        match_context_id=match,
        record=MetricRecord(
            metric_id="serve.double_fault_count",
            definition_version="1",
            subject_type="team",
            subject_id="A",
            filters={},
            bin_version=None,
            numerator=1,
            denominator=1,
            value=1,
            candidate_count=1,
            resolved_count=1,
            unknown_count=0,
            excluded_by_reason={},
            status="complete",
            source_ids={
                "numerator_point_ids": [point],
                "denominator_point_ids": [point],
            },
        ),
    )
    return {
        "court_net_profiles": profile,
        "ingest_runs": run,
        "match_contexts": match,
        "clips": clip,
        "points": point,
        "ball_events": shot,
        "shot_hitter_estimates": hitter,
        "flight_arcs": arc,
        "shot_geometry_features": geometry,
        "net_plane_crossings": crossing,
        "player_identity_maps": player,
        "team_pair_assignments": pair,
        "serve_attempts": attempt,
        "metric_snapshots": metric,
    }


def test_alembic_upgrade_downgrade_upgrade_and_metadata_match(
    postgres_db: Engine,
) -> None:
    config = Config(ALEMBIC_INI)
    command.upgrade(config, "head")
    assert set(inspect(postgres_db).get_table_names()) == {*TABLES, "alembic_version"}
    assert main(["db", "check"]) == 0
    command.downgrade(config, "base")
    assert set(inspect(postgres_db).get_table_names()) == {"alembic_version"}
    assert main(["db", "check"]) == 1
    assert main(["db", "upgrade"]) == 0
    inspector = inspect(postgres_db)
    assert set(inspector.get_table_names()) == {*TABLES, "alembic_version"}
    for name in DOMAIN_TABLES:
        assert any(
            index["column_names"][0] == "owner_team_id"
            and len(index["column_names"]) > 1
            for index in inspector.get_indexes(name)
        )
    assert main(["db", "check"]) == 0
    assert main(["db", "downgrade"]) == 0
    assert set(inspect(postgres_db).get_table_names()) == {"alembic_version"}


def test_postgres_crud_and_visibility_for_every_table(migrated_db: Engine) -> None:
    access_updates: dict[str, dict[str, Any]] = {
        "organizations": {"name": "Club revised"},
        "teams": {"name": "Team revised"},
        "users": {"is_active": False},
        "team_memberships": {"role": "editor"},
    }
    with migrated_db.begin() as connection:
        access = _access_rows(connection)
        ids = {**access, **_graph(connection, access["teams"])}
        assert set(ids) == set(TABLES)
        for table in Base.metadata.sorted_tables:
            row_id = ids[table.name]
            row = _row(connection, table, row_id)
            assert row["created_at"].utcoffset() is not None
            assert row["updated_at"].utcoffset() is not None
            if table.name in DOMAIN_TABLES:
                assert row["owner_team_id"] == access["teams"]
                assert row["visibility"] == "team"
                values = {"visibility": "private"}
            else:
                values = access_updates[table.name]
            connection.execute(
                table.update().where(table.c.id == row_id).values(**values)
            )
            updated = _row(connection, table, row_id)
            assert all(updated[key] == value for key, value in values.items())

        for table in reversed(Base.metadata.sorted_tables):
            connection.execute(table.delete().where(table.c.id == ids[table.name]))
            assert connection.execute(select(table)).first() is None


def _rejected(
    connection: Connection, table_name: str, row_id: int, sqlstate: str, **changes: Any
) -> None:
    table = TABLES[table_name]
    with pytest.raises(IntegrityError) as error, connection.begin_nested():
        connection.execute(table.update().where(table.c.id == row_id).values(**changes))
    assert isinstance(error.value.orig, Error)
    assert error.value.orig.sqlstate == sqlstate


def test_postgres_rejects_bad_ownership_and_invalid_domain_states(
    migrated_db: Engine,
) -> None:
    with migrated_db.begin() as connection:
        access = _access_rows(connection)
        team = access["teams"]
        other_team = _insert(
            connection,
            "teams",
            organization_id=access["organizations"],
            slug="other",
            name="Other",
        )
        graph = _graph(connection, team)
        foreign_profile = _profile(connection, other_team, "foreign")
        other_match = _match(
            connection,
            team,
            graph["ingest_runs"],
            graph["court_net_profiles"],
            "sibling",
        )
        other_clip = _clip(connection, team, other_match)
        other_point = _point(
            connection, team, other_match, other_clip, "p2", "match-point-11"
        )
        other_shot = _insert(
            connection,
            "ball_events",
            owner_team_id=team,
            clip_id=other_clip,
            point_id=other_point,
            source_event_id="serve-1",
            frame=10,
            time_s=10 / 30,
            kind="shot",
            event_source="rule",
            in_point=True,
        )
        other_arc = _insert(
            connection,
            "flight_arcs",
            owner_team_id=team,
            clip_id=other_clip,
            point_id=other_point,
            shot_event_id=other_shot,
            arc_index=0,
            start_frame=10,
            end_frame=20,
            valid_frame_count=11,
            total_frame_count=11,
            source_kind="observed",
        )
        same_clip_other_point = _point(
            connection,
            team,
            graph["match_contexts"],
            graph["clips"],
            "p2",
            "match-point-12",
        )
        for name in DOMAIN_TABLES:
            _rejected(connection, name, graph[name], "23514", visibility="shared")
            _rejected(connection, name, graph[name], "23502", owner_team_id=None)
            _rejected(connection, name, graph[name], "23503", owner_team_id=999999)
        _rejected(
            connection,
            "match_contexts",
            graph["match_contexts"],
            "23503",
            court_net_profile_id=foreign_profile,
        )
        _rejected(
            connection,
            "match_contexts",
            graph["match_contexts"],
            "23503",
            dataset_id="foreign-dataset",
        )
        _rejected(
            connection, "ball_events", graph["ball_events"], "23503", clip_id=other_clip
        )
        _rejected(
            connection, "ball_events", graph["ball_events"], "23514", subtype="bounce"
        )
        for table_name in ("shot_geometry_features", "net_plane_crossings"):
            _rejected(
                connection,
                table_name,
                graph[table_name],
                "23503",
                flight_arc_id=other_arc,
            )
            _rejected(
                connection,
                table_name,
                graph[table_name],
                "23503",
                ball_event_id=other_shot,
            )
        _rejected(
            connection,
            "serve_attempts",
            graph["serve_attempts"],
            "23503",
            ball_event_id=other_shot,
        )
        _rejected(
            connection,
            "serve_attempts",
            graph["serve_attempts"],
            "23503",
            point_id=other_point,
        )
        _rejected(
            connection,
            "serve_attempts",
            graph["serve_attempts"],
            "23503",
            point_id=same_clip_other_point,
        )
        _rejected(
            connection, "points", graph["points"], "23503", match_context_id=other_match
        )
        _rejected(
            connection,
            "points",
            graph["points"],
            "23514",
            terminal_cause_source="unknown",
        )
        _rejected(
            connection,
            "net_plane_crossings",
            graph["net_plane_crossings"],
            "23514",
            time_s=0,
        )
        _rejected(
            connection,
            "shot_geometry_features",
            graph["shot_geometry_features"],
            "23514",
            landing_depth_m=0,
        )
        _rejected(
            connection,
            "metric_snapshots",
            graph["metric_snapshots"],
            "23514",
            value=None,
        )
        assert (
            _row(connection, TABLES["serve_attempts"], graph["serve_attempts"])[
                "fault_cause"
            ]
            == "net"
        )
        assert (
            _row(connection, TABLES["points"], graph["points"])["terminal_cause"]
            == "double_fault"
        )
        for cause in (
            "out_long",
            "out_wide",
            "out_both",
            "unreturned",
            "other",
            "unknown",
        ):
            _rejected(
                connection, "points", graph["points"], "23514", terminal_cause=cause
            )
        confirmed = _point(
            connection,
            team,
            other_match,
            other_clip,
            "p3",
            "match-point-13",
            terminal_cause="out_long",
            terminal_cause_source="manual_confirmed",
        )
        assert (
            _row(connection, TABLES["points"], confirmed)["terminal_cause"]
            == "out_long"
        )


def test_postgres_preserves_ingest_and_v4_source_identity(migrated_db: Engine) -> None:
    with migrated_db.begin() as connection:
        access = _access_rows(connection)
        team = access["teams"]
        graph = _graph(connection, team)

        for table_name, fields in (
            ("organizations", ("slug", "name")),
            ("teams", ("organization_id", "slug", "name")),
            ("users", ("external_subject", "display_name")),
            ("team_memberships", ("team_id", "user_id", "role")),
            (
                "ingest_runs",
                (
                    "owner_team_id",
                    "dataset_id",
                    "video_id",
                    "schema_version",
                    "definition_version",
                    "v4_manifest_sha256",
                    "projects_json_sha256",
                    "manual_supplement_sha256",
                    "status",
                ),
            ),
            (
                "ball_events",
                (
                    "owner_team_id",
                    "clip_id",
                    "source_event_id",
                    "frame",
                    "time_s",
                    "frame_uncertainty",
                    "kind",
                    "subtype",
                    "status",
                    "event_source",
                    "adopted_in_v4",
                    "court_side",
                    "in_point",
                ),
            ),
        ):
            original = _row(
                connection, TABLES[table_name], (access | graph)[table_name]
            )
            with pytest.raises(IntegrityError) as error, connection.begin_nested():
                _insert(
                    connection,
                    table_name,
                    **{field: original[field] for field in fields},
                )
            assert isinstance(error.value.orig, Error)
            assert error.value.orig.sqlstate == "23505"

        cases = [
            (kind, EVENT_SUBTYPES_V4[subtype])
            for kind, subtypes in zip(
                EVENT_CLASSES_V4, EVENT_SUBTYPES_BY_KIND_V4, strict=True
            )
            for subtype in subtypes
        ]
        values = [
            {
                "owner_team_id": team,
                "clip_id": graph["clips"],
                "source_event_id": f"e-{index}",
                "frame": 1 + index,
                "time_s": (1 + index) / 30,
                "frame_uncertainty": 1,
                "kind": kind,
                "subtype": subtype,
                "status": "probable",
                "court_side": 0,
                "event_source": "annotation",
                "adopted_in_v4": True,
                "in_point": True,
            }
            for index, (kind, subtype) in enumerate(cases)
        ]
        events = TABLES["ball_events"]
        rows = connection.execute(
            events.insert().returning(
                events.c.source_event_id, events.c.kind, events.c.subtype
            ),
            values,
        ).all()
        assert {(row.kind, row.subtype) for row in rows} == set(cases)
        assert {row.source_event_id for row in rows} == {
            row["source_event_id"] for row in values
        }

        excluded = _insert(
            connection,
            "ball_events",
            owner_team_id=team,
            clip_id=graph["clips"],
            source_event_id="excluded-v4-id",
            frame=64,
            time_s=64 / 30,
            kind="shot",
            event_source="annotation",
            adopted_in_v4=False,
            in_point=False,
            excluded_reason="Outside point.",
            provenance_json={"old_subtype": "other"},
        )
        lineage = _row(connection, events, excluded)
        assert lineage["adopted_in_v4"] is False and lineage["in_point"] is False
        assert lineage["subtype"] is None and lineage["status"] is None
        assert lineage["provenance_json"] == {"old_subtype": "other"}
        _rejected(connection, "ball_events", excluded, "23514", excluded_reason=None)
        _rejected(connection, "ball_events", excluded, "23514", excluded_reason=" ")
        _rejected(connection, "ball_events", excluded, "23514", in_point=True)

        # The first bounce after a let in the fixed V4 adopted-events ledger.
        let_clip = _clip(
            connection, team, graph["match_contexts"], "clip_009", num_frames=200
        )
        bounce = _insert(
            connection,
            "ball_events",
            owner_team_id=team,
            clip_id=let_clip,
            source_event_id="e003",
            frame=132,
            time_s=132 / 30,
            frame_uncertainty=1,
            kind="bounce",
            subtype="bounce",
            status="probable",
            court_side=0,
            event_source="annotation",
            adopted_in_v4=True,
            in_point=False,
            provenance_json={"review_source": "service_let_first_bounce"},
        )
        recorded = _row(connection, events, bounce)
        assert recorded["adopted_in_v4"] is True and recorded["in_point"] is False
        assert recorded["excluded_reason"] is None
        _rejected(connection, "ball_events", bounce, "23514", adopted_in_v4=None)
        _rejected(connection, "ball_events", bounce, "23514", status=None)
        _rejected(
            connection, "ball_events", bounce, "23514", excluded_reason="not adopted"
        )


def _check_same_rows(
    json_rows: list[dict[str, Any]], csv_rows: list[dict[str, str]]
) -> None:
    assert len(csv_rows) == len(json_rows)
    for csv_row, json_row in zip(csv_rows, json_rows, strict=True):
        assert set(csv_row) == set(json_row)
        for key, value in json_row.items():
            if value is None:
                assert csv_row[key] == r"\N"
            elif isinstance(value, str):
                assert csv_row[key] == value
            else:
                assert json.loads(csv_row[key]) == value


def test_cli_query_and_scoped_exports_agree_with_the_persisted_rows(
    migrated_db: Engine, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with migrated_db.begin() as connection:
        access = _access_rows(connection)
        team = access["teams"]
        other_team = _insert(
            connection,
            "teams",
            organization_id=access["organizations"],
            slug="other",
            name="Other",
        )
        own = _graph(connection, team)
        foreign = _graph(connection, other_team)
        added_profile = _profile(connection, team, "same-team-unreferenced")
        used_profile = _profile(connection, team, "venue-profile")
        crossing_profile = _profile(connection, team, "crossing-only-profile")
        geometry = TABLES["shot_geometry_features"]
        connection.execute(
            geometry.update()
            .where(geometry.c.id == own[geometry.name])
            .values(profile_id=used_profile)
        )
        crossings = TABLES["net_plane_crossings"]
        connection.execute(
            crossings.update()
            .where(crossings.c.id == own[crossings.name])
            .values(
                court_net_profile_id=crossing_profile,
                profile_source_kind="rule_default",
            )
        )
        sibling = _match(
            connection, team, own["ingest_runs"], own["court_net_profiles"], "match-2"
        )
        sibling_clip = _clip(connection, team, sibling)

    command_prefix = [sys.executable, "-m", "src.tennis_scene.analytics"]
    query = ["query", "ball_events", "--owner-team", str(team)]
    completed = subprocess.run(
        [*command_prefix, *query], capture_output=True, text=True, check=True, timeout=8
    )
    json_events = json.loads(completed.stdout)
    assert len(json_events) == 1
    assert json_events[0]["id"] == own["ball_events"]
    assert json_events[0]["point_id"] == own["points"]
    assert json_events[0]["source_event_id"] == "v4,event-027"
    assert json_events[0]["provenance_json"] == {"frame": 9, "label": "確認済み"}
    assert main([*query, "--format", "csv"]) == 0
    _check_same_rows(
        json_events, list(csv.DictReader(StringIO(capsys.readouterr().out)))
    )

    assert main([*query, "--id", str(foreign["ball_events"])]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert main(["query", "ball_events", "--limit", "1"]) == 0
    assert json.loads(capsys.readouterr().out) == json_events

    match_export = [
        "export",
        "--match-context",
        str(own["match_contexts"]),
        "--owner-team",
        str(team),
    ]
    assert main(match_export) == 0
    json_export = capsys.readouterr().out
    assert main(match_export) == 0
    assert capsys.readouterr().out == json_export
    data = json.loads(json_export)
    assert set(data) == set(DOMAIN_TABLES)
    assert data["ball_events"] == json_events
    assert [row["id"] for row in data["court_net_profiles"]] == [
        own["court_net_profiles"],
        used_profile,
        crossing_profile,
    ]
    assert added_profile not in [row["id"] for row in data["court_net_profiles"]]
    assert all(
        [row["id"] for row in data[name]] == [row_id]
        for name, row_id in own.items()
        if name != "court_net_profiles"
    )
    assert data["shot_geometry_features"][0]["landing_depth_m"] is None
    assert data["net_plane_crossings"][0]["time_s"] is None
    assert data["net_plane_crossings"][0]["court_net_profile_id"] == crossing_profile
    assert data["ingest_runs"][0]["manual_supplement_sha256"] is None
    assert data["metric_snapshots"][0]["source_ids"]["numerator_point_ids"] == [
        own["points"]
    ]
    assert main(["query", "metric_snapshots", "--owner-team", str(team)]) == 0
    assert json.loads(capsys.readouterr().out) == data["metric_snapshots"]

    destination = tmp_path / "match"
    assert (
        main([*match_export, "--format", "csv", "--output-dir", str(destination)]) == 0
    )
    assert {file.stem for file in destination.glob("*.csv")} == set(data)
    for name, json_rows in data.items():
        with (destination / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
            _check_same_rows(json_rows, list(csv.DictReader(stream)))

    assert main(["export", "--ingest-run", str(own["ingest_runs"])]) == 0
    run = json.loads(capsys.readouterr().out)
    assert [row["id"] for row in run["match_contexts"]] == [
        own["match_contexts"],
        sibling,
    ]
    assert [row["id"] for row in run["clips"]] == [own["clips"], sibling_clip]
    assert run["ball_events"] == json_events
    assert (
        main(
            [
                "export",
                "--ingest-run",
                str(foreign["ingest_runs"]),
                "--owner-team",
                str(team),
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert output.out == "" and "No ingest_runs record" in output.err


def test_landing_states_and_crossing_profile_are_enforced(migrated_db: Engine) -> None:
    with migrated_db.begin() as connection:
        access = _access_rows(connection)
        team = access["teams"]
        graph = _graph(connection, team)
        other_team = _insert(
            connection,
            "teams",
            organization_id=access["organizations"],
            slug="other",
            name="Other",
        )
        foreign_profile = _profile(connection, other_team, "foreign")
        geometry = TABLES["shot_geometry_features"]
        missing = graph[geometry.name]
        assert _row(connection, geometry, missing)["landing_source_kind"] is None
        _rejected(
            connection, geometry.name, missing, "23514", landing_missing_reason=None
        )
        _rejected(connection, geometry.name, missing, "23514", first_bounce_time_s=0.6)
        _rejected(
            connection, geometry.name, missing, "23514", landing_result="out_long"
        )
        _rejected(
            connection,
            geometry.name,
            missing,
            "23514",
            first_bounce_frame=18,
            landing_depth_m=2.1,
        )
        _rejected(
            connection,
            geometry.name,
            missing,
            "23514",
            landing_source_kind="observed",
            first_bounce_frame=18,
            landing_depth_m=2.1,
        )
        _rejected(
            connection,
            geometry.name,
            missing,
            "23514",
            landing_source_kind="observed",
            landing_missing_reason=None,
        )

        ids = {}
        for source, depth in (("observed", -0.45), ("estimated", 1.2)):
            ids[source] = _insert(
                connection,
                geometry.name,
                owner_team_id=team,
                clip_id=graph["clips"],
                ball_event_id=graph["ball_events"],
                profile_id=graph["court_net_profiles"],
                definition_version=f"geometry-{source}",
                contact_frame=9,
                contact_time_s=0.3,
                first_bounce_frame=18,
                first_bounce_time_s=0.6,
                first_bounce_x_m=0.7,
                first_bounce_y_m=4.0,
                first_bounce_z_m=0.033,
                landing_source_kind=source,
                landing_depth_m=depth,
            )
        observed = connection.execute(
            select(geometry.c.id, geometry.c.landing_depth_m).where(
                geometry.c.landing_source_kind == "observed"
            )
        ).one()
        assert tuple(observed) == (ids["observed"], -0.45)
        assert _row(connection, geometry, ids["estimated"])["landing_depth_m"] == 1.2
        _rejected(
            connection,
            geometry.name,
            ids["observed"],
            "23514",
            landing_missing_reason="tracking_gap",
        )

        crossings = TABLES["net_plane_crossings"]
        crossing_id = graph[crossings.name]
        crossing = {
            "status": "crossed",
            "fractional_frame": 12.5,
            "time_s": 12.5 / 30,
            "x_m": 0.7,
            "y_m": 0.0,
            "z_center_m": 1.42,
            "crossing_in_net_span": True,
            "direction_y": 1,
            "net_height_m": 1.07,
            "net_clearance_center_m": 0.35,
            "net_clearance_bottom_m": 0.317,
        }
        _rejected(connection, crossings.name, crossing_id, "23514", **crossing)
        _rejected(
            connection,
            crossings.name,
            crossing_id,
            "23503",
            **crossing,
            court_net_profile_id=foreign_profile,
            profile_source_kind="rule_default",
        )
        profile = graph["court_net_profiles"]
        _rejected(
            connection,
            crossings.name,
            crossing_id,
            "23503",
            **crossing,
            court_net_profile_id=profile,
            profile_source_kind="measured",
        )
        connection.execute(
            crossings.update()
            .where(crossings.c.id == crossing_id)
            .values(
                **crossing,
                court_net_profile_id=profile,
                profile_source_kind="rule_default",
            )
        )
        observed_crossing = _row(connection, crossings, crossing_id)
        assert observed_crossing["court_net_profile_id"] == profile
        assert observed_crossing["net_clearance_center_m"] == 0.35


def test_canonical_match_points_deduplicate_overlapping_clips(
    migrated_db: Engine,
) -> None:
    with migrated_db.begin() as connection:
        team = _access_rows(connection)["teams"]
        graph = _graph(connection, team)
        match = graph["match_contexts"]
        overlap_clip = _clip(connection, team, match, "clip_overlap")
        overlap = _point(connection, team, match, overlap_clip, "p2", "match-point-11")
        other_point = _point(
            connection, team, match, overlap_clip, "p3", "match-point-12"
        )
        other_match = _match(
            connection,
            team,
            graph["ingest_runs"],
            graph["court_net_profiles"],
            "match-2",
        )
        other_clip = _clip(connection, team, other_match)
        other_match_point = _point(
            connection, team, other_match, other_clip, "p1", "match-point-11"
        )
        points = TABLES["points"]
        rows = connection.execute(
            select(points.c.id, points.c.match_context_id, points.c.match_point_key)
        ).all()
        assert {row.id for row in rows} == {
            graph["points"],
            overlap,
            other_point,
            other_match_point,
        }
        assert unique_point_bins(
            (MatchPointId(row.match_context_id, row.match_point_key), "A", "low")
            for row in rows
        ) == {
            (MatchPointId(match, "match-point-11"), "A", "low"),
            (MatchPointId(match, "match-point-12"), "A", "low"),
            (MatchPointId(other_match, "match-point-11"), "A", "low"),
        }


def test_snapshot_writes_and_cli_validate_metric_evidence(
    migrated_db: Engine,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    table = TABLES["metric_snapshots"]
    with migrated_db.begin() as connection:
        team = _access_rows(connection)["teams"]
        graph = _graph(connection, team)
        snapshot_id = graph["metric_snapshots"]
        match = graph["match_contexts"]
        good_row = _row(connection, table, snapshot_id)
        good_record = MetricRecord.from_record(good_row)
        _rejected(connection, table.name, snapshot_id, "23514", source_ids={})
        _rejected(
            connection,
            table.name,
            snapshot_id,
            "23514",
            metric_id="point.played_count",
            numerator=3,
            value=3,
        )
        changed_record = MetricRecord.from_record(_row(connection, table, snapshot_id))
        changed_record.source_ids.clear()
        with pytest.raises(ValueError, match="Missing source_ids"):
            insert_metric_snapshot(
                connection,
                owner_team_id=team,
                match_context_id=match,
                record=changed_record,
            )
        assert connection.scalars(select(table.c.id)).all() == [snapshot_id]

    with Session(migrated_db) as session:
        snapshot = session.get(MetricSnapshot, snapshot_id)
        assert snapshot is not None
        snapshot.source_ids = {}
        with pytest.raises(ValueError, match="Missing source_ids"):
            session.flush()
        session.rollback()

        invalid = good_record.to_record()
        invalid.update(metric_id="point.played_count", numerator=3, value=3)
        session.add(
            MetricSnapshot(owner_team_id=team, match_context_id=match, **invalid)
        )
        with pytest.raises(ValueError, match="Exact count"):
            session.flush()
        session.rollback()

    with migrated_db.begin() as connection:
        assert connection.scalars(select(table.c.id)).all() == [snapshot_id]
        # Insert malformed evidence through Core to exercise the defensive read path.
        point = graph["points"]
        connection.execute(
            table.update()
            .where(table.c.id == snapshot_id)
            .values(
                source_ids={
                    "numerator_point_ids": [point],
                    "denominator_point_ids": [point, point],
                }
            )
        )

    destination = tmp_path / "invalid-export"
    for args in (
        ["query", "metric_snapshots"],
        ["query", "metric_snapshots", "--format", "csv"],
        ["export", "--match-context", str(match)],
        [
            "export",
            "--match-context",
            str(match),
            "--format",
            "csv",
            "--output-dir",
            str(destination),
        ],
    ):
        assert main(args) == 1
        output = capsys.readouterr()
        assert output.out == ""
        assert f"Invalid metric_snapshots row {snapshot_id}" in output.err
        assert "must contain unique IDs" in output.err
    assert not destination.exists()
