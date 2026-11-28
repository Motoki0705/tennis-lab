from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy import Engine, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.tennis_scene.analytics import geometry_v4
from src.tennis_scene.analytics.__main__ import main
from src.tennis_scene.analytics.db.geometry import (
    persist_v4_geometry,
    save_court_net_profile,
)
from src.tennis_scene.analytics.db.models import (
    BallEvent,
    Clip,
    CourtNetProfile,
    FlightArc,
    IngestRun,
    MatchContext,
    Organization,
    Point,
    ShotGeometryFeature,
    Team,
)
from src.tennis_scene.analytics.geometry import BallTrajectory, GeometryEvent
from src.tennis_scene.analytics.geometry_v4 import (
    MEIJI_V4_MANIFEST_SHA256,
    V4GeometryClip,
)
from src.tennis_scene.analytics.profiles import resolve_profile

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


def _insert_v4_evidence(db: Engine) -> tuple[int, int, int, V4GeometryClip]:
    events = (
        GeometryEvent(
            "e1", 1, "shot", "groundstroke_forehand", court_side=-1, status="certain"
        ),
        GeometryEvent("e2", 9, "bounce", "bounce", status="certain"),
        GeometryEvent(
            "return",
            12,
            "shot",
            frame_uncertainty=None,
            in_point=False,
            excluded_reason="dead-ball return after fault",
        ),
        GeometryEvent(
            "e3", 20, "shot", "groundstroke_backhand", court_side=1, status="certain"
        ),
        GeometryEvent("e4", 26, "net", "net_fault", status="certain"),
    )
    track = np.stack(
        (
            np.linspace(-2, 1, 29),
            [
                -8,
                -7,
                -6,
                -5,
                -2,
                -1,
                1,
                3,
                4,
                6,
                7,
                8,
                7,
                6,
                5,
                1,
                -1,
                -4,
                -2,
                3,
                5,
                4,
                3,
                2,
                1.2,
                1,
                1,
                1,
                1,
            ],
            np.full(29, 1.4),
        ),
        axis=1,
    )
    track[9, 2] = 0.033
    clip = V4GeometryClip(
        "video_000/clip_000",
        "train",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        BallTrajectory(track, np.ones(len(track), dtype=bool), 30.0),
        events,
    )
    with Session(db) as session, session.begin():

        def add(row: Any) -> int:
            session.add(row)
            session.flush()
            return int(row.id)

        org = add(Organization(slug="club", name="Club"))
        owner = add(Team(organization_id=org, slug="analytics", name="Analytics"))
        run = add(
            IngestRun(
                owner_team_id=owner,
                dataset_id="meiji_4cam",
                video_id="video_000",
                v4_manifest_sha256=MEIJI_V4_MANIFEST_SHA256,
                projects_json_sha256="d" * 64,
                schema_version=1,
                definition_version="s1_test",
                status="complete",
            )
        )
        match = add(
            MatchContext(
                owner_team_id=owner,
                ingest_run_id=run,
                dataset_id="meiji_4cam",
                video_id="video_000",
                match_key="example",
                sport_type="soft_tennis",
                rules_json={"format": "doubles"},
                time_conversion_json={},
            )
        )
        clip_id = add(
            Clip(
                owner_team_id=owner,
                match_context_id=match,
                clip_id=clip.clip_id,
                clip_name="clip_000",
                split=clip.split,
                global_start_sec=17,
                global_end_sec=17 + len(track) / 30,
                fps=30,
                num_frames=len(track),
                scene_schema_version=4,
                scene_npz_sha256=clip.scene_npz_sha256,
                annotation_sha256=clip.annotation_sha256,
            )
        )
        point = add(
            Point(
                owner_team_id=owner,
                clip_id=clip_id,
                match_context_id=match,
                point_key="pt-1",
                start_frame=0,
                end_frame=len(track) - 1,
            )
        )
        for event in events:
            add(
                BallEvent(
                    owner_team_id=owner,
                    clip_id=clip_id,
                    point_id=point if event.in_point else None,
                    source_event_id=event.source_event_id,
                    kind=event.kind,
                    subtype=event.subtype,
                    frame=event.frame,
                    time_s=event.frame / 30.0,
                    frame_uncertainty=int(event.frame_uncertainty)
                    if event.frame_uncertainty is not None
                    else None,
                    status=event.status,
                    court_side=event.court_side if event.in_point else None,
                    event_source="annotation",
                    adopted_in_v4=event.excluded_reason is None,
                    in_point=event.in_point,
                    excluded_reason=event.excluded_reason,
                )
            )
    return owner, run, match, clip


def test_geometry_cli_persists_and_exports_crossing_and_missing_landing_evidence(
    postgres_db: Engine,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["db", "upgrade"]) == 0
    owner, run, match_id, clip = _insert_v4_evidence(postgres_db)
    with Session(postgres_db) as session, session.begin():
        profile = CourtNetProfile(
            owner_team_id=owner,
            profile_key="measured-venue",
            source_kind="measured",
            visibility="private",
            sport_type="soft_tennis",
            net_profile_kind="flat",
            net_height_center_m=1.02,
            net_height_post_m=1.02,
            net_post_offset_x_m=0.914,
            ball_radius_m=0.033,
            parameters_json={
                "court_length_m": 24.0,
                "source_reference": "fixture measurement",
            },
        )
        session.add(profile)
        session.flush()
        measured_id = profile.id
        match = session.get(MatchContext, match_id)
        assert match is not None
        match.court_net_profile_id = measured_id

    # Frozen array decoding is exercised by the all-clip audit. Supply the
    # bound trajectory here so the real write/query/export use an isolated DB.
    monkeypatch.setattr(geometry_v4, "read_frozen_v4", lambda root, digest: (clip,))
    extract = [
        "geometry",
        "--benchmark",
        str(tmp_path),
        "--owner-team",
        str(owner),
        "--ingest-run",
        str(run),
    ]
    assert main(extract) == 0
    audit = json.loads(capsys.readouterr().out)
    assert audit["profile"]["profile_key"] == "measured-venue"
    assert audit["overall"]["shot_crossing_statuses"] == {
        "crossed": 1,
        "no_crossing_confirmed": 1,
    }
    assert audit["overall"]["availability"]["landing_depth"]["missing_reasons"] == {
        "net_stopped": 1,
    }

    scope = ["--ingest-run", str(run), "--owner-team", str(owner)]
    assert main(["export", *scope]) == 0
    evidence = json.loads(capsys.readouterr().out)
    assert main(extract) == 0
    assert json.loads(capsys.readouterr().out) == audit
    assert main(["export", *scope]) == 0
    assert json.loads(capsys.readouterr().out) == evidence

    (profile,) = evidence["court_net_profiles"]
    assert (profile["profile_key"], profile["source_kind"]) == (
        "measured-venue",
        "measured",
    )
    assert profile["parameters_json"]["source_reference"] == "fixture measurement"
    events = {row["source_event_id"]: row["id"] for row in evidence["ball_events"]}
    assert len(evidence["flight_arcs"]) == 6
    launch = evidence["flight_arcs"][1]
    assert (launch["start_event_id"], launch["end_event_id"]) == (
        events["e1"],
        events["e2"],
    )
    assert launch["quality_json"]["sample_runs_inclusive"] == [[2, 8]]
    assert launch["quality_json"]["event_detection_sha256"] == clip.component_sha256
    returned = evidence["flight_arcs"][3]
    assert returned["start_event_id"] == events["return"]
    assert returned["shot_event_id"] is None
    assert returned["quality_json"]["sample_runs_inclusive"] == [[13, 19]]
    assert (
        len([r for r in evidence["net_plane_crossings"] if r["status"] == "crossed"])
        == 1
    )
    landed, stopped = evidence["shot_geometry_features"]
    assert landed["speed_window_frames"] == {"incoming": [], "outgoing": [2, 3]}
    assert landed["contact_height_m"] == 1.4
    assert landed["landing_depth_m"] == pytest.approx(6.0)
    assert landed["visibility"] == "private" and launch["visibility"] == "team"
    assert (
        landed["landing_source_kind"] == "observed"
        and landed["landing_missing_reason"] is None
    )
    assert landed["flight_arc_id"] == launch["id"]
    assert (
        stopped["landing_source_kind"]
        is stopped["first_bounce_frame"]
        is stopped["landing_depth_m"]
        is None
    )
    assert stopped["landing_missing_reason"] == "net_stopped"
    (crossing,) = (row for row in evidence["net_plane_crossings"] if row["is_primary"])
    assert crossing["court_net_profile_id"] == profile["id"] == measured_id
    assert crossing["profile_source_kind"] == "measured"
    assert crossing["time_to_net_s"] == pytest.approx(4.5 / 30)
    assert crossing["fractional_frame"] == pytest.approx(5.5)
    assert crossing["net_clearance_bottom_m"] == pytest.approx(0.347)
    assert crossing["used_frames_json"] == [5, 6]
    for row in evidence["net_plane_crossings"]:
        if row["status"] != "crossed":
            assert (
                row["time_s"]
                is row["x_m"]
                is row["z_center_m"]
                is row["time_to_net_s"]
                is None
            )

    target = tmp_path / "csv"
    assert main(["export", *scope, "--format", "csv", "--output-dir", str(target)]) == 0
    for name in (
        "court_net_profiles",
        "flight_arcs",
        "shot_geometry_features",
        "net_plane_crossings",
    ):
        assert main(["query", name, "--owner-team", str(owner)]) == 0
        assert json.loads(capsys.readouterr().out) == evidence[name]
        with (target / f"{name}.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert [int(row["id"]) for row in rows] == [row["id"] for row in evidence[name]]
        for actual, expected in zip(rows, evidence[name], strict=True):
            for field, value in expected.items():
                if value is None:
                    assert actual[field] == r"\N"
                elif isinstance(value, str):
                    assert actual[field] == value
                else:
                    assert json.loads(actual[field]) == value

    assert main([*extract, "--profile", "soft_tennis_standard"]) == 0
    assert (
        json.loads(capsys.readouterr().out)["profile"]["profile_key"]
        == "soft_tennis_standard"
    )
    assert main(["export", *scope]) == 0
    explicit = json.loads(capsys.readouterr().out)
    assert explicit["match_contexts"][0]["court_net_profile_id"] == measured_id
    (standard_crossing,) = (
        row for row in explicit["net_plane_crossings"] if row["is_primary"]
    )
    assert standard_crossing["profile_source_kind"] == "rule_default"
    assert standard_crossing["net_clearance_bottom_m"] == pytest.approx(0.297)


def test_measured_override_replaces_default_rows_atomically(
    postgres_db: Engine,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["db", "upgrade"]) == 0
    owner, run, _, clip = _insert_v4_evidence(postgres_db)
    monkeypatch.setattr(geometry_v4, "read_frozen_v4", lambda root, digest: (clip,))
    scope = ["--ingest-run", str(run), "--owner-team", str(owner)]
    extract = ["geometry", "--benchmark", str(tmp_path), *scope]
    assert main(extract) == 0
    assert (
        json.loads(capsys.readouterr().out)["profile"]["profile_key"]
        == "soft_tennis_standard"
    )
    assert main(["export", *scope]) == 0
    default = json.loads(capsys.readouterr().out)

    measured_overrides = {
        "profile_key": "post-survey",
        "source_kind": "measured",
        "net_height_center_m": 1.02,
        "net_height_post_m": 1.02,
        "source_reference": "venue survey",
    }
    # Force a database failure after replacing the old rows inside the savepoint.
    with postgres_db.begin() as connection:
        connection.execute(
            text(
                "ALTER TABLE net_plane_crossings ADD CONSTRAINT test_reject_measured_crossings "
                "CHECK (profile_source_kind <> 'measured')"
            )
        )
        with pytest.raises(IntegrityError, match="test_reject_measured_crossings"):
            persist_v4_geometry(
                connection,
                owner_team_id=owner,
                ingest_run_id=run,
                clips=(clip,),
                overrides=measured_overrides,
                manifest_sha256=MEIJI_V4_MANIFEST_SHA256,
            )
        connection.execute(
            text(
                "ALTER TABLE net_plane_crossings DROP CONSTRAINT test_reject_measured_crossings"
            )
        )
    assert main(["export", *scope]) == 0
    assert json.loads(capsys.readouterr().out) == default
    assert main(["query", "court_net_profiles", "--owner-team", str(owner)]) == 0
    assert json.loads(capsys.readouterr().out) == default["court_net_profiles"]

    overrides_path = tmp_path / "survey.json"
    overrides_path.write_text(json.dumps(measured_overrides))
    measured_extract = [*extract, "--profile-overrides", str(overrides_path)]
    assert main(measured_extract) == 0
    audit = json.loads(capsys.readouterr().out)
    assert audit["profile"]["profile_key"] == "post-survey"
    assert main(["export", *scope]) == 0
    measured = json.loads(capsys.readouterr().out)
    for table in (
        "ingest_runs",
        "match_contexts",
        "clips",
        "points",
        "ball_events",
        "flight_arcs",
    ):
        assert measured[table] == default[table]
    (profile,) = measured["court_net_profiles"]
    assert profile["parameters_json"]["source_reference"] == "venue survey"
    assert (
        len(measured["shot_geometry_features"])
        == len(default["shot_geometry_features"])
        == 2
    )
    assert {row["profile_id"] for row in measured["shot_geometry_features"]} == {
        profile["id"]
    }
    assert {row["court_net_profile_id"] for row in measured["net_plane_crossings"]} == {
        profile["id"]
    }
    (crossing,) = (row for row in measured["net_plane_crossings"] if row["is_primary"])
    assert crossing["profile_source_kind"] == "measured"
    assert crossing["net_clearance_bottom_m"] == pytest.approx(0.347)
    assert main(measured_extract) == 0
    assert json.loads(capsys.readouterr().out) == audit
    assert main(["export", *scope]) == 0
    assert json.loads(capsys.readouterr().out) == measured
    assert main(["query", "court_net_profiles", "--owner-team", str(owner)]) == 0
    assert json.loads(capsys.readouterr().out) == [
        *default["court_net_profiles"],
        profile,
    ]

    with postgres_db.begin() as connection:
        hard_id = save_court_net_profile(
            connection,
            owner,
            resolve_profile(dataset_id="meiji_4cam", profile_key="tennis_itf_standard"),
        )
    assert main(["query", "court_net_profiles", "--id", str(hard_id)]) == 0
    (hard,) = json.loads(capsys.readouterr().out)
    assert hard["net_profile_kind"] == "itf_linear"


def test_reextraction_refreshes_prior_s4_definition_for_the_same_pins(
    postgres_db: Engine,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["db", "upgrade"]) == 0
    owner, run, _, clip = _insert_v4_evidence(postgres_db)
    with postgres_db.begin() as connection:

        def reextract() -> None:
            persist_v4_geometry(
                connection,
                owner_team_id=owner,
                ingest_run_id=run,
                clips=(clip,),
                manifest_sha256=MEIJI_V4_MANIFEST_SHA256,
            )

        reextract()
        arc_table = FlightArc.__table__
        old_arcs = (
            connection.execute(
                update(arc_table)
                .values(
                    quality_json=arc_table.c.quality_json.op("||")(
                        {"definition_version": "s4_geometry.v1"}
                    )
                )
                .returning(arc_table.c.id)
            )
            .scalars()
            .all()
        )
        connection.execute(
            update(ShotGeometryFeature.__table__).values(
                definition_version="s4_geometry.v1",
                bin_version="s4_bins.v1",
            )
        )
        reextract()

    assert main(["export", "--ingest-run", str(run), "--owner-team", str(owner)]) == 0
    refreshed = json.loads(capsys.readouterr().out)
    updated = refreshed["flight_arcs"]
    assert len(updated) == len(old_arcs)
    assert not {row["id"] for row in updated}.intersection(old_arcs)
    assert {row["quality_json"]["manifest_sha256"] for row in updated} == {
        MEIJI_V4_MANIFEST_SHA256
    }
    assert {
        row["definition_version"] for row in refreshed["shot_geometry_features"]
    } == {"s4_geometry.v2"}
