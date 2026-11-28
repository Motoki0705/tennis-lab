from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from src.tennis_scene.analytics import geometry_v4
from src.tennis_scene.analytics.__main__ import main
from src.tennis_scene.analytics.db.geometry import persist_v4_geometry
from src.tennis_scene.analytics.db.models import (
    BallEvent,
    Clip,
    IngestRun,
    MatchContext,
    Organization,
    Point,
    Team,
)
from src.tennis_scene.analytics.geometry import BallTrajectory, GeometryEvent
from src.tennis_scene.analytics.geometry_v4 import (
    MEIJI_V4_MANIFEST_SHA256,
    V4GeometryClip,
)
from src.tennis_scene.analytics.profiles import resolve_profile

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


def _insert_v4_evidence(db: Engine) -> tuple[int, int, V4GeometryClip]:
    events = (
        GeometryEvent(
            "e1", 1, "shot", "groundstroke_forehand", court_side=-1, status="certain"
        ),
        GeometryEvent("e2", 9, "bounce", "bounce", status="certain"),
        GeometryEvent(
            "e3", 12, "shot", "groundstroke_backhand", court_side=1, status="certain"
        ),
        GeometryEvent("e4", 18, "net", "net_fault", status="certain"),
    )
    track = np.stack(
        (
            np.linspace(-2, 1, 22),
            [-8, -7, -6, -5, -2, -1, 1, 3, 4, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1],
            np.full(22, 1.4),
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
                end_frame=21,
            )
        )
        for event in events:
            add(
                BallEvent(
                    owner_team_id=owner,
                    clip_id=clip_id,
                    point_id=point,
                    source_event_id=event.source_event_id,
                    kind=event.kind,
                    subtype=event.subtype,
                    frame=event.frame,
                    time_s=event.frame / 30.0,
                    frame_uncertainty=int(event.frame_uncertainty or 0),
                    status=event.status,
                    court_side=event.court_side,
                    event_source="annotation",
                    adopted_in_v4=True,
                    in_point=True,
                )
            )
    return owner, run, clip


def test_geometry_cli_persists_and_exports_crossing_and_missing_landing_evidence(
    postgres_db: Engine,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["db", "upgrade"]) == 0
    owner, run, clip = _insert_v4_evidence(postgres_db)

    # Frozen array decoding is exercised by the all-clip audit. Supply the
    # bound trajectory here so the real write/query/export use an isolated DB.
    monkeypatch.setattr(geometry_v4, "read_frozen_v4", lambda root, digest: (clip,))
    overrides = tmp_path / "venue.json"
    overrides.write_text(
        json.dumps(
            {
                "profile_key": "measured-venue",
                "source_kind": "measured",
                "net_height_center_m": 1.02,
                "net_height_post_m": 1.02,
                "source_reference": "fixture measurement",
            }
        )
    )
    extract = [
        "geometry",
        "--benchmark",
        str(tmp_path),
        "--profile-overrides",
        str(overrides),
        "--owner-team",
        str(owner),
        "--ingest-run",
        str(run),
    ]
    assert main(extract) == 0
    audit = json.loads(capsys.readouterr().out)
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
    assert len(evidence["flight_arcs"]) == 5
    launch = evidence["flight_arcs"][1]
    assert (launch["start_event_id"], launch["end_event_id"]) == (
        events["e1"],
        events["e2"],
    )
    assert launch["quality_json"]["sample_runs_inclusive"] == [[2, 8]]
    assert launch["quality_json"]["event_detection_sha256"] == clip.component_sha256
    landed, stopped = evidence["shot_geometry_features"]
    assert landed["speed_window_frames"] == {"incoming": [], "outgoing": [2, 3]}
    assert landed["contact_height_m"] == 1.4
    assert landed["landing_depth_m"] == pytest.approx(5.885)
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
    assert crossing["court_net_profile_id"] == profile["id"]
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

    # A conflicting profile must roll back all of its proposed rows.
    with (
        postgres_db.begin() as connection,
        pytest.raises(ValueError, match="Conflicting net_plane_crossings evidence"),
    ):
        persist_v4_geometry(
            connection,
            owner_team_id=owner,
            ingest_run_id=run,
            clips=(clip,),
            profile=resolve_profile(dataset_id="meiji_4cam"),
            manifest_sha256=MEIJI_V4_MANIFEST_SHA256,
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
