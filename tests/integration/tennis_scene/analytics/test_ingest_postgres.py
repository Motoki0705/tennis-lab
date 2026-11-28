"""PostgreSQL S1 ingest, evidence queries and replay stability."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest
from sqlalchemy import Engine, func, insert, select

from src.tennis_scene.analytics.__main__ import main
from src.tennis_scene.analytics.db.models import Base
from src.tennis_scene.analytics.ingest.audit import prepare_ingest
from src.tennis_scene.analytics.ingest.score_adapter import read_scores
from src.tennis_scene.analytics.ingest.store import persist_ingest
from src.tennis_scene.analytics.ingest.v4_reader import read_v4
from tests.support.tennis_scene.analytics_ingest import (
    digest,
    make_ingest_fixture,
    write_json,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


def _create_owner(db: Engine) -> int:
    tables = Base.metadata.tables
    organizations, teams = tables["organizations"], tables["teams"]
    with db.begin() as connection:
        org = connection.execute(
            insert(organizations)
            .values(slug="club", name="Club")
            .returning(organizations.c.id)
        ).scalar_one()
        return int(
            connection.execute(
                insert(teams)
                .values(organization_id=org, slug="data", name="Data team")
                .returning(teams.c.id)
            ).scalar_one()
        )


def _cli(capsys: pytest.CaptureFixture[str], argv: list[str]) -> str:
    status = main(argv)
    captured = capsys.readouterr()
    assert status == 0, captured.err
    output: str = captured.out
    return output


def test_cli_ingest_query_export_and_replay_in_postgres(
    postgres_db: Engine,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["db", "upgrade"]) == 0
    owner = _create_owner(postgres_db)
    inputs = make_ingest_fixture(tmp_path, duplicate="shared_serve")
    ingest = ["ingest", *inputs.cli_args(), "--owner-team", str(owner)]
    original = _cli(capsys, ingest)
    run = json.loads(original)
    scope = ["--ingest-run", str(run["ingest_run_id"]), "--owner-team", str(owner)]
    snapshot = _cli(capsys, ["export", *scope])
    assert _cli(capsys, ingest) == original
    assert _cli(capsys, ["export", *scope]) == snapshot
    exported = json.loads(snapshot)
    assert {
        table: len(exported[table])
        for table in ("ingest_runs", "match_contexts", "clips", "points", "ball_events")
    } == {
        "ingest_runs": 1,
        "match_contexts": 1,
        "clips": 4,
        "points": 4,
        "ball_events": 26,
    }
    assert exported["ingest_runs"][0]["manual_supplement_sha256"] is None
    points = exported["points"]
    assert points[0]["match_point_key"] == points[1]["match_point_key"]
    assert (
        points[0]["match_context_id"]
        == points[1]["match_context_id"]
        == run["match_context_id"]
    )
    assert points[2]["score_before_json"] == {"games": [0, 0], "points": [1, 0]}
    assert (points[2]["terminal_cause"], points[2]["terminal_cause_source"]) == (
        "unknown",
        "unknown",
    )
    events = _cli(capsys, ["query", "ball_events", "--owner-team", str(owner)])
    assert json.loads(events) == exported["ball_events"]
    clip = exported["clips"][0]
    assert clip["clip_id"] == "video_000/clip_009"
    event = next(
        row
        for row in exported["ball_events"]
        if row["clip_id"] == clip["id"] and row["source_event_id"] == "e003"
    )
    assert (event["adopted_in_v4"], event["in_point"], event["point_id"]) == (
        True,
        False,
        None,
    )
    assert event["time_s"] == pytest.approx(7 / 30)
    assert event["provenance_json"]["event_evidence"]["old_in_point"] is False

    manual = tmp_path / "review.json"
    write_json(
        manual,
        {
            "version": 1,
            "points": [
                {
                    "clip_id": "video_000/clip_009",
                    "point_key": "point",
                    "reviewer": "Reviewer",
                    "terminal_cause": "out_wide",
                    "error_assessment": "unforced",
                    "serve_attempts": [
                        {
                            "source_event_id": "e001",
                            "serve_number": 1,
                            "replay_index": 0,
                            "result": "let",
                        },
                        {
                            "source_event_id": "e004",
                            "serve_number": 1,
                            "replay_index": 1,
                            "result": "in",
                        },
                    ],
                }
            ],
        },
    )
    with_manual = [*ingest, "--manual-supplement", str(manual)]
    new_run = json.loads(_cli(capsys, with_manual))
    assert new_run["ingest_run_id"] != run["ingest_run_id"]
    new_scope = [
        "--ingest-run",
        str(new_run["ingest_run_id"]),
        "--owner-team",
        str(owner),
    ]
    new_snapshot = _cli(capsys, ["export", *new_scope])
    new_export = json.loads(new_snapshot)
    assert new_export["ingest_runs"][0]["manual_supplement_sha256"] == digest(manual)
    for point in new_export["points"][:2]:
        assert point["match_point_key"] == points[0]["match_point_key"]
        assert point["terminal_cause"] == "out_wide"
        assert (
            point["terminal_cause_source"]
            == point["error_assessment_source"]
            == "manual_confirmed"
        )
        assert point["error_assessment"] == "unforced"
    attempts = new_export["serve_attempts"]
    assert [(row["result"], row["replay_index"]) for row in attempts] == [
        ("let", 0),
        ("in", 1),
    ]
    assert all(
        row["source_kind"] == "manual_confirmed"
        and row["point_id"] == new_export["points"][0]["id"]
        for row in attempts
    )
    linked = {row["id"]: row["source_event_id"] for row in new_export["ball_events"]}
    assert [linked[row["ball_event_id"]] for row in attempts] == ["e001", "e004"]
    assert json.loads(_cli(capsys, with_manual)) == new_run
    assert _cli(capsys, ["export", *new_scope]) == new_snapshot
    assert _cli(capsys, ["export", *scope]) == snapshot


def test_concurrent_identical_ingests_share_one_graph(
    postgres_db: Engine, tmp_path: Path
) -> None:
    assert main(["db", "upgrade"]) == 0
    owner = _create_owner(postgres_db)
    inputs = make_ingest_fixture(tmp_path)
    release = read_v4(inputs.root, manifest_sha256=inputs.manifest_sha256)
    batch = prepare_ingest(
        read_scores(release, inputs.projects, inputs.resolver()), None, {}
    )
    start = Barrier(2, timeout=5)

    def write() -> dict[str, Any]:
        start.wait()
        return persist_ingest(postgres_db, owner, batch)

    with ThreadPoolExecutor(max_workers=2) as workers:
        first, second = [workers.submit(write) for _ in range(2)]
        assert first.result(timeout=10) == second.result(timeout=10)
    counts = {
        "ingest_runs": 1,
        "match_contexts": 1,
        "clips": 3,
        "points": 3,
        "ball_events": 19,
    }
    with postgres_db.connect() as connection:
        for table, count in counts.items():
            records = Base.metadata.tables[table]
            assert connection.scalar(select(func.count()).select_from(records)) == count


def test_ingest_rejects_contradictory_manual_rows_before_any_insert(
    postgres_db: Engine,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["db", "upgrade"]) == 0
    owner = _create_owner(postgres_db)
    inputs = make_ingest_fixture(tmp_path)
    manual = tmp_path / "contradiction.json"
    write_json(
        manual,
        {
            "version": 1,
            "points": [
                {
                    "clip_id": "video_000/clip_012",
                    "point_key": "point",
                    "reviewer": "Reviewer",
                    "terminal_cause": "double_fault",
                    "serve_attempts": [{"serve_number": 2, "result": "in"}],
                }
            ],
        },
    )
    assert (
        main(
            [
                "ingest",
                *inputs.cli_args(),
                "--owner-team",
                str(owner),
                "--manual-supplement",
                str(manual),
            ]
        )
        == 1
    )
    assert "double_fault with a supplied serve sequence" in capsys.readouterr().err
    with postgres_db.connect() as connection:
        for name in (
            "ingest_runs",
            "match_contexts",
            "clips",
            "points",
            "ball_events",
            "serve_attempts",
        ):
            assert (
                connection.scalar(
                    select(func.count()).select_from(Base.metadata.tables[name])
                )
                == 0
            )


def test_ingest_persists_unresolved_identity_with_clip_local_score(
    postgres_db: Engine,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["db", "upgrade"]) == 0
    owner = _create_owner(postgres_db)
    inputs = make_ingest_fixture(
        tmp_path, duplicate="ambiguous", duplicate_split="test"
    )
    ingest = ["ingest", *inputs.cli_args(), "--owner-team", str(owner)]
    result = _cli(capsys, ingest)
    run = json.loads(result)
    assert _cli(capsys, ingest) == result
    export = json.loads(
        _cli(capsys, ["export", "--ingest-run", str(run["ingest_run_id"])])
    )
    clip = next(
        row for row in export["clips"] if row["clip_id"] == "video_000/clip_010"
    )
    assert clip["metadata_json"]["point_identity_hold"] == {
        "reason": "overlapping_same_score_without_shared_event",
        "candidate_clip_ids": ["video_000/clip_009"],
    }
    point = next(row for row in export["points"] if row["clip_id"] == clip["id"])
    assert point["match_point_key"] is None
    assert point["score_status"] == "confirmed"
    assert point["score_after_json"] == {"games": [0, 0], "points": [1, 0]}
