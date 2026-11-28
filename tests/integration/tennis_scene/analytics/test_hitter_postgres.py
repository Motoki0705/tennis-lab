from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy import Connection, Engine, func, select
from sqlalchemy.exc import IntegrityError

from src.tennis_scene.analytics.__main__ import main
from src.tennis_scene.analytics.db.models import Base
from src.tennis_scene.analytics.hitter import DEFAULT_THRESHOLDS
from src.tennis_scene.analytics.hitter_v4 import file_sha256
from src.utils.schema.player import COCO_KP_IDX

pytestmark = [pytest.mark.integration, pytest.mark.postgres]
TABLES = Base.metadata.tables
CLIP_ID = "video_000/clip_001"
FPS = 60.0
FRAMES = 32
CONTACT = 8
MISSING = 23
ANNOTATION_SHA = "b" * 64


def _synthetic_v4(
    root: Path, *, extra_clip_id: str | None = None
) -> tuple[str, str, str]:
    scene = root / "scene.npz"
    roots: NDArray[np.float32] = np.empty((2, FRAMES, 3), dtype=np.float32)
    roots[0] = (4, -6, 1.0)
    roots[1] = (0, -6.6, 1.0)
    poses = np.broadcast_to(roots[:, :, None, :], (2, FRAMES, 17, 3)).copy()
    poses[1, :, COCO_KP_IDX["left_wrist"]] = (0.55, -6.0, 1.5)
    ball = np.broadcast_to(np.array([0, -6, 1.5], dtype=np.float32), (FRAMES, 3)).copy()
    ball_valid = np.arange(FRAMES) < 18
    np.savez(
        scene,
        num_frames=FRAMES,
        fps=FPS,
        ball_3d=ball,
        ball_3d_valid=ball_valid,
        player_position=roots,
        player_valid=np.ones((2, FRAMES), dtype=np.bool_),
        player_track_ids=np.array([27, 119], dtype=np.int32),
        player_kp_3d=poses,
        player_kp_3d_vis=np.ones((2, FRAMES, 17), dtype=np.bool_),
        ball_event_frame=np.array([CONTACT, MISSING], dtype=np.int32),
        ball_event_kind=np.array([0, 0], dtype=np.uint8),
        ball_event_subtype=np.array([0, 0], dtype=np.uint8),
        ball_event_status=np.array([1, 0], dtype=np.uint8),
        ball_event_frame_uncertainty=np.array([1, 1], dtype=np.uint8),
        ball_event_side=np.array([-1, -1], dtype=np.int8),
    )
    evidence = root / "event_detection.json"
    evidence.write_text(
        json.dumps(
            {
                "payload": {
                    "event_evidence": {
                        "events": [
                            {
                                "source_event_id": event_id,
                                "frame": frame,
                                "kind": "shot",
                                "subtype": "serve_overhead",
                                "status": status,
                                "frame_uncertainty": 1,
                                "player_hint": "near",
                            }
                            for event_id, frame, status in (
                                ("e011", CONTACT, "probable"),
                                ("e029", MISSING, "certain"),
                            )
                        ],
                        "player_hint_reference": {"near": -1, "far": 1},
                    }
                }
            }
        )
    )
    scene_sha = file_sha256(scene)
    event_sha = file_sha256(evidence)
    placeholder = {"path": "unused.json", "sha256": "a" * 64}
    manifest: dict[str, Any] = {
        "schema": "tennis_multiview_benchmark_v1",
        "benchmark_id": "meiji_4cam/multiview-v4",
        "scene_schema_version": 4,
        "splits": {
            "train": {"clips": 0, "clip_ids": []},
            "val": {"clips": 1, "clip_ids": [CLIP_ID]},
            "test": {"clips": 0, "clip_ids": []},
        },
        "clips": [
            {
                "clip_id": CLIP_ID,
                "frames": FRAMES,
                "fps": FPS,
                "scene": placeholder,
                "clip_json": placeholder,
                "cameras": [],
                "exports": {
                    "scene/scene": {"path": scene.name, "sha256": scene_sha},
                    "scene/metadata": placeholder,
                },
                "ball_events": {
                    "component": {"path": evidence.name, "sha256": event_sha},
                    "annotation_sha256": ANNOTATION_SHA,
                    "by_kind": {"shot": 2},
                },
            }
        ],
    }
    if extra_clip_id is not None:
        manifest["splits"]["val"] = {"clips": 2, "clip_ids": [CLIP_ID, extra_clip_id]}
        manifest["clips"].append({**manifest["clips"][0], "clip_id": extra_clip_id})
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return file_sha256(manifest_path), scene_sha, event_sha


def _insert(connection: Connection, table_name: str, **values: Any) -> int:
    table = TABLES[table_name]
    return int(
        connection.execute(
            table.insert().values(**values).returning(table.c.id)
        ).scalar_one()
    )


def _seed_run(
    connection: Connection, manifest_sha: str, scene_sha: str, event_sha: str
) -> tuple[int, int, int, int]:
    organization = _insert(connection, "organizations", slug="club", name="Club")
    team = _insert(
        connection, "teams", organization_id=organization, slug="first", name="First"
    )
    run = _insert(
        connection,
        "ingest_runs",
        owner_team_id=team,
        dataset_id="meiji_4cam",
        video_id="video_000",
        v4_manifest_sha256=manifest_sha,
        projects_json_sha256="c" * 64,
        schema_version=1,
        definition_version="v1",
        status="complete",
    )
    match = _insert(
        connection,
        "match_contexts",
        owner_team_id=team,
        ingest_run_id=run,
        dataset_id="meiji_4cam",
        video_id="video_000",
        match_key="match",
        sport_type="soft_tennis",
        rules_json={},
        time_conversion_json={},
    )
    clip = _insert(
        connection,
        "clips",
        owner_team_id=team,
        match_context_id=match,
        clip_id=CLIP_ID,
        clip_name="clip_001",
        split="val",
        global_start_sec=15,
        global_end_sec=15 + FRAMES / FPS,
        fps=FPS,
        num_frames=FRAMES,
        scene_schema_version=4,
        scene_npz_sha256=scene_sha,
        annotation_sha256=ANNOTATION_SHA,
        metadata_json={"event_detection": {"sha256": event_sha}},
    )
    common = {
        "owner_team_id": team,
        "clip_id": clip,
        "frame_uncertainty": 1,
        "kind": "shot",
        "subtype": "serve_overhead",
        "court_side": -1,
        "event_source": "annotation",
    }
    resolved = _insert(
        connection,
        "ball_events",
        **common,
        source_event_id="e011",
        frame=CONTACT,
        status="probable",
        time_s=CONTACT / FPS,
        adopted_in_v4=True,
        in_point=True,
        visibility="private",
    )
    unresolved = _insert(
        connection,
        "ball_events",
        **common,
        source_event_id="e029",
        frame=MISSING,
        status="certain",
        time_s=MISSING / FPS,
        adopted_in_v4=True,
        in_point=True,
    )
    _insert(
        connection,
        "ball_events",
        **common,
        source_event_id="e099",
        frame=CONTACT + 1,
        status="certain",
        time_s=(CONTACT + 1) / FPS,
        adopted_in_v4=False,
        in_point=False,
        excluded_reason="dead_ball",
    )
    return team, run, resolved, unresolved


def _store_command(root: Path, team: int, run: int) -> list[str]:
    return [
        "hitter",
        "--benchmark",
        str(root),
        "--ingest-run",
        str(run),
        "--owner-team",
        str(team),
    ]


def test_hitter_writer_validates_pin_and_round_trips_cli_exports(
    postgres_db: Engine, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["db", "upgrade"]) == 0
    capsys.readouterr()
    root = tmp_path / "v4"
    root.mkdir()
    manifest_sha, scene_sha, event_sha = _synthetic_v4(
        root, extra_clip_id="video_001/clip_001"
    )
    events = TABLES["ball_events"]
    with postgres_db.begin() as connection:
        team, run, resolved, unresolved = _seed_run(
            connection, manifest_sha, scene_sha, event_sha
        )
        connection.execute(
            events.update().where(events.c.id == unresolved).values(frame=MISSING - 1)
        )
    store = _store_command(root, team, run)
    table = TABLES["shot_hitter_estimates"]

    assert main(store) == 1
    assert "ingested shot differs" in capsys.readouterr().err
    with postgres_db.begin() as connection:
        assert connection.scalar(select(func.count()).select_from(table)) == 0
        connection.execute(
            events.update()
            .where(events.c.id == unresolved)
            .values(
                frame=MISSING, provenance_json={"event_component_sha256": event_sha}
            )
        )

    archive = root / "scene.npz"
    original = archive.read_bytes()
    archive.write_bytes(original + b"unexpected contents")
    assert main(store) == 1
    assert "V4 file SHA-256 mismatch" in capsys.readouterr().err
    archive.write_bytes(original)

    assert main(store) == 0
    report = json.loads(capsys.readouterr().out)
    assert (
        report["rows_changed"],
        report["resolved_count"],
        report["unresolved_count"],
    ) == (2, 1, 1)

    query = ["query", "shot_hitter_estimates", "--owner-team", str(team)]
    assert main(query) == 0
    rows = json.loads(capsys.readouterr().out)
    chosen, abstained = rows
    assert chosen["ball_event_id"] == resolved
    assert (chosen["local_player_index"], chosen["local_player_id"]) == (1, 119)
    assert chosen["visibility"] == "private"
    assert chosen["selected_landmark"] == "left_wrist"
    assert chosen["rule_version"] == DEFAULT_THRESHOLDS.rule_version
    assert chosen["candidates_json"][0]["contact"]["distance_m"] == pytest.approx(0.55)
    assert chosen["evidence_json"]["source_event_id"] == "e011"
    assert chosen["evidence_json"]["v4_manifest_sha256"] == manifest_sha
    assert chosen["evidence_json"]["scene_npz_sha256"] == scene_sha
    assert chosen["evidence_json"]["event_component_sha256"] == event_sha
    assert abstained["ball_event_id"] == unresolved
    assert abstained["local_player_id"] is None
    assert abstained["rejection_reason"] == "ball_3d_invalid"

    assert main(store) == 0
    assert json.loads(capsys.readouterr().out)["rows_changed"] == 0
    assert main(query) == 0
    assert json.loads(capsys.readouterr().out) == rows

    scope = ["export", "--ingest-run", str(run), "--owner-team", str(team)]
    assert main(scope) == 0
    assert json.loads(capsys.readouterr().out)["shot_hitter_estimates"] == rows
    destination = tmp_path / "csv"
    assert main([*scope, "--format", "csv", "--output-dir", str(destination)]) == 0
    with (destination / "shot_hitter_estimates.csv").open(newline="") as handle:
        csv_chosen, csv_abstained = csv.DictReader(handle)
    assert csv_chosen["local_player_id"] == "119"
    assert json.loads(csv_chosen["candidates_json"]) == chosen["candidates_json"]
    assert json.loads(csv_chosen["evidence_json"]) == chosen["evidence_json"]
    assert csv_abstained["local_player_index"] == r"\N"

    with postgres_db.connect() as connection, pytest.raises(IntegrityError):
        connection.execute(
            table.insert().values(
                owner_team_id=team,
                ball_event_id=resolved,
                rule_version=chosen["rule_version"],
                resolution_status="resolved",
                local_player_id=119,
                local_player_index=1,
            )
        )


@pytest.mark.parametrize(
    ("mismatch", "error"),
    [
        ("missing_clip", "missing=['video_000/clip_002']"),
        ("shot_status", "ingested shot differs from V4: status"),
        ("clip_component_sha", "ingested clip event component SHA-256 differs"),
        ("event_component_sha", "ingested shot event component SHA-256 differs"),
    ],
)
def test_hitter_writer_rejects_incomplete_or_stale_ingest_without_storing_estimates(
    postgres_db: Engine,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mismatch: str,
    error: str,
) -> None:
    assert main(["db", "upgrade"]) == 0
    capsys.readouterr()
    extra_clip_id = "video_000/clip_002" if mismatch == "missing_clip" else None
    manifest_sha, scene_sha, event_sha = _synthetic_v4(
        tmp_path, extra_clip_id=extra_clip_id
    )
    events = TABLES["ball_events"]
    clips = TABLES["clips"]
    with postgres_db.begin() as connection:
        team, run, _, unresolved = _seed_run(
            connection, manifest_sha, scene_sha, event_sha
        )
        if mismatch == "shot_status":
            connection.execute(
                events.update()
                .where(events.c.id == unresolved)
                .values(status="uncertain")
            )
        elif mismatch == "clip_component_sha":
            connection.execute(
                clips.update()
                .where(clips.c.clip_id == CLIP_ID)
                .values(metadata_json={"event_detection": {"sha256": "d" * 64}})
            )
        elif mismatch == "event_component_sha":
            connection.execute(
                events.update()
                .where(events.c.id == unresolved)
                .values(provenance_json={"event_component_sha256": "d" * 64})
            )

    assert main(_store_command(tmp_path, team, run)) == 1
    assert error in capsys.readouterr().err
    with postgres_db.connect() as connection:
        table = TABLES["shot_hitter_estimates"]
        assert connection.scalar(select(func.count()).select_from(table)) == 0
