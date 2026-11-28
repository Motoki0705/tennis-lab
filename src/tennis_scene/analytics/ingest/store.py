"""Persist one checked release in a transaction keyed by the input hashes."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from sqlalchemy import Connection, Engine, Table, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.tennis_scene.analytics.db.models import SCHEMA_VERSION, Base
from src.tennis_scene.analytics.ingest.audit import (
    PreparedClip,
    PreparedIngest,
    audit_report,
)
from src.tennis_scene.analytics.ingest.score_adapter import POINT_KEY
from src.tennis_scene.analytics.metrics import DEFINITION_VERSION
from src.tennis_scene.clip_studio.project import count_to_dict

_TABLES = Base.metadata.tables


def _insert(connection: Connection, table: Table, values: dict[str, Any]) -> int:
    return int(
        connection.execute(
            insert(table).values(values).returning(table.c.id)
        ).scalar_one()
    )


def _store_clip(
    connection: Connection, owner: int, match_id: int, item: PreparedClip
) -> None:
    scored = item.scored
    clip = scored.v4
    score = scored.confirmed
    metadata = {
        **clip.provenance,
        "clip_studio_score": scored.recorded_score.to_dict()
        if scored.recorded_score is not None
        else None,
        "terminal_v4_evidence": {
            "cause": item.terminal.cause,
            "clip_id": item.terminal_clip_id,
            "source_event_ids": list(item.terminal.event_ids),
            "missing_reason": item.terminal.missing_reason,
        },
    }
    if item.manual_point is not None:
        metadata["manual_point_supplement"] = {
            "clip_id": item.manual_point.clip_id,
            "point_key": item.manual_point.point_key,
            "reviewer": item.manual_point.reviewer,
            "note": item.manual_point.note,
            "terminal_cause": item.manual_point.terminal_cause,
            "error_assessment": item.manual_point.error_assessment,
        }
    clip_pk = _insert(
        connection,
        _TABLES["clips"],
        {
            "owner_team_id": owner,
            "match_context_id": match_id,
            "clip_id": clip.clip_id,
            "clip_name": clip.clip.clip_name,
            "split": clip.split,
            "global_start_sec": clip.global_start_sec,
            "global_end_sec": clip.global_end_sec,
            "fps": clip.clip.fps,
            "num_frames": clip.clip.num_frames,
            "scene_schema_version": 4,
            "scene_npz_sha256": clip.provenance["scene_npz"]["sha256"],
            "annotation_sha256": clip.provenance["annotation_provenance"][
                "annotation_sha256"
            ],
            "metadata_json": metadata,
        },
    )
    point_pk = None
    if score is not None:
        assert score.before is not None and scored.score_after is not None
        point_pk = _insert(
            connection,
            _TABLES["points"],
            {
                "owner_team_id": owner,
                "clip_id": clip_pk,
                "match_context_id": match_id,
                "point_key": POINT_KEY,
                "match_point_key": scored.match_point_key,
                "score_status": score.status,
                "score_before_json": count_to_dict(score.before),
                "score_after_json": count_to_dict(scored.score_after),
                "server_team": score.server,
                "winner_team": score.winner,
                "score_source": "projects_json",
                "terminal_cause": item.terminal_cause,
                "terminal_cause_source": item.terminal_cause_source,
                "error_assessment": item.error_assessment,
                "error_assessment_source": item.error_assessment_source,
                "note": score.note,
            },
        )
    event_values = []
    for event in clip.events:
        in_point = event.in_point and scored.score_status != "not_point"
        event_values.append(
            {
                "owner_team_id": owner,
                "clip_id": clip_pk,
                "point_id": point_pk if in_point else None,
                "source_event_id": event.source_event_id,
                "frame": event.frame,
                "time_s": event.frame / clip.clip.fps,
                "frame_uncertainty": event.frame_uncertainty,
                "kind": event.kind,
                "subtype": event.subtype,
                "status": event.status,
                "court_side": event.court_side,
                "event_source": "annotation",
                "adopted_in_v4": event.adopted_in_v4,
                "in_point": in_point,
                "excluded_reason": event.excluded_reason,
                "provenance_json": {
                    "event_evidence": event.evidence,
                    "v4_point_eligibility": event.in_point,
                    "outside_point_reason": (
                        "projects_not_point"
                        if event.adopted_in_v4 and scored.score_status == "not_point"
                        else event.outside_point_reason
                    ),
                },
            }
        )
    event_ids = {}
    if event_values:
        events = _TABLES["ball_events"]
        event_ids = dict(
            connection.execute(
                insert(events).returning(events.c.source_event_id, events.c.id),
                event_values,
            )
            .tuples()
            .all()
        )
    manual = item.manual_attempts
    if manual is not None and manual.serve_attempts:
        assert point_pk is not None
        values = []
        for index, attempt in enumerate(manual.serve_attempts):
            details = asdict(attempt)
            event_id = details.pop("source_event_id")
            values.append(
                {
                    **details,
                    "owner_team_id": owner,
                    "clip_id": clip_pk,
                    "point_id": point_pk,
                    "ball_event_id": event_ids[event_id]
                    if event_id is not None
                    else None,
                    "attempt_index": index,
                    "source_kind": "manual_confirmed",
                    "evidence_json": {
                        "reviewer": manual.reviewer,
                        "note": manual.note,
                        "source_event_id": event_id,
                    },
                }
            )
        connection.execute(insert(_TABLES["serve_attempts"]), values)


def persist_ingest(
    engine: Engine, owner_team_id: int, batch: PreparedIngest
) -> dict[str, int]:
    """Insert all rows exactly once, even when identical runs start concurrently."""
    video, release = batch.video, batch.video.release
    summary = audit_report(batch)
    identity = {
        "owner_team_id": owner_team_id,
        "dataset_id": release.dataset_id,
        "video_id": release.video_id,
        "v4_manifest_sha256": release.manifest_sha256,
        "projects_json_sha256": video.projects_sha256,
        "manual_supplement_sha256": batch.manual_supplement_sha256,
        "schema_version": SCHEMA_VERSION,
        "definition_version": DEFINITION_VERSION,
    }
    run = _TABLES["ingest_runs"]
    matches = _TABLES["match_contexts"]
    with engine.begin() as connection:
        if (
            connection.execute(
                select(_TABLES["teams"].c.id).filter_by(id=owner_team_id)
            ).scalar_one_or_none()
            is None
        ):
            raise ValueError(f"Unknown owner team {owner_team_id}")
        # The unique constraint waits for an identical concurrent transaction.
        # At PostgreSQL READ COMMITTED, the next statement can see its commit.
        run_id = connection.execute(
            pg_insert(run)
            .values(**identity, status="complete", summary_json=summary)
            .on_conflict_do_nothing(constraint="uq_ingest_runs_inputs")
            .returning(run.c.id)
        ).scalar_one_or_none()
        if run_id is None:
            existing = (
                connection.execute(select(run).filter_by(**identity)).mappings().one()
            )
            if existing["status"] != "complete" or existing["summary_json"] != summary:
                raise ValueError(
                    "Stored ingest run disagrees with the verified input summary"
                )
            run_id = existing["id"]
            match_id = connection.execute(
                select(matches.c.id).where(
                    matches.c.owner_team_id == owner_team_id,
                    matches.c.ingest_run_id == run_id,
                    matches.c.match_key == release.video_id,
                )
            ).scalar_one_or_none()
            if match_id is None:
                raise ValueError(f"Stored ingest run {run_id} has no match context")
        else:
            raw_match = video.match.to_dict()
            match_id = _insert(
                connection,
                matches,
                {
                    "owner_team_id": owner_team_id,
                    "ingest_run_id": run_id,
                    "dataset_id": release.dataset_id,
                    "video_id": release.video_id,
                    "match_key": release.video_id,
                    "sport_type": raw_match["rules"]["type"],
                    "rules_json": raw_match["rules"],
                    "side_a_name": video.match.side_a,
                    "side_b_name": video.match.side_b,
                    "first_server": video.match.first_server,
                    "time_conversion_json": {
                        "convention": "local_time = global_time + offset_sec",
                        "sources": list(video.sources),
                    },
                },
            )
            for clip in batch.clips:
                _store_clip(connection, owner_team_id, match_id, clip)
    return {"ingest_run_id": int(run_id), "match_context_id": int(match_id)}
