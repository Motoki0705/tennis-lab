"""Persist estimates against adopted V4 shot IDs in an existing ingest run."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import Connection, RowMapping, func, or_, select
from sqlalchemy.dialects.postgresql import insert

from src.tennis_scene.analytics.db.models import Base
from src.tennis_scene.analytics.hitter import (
    DEFAULT_THRESHOLDS,
    HitterScene,
    estimate_hitter,
)
from src.tennis_scene.analytics.hitter_v4 import V4Benchmark, V4Clip

_TABLES = Base.metadata.tables
_KEY_FIELDS = ("owner_team_id", "ball_event_id", "rule_version")


def _make_clip_rows(
    clip: V4Clip,
    events: dict[str, RowMapping],
    owner_team_id: int,
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    with np.load(clip.scene.verify(), allow_pickle=False) as archive:
        shots = clip.shots(archive)
        scene = HitterScene.from_archive(archive)
    if len(scene.ball) != clip.frames or events.keys() != {
        shot.source_event_id for shot in shots
    }:
        raise ValueError(
            f"{clip.clip_id}: ingested shot IDs or scene timeline disagree with V4"
        )
    rows = []
    for shot in shots:
        event = events[shot.source_event_id]
        if (
            event["frame"],
            event["subtype"],
            event["frame_uncertainty"],
            event["court_side"],
        ) != (shot.frame, shot.subtype, shot.frame_uncertainty, shot.court_side):
            raise ValueError(
                f"{clip.clip_id}/{shot.source_event_id}: ingested shot differs from V4"
            )
        estimate = estimate_hitter(
            scene,
            frame=shot.frame,
            frame_uncertainty=shot.frame_uncertainty,
            court_side=shot.court_side,
        ).to_record()
        estimate["evidence_json"].update(
            {
                "clip_id": clip.clip_id,
                "source_event_id": shot.source_event_id,
                "v4_manifest_sha256": manifest_sha256,
                "scene_npz_sha256": clip.scene.sha256,
                "event_component_sha256": clip.event_component.sha256,
            }
        )
        rows.append(
            {
                "owner_team_id": owner_team_id,
                "ball_event_id": event["id"],
                "visibility": event["visibility"],
                **estimate,
            }
        )
    return rows


def _upsert(connection: Connection, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    table = _TABLES["shot_hitter_estimates"]
    statement = insert(table)
    updates = {
        name: statement.excluded[name] for name in rows[0] if name not in _KEY_FIELDS
    }
    statement = statement.on_conflict_do_update(
        index_elements=list(_KEY_FIELDS),
        set_={**updates, "updated_at": func.now()},
        where=or_(
            *(table.c[name].is_distinct_from(value) for name, value in updates.items())
        ),
    )
    return len(
        connection.execute(statement.returning(table.c.id), rows).scalars().all()
    )


def store_v4_hitters(
    connection: Connection,
    benchmark_root: Path,
    *,
    ingest_run_id: int,
    owner_team_id: int,
) -> dict[str, Any]:
    """Validate a run and store the current rule's rows in the caller's transaction."""
    runs = _TABLES["ingest_runs"]
    matches = _TABLES["match_contexts"]
    clips = _TABLES["clips"]
    events = _TABLES["ball_events"]
    run = (
        connection.execute(
            select(runs)
            .where(runs.c.id == ingest_run_id, runs.c.owner_team_id == owner_team_id)
            .with_for_update(read=True)
        )
        .mappings()
        .one_or_none()
    )
    if run is None:
        raise ValueError(
            f"Ingest run {ingest_run_id} was not found for team {owner_team_id}"
        )
    manifest_sha256 = run["v4_manifest_sha256"]
    benchmark = V4Benchmark.load(benchmark_root, manifest_sha256)
    if benchmark.benchmark_id.split("/")[0] != run["dataset_id"]:
        raise ValueError("Ingest run and V4 manifest have different dataset IDs")
    clip_rows = (
        connection.execute(
            select(clips)
            .join(matches, matches.c.id == clips.c.match_context_id)
            .where(
                matches.c.ingest_run_id == ingest_run_id,
                clips.c.owner_team_id == owner_team_id,
            )
            .order_by(clips.c.id)
            .with_for_update(read=True)
        )
        .mappings()
        .all()
    )
    if not clip_rows or len({row["clip_id"] for row in clip_rows}) != len(clip_rows):
        raise ValueError("An ingest run must have a nonempty set of distinct V4 clips")
    shot_rows = connection.execute(
        select(events)
        .where(
            events.c.owner_team_id == owner_team_id,
            events.c.clip_id.in_([row["id"] for row in clip_rows]),
            events.c.kind == "shot",
            events.c.adopted_in_v4.is_(True),
        )
        .order_by(events.c.id)
        .with_for_update(read=True)
    ).mappings()
    by_clip: dict[int, dict[str, RowMapping]] = defaultdict(dict)
    for event in shot_rows:
        by_clip[event["clip_id"]][event["source_event_id"]] = event

    rows = []
    for row in clip_rows:
        clip_id = row["clip_id"]
        clip = benchmark.clips.get(clip_id)
        if (
            clip is None
            or clip_id.split("/")[0] != run["video_id"]
            or (
                row["scene_schema_version"],
                row["scene_npz_sha256"],
                row["annotation_sha256"],
            )
            != (4, clip.scene.sha256, clip.annotation_sha256)
            or (row["num_frames"], row["fps"], row["split"])
            != (clip.frames, clip.fps, clip.split)
        ):
            raise ValueError(
                f"{clip_id}: ingested clip does not match the pinned V4 manifest"
            )
        rows.extend(
            _make_clip_rows(clip, by_clip[row["id"]], owner_team_id, manifest_sha256)
        )

    changed = _upsert(connection, rows)
    counts = Counter(row["resolution_status"] for row in rows)
    return {
        "ingest_run_id": ingest_run_id,
        "owner_team_id": owner_team_id,
        "clips": len(clip_rows),
        "total_shots": len(rows),
        "rows_changed": changed,
        "rule_version": DEFAULT_THRESHOLDS.rule_version,
        "resolved_count": counts["resolved"],
        "ambiguous_count": counts["ambiguous"],
        "unresolved_count": counts["unresolved"],
    }
