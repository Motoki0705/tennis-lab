"""Persist S4 evidence against an existing V4 ingest run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from math import isclose
from typing import Any

from sqlalchemy import Connection, Table, select
from sqlalchemy.dialects.postgresql import insert

from src.tennis_scene.analytics.db.models import (
    VISIBILITIES,
    BallEvent,
    Clip,
    IngestRun,
    MatchContext,
    Point,
    ShotGeometryFeature,
)
from src.tennis_scene.analytics.db.models import (
    CourtNetProfile as ProfileModel,
)
from src.tennis_scene.analytics.db.models import (
    FlightArc as ArcModel,
)
from src.tennis_scene.analytics.db.models import (
    NetPlaneCrossing as CrossingModel,
)
from src.tennis_scene.analytics.geometry import (
    BIN_VERSION,
    DEFINITION_VERSION,
    FlightArc,
    GeometryEvent,
    GeometryResult,
    NetPlaneCrossing,
    ShotGeometry,
    extract_geometry,
)
from src.tennis_scene.analytics.geometry_v4 import V4GeometryClip
from src.tennis_scene.analytics.profiles import CourtNetProfile


def _write_once(
    connection: Connection, table: Table, keys: tuple[str, ...], values: dict[str, Any]
) -> int:
    """Retry on the same unique key only when the evidence agrees."""
    statement = (
        insert(table)
        .values(**values)
        .on_conflict_do_nothing(index_elements=[table.c[key] for key in keys])
        .returning(table.c.id)
    )
    created = connection.execute(statement).scalar_one_or_none()
    if created is not None:
        return int(created)

    predicate = [table.c[key] == values[key] for key in keys]
    existing = connection.execute(select(table).where(*predicate)).mappings().one()
    differences = sorted(key for key, value in values.items() if existing[key] != value)
    if differences:
        raise ValueError(
            f"Conflicting {table.name} evidence for {dict((key, values[key]) for key in keys)} "
            f"in {', '.join(differences)}"
        )
    return int(existing["id"])


def save_court_net_profile(
    connection: Connection,
    owner_team_id: int,
    profile: CourtNetProfile,
) -> int:
    return _write_once(
        connection,
        ProfileModel.__table__,
        ("owner_team_id", "profile_key"),
        {"owner_team_id": owner_team_id, **profile.to_record()},
    )


def _reconcile_events(
    connection: Connection,
    *,
    owner_team_id: int,
    clip_db_id: int,
    clip: V4GeometryClip,
) -> tuple[tuple[GeometryEvent, ...], dict[str, int], list[str]]:
    event_table, points = BallEvent.__table__, Point.__table__
    rows = (
        connection.execute(
            select(event_table, points.c.end_frame.label("point_end_frame"))
            .outerjoin(
                points,
                (event_table.c.owner_team_id == points.c.owner_team_id)
                & (event_table.c.clip_id == points.c.clip_id)
                & (event_table.c.point_id == points.c.id),
            )
            .where(
                event_table.c.owner_team_id == owner_team_id,
                event_table.c.clip_id == clip_db_id,
                event_table.c.adopted_in_v4.is_(True),
            )
        )
        .mappings()
        .all()
    )
    by_id = {row["source_event_id"]: row for row in rows}
    if set(by_id) != {event.source_event_id for event in clip.events}:
        raise ValueError(
            f"V4 event IDs differ from the ingested events for {clip.clip_id}"
        )
    events = []
    for event in clip.events:
        row = by_id[event.source_event_id]
        if (
            event.frame,
            event.kind,
            event.subtype,
            event.status,
            event.court_side,
            event.frame_uncertainty,
        ) != (
            row["frame"],
            row["kind"],
            row["subtype"],
            row["status"],
            row["court_side"],
            row["frame_uncertainty"],
        ) or not isclose(
            event.frame / clip.trajectory.fps, row["time_s"], rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError(
                f"V4 event evidence differs for {clip.clip_id}/{event.source_event_id}"
            )
        events.append(
            replace(
                event,
                in_point=row["in_point"],
                point_id=row["point_id"],
                point_end_frame=row["point_end_frame"] if row["in_point"] else None,
            )
        )
    return (
        tuple(events),
        {key: row["id"] for key, row in by_id.items()},
        [row["visibility"] for row in rows],
    )


def _arc_record(arc: FlightArc, event_ids: Mapping[str, int]) -> dict[str, Any]:
    start, end, shot = arc.start_event, arc.end_event, arc.shot_event
    boundary_point = (
        start.point_id
        if start is not None
        and end is not None
        and start.in_point
        and end.in_point
        and start.point_id == end.point_id
        else None
    )
    return {
        "arc_index": arc.arc_index,
        "point_id": shot.point_id if shot is not None else boundary_point,
        "shot_event_id": None if shot is None else event_ids[shot.source_event_id],
        "start_event_id": None if start is None else event_ids[start.source_event_id],
        "end_event_id": None if end is None else event_ids[end.source_event_id],
        "start_frame": arc.start_frame,
        "end_frame": arc.end_frame,
        "valid_frame_count": arc.valid_frame_count,
        "total_frame_count": arc.total_frame_count,
        "source_kind": arc.source_kind,
    }


def _crossing_record(crossing: NetPlaneCrossing) -> dict[str, Any]:
    record = asdict(crossing)
    del record["arc_index"]
    record["used_frames_json"] = list(record.pop("used_frames"))
    return record


def _shot_record(shot: ShotGeometry, fps: float) -> dict[str, Any]:
    contact = shot.contact_position.xyz if shot.contact_position else (None, None, None)
    landing = shot.landing
    bounce = landing.bounce_event
    point = landing.position.xyz if landing.position else (None, None, None)
    return {
        "definition_version": DEFINITION_VERSION,
        "contact_frame": shot.shot_event.frame,
        "contact_time_s": shot.contact_time_s,
        "contact_frame_uncertainty": shot.shot_event.frame_uncertainty,
        "contact_x_m": contact[0],
        "contact_y_m": contact[1],
        "contact_z_m": contact[2],
        "contact_height_m": shot.contact_height_m,
        "incoming_speed_mps": None
        if shot.incoming is None
        else shot.incoming.speed_mps,
        "outgoing_speed_mps": None
        if shot.outgoing is None
        else shot.outgoing.speed_mps,
        "outgoing_heading_rad": shot.outgoing_heading_rad,
        "speed_method": "one_sided_secant" if shot.incoming or shot.outgoing else None,
        "speed_window_frames": {
            "incoming": [] if shot.incoming is None else list(shot.incoming.frames),
            "outgoing": [] if shot.outgoing is None else list(shot.outgoing.frames),
        },
        "first_bounce_frame": None if bounce is None else bounce.frame,
        "first_bounce_time_s": None if bounce is None else bounce.center / fps,
        "first_bounce_x_m": point[0],
        "first_bounce_y_m": point[1],
        "first_bounce_z_m": point[2],
        "landing_depth_m": landing.depth_m,
        "landing_lane": landing.lane,
        "landing_course": landing.course,
        "line_margin_m": landing.line_margin_m,
        "landing_result": landing.result,
        "landing_source_kind": None
        if landing.position is None
        else landing.position.source_kind,
        "landing_missing_reason": landing.missing_reason,
        "bin_version": BIN_VERSION,
        "bins_json": shot.bins,
        "quality_json": shot.quality,
    }


def _persist_clip(
    connection: Connection,
    *,
    owner_team_id: int,
    clip_db_id: int,
    clip: V4GeometryClip,
    profile: CourtNetProfile,
    manifest_sha256: str,
) -> GeometryResult:
    clips, matches = Clip.__table__, MatchContext.__table__
    record = (
        connection.execute(
            select(
                clips,
                matches.c.sport_type,
                matches.c.visibility.label("match_visibility"),
            )
            .join(
                matches,
                (clips.c.owner_team_id == matches.c.owner_team_id)
                & (clips.c.match_context_id == matches.c.id),
            )
            .where(clips.c.owner_team_id == owner_team_id, clips.c.id == clip_db_id)
            .with_for_update(of=clips)
        )
        .mappings()
        .one()
    )
    if (
        record["clip_id"],
        record["scene_schema_version"],
        record["split"],
        record["scene_npz_sha256"],
        record["annotation_sha256"],
        record["num_frames"],
        record["fps"],
    ) != (
        clip.clip_id,
        4,
        clip.split,
        clip.scene_npz_sha256,
        clip.annotation_sha256,
        len(clip.trajectory.valid),
        clip.trajectory.fps,
    ):
        raise ValueError(f"V4 clip identity or timebase differs for {clip.clip_id}")
    if record["sport_type"] != profile.sport_type:
        raise ValueError(
            f"Profile sport differs from the ingested match for {clip.clip_id}"
        )

    events, event_ids, visibility = _reconcile_events(
        connection,
        owner_team_id=owner_team_id,
        clip_db_id=clip_db_id,
        clip=clip,
    )
    result = extract_geometry(
        clip.trajectory, events, profile=profile, match_format="doubles"
    )
    profile_id = save_court_net_profile(connection, owner_team_id, profile)
    common = {
        "owner_team_id": owner_team_id,
        "clip_id": clip_db_id,
        "visibility": min(
            [record["visibility"], record["match_visibility"], *visibility],
            key=VISIBILITIES.index,
        ),
    }
    input_evidence = {
        "definition_version": DEFINITION_VERSION,
        "manifest_sha256": manifest_sha256,
        "event_detection_sha256": clip.component_sha256,
    }
    arc_ids = {
        arc.arc_index: _write_once(
            connection,
            ArcModel.__table__,
            ("owner_team_id", "clip_id", "arc_index"),
            {
                **common,
                **_arc_record(arc, event_ids),
                "quality_json": {**arc.quality_record(), **input_evidence},
            },
        )
        for arc in result.arcs
    }
    for crossing in result.crossings:
        shot = result.arcs[crossing.arc_index].shot_event
        _write_once(
            connection,
            CrossingModel.__table__,
            ("owner_team_id", "flight_arc_id", "crossing_index"),
            {
                **common,
                **_crossing_record(crossing),
                "flight_arc_id": arc_ids[crossing.arc_index],
                "ball_event_id": None
                if shot is None
                else event_ids[shot.source_event_id],
                "court_net_profile_id": profile_id,
                "profile_source_kind": profile.source_kind,
            },
        )
    for shot_geometry in result.shots:
        _write_once(
            connection,
            ShotGeometryFeature.__table__,
            (
                "owner_team_id",
                "ball_event_id",
                "profile_id",
                "definition_version",
                "bin_version",
            ),
            {
                **common,
                **_shot_record(shot_geometry, clip.trajectory.fps),
                "profile_id": profile_id,
                "ball_event_id": event_ids[shot_geometry.shot_event.source_event_id],
                "flight_arc_id": arc_ids[shot_geometry.flight_arc_index],
            },
        )
    return result


def persist_v4_geometry(
    connection: Connection,
    *,
    owner_team_id: int,
    ingest_run_id: int,
    manifest_sha256: str,
    clips: Sequence[V4GeometryClip],
    profile: CourtNetProfile,
) -> dict[str, GeometryResult]:
    """Write one run atomically, joining existing clips and adopted event IDs."""
    runs, matches, table = IngestRun.__table__, MatchContext.__table__, Clip.__table__
    run = (
        connection.execute(
            select(runs).where(
                runs.c.owner_team_id == owner_team_id,
                runs.c.id == ingest_run_id,
            )
        )
        .mappings()
        .one_or_none()
    )
    if (
        run is None
        or run["dataset_id"] != "meiji_4cam"
        or run["v4_manifest_sha256"] != manifest_sha256
    ):
        raise ValueError(
            "Ingest run is missing or belongs to a different owner or V4 release"
        )
    rows = (
        connection.execute(
            select(table.c.id, table.c.clip_id)
            .join(
                matches,
                (table.c.owner_team_id == matches.c.owner_team_id)
                & (table.c.match_context_id == matches.c.id),
            )
            .where(
                table.c.owner_team_id == owner_team_id,
                matches.c.ingest_run_id == ingest_run_id,
            )
            .order_by(table.c.id)
        )
        .mappings()
        .all()
    )
    by_name = {clip.clip_id: clip for clip in clips}
    names = [row["clip_id"] for row in rows]
    if (
        not names
        or len(set(names)) != len(names)
        or len(by_name) != len(clips)
        or not set(names) <= by_name.keys()
    ):
        raise ValueError("Ingest run must contain unique clips in the pinned V4 inputs")
    with connection.begin_nested():
        return {
            row["clip_id"]: _persist_clip(
                connection,
                owner_team_id=owner_team_id,
                clip_db_id=row["id"],
                clip=by_name[row["clip_id"]],
                profile=profile,
                manifest_sha256=manifest_sha256,
            )
            for row in rows
        }
