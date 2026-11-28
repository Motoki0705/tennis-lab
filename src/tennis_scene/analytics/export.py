"""Read and serialize normalized analytics records."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TextIO

from sqlalchemy import ColumnElement, Connection, Select, Table, select, union_all

from src.tennis_scene.analytics.db.models import Base

Record = dict[str, Any]
DOMAIN_TABLES = {
    table.name: table
    for table in Base.metadata.tables.values()
    if "owner_team_id" in table.c
}


def _json_default(value: Any) -> str:
    if isinstance(value, datetime) and value.utcoffset() is not None:
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        default=_json_default,
        indent=indent,
    )


def write_json(stream: TextIO, value: Any) -> None:
    stream.write(_json(value, indent=2) + "\n")


def _csv_cell(value: Any) -> str:
    if value is None:
        return r"\N"
    if isinstance(value, str):
        return "\\" + value if value.startswith("\\") else value
    if isinstance(value, datetime):
        return _json_default(value)
    return _json(value)


def write_csv(
    stream: TextIO, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    writer = csv.writer(stream, lineterminator="\r\n")
    writer.writerow(fields)
    writer.writerows([_csv_cell(row[field]) for field in fields] for row in rows)


def write_records(
    stream: TextIO, fields: Sequence[str], rows: Sequence[Record], format: str
) -> None:
    if format == "json":
        write_json(stream, rows)
    elif format == "csv":
        write_csv(stream, fields, rows)
    else:
        raise ValueError(f"Unsupported export format: {format}")


def _rows(connection: Connection, statement: Select[Any]) -> list[Record]:
    return [dict(row) for row in connection.execute(statement).mappings()]


def query_records(
    connection: Connection,
    table: Table,
    *,
    owner_team_id: int | None = None,
    row_id: int | None = None,
    limit: int | None = None,
) -> list[Record]:
    statement = select(table).order_by(table.c.id)
    if owner_team_id is not None:
        statement = statement.where(table.c.owner_team_id == owner_team_id)
    if row_id is not None:
        statement = statement.where(table.c.id == row_id)
    return _rows(connection, statement.limit(limit))


def export_records(
    connection: Connection,
    *,
    ingest_run_id: int | None = None,
    match_context_id: int | None = None,
    owner_team_id: int | None = None,
) -> dict[str, list[Record]]:
    """Fetch a scope and its evidence using the caller's database snapshot."""
    if (ingest_run_id is None) == (match_context_id is None):
        raise ValueError("Select exactly one ingest run or match context")

    tables = DOMAIN_TABLES
    runs, matches, clips, points, events, arcs, geometry = (
        tables[name]
        for name in (
            "ingest_runs",
            "match_contexts",
            "clips",
            "points",
            "ball_events",
            "flight_arcs",
            "shot_geometry_features",
        )
    )

    root = matches if match_context_id is not None else runs
    root_id = match_context_id if match_context_id is not None else ingest_run_id
    root_rows = query_records(
        connection, root, owner_team_id=owner_team_id, row_id=root_id
    )
    if not root_rows:
        raise ValueError(
            f"No {root.name} record with ID {root_id} in the requested scope"
        )

    owner_team_id = root_rows[0]["owner_team_id"]
    run_id = (
        root_rows[0]["ingest_run_id"] if match_context_id is not None else ingest_run_id
    )
    match_scope = (
        matches.c.ingest_run_id == run_id
        if match_context_id is None
        else matches.c.id == match_context_id
    )
    match_ids = select(matches.c.id).where(
        matches.c.owner_team_id == owner_team_id, match_scope
    )
    clip_ids = select(clips.c.id).where(clips.c.match_context_id.in_(match_ids))
    point_ids = select(points.c.id).where(points.c.clip_id.in_(clip_ids))
    event_ids = select(events.c.id).where(events.c.clip_id.in_(clip_ids))
    arc_ids = select(arcs.c.id).where(arcs.c.clip_id.in_(clip_ids))
    profile_ids = union_all(
        select(matches.c.court_net_profile_id).where(matches.c.id.in_(match_ids)),
        select(geometry.c.profile_id).where(geometry.c.ball_event_id.in_(event_ids)),
    )

    conditions: dict[str, ColumnElement[bool]] = {
        "court_net_profiles": tables["court_net_profiles"].c.id.in_(profile_ids),
        "ingest_runs": runs.c.id == run_id,
        "match_contexts": match_scope,
        "clips": clips.c.match_context_id.in_(match_ids),
        "points": points.c.clip_id.in_(clip_ids),
        "ball_events": events.c.clip_id.in_(clip_ids),
        "shot_hitter_estimates": tables["shot_hitter_estimates"].c.ball_event_id.in_(
            event_ids
        ),
        "flight_arcs": arcs.c.clip_id.in_(clip_ids),
        "shot_geometry_features": geometry.c.ball_event_id.in_(event_ids),
        "net_plane_crossings": tables["net_plane_crossings"].c.flight_arc_id.in_(
            arc_ids
        ),
        "player_identity_maps": tables["player_identity_maps"].c.clip_id.in_(clip_ids),
        "team_pair_assignments": tables["team_pair_assignments"].c.clip_id.in_(
            clip_ids
        ),
        "serve_attempts": tables["serve_attempts"].c.point_id.in_(point_ids),
        "metric_snapshots": tables["metric_snapshots"].c.match_context_id.in_(
            match_ids
        ),
    }
    return {
        name: root_rows
        if name == root.name
        else _rows(
            connection,
            select(table)
            .where(table.c.owner_team_id == owner_team_id, conditions[name])
            .order_by(table.c.id),
        )
        for name, table in tables.items()
    }


def write_csv_export(destination: Path, records: Mapping[str, list[Record]]) -> None:
    """Publish the set of normalized CSV tables together."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Export destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".analytics-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "export"
        staging.mkdir()
        for name, rows in records.items():
            fields = tuple(DOMAIN_TABLES[name].c.keys())
            with (staging / f"{name}.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                write_csv(stream, fields, rows)
        staging.rename(destination)
