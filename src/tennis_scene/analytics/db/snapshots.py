"""Validate and write a metric snapshot in the caller's transaction."""

from sqlalchemy import Connection

from src.tennis_scene.analytics.db.models import Base
from src.tennis_scene.analytics.metrics import MetricRecord


def insert_metric_snapshot(
    connection: Connection,
    *,
    owner_team_id: int,
    match_context_id: int,
    record: MetricRecord,
    visibility: str = "team",
) -> int:
    table = Base.metadata.tables["metric_snapshots"]
    statement = (
        table.insert()
        .values(
            owner_team_id=owner_team_id,
            match_context_id=match_context_id,
            visibility=visibility,
            **record.to_record(),
        )
        .returning(table.c.id)
    )
    return int(connection.execute(statement).scalar_one())
