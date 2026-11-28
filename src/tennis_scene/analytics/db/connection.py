"""Connection settings shared by the CLI, migrations and downstream writers."""

from __future__ import annotations

import os
from pathlib import Path

from alembic.config import Config
from sqlalchemy import URL, Engine, create_engine, make_url
from sqlalchemy.exc import ArgumentError


def database_url(url: str | URL | None = None) -> URL:
    raw = os.environ.get("DATABASE_URL") if url is None else url
    if not raw:
        raise ValueError("Set DATABASE_URL to a PostgreSQL connection URL")
    try:
        target = make_url(raw)
    except ArgumentError:
        raise ValueError(
            "DATABASE_URL is not a valid PostgreSQL connection URL"
        ) from None
    if target.drivername in {"postgresql", "postgres"}:
        target = target.set(drivername="postgresql+psycopg")
    if target.drivername != "postgresql+psycopg":
        raise ValueError("DATABASE_URL must use PostgreSQL with psycopg v3")
    return target


def analytics_engine(url: str | URL | None = None) -> Engine:
    return create_engine(database_url(url), hide_parameters=True)


def migration_config() -> Config:
    config = Config()
    location = Path(__file__).with_name("migrations")
    config.set_main_option("script_location", str(location).replace("%", "%%"))
    return config
