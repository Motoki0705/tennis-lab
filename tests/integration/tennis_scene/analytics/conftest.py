from __future__ import annotations

import os
from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import Engine
from sqlalchemy.schema import CreateSchema, DropSchema

from src.tennis_scene.analytics.db.connection import analytics_engine, database_url


@pytest.fixture
def postgres_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    if not os.environ.get("DATABASE_URL"):
        message = "Set DATABASE_URL to run the PostgreSQL integration tests"
        if os.environ.get("CI_REQUIRE_POSTGRES") == "1":
            pytest.fail(message)
        pytest.skip(message)

    service_url = database_url()
    service = analytics_engine(service_url)
    schema = f"analytics_test_{uuid4().hex}"
    try:
        with service.begin() as connection:
            connection.execute(CreateSchema(schema))
        options = " ".join(
            (
                *service_url.normalized_query.get("options", ()),
                f"-csearch_path={schema}",
            )
        )
        test_url = service_url.update_query_dict({"options": options})
        monkeypatch.setenv(
            "DATABASE_URL", test_url.render_as_string(hide_password=False)
        )
        engine = analytics_engine(test_url)
        try:
            yield engine
        finally:
            engine.dispose()
            with service.begin() as connection:
                connection.execute(DropSchema(schema, cascade=True))
    finally:
        service.dispose()
