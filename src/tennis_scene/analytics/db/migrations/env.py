"""Alembic environment backed by the analytics PostgreSQL connection settings."""

from alembic import context

from src.tennis_scene.analytics.db.connection import analytics_engine, database_url
from src.tennis_scene.analytics.db.models import Base


def run_migrations() -> None:
    if context.is_offline_mode():
        context.configure(
            url=database_url(),
            target_metadata=Base.metadata,
            literal_binds=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()
        return

    engine = analytics_engine()
    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=Base.metadata,
                compare_server_default=True,
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


run_migrations()
