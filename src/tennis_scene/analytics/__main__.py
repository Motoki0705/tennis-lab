"""Inspect the tennis scene analytics store and metric definitions."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from alembic import command
from alembic.util import CommandError
from sqlalchemy.exc import SQLAlchemyError

from src.tennis_scene.analytics.db.connection import analytics_engine, migration_config
from src.tennis_scene.analytics.export import (
    DOMAIN_TABLES,
    export_records,
    query_records,
    write_csv_export,
    write_json,
    write_records,
)
from src.tennis_scene.analytics.metrics import (
    METRIC_DEFINITION_FIELDS,
    METRIC_DEFINITIONS,
)
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.analytics",
    fields=(
        BoundaryPathField(
            "output_dir",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
    ),
)


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def _format_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("json", "csv"), default="json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    db = commands.add_parser("db", help="manage the Alembic schema")
    db_commands = db.add_subparsers(dest="db_command", required=True)
    for name, default in (("upgrade", "head"), ("downgrade", "base")):
        migration = db_commands.add_parser(name)
        migration.add_argument("revision", nargs="?", default=default)
    db_commands.add_parser(
        "check", help="check the revision and model for schema drift"
    )

    metrics = commands.add_parser(
        "metrics", help="read the versioned metric dictionary"
    )
    metric_commands = metrics.add_subparsers(dest="metrics_command", required=True)
    metric_commands.add_parser("list")
    _format_arg(metric_commands.add_parser("export"))

    query = commands.add_parser("query", help="read rows from a domain table")
    query.add_argument("table", choices=sorted(DOMAIN_TABLES))
    query.add_argument("--owner-team", type=_positive_int)
    query.add_argument("--id", type=_positive_int)
    query.add_argument("--limit", type=_positive_int)
    _format_arg(query)

    export = commands.add_parser(
        "export", help="export a run or match and its evidence"
    )
    scope = export.add_mutually_exclusive_group(required=True)
    scope.add_argument("--ingest-run", type=_positive_int)
    scope.add_argument("--match-context", type=_positive_int)
    export.add_argument("--owner-team", type=_positive_int)
    export.add_argument(
        "--output-dir",
        type=Path,
        help="new directory for CSV tables (relative paths use the working directory)",
    )
    _format_arg(export)

    hitter = commands.add_parser("hitter", help="store V4 shot hitter estimates")
    hitter.add_argument("--benchmark", required=True, type=Path)
    hitter.add_argument("--ingest-run", required=True, type=_positive_int)
    hitter.add_argument("--owner-team", required=True, type=_positive_int)
    return parser


def _run(args: argparse.Namespace) -> None:
    if args.command == "db":
        config = migration_config()
        if args.db_command == "upgrade":
            command.upgrade(config, args.revision)
        elif args.db_command == "downgrade":
            command.downgrade(config, args.revision)
        else:
            command.check(config)
        return

    if args.command == "metrics":
        definitions = sorted(
            METRIC_DEFINITIONS.values(), key=lambda item: item.metric_id
        )
        if args.metrics_command == "list":
            for definition in definitions:
                print(f"{definition.metric_id}\t{definition.definition_version}")
        else:
            rows = [definition.to_record() for definition in definitions]
            write_records(sys.stdout, METRIC_DEFINITION_FIELDS, rows, args.format)
        return

    engine = analytics_engine()
    try:
        if args.command == "hitter":
            from src.tennis_scene.analytics.db.hitters import store_v4_hitters

            with engine.begin() as transaction:
                report = store_v4_hitters(
                    transaction,
                    args.benchmark,
                    ingest_run_id=args.ingest_run,
                    owner_team_id=args.owner_team,
                )
            write_json(sys.stdout, report)
            return
        with engine.connect() as connection:
            if args.command == "query":
                table = DOMAIN_TABLES[args.table]
                rows = query_records(
                    connection,
                    table,
                    owner_team_id=args.owner_team,
                    row_id=args.id,
                    limit=args.limit,
                )
                write_records(sys.stdout, table.c.keys(), rows, args.format)
                return

            connection.execution_options(isolation_level="REPEATABLE READ")
            records = export_records(
                connection,
                ingest_run_id=args.ingest_run,
                match_context_id=args.match_context,
                owner_team_id=args.owner_team,
            )
    finally:
        engine.dispose()

    if args.format == "csv":
        write_csv_export(args.output_dir, records)
    else:
        write_json(sys.stdout, records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "export" and (args.format == "csv") != (
        args.output_dir is not None
    ):
        parser.error("--output-dir is required for CSV exports and only applies to CSV")
    try:
        if args.command == "export" and args.output_dir is not None:
            destination = args.output_dir
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(
                    f"Export destination already exists: {destination}"
                )
            output_dir = destination.parent.resolve() / destination.name
            roots = RuntimePathRoots(
                project_root=PROJECT_ROOT,
                data_root=PROJECT_ROOT,
                checkpoint_root=PROJECT_ROOT,
                artifact_root=PROJECT_ROOT,
                output_root=output_dir,
                cache_root=PROJECT_ROOT,
                external_asset_root=PROJECT_ROOT,
            )
            paths = PATH_BOUNDARY.validate(
                {"output_dir": output_dir}, resolver=PathResolver(roots)
            )
            args.output_dir = paths.declared("output_dir").path
        _run(args)
    except (
        CommandError,
        KeyError,
        OSError,
        SQLAlchemyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"analytics: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
