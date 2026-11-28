"""Command line arguments shared by the audit and PostgreSQL ingest."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from src.tennis_scene.analytics.db.connection import analytics_engine
from src.tennis_scene.analytics.export import write_json
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

_PATHS = NonHydraPathBoundary(
    name="tennis_scene.analytics.ingest",
    fields=(
        BoundaryPathField(
            "v4_root",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "projects",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "manual_supplement",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
            allow_role_root=True,
            required=False,
        ),
    ),
)


def add_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    positive_int: Callable[[str], int],
) -> None:
    for name, help_text in (
        ("audit", "audit the frozen V4 annotation and score inputs"),
        ("ingest", "store checked V4 annotation and score inputs in PostgreSQL"),
    ):
        parser = commands.add_parser(name, help=help_text)
        parser.add_argument("--v4-root", required=True, type=Path)
        parser.add_argument("--projects", required=True, type=Path)
        parser.add_argument(
            "--data-root",
            type=Path,
            default=PROJECT_ROOT / "data",
            help="root for Clip Studio's data-relative camera paths",
        )
        parser.add_argument("--manual-supplement", type=Path)
        parser.add_argument("--manifest-sha256", help="expected V4 manifest SHA-256")
        parser.add_argument("--projects-sha256", help="expected projects.json SHA-256")
        if name == "ingest":
            parser.add_argument("--owner-team", required=True, type=positive_int)
        else:
            parser.add_argument("--format", choices=("json", "csv"), default="json")


def run(args: argparse.Namespace) -> None:
    from src.tennis_scene.analytics.ingest.audit import (
        audit_report,
        prepare_ingest,
        write_audit,
    )
    from src.tennis_scene.analytics.ingest.score_adapter import read_scores
    from src.tennis_scene.analytics.ingest.store import persist_ingest
    from src.tennis_scene.analytics.ingest.supplement import read_supplement
    from src.tennis_scene.analytics.ingest.v4_reader import read_v4

    values = {
        name: value.expanduser().resolve()
        for name in ("v4_root", "projects", "manual_supplement")
        if (value := getattr(args, name)) is not None
    }
    resolver = PathResolver(
        RuntimePathRoots(
            project_root=PROJECT_ROOT,
            data_root=args.data_root.expanduser().resolve(),
            checkpoint_root=PROJECT_ROOT,
            artifact_root=PROJECT_ROOT,
            output_root=PROJECT_ROOT,
            cache_root=PROJECT_ROOT,
            external_asset_root=PROJECT_ROOT,
        )
    )
    paths = _PATHS.validate(values, resolver=resolver, independent_artifact_inputs=True)
    release = read_v4(
        paths.declared("v4_root").path, manifest_sha256=args.manifest_sha256
    )
    video = read_scores(
        release,
        paths.declared("projects").path,
        resolver,
        projects_sha256=args.projects_sha256,
    )
    supplement_path = (
        paths.declared("manual_supplement").path
        if "manual_supplement" in paths
        else None
    )
    batch = prepare_ingest(video, *read_supplement(supplement_path))
    if args.command == "audit":
        write_audit(sys.stdout, audit_report(batch), args.format)
        return
    engine = analytics_engine()
    try:
        result = persist_ingest(engine, args.owner_team, batch)
    finally:
        engine.dispose()
    write_json(sys.stdout, result)
