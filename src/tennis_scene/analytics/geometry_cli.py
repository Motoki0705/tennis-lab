"""CLI options and orchestration for V4 geometry extraction."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

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

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.analytics.geometry",
    fields=(
        BoundaryPathField(
            "benchmark",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "profile_overrides",
            PathRole.EXTERNAL_ASSET,
            PathDirection.INPUT,
            PathKind.FILE,
            required=False,
            must_exist=True,
        ),
    ),
)


def add_geometry_arguments(
    parser: argparse.ArgumentParser,
    positive_int: Callable[[str], int],
) -> None:
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument(
        "--manifest-sha256", help="expected SHA-256 (defaults to frozen Meiji V4)"
    )
    parser.add_argument(
        "--profile",
        help="standard profile key (Meiji defaults to soft_tennis_standard)",
    )
    parser.add_argument(
        "--profile-overrides",
        type=Path,
        help="JSON parameters for a measured/custom profile",
    )
    parser.add_argument(
        "--ingest-run",
        type=positive_int,
        help="write geometry against this existing V4 ingest run",
    )
    parser.add_argument(
        "--owner-team", type=positive_int, help="owner of the existing ingest run"
    )


def run_geometry_command(args: argparse.Namespace) -> None:
    from src.tennis_scene.analytics.db.connection import analytics_engine
    from src.tennis_scene.analytics.db.geometry import persist_v4_geometry
    from src.tennis_scene.analytics.geometry import extract_geometry
    from src.tennis_scene.analytics.geometry_v4 import (
        MEIJI_V4_MANIFEST_SHA256,
        audit_geometry,
        read_frozen_v4,
    )
    from src.tennis_scene.analytics.profiles import resolve_profile

    if (args.ingest_run is None) != (args.owner_team is None):
        raise ValueError("--ingest-run and --owner-team must be supplied together")
    benchmark = args.benchmark.expanduser().resolve()
    override_path = (
        args.profile_overrides.expanduser().resolve()
        if args.profile_overrides is not None
        else None
    )
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=benchmark,
        checkpoint_root=PROJECT_ROOT,
        artifact_root=PROJECT_ROOT,
        output_root=PROJECT_ROOT,
        cache_root=PROJECT_ROOT,
        external_asset_root=PROJECT_ROOT
        if override_path is None
        else override_path.parent,
    )
    inputs = {"benchmark": benchmark}
    if override_path is not None:
        inputs["profile_overrides"] = override_path
    paths = PATH_BOUNDARY.validate(inputs, resolver=PathResolver(roots))
    overrides = None
    if override_path is not None:
        overrides = json.loads(
            paths.declared("profile_overrides").path.read_text(encoding="utf-8")
        )
        if not isinstance(overrides, dict):
            raise ValueError("Profile overrides must be a JSON object")
    profile = resolve_profile(
        dataset_id="meiji_4cam", profile_key=args.profile, overrides=overrides
    )
    manifest_sha256 = (
        MEIJI_V4_MANIFEST_SHA256
        if args.manifest_sha256 is None
        else args.manifest_sha256
    )
    clips = read_frozen_v4(paths.declared("benchmark").path, manifest_sha256)
    if args.ingest_run is None:
        results = {
            clip.clip_id: extract_geometry(
                clip.trajectory, clip.events, profile=profile, match_format="doubles"
            )
            for clip in clips
        }
    else:
        engine = analytics_engine()
        try:
            with engine.begin() as connection:
                results = persist_v4_geometry(
                    connection,
                    owner_team_id=args.owner_team,
                    ingest_run_id=args.ingest_run,
                    manifest_sha256=manifest_sha256,
                    clips=clips,
                    profile=profile,
                )
        finally:
            engine.dispose()
    write_json(
        sys.stdout,
        audit_geometry(
            [clip for clip in clips if clip.clip_id in results],
            results,
            profile=profile,
            manifest_sha256=manifest_sha256,
            ingest_run_id=args.ingest_run,
        ),
    )
