"""Move completed annotation clips into done, once or on an interval."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.utils.configuration.paths import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

from ..artifacts.completion import sync_done
from ..artifacts.configuration import artifact_path_resolver

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.chat_annotation.sync_done",
    fields=(
        BoundaryPathField(
            "root",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Absolute annotation output directory"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=0,
        help="0: one pass; positive: poll interval",
    )
    args = parser.parse_args()
    if not args.root.is_absolute():
        parser.error("--root must be an absolute directory")
    root = args.root.resolve()
    paths = PATH_BOUNDARY.validate(
        {"root": root}, resolver=artifact_path_resolver(root)
    )
    if args.watch_seconds < 0 or (args.watch_seconds and args.dry_run):
        parser.error(
            "watch requires a positive interval and cannot be combined with dry-run"
        )
    while True:
        report = sync_done(paths.declared("root").path, dry_run=args.dry_run)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if not args.watch_seconds:
            raise SystemExit(1 if report["errors"] else 0)
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()
