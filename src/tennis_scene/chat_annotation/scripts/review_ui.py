"""Serve the read-only annotation progress and quality dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from src.utils.configuration.paths import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

from ..artifacts.configuration import artifact_path_resolver
from ..web.app import create_app

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.chat_annotation.review_ui",
    fields=(
        BoundaryPathField(
            "root",
            PathRole.OUTPUT,
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
        "--root", required=True, type=Path, help="outputs/chat_annotation の絶対パス"
    )
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    if not args.root.is_absolute():
        parser.error("--root must be absolute")
    root = args.root.resolve()
    paths = PATH_BOUNDARY.validate(
        {"root": root}, resolver=artifact_path_resolver(root)
    )
    uvicorn.run(create_app(paths.declared("root").path), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
