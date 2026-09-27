"""Run the private annotation ZIP intake MCP server."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.utils.configuration.paths import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

from ..artifacts.configuration import artifact_path_resolver
from ..artifacts.server import create_server

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.chat_annotation.serve_artifacts",
    fields=(
        BoundaryPathField(
            "root",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Absolute raw ZIP directory"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not args.root.is_absolute():
        parser.error("--root must be an absolute directory")
    root = args.root.resolve()
    paths = PATH_BOUNDARY.validate(
        {"root": root}, resolver=artifact_path_resolver(root)
    )
    # httpx INFO messages include signed URLs; suppress those access logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    create_server(paths.declared("root").path, args.host, args.port).run(
        transport="streamable-http"
    )


if __name__ == "__main__":
    main()
