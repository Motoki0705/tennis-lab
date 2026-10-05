"""Review existing player frame stores in a read-only local browser UI."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.base.visualization.inference_queue import shared_repository_root
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="player_detection.review_dataset",
    fields=(
        BoundaryPathField("project_root", PathRole.PROJECT, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True, allow_role_root=True),
        BoundaryPathField("data_root", PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True, allow_role_root=True),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=shared_repository_root())
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--port", type=int, default=8895)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    project = args.project_root.expanduser().resolve()
    data = (args.data_root if args.data_root is not None else project / "data").expanduser().resolve()
    roots = RuntimePathRoots(
        project_root=project, data_root=data, output_root=project / "outputs",
        artifact_root=project / "outputs", checkpoint_root=project / "ckpt",
        cache_root=project / ".cache", external_asset_root=project / "third_party",
    )
    PATH_BOUNDARY.validate({"project_root": project, "data_root": data}, resolver=PathResolver(roots))
    # The path boundary is validated before loading data or starting a server.
    import uvicorn

    from src.tasks.player_detection.review.service import PlayerReviewService
    from src.tasks.player_detection.review.web import create_app

    service = PlayerReviewService(data)
    print(f"Player Dataset Review (read-only): http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(create_app(service), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
