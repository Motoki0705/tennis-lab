"""Execute one web inference request inside the shared GPU queue."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.base.visualization.inference_queue import (
    process_request,
    shared_repository_root,
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

PATH_BOUNDARY = NonHydraPathBoundary(
    name="base.inference_worker",
    fields=(
        BoundaryPathField(
            "request",
            PathRole.ARTIFACT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    root = shared_repository_root()
    requests = root / ".training_queue" / "ui_requests"
    roots = RuntimePathRoots(
        project_root=root,
        data_root=root / "data",
        output_root=root / "outputs",
        checkpoint_root=root / "ckpt",
        artifact_root=requests,
        cache_root=root / ".cache",
        external_asset_root=root / "third_party",
    )
    paths = PATH_BOUNDARY.validate(
        {"request": args.request.expanduser().resolve()}, resolver=PathResolver(roots)
    )
    process_request(paths.declared("request").path)


if __name__ == "__main__":
    main()
