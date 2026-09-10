"""Open a local human-alignment editor against an explicit canonical scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from src.synthetic_data_generation.alignment.manual.service import AlignmentEditor
from src.synthetic_data_generation.alignment.manual.web import create_app
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
    name="synthetic.manual_court_alignment",
    fields=(
        BoundaryPathField(
            "data_root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "scene_root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--recover-ground-frame",
        action="store_true",
        help="Explicitly recover a missing frame from verified paired UV/3D observations.",
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    roots = RuntimePathRoots(
        project_root=data_root.parent,
        data_root=data_root,
        checkpoint_root=data_root,
        artifact_root=data_root,
        output_root=data_root,
        cache_root=data_root,
        external_asset_root=data_root,
    )
    paths = PATH_BOUNDARY.validate(
        {"data_root": data_root, "scene_root": args.scene_root},
        resolver=PathResolver(roots),
    )
    editor = AlignmentEditor(
        paths.declared("scene_root").path,
        recover_ground_frame=args.recover_ground_frame,
    )
    print(
        f"Court Alignment Studio · {editor.root.name} · http://localhost:{args.port}",
        flush=True,
    )
    print(f"Ground frame: {editor.source.provenance}", flush=True)
    uvicorn.run(create_app(editor), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
