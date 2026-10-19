"""Render every adopted component artifact of a clip store into a static review gallery.

The gallery reads ``scene.json`` and the artifacts it adopts, read-only; it never
runs inference or writes to the store. Components appear in
``contracts.STANDARD_COMPONENTS`` order with every camera-scoped node found in
the index. Each renderer declares the output schema/version it understands; an
artifact of another schema is reported, not guessed. Imported artifacts are
rendered like any other, with their ``provenance.origin`` shown.

Usage: python -m src.tennis_scene.scripts.visualize_component_store --store <store> --output <dir> [--videos]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.tennis_scene.visualization.review.gallery import Review
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
    name="tennis_scene.component_gallery",
    fields=(
        BoundaryPathField("store", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True, allow_role_root=True),
        BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY, allow_role_root=True),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True, help="Component store directory holding scene.json")
    parser.add_argument("--output", type=Path, required=True, help="Gallery directory outside the component store")
    parser.add_argument("--videos", action="store_true", help="Also render complete 2D component overlay videos")
    parser.add_argument("--serve", action="store_true", help="Serve the gallery and exact source RGB frames read-only on loopback")
    parser.add_argument("--port", type=int, default=8903, help="Loopback HTTP port with --serve (default: 8903)")
    args = parser.parse_args()
    store, output = args.store.expanduser().resolve(), args.output.expanduser().resolve()
    roots = RuntimePathRoots(project_root=store, data_root=store, checkpoint_root=store, artifact_root=store,
                             output_root=output, cache_root=output, external_asset_root=store)
    paths = PATH_BOUNDARY.validate({"store": store, "output": output}, resolver=PathResolver(roots))
    index = paths.declared("store").path / "scene.json"
    if not index.is_file():
        raise FileNotFoundError(index)
    cv2.setNumThreads(2)
    review = Review(index, paths.declared("output").path, videos=args.videos, online=args.serve)
    print(review.build(), flush=True)
    if args.serve:
        from src.tennis_scene.visualization.review.web import serve
        serve(review, port=args.port)


if __name__ == "__main__":
    main()
