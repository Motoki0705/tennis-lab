"""Run the local BLCS inference web UI.

Usage:
    .venv/bin/python -m src.tasks.blcs.scripts.inference_ui
    .venv/bin/python -m src.tasks.blcs.scripts.inference_ui --port 8770 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from src.tasks.blcs.visualization.inference.service import InferenceService
from src.tasks.blcs.visualization.inference.web import create_app
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
    name="blcs.inference_ui",
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
            "outputs_root",
            PathRole.OUTPUT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=False,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "checkpoints_root",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=False,
            allow_role_root=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Dataset root holding blcs/<form>/scenes.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "blcs",
        help="Recursively scanned for *.ckpt.",
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=PROJECT_ROOT / "ckpt" / "blcs",
        help="Recursively scanned for curated *.ckpt files.",
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    outputs_root = args.outputs_root.expanduser().resolve()
    checkpoints_root = args.checkpoints_root.expanduser().resolve()
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=data_root,
        checkpoint_root=checkpoints_root,
        artifact_root=PROJECT_ROOT,
        output_root=outputs_root,
        cache_root=PROJECT_ROOT / ".cache",
        external_asset_root=PROJECT_ROOT / "third_party",
    )
    paths = PATH_BOUNDARY.validate(
        {
            "data_root": data_root,
            "outputs_root": outputs_root,
            "checkpoints_root": checkpoints_root,
        },
        resolver=PathResolver(roots),
    )
    service = InferenceService(
        paths.declared("outputs_root").path,
        paths.declared("checkpoints_root").path,
        paths.declared("data_root").path,
        device=args.device,
    )
    print(f"BLCS Inference UI · http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(create_app(service), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
