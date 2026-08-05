"""Import real ball RGB, mask, and camera records into a BLCS bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.synthetic_data_generation.configuration import (
    add_path_roots_argument,
    non_hydra_path_resolver,
)
from src.synthetic_data_generation.dataset.blcs.artifacts.calibration import (  # noqa: E402
    import_ball_calibration_capture,
    load_ball_calibration_import,
)
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="synthetic.blcs.calibration_import",
    fields=(
        BoundaryPathField(
            "capture_manifest",
            PathRole.EXTERNAL_ASSET,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "output_dir",
            PathRole.ARTIFACT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
        ),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_path_roots_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = PATH_BOUNDARY.validate(
        {
            "capture_manifest": args.capture_manifest,
            "output_dir": args.output_dir,
        },
        resolver=non_hydra_path_resolver(args.path_roots),
    )
    manifest_path = import_ball_calibration_capture(
        paths.declared("capture_manifest").path,
        paths.declared("output_dir").path,
        bundle_id=args.bundle_id,
    )
    imported = load_ball_calibration_import(manifest_path)
    print(json.dumps(imported.manifest, sort_keys=True))
