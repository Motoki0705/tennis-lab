"""Render a saved player prediction receipt into the current worktree's tmp/."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from src.tasks.base.annotation.artifacts import annotation_reference
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.player_detection.annotation.preview_video import export_prediction_video
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
)
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="player_detection.preview_predictions",
    fields=(
        BoundaryPathField(
            "prediction",
            PathRole.OUTPUT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "output", PathRole.CACHE, PathDirection.OUTPUT, PathKind.DIRECTORY
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    layout = AnnotationLayout.production()
    resolver = PathResolver(
        replace(layout.resolver.roots, cache_root=PROJECT_ROOT / "tmp")
    )
    paths = PATH_BOUNDARY.validate(
        {"prediction": args.prediction, "output": args.output}, resolver=resolver
    )
    prediction = annotation_reference(layout, paths.declared("prediction").path)
    if prediction.digest != args.sha256:
        raise ValueError("player prediction receipt hash changed")
    output = paths.declared("output").path
    result = export_prediction_video(layout, prediction, output, reason=args.reason)
    print(
        json.dumps(
            {
                "frames": len(result.frames),
                "output": str(output),
                "video_sha256": result.video.digest,
            }
        )
    )


if __name__ == "__main__":
    main()
