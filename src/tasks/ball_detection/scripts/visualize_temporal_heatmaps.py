"""Render the same evaluated clip range across explicit model/T selections."""

import argparse
from pathlib import Path

from src.tasks.ball_detection.visualization.presentation.temporal_heatmap import (
    render_temporal_heatmap_comparison,
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
    name="ball_detection.visualize_temporal_heatmaps",
    fields=(
        BoundaryPathField("evaluations", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE,
                          must_exist=True, allow_role_root=True, many=True),
        BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.FILE),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=480)
    args = parser.parse_args()
    if not all(path.is_absolute() for path in (*args.evaluations, args.output)):
        parser.error("Comparison evaluation and output paths must be absolute")
    output_root = args.output.parent.resolve()
    roots = RuntimePathRoots(project_root=output_root, data_root=output_root, artifact_root=output_root,
                             checkpoint_root=output_root, output_root=output_root,
                             cache_root=output_root, external_asset_root=output_root)
    paths = PATH_BOUNDARY.validate(dict(evaluations=args.evaluations, output=args.output),
                                   resolver=PathResolver(roots), independent_artifact_inputs=True)
    render_temporal_heatmap_comparison([entry.path for entry in paths.declared_many("evaluations")],
        clip_id=args.clip_id, start=args.start, stop=args.stop, output=paths.declared("output").path,
        panel_width=args.panel_width)


if __name__ == "__main__":
    main()
