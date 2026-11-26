"""Build the fixed Meiji multiview-v3 release on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tennis_scene.ball_events.meiji_benchmark import build_meiji_benchmark
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
    name="tennis_scene.meiji_event_benchmark",
    fields=(
        BoundaryPathField(
            "base", PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY,
            must_exist=True, allow_role_root=True,
        ),
        BoundaryPathField(
            "annotations", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY,
            must_exist=True, allow_role_root=True,
        ),
        BoundaryPathField(
            "output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY,
            allow_role_root=True,
        ),
    ),
)


def main(argv: list[str] | None = None) -> None:
    meiji = PROJECT_ROOT / "data/tennis_multiview/processed/meiji_4cam"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=meiji / "benchmarks/multiview-v2")
    parser.add_argument("--annotations", type=Path, default=meiji / "dataset/videos/video_000/clip_000-068_ball_events")
    parser.add_argument("--output", type=Path, default=meiji / "benchmarks/multiview-v3")
    args = parser.parse_args(argv)
    base = args.base.expanduser().resolve()
    annotations = args.annotations.expanduser().resolve()
    output = args.output.expanduser().resolve()
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=base,
        checkpoint_root=PROJECT_ROOT,
        artifact_root=annotations,
        output_root=output,
        cache_root=PROJECT_ROOT,
        external_asset_root=PROJECT_ROOT,
    )
    paths = PATH_BOUNDARY.validate(
        {"base": base, "annotations": annotations, "output": output},
        resolver=PathResolver(roots),
    )
    output = paths.declared("output").path
    existed = output.exists()
    manifest = build_meiji_benchmark(
        paths.declared("base").path, paths.declared("annotations").path, output,
    )
    print(json.dumps({
        "status": "verified" if existed else "created",
        "output": str(output),
        "provenance": manifest["provenance"],
        "verification": manifest["verification"],
        "ball_events": manifest["ball_events"],
        "splits": {split: {key: value for key, value in manifest["splits"][split].items() if key != "clip_ids"}
                   for split in ("train", "val", "test")},
    }, indent=2))


if __name__ == "__main__":
    main()
