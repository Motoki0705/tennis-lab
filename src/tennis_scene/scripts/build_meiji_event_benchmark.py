"""Build the fixed Meiji multiview-v3 or multiview-v4 release on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tennis_scene.ball_events.meiji_benchmark import (
    build_meiji_benchmark,
    build_meiji_v4_benchmark,
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
    parser.add_argument("--version", choices=("v3", "v4"), default="v3")
    parser.add_argument("--base", type=Path, default=None)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.version == "v4":
        default_base = meiji / "benchmarks/multiview-v3"
        default_annotations = meiji / "dataset/videos/video_000/clip_000-068_ball_events_v2"
        default_output = meiji / "benchmarks/multiview-v4"
        builder = build_meiji_v4_benchmark
    else:
        default_base = meiji / "benchmarks/multiview-v2"
        default_annotations = meiji / "dataset/videos/video_000/clip_000-068_ball_events"
        default_output = meiji / "benchmarks/multiview-v3"
        builder = build_meiji_benchmark
    base = (args.base or default_base).expanduser().resolve()
    annotations = (args.annotations or default_annotations).expanduser().resolve()
    output = (args.output or default_output).expanduser().resolve()
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
    manifest = builder(
        paths.declared("base").path, paths.declared("annotations").path, output,
    )
    print(json.dumps({
        "status": "verified" if existed else "created",
        "version": args.version,
        "output": str(output),
        "provenance": manifest["provenance"],
        "verification": manifest["verification"],
        "ball_events": manifest["ball_events"],
        "splits": {split: {key: value for key, value in manifest["splits"][split].items() if key != "clip_ids"}
                   for split in ("train", "val", "test")},
    }, indent=2))


if __name__ == "__main__":
    main()
