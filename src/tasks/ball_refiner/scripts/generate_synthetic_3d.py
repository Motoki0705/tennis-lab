"""Generate a CPU-only, immutable 3D-refiner synthetic dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import load_plan
from src.tasks.ball_refiner.refiner_3d.synthetic.generator import generate_dataset
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
    name="ball_refiner.generate_synthetic_3d",
    fields=(
        BoundaryPathField("plan", PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("output", PathRole.DATA, PathDirection.OUTPUT, PathKind.DIRECTORY),
        BoundaryPathField("calibration_report", PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, required=False, must_exist=True),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "output", "project-root", "data-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "dev", "pilot"), required=True)
    parser.add_argument("--calibration-report", type=Path)
    args = parser.parse_args()
    if not all(p.is_absolute() for p in (args.plan, args.output, args.project_root, args.data_root)):
        parser.error("All paths must be absolute")
    roots = RuntimePathRoots(
        project_root=args.project_root, data_root=args.data_root,
        artifact_root=args.output.parent, output_root=args.output.parent,
        checkpoint_root=args.output.parent, cache_root=args.output.parent,
        external_asset_root=args.output.parent,
    )
    resolver = PathResolver(roots)
    arguments = {"plan": args.plan, "output": args.output}
    if args.calibration_report is not None:
        arguments["calibration_report"] = args.calibration_report
    paths = PATH_BOUNDARY.validate(arguments, resolver=resolver)
    report = paths.declared("calibration_report") if "calibration_report" in paths else None
    plan = load_plan(paths.declared("plan").path, resolver, calibration_report=report.path if report is not None else None)
    generate_dataset(plan, paths.declared("output").path, mode=args.mode)


if __name__ == "__main__":
    main()
