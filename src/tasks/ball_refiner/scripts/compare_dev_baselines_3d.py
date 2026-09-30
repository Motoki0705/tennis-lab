"""CPU comparison of saved 2k dev predictions and untrained condition baselines."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.tasks.ball_refiner.refiner_3d.baseline_comparison import compare_saved_dev
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
    name='ball_refiner.compare_dev_baselines_3d', fields=(
        BoundaryPathField('dataset', PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField('training_output', PathRole.OUTPUT, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField('output', PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'training-output', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if not all(p.is_absolute() for p in (args.dataset, args.training_output, args.output)):
        parser.error('All paths must be absolute')
    roots = RuntimePathRoots(project_root=args.dataset.parent, data_root=args.dataset.parent,
        artifact_root=args.output.parent, output_root=Path(os.path.commonpath([args.training_output, args.output])),
        checkpoint_root=args.output.parent,
        cache_root=args.output.parent, external_asset_root=args.dataset.parent)
    paths = PATH_BOUNDARY.validate({'dataset': args.dataset, 'training_output': args.training_output,
                                   'output': args.output}, resolver=PathResolver(roots))
    result = compare_saved_dev(paths.declared('dataset').path, paths.declared('training_output').path,
                               paths.declared('output').path)
    print(f"Compared {len(result['rallies'])} val rallies, {result['frames']} frames; historical metrics match")


if __name__ == '__main__':
    main()
