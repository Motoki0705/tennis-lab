"""Verify the fixed H development set and its train/val 2D calibration."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.ball_refiner.refiner_3d.verification import verify_dev_dataset
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)

PATH_BOUNDARY = NonHydraPathBoundary(name='ball_refiner.verify_synthetic_3d', fields=(
    BoundaryPathField('dataset', PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
    BoundaryPathField('output', PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if not args.dataset.is_absolute() or not args.output.is_absolute():
        parser.error('All paths must be absolute')
    roots = RuntimePathRoots(project_root=args.output.parent, data_root=args.dataset.parent,
        artifact_root=args.output.parent, output_root=args.output.parent, checkpoint_root=args.output.parent,
        cache_root=args.output.parent, external_asset_root=args.dataset.parent)
    paths = PATH_BOUNDARY.validate({'dataset': args.dataset, 'output': args.output}, resolver=PathResolver(roots))
    verify_dev_dataset(paths.declared('dataset').path, paths.declared('output').path)


if __name__ == '__main__':
    main()
