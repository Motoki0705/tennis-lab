"""CPU audit of a complete dev dataset and its train/val 3D conditions."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.ball_refiner.refiner_3d.condition_audit import run_condition_audit
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)

PATH_BOUNDARY = NonHydraPathBoundary(name='ball_refiner.audit_conditions_3d', fields=(
    BoundaryPathField('dataset', PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
    BoundaryPathField('output', PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if not args.dataset.is_absolute() or not args.output.is_absolute():
        parser.error('All paths must be absolute')
    roots = RuntimePathRoots(project_root=args.output.parent, data_root=args.dataset.parent,
        artifact_root=args.output.parent, output_root=args.output.parent, checkpoint_root=args.output.parent,
        cache_root=args.output.parent, external_asset_root=args.dataset.parent)
    paths = PATH_BOUNDARY.validate({'dataset': args.dataset, 'output': args.output}, resolver=PathResolver(roots))
    result = run_condition_audit(paths.declared('dataset').path, paths.declared('output').path,
                                 samples=args.samples, audit_only=args.audit_only)
    print(result['condition_metrics'])


if __name__ == '__main__':
    main()
