"""Run the bounded synthetic development comparison (CUDA via training queue)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tasks.ball_refiner.refiner_3d.diffusion.dev_training import run_dev_training
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
    name='ball_refiner.training_dev_3d', fields=(
        BoundaryPathField('dataset', PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField('config', PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField('output', PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    if not all(p.is_absolute() for p in (args.dataset, args.config, args.output)):
        parser.error('All paths must be absolute')
    roots = RuntimePathRoots(project_root=args.config.parent, data_root=args.dataset.parent,
        artifact_root=args.config.parent, output_root=args.output.parent, checkpoint_root=args.output.parent,
        cache_root=args.output.parent, external_asset_root=args.dataset.parent)
    paths = PATH_BOUNDARY.validate({'dataset': args.dataset, 'config': args.config, 'output': args.output}, resolver=PathResolver(roots))
    result = run_dev_training(paths.declared('dataset').path, paths.declared('config').path,
                              paths.declared('output').path, device=args.device)
    print(json.dumps({'status': result['status'], 'resources': result['resources']}, indent=2))


if __name__ == '__main__':
    main()
