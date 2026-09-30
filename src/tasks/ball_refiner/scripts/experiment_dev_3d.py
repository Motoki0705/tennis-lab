"""Run a pinned one-factor dev experiment (CUDA only through the shared queue)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tasks.ball_refiner.refiner_3d.diffusion.experiment import run_experiment
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
    name='ball_refiner.experiment_dev_3d', fields=(
        BoundaryPathField('plan', PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if not args.plan.is_absolute():
        parser.error('Plan path must be absolute')
    if args.audit_only and args.device != 'cpu':
        parser.error('Audit-only execution is CPU-only')
    roots = RuntimePathRoots(project_root=args.plan.parent, data_root=args.plan.parent,
        artifact_root=args.plan.parent, output_root=args.plan.parent, checkpoint_root=args.plan.parent,
        cache_root=args.plan.parent, external_asset_root=args.plan.parent)
    paths = PATH_BOUNDARY.validate({'plan': args.plan}, resolver=PathResolver(roots))
    result = run_experiment(paths.declared('plan').path, device=args.device, audit_only=args.audit_only)
    print(json.dumps({'status': result['status'], 'resources': result.get('resources')}, indent=2))


if __name__ == '__main__':
    main()
