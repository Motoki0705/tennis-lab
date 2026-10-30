"""Freeze video-disjoint val/test or the original validation without modifying data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tasks.court_detection.scaling.real_split import (
    make_original_val_registry,
    make_registry,
    real_source,
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
    name="court_detection.freeze_real_evaluation",
    fields=(
        BoundaryPathField(
            "root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--policy", choices=("video-disjoint", "original-val"), default="video-disjoint"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--val-fraction", type=float)
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError(
            "Evaluation registry output and the original store must be disjoint."
        )
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=root,
        checkpoint_root=PROJECT_ROOT,
        external_asset_root=PROJECT_ROOT,
        artifact_root=root,
        output_root=output.parent,
        cache_root=PROJECT_ROOT,
    )
    paths = PATH_BOUNDARY.validate(
        {"root": root, "output": output},
        resolver=PathResolver(roots),
        independent_artifact_inputs=True,
    )
    source = real_source(paths.declared("root").path)
    if args.policy == "original-val":
        if args.seed is not None or args.val_fraction is not None:
            raise ValueError(
                "original-val preserves the existing split; seed/fraction are not applicable."
            )
        registry = make_original_val_registry(source, version=args.version)
    else:
        if args.seed is None:
            raise ValueError("video-disjoint requires an explicit --seed.")
        registry = make_registry(
            source,
            version=args.version,
            seed=args.seed,
            val_fraction=0.5 if args.val_fraction is None else args.val_fraction,
        )
    destination = paths.declared("output").path
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    print(
        json.dumps(
            {key: registry[key] for key in ("schema", "version", "sha256", "summary")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
