"""Extract foreground-player GVHMR motions from a structured tennis dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.tasks.plcs.motion.gvhmr_extraction import (
    GvhmrMotionExtractor,
    GvhmrSelectionConfig,
    load_model_runtime,
    sha256_file,
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

LOGGER = logging.getLogger(__name__)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="plcs.gvhmr_motion_extraction",
    fields=(
        BoundaryPathField(
            "dataset_root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "output_root",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "selection_config",
            PathRole.PROJECT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "model_config",
            PathRole.PROJECT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "asset_repository_root",
            PathRole.PROJECT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "checkpoint_root",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
        ),
        BoundaryPathField(
            "dino_checkpoint",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
    ),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--selection-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--asset-repository-root",
        type=Path,
        required=True,
        help="Repository root whose data/, ckpt/, and third_party/ hold model assets.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Explicit GVHMR checkpoint root.",
    )
    parser.add_argument(
        "--dino-checkpoint",
        type=Path,
        required=True,
        help="Explicit DINO 4-scale Swin-L checkpoint.",
    )
    parser.add_argument("--clip-id", action="append", dest="clip_ids")
    parser.add_argument("--camera-id", action="append", dest="camera_ids")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-preview", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parser().parse_args(argv)
    root_arguments = {
        "dataset_root": args.dataset_root,
        "output_root": args.output_root,
        "asset_repository_root": args.asset_repository_root,
    }
    relative_roots = tuple(
        name for name, value in root_arguments.items() if not value.is_absolute()
    )
    if relative_roots:
        raise ValueError(
            "GVHMR extraction root paths must be absolute: " + ", ".join(relative_roots)
        )
    repository_root = args.asset_repository_root.resolve(strict=False)
    dataset_root = args.dataset_root.resolve(strict=False)
    output_root = args.output_root.resolve(strict=False)
    resolver = PathResolver(
        RuntimePathRoots(
            project_root=repository_root,
            data_root=dataset_root,
            checkpoint_root=repository_root,
            artifact_root=output_root,
            output_root=output_root,
            cache_root=(repository_root / ".cache").resolve(strict=False),
            external_asset_root=repository_root,
        )
    )
    paths = PATH_BOUNDARY.validate(
        {
            "dataset_root": args.dataset_root,
            "output_root": args.output_root,
            "selection_config": args.selection_config,
            "model_config": args.model_config,
            "asset_repository_root": args.asset_repository_root,
            "checkpoint_root": args.checkpoint_root,
            "dino_checkpoint": args.dino_checkpoint,
        },
        resolver=resolver,
    )
    selection_config = paths.declared("selection_config").path
    selection = GvhmrSelectionConfig.load(selection_config)
    runtime = load_model_runtime(
        paths.declared("model_config").path,
        repository_root=paths.declared("asset_repository_root").path,
        checkpoint_root=paths.declared("checkpoint_root").path,
        dino_checkpoint=paths.declared("dino_checkpoint").path,
    )
    extractor = GvhmrMotionExtractor(
        dataset_root=paths.declared("dataset_root").path,
        output_root=paths.declared("output_root").path,
        selection=selection,
        selection_digest=sha256_file(selection_config),
        model_runtime=runtime,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
        write_preview=args.write_preview,
    )
    summary = extractor.run(
        clip_ids=None if args.clip_ids is None else tuple(args.clip_ids),
        camera_ids=None if args.camera_ids is None else tuple(args.camera_ids),
        max_clips=args.max_clips,
    )
    LOGGER.info(
        "GVHMR extraction complete: generated=%d skipped=%d",
        summary["generated"],
        summary["skipped"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
