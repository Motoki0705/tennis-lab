"""Queue payload: require a separate-process GPU freeze/unfreeze probe first."""
from __future__ import annotations

import json
import subprocess
import sys

from src.tasks.ball_detection.training.posttraining.checkpoint import (
    completed_pretraining,
)
from src.tasks.ball_detection.training.posttraining.paths import resolver
from src.tasks.ball_detection.training.posttraining.runner import arguments
from src.utils.checksum import dual_sha256
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

PATH_BOUNDARY = NonHydraPathBoundary(name="ball_detection.train_best_mdd_posttraining", fields=(
    BoundaryPathField("manifest", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("pretraining_run", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
    BoundaryPathField("augmentation_config", PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    BoundaryPathField("resume", PathRole.OUTPUT, PathDirection.INPUT, PathKind.FILE, must_exist=True, required=False),
))


def main() -> None:
    args = arguments()
    declared = {k: getattr(args, k) for k in ("manifest", "pretraining_run", "augmentation_config", "output", "resume")
                if getattr(args, k) is not None}
    paths = PATH_BOUNDARY.validate(declared, resolver=resolver(args.augmentation_config.parent,
        (args.manifest, args.pretraining_run), (args.output,), args.output.parent))
    for key in declared:
        setattr(args, key, paths.declared(key).path)
    probe = args.output.with_name(args.output.name + "-probe")
    if not (probe / "PROBE_COMPLETED.json").exists():
        if probe.exists():
            raise ValueError("Incomplete GPU probe requires inspection before retrying")
        subprocess.run([sys.executable, "-m", "src.tasks.ball_detection.scripts.probe_mdd_posttraining",
            "--manifest", str(args.manifest), "--pretraining-run", str(args.pretraining_run),
            "--augmentation-config", str(args.augmentation_config), "--output", str(probe),
            *(["--image-prefetch"] if args.image_prefetch else [])], check=True)
    receipt = json.loads((probe / "PROBE_COMPLETED.json").read_text())
    path, _ = completed_pretraining(args.pretraining_run, args.manifest)
    expected = dict(checkpoint_sha256=dual_sha256(path), manifest_sha256=dual_sha256(args.manifest),
                    augmentation_sha256=dual_sha256(args.augmentation_config), image_prefetch=args.image_prefetch)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("GPU probe does not match the selected CNN/data/augmentation")
    # Separate process also ensures full CPU input verification precedes CUDA startup.
    subprocess.run([sys.executable, "-m", "src.tasks.ball_detection.scripts.posttrain_mdd_query", *sys.argv[1:]], check=True)


if __name__ == "__main__":
    main()
