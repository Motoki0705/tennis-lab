"""Download or verify the published court and ball inference checkpoints."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.utils.checksum import dual_sha256
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

MODEL_REPO = "Motoki098/tennis-lab-models"


@dataclass(frozen=True)
class PublishedWeight:
    name: str
    filename: str
    sha256: str


PUBLISHED_WEIGHTS = (
    PublishedWeight(
        "court",
        "court_detection/multiscale_depth3/b863df1f01f0.ckpt",
        "bca48cdbdf0b26847b3f650cc4249397b2f7c0a963ba4801b7cd111c6e81522c",
    ),
    PublishedWeight(
        "ball",
        "ball_detection/i935-mixed-ft-s42-epoch09.ckpt",
        "489bb90fb6bb4deab81e507b17e698fbe40ee945533196e78d6c69e4ee3f5828",
    ),
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="model_weights.download",
    fields=(
        BoundaryPathField(
            "checkpoint_root",
            PathRole.CHECKPOINT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
    ),
)


def _checkpoint_resolver(checkpoint_root: Path) -> PathResolver:
    if not checkpoint_root.is_absolute():
        raise ValueError("checkpoint_root must be an absolute path")
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=PROJECT_ROOT / "data",
        checkpoint_root=checkpoint_root.resolve(),
        artifact_root=PROJECT_ROOT / "outputs",
        output_root=PROJECT_ROOT / "outputs",
        cache_root=PROJECT_ROOT / ".cache",
        external_asset_root=PROJECT_ROOT / "third_party",
    )
    resolver = PathResolver(roots)
    PATH_BOUNDARY.validate({"checkpoint_root": checkpoint_root}, resolver=resolver)
    return resolver


def _verify(path: Path, weight: PublishedWeight) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    actual = dual_sha256(path)
    if actual != weight.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {weight.sha256}, got {actual}. "
            "Move the conflicting file explicitly before downloading again."
        )


def ensure_weights(
    checkpoint_root: Path,
    *,
    model: str = "all",
    revision: str = "main",
    verify_only: bool = False,
) -> tuple[Path, ...]:
    """Authenticate cached and copied bytes; never replace an existing file.

    Hub/network errors propagate. An invalid existing checkpoint stops the
    complete selection before any download. Temporary copies are authenticated
    before an atomic, no-clobber hard link publishes them under checkpoint_root.
    """
    if model not in {"all", *(weight.name for weight in PUBLISHED_WEIGHTS)}:
        raise ValueError(f"Unknown model: {model}")
    if not revision.strip():
        raise ValueError("revision must not be empty")
    resolver = _checkpoint_resolver(checkpoint_root)
    selected = tuple(
        (weight, resolver.resolve(PathRole.CHECKPOINT, weight.filename))
        for weight in PUBLISHED_WEIGHTS
        if model in {"all", weight.name}
    )
    pending = []
    for weight, destination in selected:
        if verify_only or destination.exists():
            _verify(destination, weight)
        else:
            pending.append((weight, destination))
    for weight, destination in pending:
        cached = Path(
            hf_hub_download(
                repo_id=MODEL_REPO,
                filename=weight.filename,
                repo_type="model",
                revision=revision,
            )
        )
        _verify(cached, weight)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(cached, temporary)
            _verify(temporary, weight)
            os.link(temporary, destination)
        finally:
            temporary.unlink()
    return tuple(destination for _, destination in selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Absolute checkpoint root (normally $PWD/ckpt)",
    )
    parser.add_argument("--model", choices=("all", "court", "ball"), default="all")
    parser.add_argument("--revision", default="main", help="Hub branch, tag or commit")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check existing files without accessing the network",
    )
    args = parser.parse_args()
    for path in ensure_weights(
        args.checkpoint_root,
        model=args.model,
        revision=args.revision,
        verify_only=args.verify_only,
    ):
        print(f"Verified {path}")


if __name__ == "__main__":
    main()
