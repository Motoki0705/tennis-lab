"""Fetch the published synchronized sample without replacing local inputs."""

from __future__ import annotations

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

DATASET_REPO = "Motoki098/tennis-lab-multiview-sample"


@dataclass(frozen=True)
class PublishedSampleFile:
    filename: str
    sha256: str


PUBLISHED_FILES = (
    PublishedSampleFile(
        "cam0.mp4",
        "f99f0024067da49805654d65eae848407c47ee710411d91892f0921fe8331f39",
    ),
    PublishedSampleFile(
        "cam1.mp4",
        "ac333c334b81141ee72098be6c95f5a6f6e1ccc5faf6e058f6fd3491cc1ac96e",
    ),
    PublishedSampleFile(
        "cam2.mp4",
        "fd82f0b573e9f8788cc56567dabdaf0c0c3775fd5dbede4835707277ef70c461",
    ),
    PublishedSampleFile(
        "clip.json",
        "410d77571f553123ffd57ebb457f45f4798f57d7ab6a0828e70b0a47ec42eaf2",
    ),
    PublishedSampleFile(
        "LICENSE",
        "38762e3777f4ec00a6f769062a7c3f704fb78ce08303ecff88558da4c49cf9ea",
    ),
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="sample_dataset.download",
    fields=(
        BoundaryPathField(
            "data_root",
            PathRole.DATA,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
    ),
)


def _data_resolver(data_root: Path) -> PathResolver:
    if not data_root.is_absolute():
        raise ValueError("data_root must be an absolute path")
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=data_root.resolve(),
        checkpoint_root=PROJECT_ROOT / "ckpt",
        artifact_root=PROJECT_ROOT / "outputs",
        output_root=PROJECT_ROOT / "outputs",
        cache_root=PROJECT_ROOT / ".cache",
        external_asset_root=PROJECT_ROOT / "third_party",
    )
    resolver = PathResolver(roots)
    PATH_BOUNDARY.validate({"data_root": data_root}, resolver=resolver)
    return resolver


def _verify(path: Path, published: PublishedSampleFile) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing sample file: {path}")
    actual = dual_sha256(path)
    if actual != published.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {published.sha256}, got {actual}. "
            "Move the conflicting file explicitly before downloading again."
        )


def ensure_sample_dataset(
    data_root: Path, *, revision: str = "main", verify_only: bool = False
) -> tuple[Path, ...]:
    """Authenticate the release and install missing files below data_root/samples.

    Check all existing files before making requests. Missing files are fetched
    afresh even with a Hub cache or pinned revision; network failures propagate.
    Each file is published atomically from a verified temporary copy, with no
    overwrite. A later failure can leave earlier verified files installed; reruns
    verify and retain them. Offline verification never downloads or creates files.
    """
    if not revision.strip():
        raise ValueError("revision must not be empty")
    resolver = _data_resolver(data_root)
    selected = tuple(
        (published, resolver.resolve(PathRole.DATA, "samples", published.filename))
        for published in PUBLISHED_FILES
    )
    pending = []
    for published, destination in selected:
        if verify_only or destination.exists():
            _verify(destination, published)
        else:
            pending.append((published, destination))
    for published, destination in pending:
        downloaded = Path(
            hf_hub_download(
                repo_id=DATASET_REPO,
                filename=published.filename,
                repo_type="dataset",
                revision=revision,
                force_download=True,
            )
        )
        _verify(downloaded, published)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(downloaded, temporary)
            _verify(temporary, published)
            os.link(temporary, destination)
        finally:
            temporary.unlink()
    return tuple(destination for _, destination in selected)
