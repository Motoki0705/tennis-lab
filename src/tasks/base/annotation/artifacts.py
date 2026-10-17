"""Verified immutable artifacts and process-scoped filesystem exclusion."""

from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

from .contracts import ArtifactRef, Document, content_digest
from .paths import AnnotationLayout
from .plan import file_digest

D = TypeVar("D", bound=Document)


def verify_reference(layout: AnnotationLayout, reference: ArtifactRef) -> Path:
    path: Path = layout.artifact_path(reference)
    if not path.is_file() or path.stat().st_size != reference.size_bytes:
        raise ValueError(f"artifact missing or size changed: {path}")
    if file_digest(path) != reference.digest:
        raise ValueError(f"artifact hash changed: {path}")
    return path


def annotation_reference(layout: AnnotationLayout, path: Path) -> ArtifactRef:
    relative = path.relative_to(layout.root).as_posix()
    layout.artifact_path(
        ArtifactRef(role="annotation", path=relative, digest="0" * 64, size_bytes=0)
    )
    reference = ArtifactRef(
        role="annotation",
        path=relative,
        digest=file_digest(path),
        size_bytes=path.stat().st_size,
    )
    return reference


def read_document(layout: AnnotationLayout, reference: ArtifactRef, kind: type[D]) -> D:
    document: D = kind.model_validate_json(
        verify_reference(layout, reference).read_bytes()
    )
    return document


@contextmanager
def artifact_lock(
    layout: AnnotationLayout, name: str, *, blocking: bool = True
) -> Iterator[None]:
    path = layout.path("cache", "locks", name + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a") as guard:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(guard, flags)
        yield


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_document(
    layout: AnnotationLayout, path: Path, document: Document
) -> ArtifactRef:
    """Write once. Same bytes recover; different bytes never replace history."""
    relative = path.relative_to(layout.root).as_posix()
    layout.artifact_path(
        ArtifactRef(role="annotation", path=relative, digest="0" * 64, size_bytes=0)
    )
    payload = document.canonical_bytes() + b"\n"
    with artifact_lock(layout, "document-" + content_digest(relative)):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(
                    f"immutable document already has different content: {path}"
                )
        else:
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=path.parent, delete=False
                ) as output:
                    temporary = Path(output.name)
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, path)
                sync_directory(path.parent)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return annotation_reference(layout, path)
