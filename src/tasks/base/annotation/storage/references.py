"""Explicit reference collectors for durable workflow state, without expiry policy."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from ..contracts import ArtifactRef, Document
from ..paths import AnnotationLayout
from ..plan import references
from .contracts import Protection


def distinct(
    layout: AnnotationLayout, values: Iterable[ArtifactRef]
) -> tuple[ArtifactRef, ...]:
    by_path: dict[str, ArtifactRef] = {}
    for reference in sorted(values, key=lambda r: (r.role, r.path, r.digest)):
        path = str(layout.artifact_path(reference))
        old = by_path.setdefault(path, reference)
        if (old.digest, old.size_bytes) != (reference.digest, reference.size_bytes):
            raise ValueError(
                "retained references disagree about the same physical path"
            )
    return tuple(by_path[path] for path in sorted(by_path))


def protection(
    layout: AnnotationLayout,
    *,
    identity: str,
    kind: Literal["adoption", "publication", "training", "model", "active", "reader"],
    reason: str,
    roots: tuple[ArtifactRef, ...],
    documents: tuple[Document, ...] = (),
    dependencies: tuple[ArtifactRef, ...] = (),
) -> Protection:
    """Only typed, explicitly supplied documents are traversed; opaque refs stay opaque."""
    return Protection(
        protection_id=identity,
        kind=kind,
        reason=reason,
        references=distinct(
            layout,
            (
                *roots,
                *dependencies,
                *(ref for doc in documents for ref in references(doc)),
            ),
        ),
    )
