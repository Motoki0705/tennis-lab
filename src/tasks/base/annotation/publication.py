"""Task-neutral, owned staging and atomic publication of immutable directories."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from .artifacts import artifact_lock, data_reference, sync_directory
from .contracts import ArtifactRef, Document, content_digest
from .paths import AnnotationLayout

D = TypeVar("D", bound=Document)


def durable_document(path: Path, document: Document) -> None:
    """Write once; completed documents are never repaired or overwritten."""
    payload = document.canonical_bytes() + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Immutable publication document differs: {path}")
        return
    temporary = path.with_name(".write-" + path.name)
    with temporary.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.link(temporary, path)
    temporary.unlink()
    sync_directory(path.parent)


def link_artifact(
    layout: AnnotationLayout, reference: ArtifactRef, target: Path
) -> None:
    """Share exact immutable bytes; cross-filesystem copying is never implicit."""
    from .artifacts import verify_reference

    source = verify_reference(layout, reference)
    os.link(source, target)
    actual = data_reference(layout, target)
    if (actual.digest, actual.size_bytes) != (reference.digest, reference.size_bytes):
        raise ValueError("Shared publication input changed while linking")


def final_reference(
    layout: AnnotationLayout, path: Path, staging: Path, destination: Path
) -> ArtifactRef:
    actual = data_reference(layout, path)
    return actual.model_copy(
        update={
            "path": (destination / path.relative_to(staging))
            .relative_to(layout.repository / "data")
            .as_posix()
        }
    )


def publish_directory(
    layout: AnnotationLayout,
    relative: str,
    intent: ArtifactRef,
    *,
    kind: type[D],
    build: Callable[[Path, Path], D],
    validate: Callable[[D, Path | None], None],
) -> tuple[ArtifactRef, D]:
    """Recover incomplete staging, rename, and post-rename interruption gaps."""
    destination = layout.artifact_path(
        ArtifactRef(
            role="data", path=relative + "/snapshot.json", digest="0" * 64, size_bytes=0
        )
    ).parent
    staging = destination.with_name("." + destination.name + ".staging")
    with artifact_lock(layout, "dataset-publication-" + content_digest(relative)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Validate the staging boundary before reading or cleaning any entry.
        layout.artifact_path(
            ArtifactRef(
                role="data",
                path=(staging / "owner.json")
                .relative_to(layout.repository / "data")
                .as_posix(),
                digest="0" * 64,
                size_bytes=0,
            )
        )
        if destination.exists():
            owner = ArtifactRef.model_validate_json(
                (destination / "owner.json").read_bytes()
            )
            if owner != intent:
                raise ValueError("Dataset destination is owned by another publication")
            snapshot = kind.model_validate_json(
                (destination / "snapshot.json").read_bytes()
            )
        else:
            staging.mkdir(exist_ok=True)
            owner_path = staging / "owner.json"
            if not owner_path.exists():
                if any(
                    p.name != ".write-owner.json" or p.is_symlink() or not p.is_file()
                    for p in staging.iterdir()
                ):
                    raise ValueError("Publication staging has no verifiable owner")
                durable_document(owner_path, intent)
            if ArtifactRef.model_validate_json(owner_path.read_bytes()) != intent:
                raise ValueError("Dataset staging is owned by another publication")
            if not (staging / "snapshot.json").exists():
                # Keep the owner throughout cleanup, including cleanup failures.
                for path in staging.iterdir():
                    if path == owner_path:
                        continue
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                snapshot = build(staging, destination)
                durable_document(staging / "snapshot.json", snapshot)
            else:
                snapshot = kind.model_validate_json(
                    (staging / "snapshot.json").read_bytes()
                )
            validate(snapshot, staging)
            sync_directory(staging)
            os.rename(staging, destination)
            sync_directory(destination.parent)
        validate(snapshot, None)
        return data_reference(layout, destination / "snapshot.json"), snapshot
