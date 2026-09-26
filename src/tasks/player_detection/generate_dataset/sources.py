"""Collect validated chat-annotation player labels with their prepared clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.tennis_scene.chat_annotation.layout import published_video_path
from src.tennis_scene.chat_annotation.manifests import (
    load_prepared_manifests,
    require_regular_path,
)
from src.tennis_scene.chat_annotation.runtime.contracts import (
    ClipManifest,
    PlayerAnnotation,
    parse_annotation,
    read_json,
    sha256_file,
)
from src.tennis_scene.chat_annotation.runtime.validation import validate_annotation

PROCESSED_PLAYER_DIR = Path("annotated/processed/player")


@dataclass(frozen=True, slots=True)
class ClipSource:
    """One adopted player annotation and the exact clip it describes."""

    clip_id: str
    manifest: ClipManifest
    annotation: PlayerAnnotation
    annotation_path: Path
    annotation_sha256: str
    video_path: Path


def collect_clip_sources(
    annotation_root: Path, *, allowed_statuses: frozenset[str]
) -> list[ClipSource]:
    """Return every processed player annotation, validated against its manifest.

    Any schema, identity, coverage, or coordinate error aborts the build: a
    processed annotation that cannot be used is a data problem to resolve in
    the annotation workflow, never something to skip silently.
    """
    root = annotation_root.resolve()
    manifests = load_prepared_manifests(root)
    directory = root / PROCESSED_PLAYER_DIR
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No processed player annotations in {directory}")
    sources: list[ClipSource] = []
    for path in paths:
        require_regular_path(path, root)
        annotation = parse_annotation(read_json(path))
        if not isinstance(annotation, PlayerAnnotation):
            raise ValueError(f"{path}: expected a player annotation schema")
        if annotation.clip_id != path.stem:
            raise ValueError(f"{path}: clip_id {annotation.clip_id!r} differs from file name")
        manifest = manifests.get(annotation.clip_id)
        if manifest is None:
            raise ValueError(f"{path}: no prepared clip manifest for {annotation.clip_id}")
        report = validate_annotation(annotation, manifest)
        if report.errors:
            raise ValueError(f"{path}: invalid annotation: {'; '.join(report.errors)}")
        if annotation.status not in allowed_statuses:
            raise ValueError(
                f"{path}: status {annotation.status!r} is not in {sorted(allowed_statuses)}"
            )
        video = published_video_path(root, manifest)
        require_regular_path(video, root)
        if not video.is_file():
            raise FileNotFoundError(f"Prepared clip video is missing: {video}")
        sources.append(
            ClipSource(
                clip_id=annotation.clip_id,
                manifest=manifest,
                annotation=annotation,
                annotation_path=path,
                annotation_sha256=sha256_file(path),
                video_path=video,
            )
        )
    return sources
