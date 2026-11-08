"""Validated v3 reads and immutable writes through the common artifact store."""

from __future__ import annotations

from pathlib import Path

from src.tasks.base.annotation.artifacts import write_document
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.paths import AnnotationLayout

from .contracts import VideoAnnotationInput, VideoBallAnnotation
from .usage import summarize_annotation


def _validated(
    payload: bytes,
    expected_input: VideoAnnotationInput | None,
    require_complete: bool,
) -> VideoBallAnnotation:
    annotation: VideoBallAnnotation = VideoBallAnnotation.model_validate_json(payload)
    if expected_input is not None and annotation.input_identity != expected_input:
        raise ValueError(
            "annotation source/recording/camera/target/frame/PTS differs from the expected input"
        )
    if require_complete and not summarize_annotation(annotation).complete:
        raise ValueError("annotation contains unprocessed frames")
    return annotation


def read_annotation(
    path: Path,
    *,
    expected_input: VideoAnnotationInput | None = None,
    require_complete: bool = False,
) -> VideoBallAnnotation:
    """Read only explicit v3. Pass pinned expected_input when attaching downstream."""
    return _validated(path.read_bytes(), expected_input, require_complete)


def write_annotation(
    layout: AnnotationLayout,
    path: Path,
    annotation: VideoBallAnnotation,
    *,
    expected_input: VideoAnnotationInput | None = None,
    require_complete: bool = False,
) -> ArtifactRef:
    """Write a new immutable revision; an identical retry reuses the artifact.

    Partial annotations are valid snapshots. A correction needs a new path;
    neither old v2 files nor previous v3 revisions are overwritten.
    """
    validated = _validated(
        annotation.canonical_bytes(), expected_input, require_complete
    )
    return write_document(layout, path, validated)
