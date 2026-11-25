"""Generate auditable pseudo annotations using shared clip lifecycle functions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.tennis_scene.pipeline.clip_inputs import (
    resolve_dataset_clip,
    select_dataset_clips,
)
from src.tennis_scene.pipeline.dataset_publication import (
    ANNOTATION_RELATIVE_DIR,
    ANNOTATION_SCHEMA_VERSION,
    completed_annotation,
    publish_annotation,
    record_annotation_failure,
    scene_array_manifest,
)
from src.tennis_scene.schema import SceneResult
from src.utils.inference_recording import recording_context, span

# (video paths, camera IDs, clip directory) -> scene; the clip directory owns the component store.
SceneRunner = Callable[[Sequence[Path], Sequence[str], Path], SceneResult]


@dataclass(frozen=True)
class AnnotationGenerationResult:
    """Outcome for one requested clip."""

    clip_id: str
    status: Literal["generated", "skipped", "failed"]
    annotation_path: Path | None = None
    error: str | None = None


def generate_pseudo_annotations(
    dataset_dir: Path,
    runner: SceneRunner,
    *,
    pipeline_config_yaml: str,
    publication_identity: Mapping[str, object],
    clip_ids: Sequence[str] | None = None,
    overwrite: bool = False,
    continue_on_error: bool = True,
    scene_export: Callable[[], Path] | None = None,
) -> list[AnnotationGenerationResult]:
    """Run selected clips, reusing the common input and publication boundary.

    ``runner`` must leave the scene export in the clip-owned component store.
    Selection and error continuation belong to this entry point; cache/load
    decisions belong to the execution plan and marker rules to the publisher.
    Orchestrator callers pass ``completed_scene_export`` to pin the exact export
    across concurrent writers, even if another lineage has identical arrays.
    """
    records = select_dataset_clips(dataset_dir, clip_ids)
    outcomes: list[AnnotationGenerationResult] = []
    for record in records:
        try:
            inputs = resolve_dataset_clip(dataset_dir, record)
            marker = completed_annotation(inputs, publication_identity, overwrite=overwrite)
            if marker is not None:
                outcomes.append(AnnotationGenerationResult(record.clip_id, "skipped", marker))
                continue
            with recording_context(clip_id=record.clip_id), span("clip"):
                result = runner(inputs.video_paths, inputs.camera_ids, inputs.clip_dir)
            marker = publish_annotation(inputs, result, pipeline_config_yaml=pipeline_config_yaml,
                                        publication_identity=publication_identity,
                                        expected_scene=None if scene_export is None else scene_export())
            outcomes.append(AnnotationGenerationResult(record.clip_id, "generated", marker))
        except Exception as error:
            message = record_annotation_failure(dataset_dir / record.path, record.clip_id, error,
                                                publication_identity=publication_identity)
            outcomes.append(AnnotationGenerationResult(record.clip_id, "failed", error=message))
            if not continue_on_error:
                raise
    return outcomes


__all__ = [
    "ANNOTATION_RELATIVE_DIR",
    "ANNOTATION_SCHEMA_VERSION",
    "AnnotationGenerationResult",
    "SceneRunner",
    "generate_pseudo_annotations",
    "scene_array_manifest",
]
