"""Incremental real-video dataset construction for ``tennis_scene``."""

from src.tennis_scene.generate_dataset.manifest import (
    DATASET_MANIFEST_FILENAME,
    DatasetClipRecord,
    DatasetManifest,
    load_dataset_manifest,
)
from src.tennis_scene.generate_dataset.pseudo_annotation import (
    AnnotationGenerationResult,
    generate_pseudo_annotations,
)

__all__ = [
    "DATASET_MANIFEST_FILENAME",
    "AnnotationGenerationResult",
    "DatasetClipRecord",
    "DatasetManifest",
    "generate_pseudo_annotations",
    "load_dataset_manifest",
]
