"""Web-derived tennis image collection pipeline for DINOv3 SSL.

The pipeline ingests tennis imagery from heterogeneous sources (local/remote
videos, image directories, and image URLs), normalises them into a flat image
folder, and writes a ``meta.json`` manifest consumed by the SSL datamodule.
"""

from src.tasks.dino_ssl.generate_dataset.collectors import (
    CollectedImage,
    collect_from_source,
)
from src.tasks.dino_ssl.generate_dataset.manifest import (
    DatasetManifest,
    write_manifest,
)
from src.tasks.dino_ssl.generate_dataset.runner import DinoSSLCollectionRunner

__all__ = [
    "CollectedImage",
    "collect_from_source",
    "DatasetManifest",
    "write_manifest",
    "DinoSSLCollectionRunner",
]
