"""
Export explicit read-only COLMAP/3DGS inputs into a verified provider bundle
for tennis-scene alignment without importing the provider application.

Usage:
    python -m src.synthetic_data_generation.scripts.alignment.export_scene_provider
    python -m src.synthetic_data_generation.scripts.alignment.export_scene_provider output_dir=/tmp/b00-provider

Notes:
    - Configuration is loaded from
      `src/synthetic_data_generation/configs/alignment/export_scene_provider.yaml`.
    - Source files are verified by SHA-256 and are never modified.
    - Publication is atomic and refuses to replace an existing bundle.
"""

from __future__ import annotations

import logging
import shlex
import sys
from pathlib import Path

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig

from src.synthetic_data_generation.alignment.scene_provider.export import (
    ProviderExportExpectations,
    ProviderExportSettings,
    SourceArtifactInput,
    collect_exporter_provenance,
    export_scene_provider_bundle,
)
from src.utils.hydra import hydra_main

LOGGER = logging.getLogger(__name__)


def _path(value: object) -> Path:
    return Path(to_absolute_path(str(value)))


def _source_artifacts(cfg: DictConfig) -> tuple[SourceArtifactInput, ...]:
    raw_artifacts = cfg.source_artifacts
    if not isinstance(raw_artifacts, (list, tuple, ListConfig)):
        raise TypeError("source_artifacts must be a list.")
    return tuple(
        SourceArtifactInput(
            artifact_id=str(item.artifact_id),
            path=_path(item.path),
            sha256=str(item.sha256),
        )
        for item in raw_artifacts
    )


@hydra_main(
    version_base="1.3",
    config_path="../../configs/alignment",
    config_name="export_scene_provider",
)
def main(cfg: DictConfig) -> int:
    """Export a verified provider bundle from explicitly configured files."""
    repo_root = Path(to_absolute_path("."))
    geometry_bridge = (
        repo_root
        / "src/synthetic_data_generation/alignment/scene_provider/geometry_bridge.py"
    )
    code_paths = (
        repo_root / "src/synthetic_data_generation/alignment/scene_provider/bundle.py",
        repo_root / "src/synthetic_data_generation/alignment/scene_provider/export.py",
        geometry_bridge,
        repo_root / "src/synthetic_data_generation/scene_contract.py",
        repo_root
        / "src/synthetic_data_generation/scripts/alignment/export_scene_provider.py",
        repo_root
        / "src/synthetic_data_generation/configs/alignment/export_scene_provider.yaml",
    )
    settings = ProviderExportSettings(
        bundle_id=str(cfg.bundle_id),
        provider_backend=str(cfg.provider_backend),
        output_dir=_path(cfg.output_dir),
        cameras_bin=_path(cfg.cameras_bin),
        images_bin=_path(cfg.images_bin),
        points3d_bin=_path(cfg.points3d_bin),
        original_image_dir=_path(cfg.original_image_dir),
        factor_image_dir=_path(cfg.factor_image_dir),
        geometry_python=_path(cfg.geometry_python),
        geometry_bridge=geometry_bridge,
        factor=int(cfg.factor),
        group_size=int(cfg.group_size),
        source_artifacts=_source_artifacts(cfg),
        expectations=ProviderExportExpectations(
            camera_count=int(cfg.expectations.camera_count),
            image_width=int(cfg.expectations.image_width),
            image_height=int(cfg.expectations.image_height),
            camera_array_sha256=str(cfg.expectations.camera_array_sha256),
            shared_intrinsics_sha256=str(cfg.expectations.shared_intrinsics_sha256),
            normalization_sha256=str(cfg.expectations.normalization_sha256),
        ),
    )
    exporter = collect_exporter_provenance(
        repo_root=repo_root,
        code_paths=code_paths,
        command=shlex.join(
            [
                sys.executable,
                "-m",
                "src.synthetic_data_generation.scripts.alignment.export_scene_provider",
                *sys.argv[1:],
            ]
        ),
        geometry_python=settings.geometry_python,
        geometry_bridge=settings.geometry_bridge,
    )
    output_dir = export_scene_provider_bundle(settings, exporter=exporter)
    LOGGER.info("Published verified provider bundle: %s", output_dir)
    return 0


if __name__ == "__main__":
    main()
