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

from omegaconf import DictConfig, ListConfig

from src.synthetic_data_generation.alignment.scene_provider.export import (
    ProviderExportExpectations,
    ProviderExportSettings,
    SourceArtifactInput,
    collect_exporter_provenance,
    export_scene_provider_bundle,
)
from src.synthetic_data_generation.configuration import (
    SyntheticRuntimeConfig,
    validate_config,
)
from src.utils.configuration import PathRole
from src.utils.hydra import hydra_main

LOGGER = logging.getLogger(__name__)


def _source_artifacts(
    cfg: DictConfig,
    runtime: SyntheticRuntimeConfig,
) -> tuple[SourceArtifactInput, ...]:
    raw_artifacts = cfg.source_artifacts
    if not isinstance(raw_artifacts, (list, tuple, ListConfig)):
        raise TypeError("source_artifacts must be a list.")
    return tuple(
        SourceArtifactInput(
            artifact_id=str(item.artifact_id),
            path=runtime.resolver.resolve(PathRole.EXTERNAL_ASSET, str(item.path)),
            sha256=str(item.sha256),
        )
        for item in raw_artifacts
    )


@hydra_main(
    version_base="1.3",
    config_path="../../configs",
    config_name="alignment/export_scene_provider",
    validation_boundary="synthetic.alignment.export_scene_provider",
)
def main(cfg: DictConfig) -> int:
    """Export a verified provider bundle from explicitly configured files."""
    runtime = validate_config("synthetic.alignment.export_scene_provider", cfg)
    repo_root = runtime.resolver.roots.project_root
    geometry_bridge = runtime.path(PathRole.PROJECT, "geometry_bridge")
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
        output_dir=runtime.path(PathRole.DATA, "output_dir"),
        external_asset_scope=runtime.path(
            PathRole.EXTERNAL_ASSET,
            "external_asset_scope",
        ),
        cameras_bin=runtime.path(PathRole.EXTERNAL_ASSET, "cameras_bin"),
        images_bin=runtime.path(PathRole.EXTERNAL_ASSET, "images_bin"),
        points3d_bin=runtime.path(PathRole.EXTERNAL_ASSET, "points3d_bin"),
        original_image_dir=runtime.path(PathRole.EXTERNAL_ASSET, "original_image_dir"),
        factor_image_dir=runtime.path(PathRole.EXTERNAL_ASSET, "factor_image_dir"),
        geometry_executable=runtime.system_executable("geometry_executable"),
        geometry_bridge=geometry_bridge,
        resolver=runtime.resolver,
        factor=int(cfg.factor),
        group_size=int(cfg.group_size),
        source_artifacts=_source_artifacts(cfg, runtime),
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
        geometry_executable=settings.geometry_executable,
        geometry_bridge=settings.geometry_bridge,
    )
    output_dir = export_scene_provider_bundle(settings, exporter=exporter)
    LOGGER.info("Published verified provider bundle: %s", output_dir)
    return 0


if __name__ == "__main__":
    main()
