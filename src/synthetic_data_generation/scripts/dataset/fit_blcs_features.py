"""Fit one external ball Gaussian source into the shared NHT appearance space.

Usage:
    python -m src.synthetic_data_generation.scripts.dataset.fit_blcs_features source=nht/data/ball.pt

Notes:
    - Configuration is composed from configs/dataset/blcs_feature_fit.yaml.
    - Source, artifact, and output paths use the shared role-based resolver.
"""

from __future__ import annotations

from typing import cast

from omegaconf import DictConfig

from src.synthetic_data_generation.configuration import validate_config
from src.synthetic_data_generation.dataset.blcs.rendering.feature_fit import (
    FeatureFitRequest,
    FeatureFitRuntimeAssets,
    run_feature_fit,
)
from src.utils.configuration import PathRole
from src.utils.hydra import hydra_main


@hydra_main(
    version_base="1.3",
    config_path="../../configs",
    config_name="dataset/blcs_feature_fit",
    validation_boundary="synthetic.dataset.blcs.feature_fit",
)
def main(cfg: DictConfig) -> None:
    """Resolve the strict feature-fit configuration and run the worker."""
    runtime = validate_config("synthetic.dataset.blcs.feature_fit", cfg)
    run_feature_fit(
        FeatureFitRequest(
            source=runtime.path(PathRole.EXTERNAL_ASSET, "source"),
            source_format=cast(str, runtime.values["source_format"]),
            calibration_bundle=runtime.path(PathRole.ARTIFACT, "calibration_bundle"),
            target_appearance=runtime.path(PathRole.ARTIFACT, "target_appearance"),
            target_appearance_space_sha256=cast(
                str, runtime.values["target_appearance_space_sha256"]
            ),
            output_dir=runtime.path(PathRole.ARTIFACT, "output_dir"),
            optimization_steps=cast(int, runtime.values["optimization_steps"]),
            feature_lr=cast(float, runtime.values["feature_lr"]),
            final_lr_fraction=cast(float, runtime.values["final_lr_fraction"]),
            min_validation_psnr_db=cast(
                float, runtime.values["min_validation_psnr_db"]
            ),
            seed=cast(int, runtime.values["seed"]),
            device=cast(str, runtime.values["device"]),
            runtime_assets=FeatureFitRuntimeAssets(
                pins=runtime.path(PathRole.EXTERNAL_ASSET, "runtime_pins"),
                nht_repository=runtime.path(PathRole.EXTERNAL_ASSET, "nht_repository"),
                gsplat_repository=runtime.path(
                    PathRole.EXTERNAL_ASSET, "gsplat_repository"
                ),
                worker_source=runtime.path(PathRole.PROJECT, "worker_source"),
            ),
        )
    )


if __name__ == "__main__":
    main()
