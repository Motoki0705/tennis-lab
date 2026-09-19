"""Prepare recording-disjoint BLCS pseudo-labels from an explicit source recipe."""

from omegaconf import DictConfig

from src.tennis_scene.dataset_pipeline.preparation import (
    BLCSPreparationConfig,
    prepare_blcs,
    validate_blcs_preparation_config,
)
from src.utils.hydra import hydra_main, register_boundary_validator

register_boundary_validator(
    "tennis_scene.prepare_blcs_real_dataset", validate_blcs_preparation_config
)


@hydra_main(
    config_path="../configs",
    config_name="prepare_blcs_real_dataset",
    version_base="1.3",
    validation_boundary="tennis_scene.prepare_blcs_real_dataset",
)
def main(cfg: DictConfig) -> None:
    prepare_blcs(BLCSPreparationConfig.from_config(cfg))


if __name__ == "__main__":
    main()
