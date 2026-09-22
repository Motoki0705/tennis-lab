"""Train camera-side and object identity association before triangulation."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from src.tasks.base.association.training import run_association_training
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.data.tracking_dataset import PLCSTrackingDataset
from src.utils.hydra import hydra_main, register_boundary_validator


def validate_config(config: DictConfig) -> None:
    base = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    del base["association"]
    PLCSTrainingConfig.from_config(base)


register_boundary_validator("plcs.association", validate_config)


@hydra_main(
    config_path="../configs",
    config_name="train_association",
    version_base="1.3",
    validation_boundary="plcs.association",
)
def main(config: DictConfig) -> None:
    run_association_training(config, task="plcs", dataset_factory=PLCSTrackingDataset)


if __name__ == "__main__":
    main()
