"""Train camera-side and object identity association before triangulation."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from src.tasks.base.association.training import run_association_training
from src.tasks.blcs.configuration import validate_training_boundary
from src.tasks.blcs.data.tracking_dataset import BLCSTrackingDataset
from src.utils.hydra import hydra_main, register_boundary_validator


def validate_config(config: DictConfig) -> None:
    base = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    del base["association"]
    validate_training_boundary(base)


register_boundary_validator("blcs.association", validate_config)


@hydra_main(
    config_path="../configs",
    config_name="train_association",
    version_base="1.3",
    validation_boundary="blcs.association",
)
def main(config: DictConfig) -> None:
    run_association_training(config, task="blcs", dataset_factory=BLCSTrackingDataset)


if __name__ == "__main__":
    main()
