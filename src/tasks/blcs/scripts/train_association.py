"""Train BLCS camera-local association with Global MHA and mHC."""

from omegaconf import DictConfig

from src.tasks.blcs.configuration import validate_association_config
from src.tasks.blcs.training.runner import BLCSTrainingRunner
from src.utils.hydra import hydra_main, register_boundary_validator

register_boundary_validator("blcs.association", validate_association_config)


@hydra_main(
    config_path="../configs",
    config_name="train_association",
    version_base="1.3",
    validation_boundary="blcs.association",
)
def main(config: DictConfig) -> None:
    BLCSTrainingRunner().run(config)


if __name__ == "__main__":
    main()
