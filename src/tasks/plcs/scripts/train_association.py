"""Train PLCS camera-local association with Global MHA and mHC."""

from omegaconf import DictConfig

from src.tasks.plcs.configuration import validate_association_config
from src.tasks.plcs.training.runner import PLCSTrainingRunner
from src.utils.hydra import hydra_main, register_boundary_validator

register_boundary_validator("plcs.association", validate_association_config)


@hydra_main(
    config_path="../configs",
    config_name="train_association",
    version_base="1.3",
    validation_boundary="plcs.association",
)
def main(config: DictConfig) -> None:
    PLCSTrainingRunner().run(config)


if __name__ == "__main__":
    main()
