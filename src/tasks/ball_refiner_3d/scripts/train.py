"""Train one coordinate refiner condition; use the shared queue for CUDA."""

from omegaconf import DictConfig

from src.tasks.ball_refiner_3d.configuration.training import validate_training
from src.tasks.ball_refiner_3d.training.runner import run_training
from src.utils.hydra import hydra_main, register_boundary_validator

register_boundary_validator("ball_refiner_3d.train", validate_training)


@hydra_main(
    version_base=None,
    config_path="../configs",
    config_name="train",
    validation_boundary="ball_refiner_3d.train",
)
def main(config: DictConfig) -> None:
    run_training(config)


if __name__ == "__main__":
    main()
