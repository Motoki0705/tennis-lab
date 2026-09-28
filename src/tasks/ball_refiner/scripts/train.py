"""Run the detector-only pilot; CUDA execution requires the shared training queue."""

from omegaconf import DictConfig

from src.tasks.ball_refiner.training.configuration import validate_training_boundary
from src.tasks.ball_refiner.training.runner import run_training
from src.utils.hydra import hydra_main, register_boundary_validator

register_boundary_validator("ball_refiner.train", validate_training_boundary)


@hydra_main(version_base=None, config_path="../configs", config_name="train", validation_boundary="ball_refiner.train")
def main(config: DictConfig) -> None:
    run_training(config)


if __name__ == "__main__":
    main()
