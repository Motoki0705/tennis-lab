"""Generate a shared, rally-disjoint coordinate dataset on CPU."""

from omegaconf import DictConfig

from src.tasks.ball_refiner.coordinates.generation import (
    generate_dataset,
    validate_generation,
)
from src.utils.hydra import hydra_main, register_boundary_validator

register_boundary_validator("ball_refiner.generate_coordinates", validate_generation)


@hydra_main(version_base=None, config_path="../configs", config_name="generate_coordinates", validation_boundary="ball_refiner.generate_coordinates")
def main(config: DictConfig) -> None:
    generate_dataset(config)


if __name__ == "__main__":
    main()
