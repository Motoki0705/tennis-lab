"""Generate a shared, rally-disjoint coordinate dataset on CPU."""

from omegaconf import DictConfig

from src.tasks.ball_refiner_3d.configuration.generation import validate_generation
from src.tasks.ball_refiner_3d.generate_dataset.generator import generate_dataset
from src.utils.hydra import hydra_main, register_boundary_validator

register_boundary_validator("ball_refiner_3d.generate_dataset", validate_generation)


@hydra_main(
    version_base=None,
    config_path="../configs",
    config_name="generate_dataset",
    validation_boundary="ball_refiner_3d.generate_dataset",
)
def main(config: DictConfig) -> None:
    generate_dataset(config)


if __name__ == "__main__":
    main()
