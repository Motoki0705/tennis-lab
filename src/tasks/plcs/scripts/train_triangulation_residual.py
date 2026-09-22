"""Train PLCS world-joint residuals from triangulated observations."""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.plcs.configuration import validate_residual_config
from src.tasks.plcs.training.runner import PLCSTrainingRunner
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs",
    config_name="train_triangulation_residual",
    version_base="1.3",
    validation_boundary="plcs.triangulation_residual.train",
)
def main(config: DictConfig) -> None:
    """Validate the PLCS contract and execute the shared training runner."""
    validate_residual_config(config)
    PLCSTrainingRunner().run(config)


if __name__ == "__main__":
    main()
