"""Train PLCS world-joint residuals from triangulated observations."""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.base.triangulation_residual.configuration import validate_config
from src.tasks.base.triangulation_residual.training import ResidualTrainingRunner
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs",
    config_name="train_triangulation_residual",
    version_base="1.3",
    validation_boundary="plcs.triangulation_residual.train",
)
def main(config: DictConfig) -> None:
    """Validate the PLCS contract and execute the shared training runner."""
    validate_config(config, expected_task="plcs")
    ResidualTrainingRunner().run(config)


if __name__ == "__main__":
    main()
