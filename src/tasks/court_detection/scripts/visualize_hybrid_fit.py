"""Search measured KP-only failures and render a selected hybrid court fit."""

from omegaconf import DictConfig

from src.tasks.court_detection.visualization.presentation.hybrid_fit_runner import (
    run_hybrid_fit,
)
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs",
    config_name="visualize_hybrid_fit",
    version_base="1.3",
    validation_boundary="court_detection.hybrid_fit",
)
def main(config: DictConfig) -> None:
    run_hybrid_fit(config)


if __name__ == "__main__":
    main()
