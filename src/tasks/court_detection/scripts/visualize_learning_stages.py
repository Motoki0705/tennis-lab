"""Render the court architecture and measured B-run learning curves on CPU.

Usage:
    python -m src.tasks.court_detection.scripts.visualize_learning_stages
    python -m src.tasks.court_detection.scripts.visualize_learning_stages paths.output_root=/absolute/outputs
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.court_detection.visualization.presentation.configuration import (
    build_request,
)
from src.tasks.court_detection.visualization.presentation.learning_stages import (
    render_learning_stages,
)
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs",
    config_name="visualize_learning_stages",
    version_base="1.3",
    validation_boundary="court_detection.learning_stages",
)
def main(config: DictConfig) -> int:  # pragma: no cover - Hydra CLI
    print(render_learning_stages(build_request(config)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    main()
