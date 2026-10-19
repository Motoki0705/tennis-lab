"""Export ball learning stages without training, inference or checkpoint loads.

Usage:
    python -m src.tasks.ball_detection.scripts.visualize_learning_stages input_panel=schematic
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.ball_detection.visualization.presentation.configuration import (
    build_learning_stages_config,
)
from src.tasks.ball_detection.visualization.presentation.learning_stages import (
    run_learning_stages,
)
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs",
    config_name="visualize_learning_stages",
    version_base="1.3",
    validation_boundary="ball.learning_stages",
)
def main(config: DictConfig) -> int:
    runtime = build_learning_stages_config(config)
    run_learning_stages(runtime, config)
    print(f"presentation={runtime.output_dir}")
    return 0


if __name__ == "__main__":
    main()
