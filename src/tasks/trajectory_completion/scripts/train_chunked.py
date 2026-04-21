"""Train the trajectory-completion model with background-generated chunk rotation.

Usage:
    python -m src.tasks.trajectory_completion.scripts.train_chunked
    python -m src.tasks.trajectory_completion.scripts.train_chunked data.chunk.scenes_per_chunk=500
    python -m src.tasks.trajectory_completion.scripts.train_chunked data.generator_device=cuda

Notes:
    - Configuration is loaded from `src/tasks/trajectory_completion/configs/train_chunked.yaml`.
    - The script uses Hydra for configuration loading.
    - Training chunks are generated from the default BLCS generator configuration.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.tasks.trajectory_completion.training.runner import (
    TrajectoryCompletionTrainingRunner,
)


@hydra.main(config_path="../configs", config_name="train_chunked", version_base="1.3")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for chunked trajectory-completion training."""
    runner = TrajectoryCompletionTrainingRunner()
    runner.run(cfg)


if __name__ == "__main__":
    main()