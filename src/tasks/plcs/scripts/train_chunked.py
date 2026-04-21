"""Train the PLCS model with background-generated chunk rotation.

Usage:
    python -m src.tasks.plcs.scripts.train_chunked
    python -m src.tasks.plcs.scripts.train_chunked data.chunk.scenes_per_chunk=500
    python -m src.tasks.plcs.scripts.train_chunked run.category=forehand

Notes:
    - Configuration is loaded from `src/tasks/plcs/configs/train_chunked.yaml`.
    - The script uses Hydra for configuration loading.
    - Training chunks are generated on the fly and stored under `data/plcs/chunks/` by default.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.tasks.plcs.training.runner import PLCSTrainingRunner


@hydra.main(config_path="../configs", config_name="train_chunked", version_base="1.3")
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for chunked PLCS training."""
    runner = PLCSTrainingRunner()
    runner.run(config)


if __name__ == "__main__":
    main()