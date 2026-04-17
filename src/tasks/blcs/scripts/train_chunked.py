"""Train a BLCS model with background-generated chunk rotation.

Training data is generated in the background as NPZ scene chunks.  Each chunk
is used for a configurable number of epochs before rotating to the next one.
Validation and test sets remain fixed from ``data/blcs/``.

Usage:
    python -m src.tasks.blcs.scripts.train_chunked
    python -m src.tasks.blcs.scripts.train_chunked data.chunk.scenes_per_chunk=500
    python -m src.tasks.blcs.scripts.train_chunked data.chunk.epochs_per_chunk=5

Notes:
    - Hydra loads configuration from ``src/tasks/blcs/configs/train_chunked.yaml``.
    - GeneratorConfig is assembled from the same config groups used by ``generate_dataset.py``.
    - Chunks are stored under ``data/blcs/chunks/`` and deleted after use.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.tasks.blcs.scripts.generate_dataset import build_generator_config
from src.tasks.blcs.training.runner import BLCSTrainingRunner


@hydra.main(config_path="../configs", config_name="train_chunked", version_base="1.3")
def main(config: DictConfig) -> None:
    """Hydra entry point for chunked BLCS training."""
    generator_config = build_generator_config(config)
    runner = BLCSTrainingRunner(generator_config=generator_config)
    runner.run(config)


if __name__ == "__main__":
    main()
